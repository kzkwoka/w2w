from collections import defaultdict
from dataclasses import asdict
from typing import Sized
from matplotlib import pyplot as plt

import torch
import torch.distributed as dist
from datasets import Dataset as HfDataset
from fnmatch import fnmatchcase
from natsort import natsorted
from safetensors.torch import load_model
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, get_linear_schedule_with_warmup

from sae.metrics import calculate_similarity, extract_input_batch, generate_images, get_latents, unflatten_batch, update_extracted_batch

from .config import TrainConfig
from .data import MemmapDataset
from .sae import Sae
from .utils import geometric_median, get_layer_list, resolve_widths

plt.style.use("seaborn-v0_8-colorblind")
class SaeTrainer:
    def __init__(
        self, cfg: TrainConfig, dataset: HfDataset | MemmapDataset, eval_dataset: Tensor, 
    ):

        self.cfg = cfg
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.distribute_modules()

        assert isinstance(dataset, Sized)
        num_examples = len(dataset)
        self.training_steps = (num_examples // cfg.batch_size) * cfg.num_epochs

        self.device = self.cfg.device
        input_widths = self.cfg.input_width

        self.saes = {
            0: Sae(input_widths, cfg.sae, self.device)
        }
        # Re-initialize the decoder for transcoder training. By default the Sae class
        # initializes the decoder with the transpose of the encoder.
        if cfg.transcode:
            for sae in self.saes.values():
                torch.nn.init.kaiming_uniform_(sae.W_dec, a=5**0.5)

        pgs = [
            {
                "params": sae.parameters(),
                # Auto-select LR using 1 / sqrt(d) scaling law from Fig 3 of the paper
                "lr": cfg.lr or 2e-4 / (sae.num_latents / (2**14)) ** 0.5
            }
            for sae in self.saes.values()
        ]
        # Dedup the learning rates we're using, sort them, round to 2 decimal places
        lrs = [f"{lr:.2e}" for lr in sorted(set(pg["lr"] for pg in pgs))]
        print(f"Learning rates: {lrs}" if len(lrs) > 1 else f"Learning rate: {lrs[0]}")

        try:
            from bitsandbytes.optim import Adam8bit as Adam

            print("Using 8-bit Adam from bitsandbytes")
        except ImportError:
            from torch.optim import Adam

            print("bitsandbytes 8-bit Adam not available, using torch.optim.Adam")
            print("Run `pip install bitsandbytes` for less memory usage.")

        self.global_step = 0
        self.num_tokens_since_fired = {
            name: torch.zeros(sae.num_latents, device=self.device, dtype=torch.long)
            for name, sae in self.saes.items()
        }
        self.optimizer = Adam(pgs)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, cfg.lr_warmup_steps, self.training_steps
        )

    def load_state(self, path: str):
        """Load the trainer state from disk."""
        device = self.device

        # Load the train state first so we can print the step number
        train_state = torch.load(f"{path}/state.pt", map_location=device, weights_only=True)
        self.global_step = train_state["global_step"]
        self.num_tokens_since_fired = train_state["num_tokens_since_fired"]

        print(f"\033[92mResuming training at step {self.global_step} from '{path}'\033[0m")

        lr_state = torch.load(f"{path}/lr_scheduler.pt", map_location=device, weights_only=True)
        opt_state = torch.load(f"{path}/optimizer.pt", map_location=device, weights_only=True)
        self.optimizer.load_state_dict(opt_state)
        self.lr_scheduler.load_state_dict(lr_state)

        for name, sae in self.saes.items():
            load_model(sae, f"{path}/{name}/sae.safetensors", device=str(device))

    def fit(self):
        # Use Tensor Cores even for fp32 matmuls
        torch.set_float32_matmul_precision("high")

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        ddp = dist.is_initialized() and not self.cfg.distribute_modules

        if self.cfg.log_to_wandb and rank_zero:
            try:
                import wandb

                wandb.init(
                    name=self.cfg.run_name,
                    project="sae",
                    config=asdict(self.cfg),
                    save_code=True,
                )
            except ImportError:
                print("Weights & Biases not installed, skipping logging.")
                self.cfg.log_to_wandb = False

        num_sae_params = sum(
            p.numel() for s in self.saes.values() for p in s.parameters()
        )
        # num_model_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of SAE parameters: {num_sae_params:_}")
        # print(f"Number of model parameters: {num_model_params:_}")

        num_batches = self.training_steps
        if self.global_step > 0:
            assert hasattr(self.dataset, "select"), "Dataset must implement `select`"

            n = self.global_step * self.cfg.batch_size
            ds = self.dataset.select(range(n, len(self.dataset)))  # type: ignore
        else:
            ds = self.dataset

        device = self.device
        dl = DataLoader(
            ds, # type: ignore
            batch_size=self.cfg.batch_size,
            # NOTE: We do not shuffle here for reproducibility; the dataset should
            # be shuffled before passing it to the trainer.
            shuffle=False,
        )
        
        eval_weights = unflatten_batch(self.eval_dataset)
        #TODO: fix passing this parameter
        keys, weights_in, shapes = extract_input_batch(eval_weights, None ) # keyword="mid_block")
        latents = get_latents(self.device)
        images0 = generate_images(eval_weights, latents, device)
        
        pbar = tqdm(
            desc="Training", 
            disable=not rank_zero, 
            initial=self.global_step, 
            total=num_batches,
        )

        did_fire = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.bool)
            for name, sae in self.saes.items()
        }
        num_tokens_in_step = 0

        # For logging purposes
        avg_auxk_loss = defaultdict(float)
        avg_fvu = defaultdict(float)
        avg_multi_topk_fvu = defaultdict(float)

        input_dict: dict[str, Tensor] = {}
        output_dict: dict[str, Tensor] = {}
        
        maybe_wrapped: dict[str, DDP] | dict[str, Sae] = {}
        frac_active_list = []  # track active features

        for _ in range(self.cfg.num_epochs):
            
            for batch in dl:
                input_dict.clear()
                output_dict.clear()
                
                data = torch.tensor(batch["data"]) if not isinstance(batch["data"], torch.Tensor) else batch["data"]
                
                # Bookkeeping for dead feature detection
                num_tokens_in_step += data.shape[0]
                
                # Load data to input and output dict
                input_dict = {0: data.to(self.device)}
                output_dict = {0: data.to(self.device)}

                if self.cfg.distribute_modules:
                    input_dict = self.scatter_hiddens(input_dict)
                    output_dict = self.scatter_hiddens(output_dict)

                for name, outputs in output_dict.items():
                    # 'inputs' is distinct from outputs iff we're transcoding
                    inputs = input_dict.get(name, outputs)
                    raw = self.saes[name]           # 'raw' never has a DDP wrapper

                    # On the first iteration, initialize the decoder bias
                    if self.global_step == 0:
                        # NOTE: The all-cat here could conceivably cause an OOM in some
                        # cases, but it's unlikely to be a problem with small world sizes.
                        # We could avoid this by "approximating" the geometric median
                        # across all ranks with the mean (median?) of the geometric medians
                        # on each rank. Not clear if that would hurt performance.
                        median = geometric_median(self.maybe_all_cat(outputs))
                        raw.b_dec.data = median.to(raw.dtype)

                    if not maybe_wrapped:
                        # Wrap the SAEs with Distributed Data Parallel. We have to do this
                        # after we set the decoder bias, otherwise DDP will not register
                        # gradients flowing to the bias after the first step.
                        maybe_wrapped = (
                            {
                                name: DDP(sae, device_ids=[dist.get_rank()])
                                for name, sae in self.saes.items()
                            }
                            if ddp
                            else self.saes
                        )

                    # Make sure the W_dec is still unit-norm if we're autoencoding
                    if raw.cfg.normalize_decoder and not self.cfg.transcode:
                        raw.set_decoder_norm_to_unit_norm()

                    acc_steps = self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
                    denom = acc_steps * self.cfg.wandb_log_frequency
                    wrapped = maybe_wrapped[name]

                    # Save memory by chunking the activations
                    in_chunks = inputs.chunk(self.cfg.micro_acc_steps)
                    out_chunks = outputs.chunk(self.cfg.micro_acc_steps)
                    for in_chunk, out_chunk in zip(in_chunks, out_chunks):
                        out = wrapped(
                            x=in_chunk,
                            y=out_chunk,
                            dead_mask=(
                                self.num_tokens_since_fired[name]
                                > self.cfg.dead_feature_threshold
                                if self.cfg.auxk_alpha > 0
                                else None
                            ),
                        )

                        avg_fvu[name] += float(
                            self.maybe_all_reduce(out.fvu.detach()) / denom
                        )
                        if self.cfg.auxk_alpha > 0:
                            avg_auxk_loss[name] += float(
                                self.maybe_all_reduce(out.auxk_loss.detach()) / denom
                            )
                        if self.cfg.sae.multi_topk:
                            avg_multi_topk_fvu[name] += float(
                                self.maybe_all_reduce(out.multi_topk_fvu.detach()) / denom
                            )

                        loss = out.fvu + self.cfg.auxk_alpha * out.auxk_loss + out.multi_topk_fvu / 8
                        loss.div(acc_steps).backward()

                        # Update the did_fire mask
                        did_fire[name][out.latent_indices.flatten()] = True
                        self.maybe_all_reduce(did_fire[name], "max")  # max is boolean "any"

                    # Clip gradient norm independently for each SAE
                    torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)

                # Check if we need to actually do a training step
                step, substep = divmod(self.global_step + 1, self.cfg.grad_acc_steps)
                if substep == 0:
                    if self.cfg.sae.normalize_decoder and not self.cfg.transcode:
                        for sae in self.saes.values():
                            sae.remove_gradient_parallel_to_decoder_directions()

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()

                    ###############
                    with torch.no_grad():
                        # Update the dead feature mask
                        for name, counts in self.num_tokens_since_fired.items():
                            counts += num_tokens_in_step
                            counts[did_fire[name]] = 0

                        # Reset stats for this step
                        num_tokens_in_step = 0
                        for mask in did_fire.values():
                            mask.zero_()

                    if (
                        self.cfg.log_to_wandb
                        and (step + 1) % self.cfg.wandb_log_frequency == 0
                    ):
                        info = {}

                        for name in self.saes:
                            mask = (
                                self.num_tokens_since_fired[name]
                                > self.cfg.dead_feature_threshold
                            )
                            fire_count = torch.zeros(
                                self.saes[name].num_latents, dtype=torch.long
                            )
                            unique, unique_counts = torch.unique(
                                out.latent_indices.flatten(),
                                return_counts=True,
                            )
                            fire_count[unique] = unique_counts.cpu()
                            frac_active_list.append(fire_count)

                            if len(frac_active_list) > self.cfg.feature_sampling_window:
                                frac_active_in_window = torch.stack(
                                    frac_active_list[
                                        -self.cfg.feature_sampling_window :
                                    ],
                                    dim=0,
                                )
                                feature_sparsity = frac_active_in_window.sum(0) / (
                                    self.cfg.feature_sampling_window
                                    * self.cfg.batch_size
                                )
                            else:
                                frac_active_in_window = torch.stack(
                                    frac_active_list, dim=0
                                )
                                feature_sparsity = frac_active_in_window.sum(0) / (
                                    len(frac_active_list) * self.cfg.batch_size
                                )

                            log_feature_sparsity = torch.log10(feature_sparsity + 1e-8)

                            info.update(
                                {
                                    f"fvu/{name}": avg_fvu[name],
                                    f"dead_pct/{name}": mask.mean(
                                        dtype=torch.float32
                                    ).item(),
                                }
                            )
                            if self.cfg.auxk_alpha > 0:
                                info[f"auxk/{name}"] = avg_auxk_loss[name]
                            if self.cfg.sae.multi_topk:
                                info[f"multi_topk_fvu/{name}"] = avg_multi_topk_fvu[name]
                            if (step + 1) % (self.cfg.wandb_log_frequency * 10) == 0:
                                plt.hist(
                                    log_feature_sparsity.tolist(),
                                    bins=50,
                                    color="blue",
                                    alpha=0.7,
                                )
                                plt.title("Feature Density")
                                plt.xlabel("Log Feature Density")
                                plt.tight_layout()
                                info[f"feature_density/{name}"] = wandb.Image(plt.gcf())
                                plt.close()
                                
                            if (step == 0) or ((step + 1) % (self.cfg.wandb_log_frequency * (10 if self.training_steps < 600 else 100)) == 0):
                                self.saes[name].eval()
                                weights_out = self.saes[name](weights_in.to(self.device)).sae_out
                                self.saes[name].train()
                            
                                new_weights = update_extracted_batch(eval_weights, weights_out, keys, shapes)
                                
                                images1 = generate_images(new_weights, latents, device)
                                mse, lpips = calculate_similarity(images0, images1)

                                info[f"eval_mse/{name}"] = mse
                                info[f"eval_lpips/{name}"] = lpips

                                for i, (img0, img1) in enumerate(zip(images0, images1)):
                                    img0 = img0.permute(1, 2, 0).numpy()  # Rearrange to [Height, Width, Channels]
                                    img1 = img1.permute(1, 2, 0).numpy()

                                    plt.subplot(1, 2, 1)
                                    plt.imshow(img0)
                                    plt.title("Original")
                                    plt.axis('off')
                                    plt.subplot(1, 2, 2)
                                    plt.imshow(img1)
                                    plt.title("Reconstructed")
                                    plt.axis('off')

                                    plt.tight_layout()
                                    info[f"reconstruction_image/{name}/{i}"] = wandb.Image(plt.gcf())
                                    plt.close()

                        avg_auxk_loss.clear()
                        avg_fvu.clear()
                        avg_multi_topk_fvu.clear()

                        if self.cfg.distribute_modules:
                            outputs = [{} for _ in range(dist.get_world_size())]
                            dist.gather_object(info, outputs if rank_zero else None)
                            info.update({k: v for out in outputs for k, v in out.items()})

                        if rank_zero:
                            wandb.log(info, step=step)

                    if (step + 1) % self.cfg.save_every == 0:
                        self.save()
                    
                self.global_step += 1
                pbar.update()

        self.save()
        pbar.close()

    def local_hookpoints(self) -> list[str]:
        return (
            self.module_plan[dist.get_rank()]
            if self.module_plan
            else self.cfg.hookpoints
        )

    def maybe_all_cat(self, x: Tensor) -> Tensor:
        """Concatenate a tensor across all processes."""
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        buffer = x.new_empty([dist.get_world_size() * x.shape[0], *x.shape[1:]])
        dist.all_gather_into_tensor(buffer, x)
        return buffer

    def maybe_all_reduce(self, x: Tensor, op: str = "mean") -> Tensor:
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        if op == "sum":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        elif op == "mean":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            x /= dist.get_world_size()
        elif op == "max":
            dist.all_reduce(x, op=dist.ReduceOp.MAX)
        else:
            raise ValueError(f"Unknown reduction op '{op}'")

        return x

    def distribute_modules(self):
        """Prepare a plan for distributing modules across ranks."""
        if not self.cfg.distribute_modules:
            self.module_plan = []
            print(f"Training on modules: {self.cfg.hookpoints}")
            return

        layers_per_rank, rem = divmod(len(self.cfg.hookpoints), dist.get_world_size())
        assert rem == 0, "Number of modules must be divisible by world size"

        # Each rank gets a subset of the layers
        self.module_plan = [
            self.cfg.hookpoints[start : start + layers_per_rank]
            for start in range(0, len(self.cfg.hookpoints), layers_per_rank)
        ]
        for rank, modules in enumerate(self.module_plan):
            print(f"Rank {rank} modules: {modules}")

    def scatter_hiddens(self, hidden_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Scatter & gather the hidden states across ranks."""
        # Short-circuit if we have no data
        if not hidden_dict:
            return hidden_dict

        outputs = [
            # Add a new leading "layer" dimension to each tensor
            torch.stack([hidden_dict[hook] for hook in hookpoints], dim=1)
            for hookpoints in self.module_plan
        ]
        local_hooks = self.module_plan[dist.get_rank()]
        shape = next(iter(hidden_dict.values())).shape

        # Allocate one contiguous buffer to minimize memcpys
        buffer = outputs[0].new_empty(
            # The (micro)batch size times the world size
            shape[0] * dist.get_world_size(),
            # The number of layers we expect to receive
            len(local_hooks),
            # All other dimensions
            *shape[1:],
        )

        # Perform the all-to-all scatter
        inputs = buffer.split([len(output) for output in outputs])
        dist.all_to_all([x for x in inputs], outputs)

        # Return a list of results, one for each layer
        return {hook: buffer[:, i] for i, hook in enumerate(local_hooks)}

    def save(self):
        """Save the SAEs to disk."""

        path = f"runs/{self.cfg.run_name}" or "sae-ckpts"
        rank_zero = not dist.is_initialized() or dist.get_rank() == 0

        if rank_zero or self.cfg.distribute_modules:
            print("Saving checkpoint")

            for hook, sae in self.saes.items():
                assert isinstance(sae, Sae)

                sae.save_to_disk(f"{path}/{hook}")
    
        if rank_zero:
            torch.save(self.lr_scheduler.state_dict(), f"{path}/lr_scheduler.pt")
            torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")
            torch.save({
                "global_step": self.global_step,
                "num_tokens_since_fired": self.num_tokens_since_fired,
            }, f"{path}/state.pt")

            self.cfg.save_json(f"{path}/config.json")

        # Barrier to ensure all ranks have saved before continuing
        if dist.is_initialized():
            dist.barrier()
