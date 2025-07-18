import os
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from multiprocessing import cpu_count
from safetensors.torch import load_model

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from simple_parsing import field, parse
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel

from .data import chunk_and_tokenize, MemmapDataset
from .trainer import SaeTrainer, TrainConfig


@dataclass
class RunConfig(TrainConfig):
    device: str = "cuda"
    """Device to be used for training."""
    
    num_epochs: int=1
    """Epochs for training"""
    
    dataset: str = field(
        default="/net/tscratch/people/plgkingak/weights2weights/weights_datasets/mid_block",
        positional=True,
    )
    """Path to the dataset to use for training."""
    
    eval_dataset: str = field(
        default="/net/tscratch/people/plgkingak/weights2weights/weights_datasets/evaluation_samples.pt",
        positional=True,
    )
    """Path to the dataset to use for evaluating reconstruction."""
    
    normalized: bool=True
    """Whether training data used is normalized"""
    
    normalization_params: str = field(
        default="/net/tscratch/people/plgkingak/weights2weights/weights_datasets/normalization",
    )
    """Path to normalization mean and std needed in evaluating reconstruction"""

    input_width: int = 9728
    """Input tensor length to use for training."""

    split: str = "train"
    """Dataset split to use for training."""

    ctx_len: int = 2048
    """Context length to use for training."""

    hf_token: str | None = None
    """Huggingface API token for downloading models."""

    revision: str | None = None
    """Model revision to use for training."""

    load_in_8bit: bool = False
    """Load the model in 8-bit mode."""

    max_examples: int | None = None
    """Maximum number of examples to use for training."""

    resume: bool = False
    """Whether to try resuming from the checkpoint present at `run_name`."""

    finetune: str | None = None
    """Path to pretrained SAEs to finetune."""

    seed: int = 42
    """Random seed for shuffling the dataset."""

    data_preprocessing_num_proc: int = field(
        default_factory=lambda: cpu_count() // 2,
    )
    """Number of processes to use for preprocessing data"""


# def load_artifacts(args: RunConfig, rank: int) -> tuple[PreTrainedModel, Dataset | MemmapDataset]:
#     if args.load_in_8bit:
#         dtype = torch.float16
#     elif torch.cuda.is_bf16_supported():
#         dtype = torch.bfloat16
#     else:
#         dtype = "auto"

#     model = AutoModel.from_pretrained(
#         args.model,
#         device_map={"": f"cuda:{rank}"},
#         quantization_config=(
#             BitsAndBytesConfig(load_in_8bit=args.load_in_8bit)
#             if args.load_in_8bit
#             else None
#         ),
#         revision=args.revision,
#         torch_dtype=dtype,
#         token=args.hf_token,
#     )

#     # For memmap-style datasets
#     if args.dataset.endswith(".bin"):
#         dataset = MemmapDataset(args.dataset, args.ctx_len, args.max_examples)
#     else:
#         # For Huggingface datasets
#         try:
#             dataset = load_dataset(
#                 args.dataset,
#                 split=args.split,
#                 # TODO: Maybe set this to False by default? But RPJ requires it.
#                 trust_remote_code=True,
#             )
#         except ValueError as e:
#             # Automatically use load_from_disk if appropriate
#             if "load_from_disk" in str(e):
#                 dataset = Dataset.load_from_disk(args.dataset, keep_in_memory=False)
#             else:
#                 raise e

#         assert isinstance(dataset, Dataset)
#         if "input_ids" not in dataset.column_names:
#             tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
#             dataset = chunk_and_tokenize(
#                 dataset,
#                 tokenizer,
#                 max_seq_len=args.ctx_len,
#                 num_proc=args.data_preprocessing_num_proc,
#             )
#         else:
#             print("Dataset already tokenized; skipping tokenization.")

#         print(f"Shuffling dataset with seed {args.seed}")
#         dataset = dataset.shuffle(args.seed)

#         dataset = dataset.with_format("torch")
#         if limit := args.max_examples:
#             dataset = dataset.select(range(limit))

#     return model, dataset


def load_custom_dataset(args: RunConfig, eval=False, in_memory=False) -> Dataset | MemmapDataset:
    d = args.dataset if not eval else args.eval_dataset
    print(d)
    try:
        dataset = load_dataset(
            d,
            split=args.split,
            # TODO: Maybe set this to False by default? But RPJ requires it.
            trust_remote_code=True,
        )
    except ValueError as e:
        # Automatically use load_from_disk if appropriate
        if "load_from_disk" in str(e):
            dataset = Dataset.load_from_disk(d, keep_in_memory=in_memory)
        else:
            raise e

    assert isinstance(dataset, Dataset)
    print(f"Shuffling dataset with seed {args.seed}")
    dataset = dataset.shuffle(args.seed)

    dataset = dataset.with_format("torch")
    if limit := args.max_examples:
        dataset = dataset.select(range(limit))
    return dataset

def run():
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    args = parse(RunConfig)

    # Awkward hack to prevent other ranks from duplicating data preprocessing
    if not ddp or rank == 0:
        # model, dataset = load_artifacts(args, rank)
        dataset = load_custom_dataset(args)
        eval_dataset = torch.load(args.eval_dataset)
    if ddp:
        dist.barrier()
        if rank != 0:
            # model, dataset = load_artifacts(args, rank)
            dataset = load_custom_dataset(args)
            eval_dataset = torch.load(args.eval_dataset)
        dataset = dataset.shard(dist.get_world_size(), rank)

    # Prevent ranks other than 0 from printing
    with nullcontext() if rank == 0 else redirect_stdout(None):
        print(f"Training on '{args.dataset}' (split '{args.split}')")
        # print(f"Storing model weights in {model.dtype}")

        trainer = SaeTrainer(args, dataset, eval_dataset)
        if args.resume:
            trainer.load_state(args.run_name or "sae-ckpts")
        elif args.finetune:
            for name, sae in trainer.saes.items():
                load_model(sae, f"{args.finetune}/{name}/sae.safetensors", device=str(args.device))

        trainer.fit()


if __name__ == "__main__":
    run()
