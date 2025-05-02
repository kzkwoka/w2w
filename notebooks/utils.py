import os
import torch
from copy import deepcopy

from safetensors.torch import load_file
from diffusers import DiffusionPipeline 
from peft import PeftModel

from sae import Sae
base_path = "weights2weights/weights_datasets"
device = "cuda"

def load_sae(path):
    sae = Sae.load_from_disk(path, device=device)
    sae.eval()
    return sae

def load_diffusion():
    model_path = os.path.join(os.getcwd(), "base_model")
    base_weights = load_file(f"{model_path}/unet/adapter_model.safetensors")

    pipe = DiffusionPipeline.from_pretrained("stablediffusionapi/realistic-vision-v51", 
                                            torch_dtype=torch.float16,safety_checker = None,
                                            requires_safety_checker = False).to(device)
    
    pipe.unet = PeftModel.from_pretrained(pipe.unet, f"{model_path}/unet", adapter_name="identity1")
    pipe.unet.load_state_dict(base_weights, strict = False)
    pipe.to(device)
    return base_weights, pipe

def get_diffusion_params(pipe):
    generator = torch.Generator(device=device)
    generator = generator.manual_seed(5)
    latents = torch.randn(
            (1, pipe.unet.in_channels, 512 // 8, 512 // 8),
            generator = generator,
            device = device
        ).half()
    
    #inference parameters
    prompt = "sks person" #"sks person in a hat in a forest"
    negative_prompt = "low quality, blurry, unfinished"
    guidance_scale = 3.0
    ddim_steps = 50

    return prompt, ddim_steps, guidance_scale, negative_prompt, latents

def array_to_dict(weights, base_weights):
    new_weights = deepcopy(base_weights)
    start_idx = 0
    for key in base_weights.keys():
        shape = base_weights[key].shape
        end_idx = start_idx + torch.prod(torch.tensor(shape)).item()  # Calculate end index
        new_weights[key] = weights[start_idx:end_idx].reshape(*shape)
        start_idx = end_idx
    return new_weights

def get_error(sae, sample, base_weights, with_weights=False):
    gt_weights = array_to_dict(sample, base_weights)

    weights_out = sae(sample).sae_out
    rec_weights = array_to_dict(weights_out, base_weights)

    err = {key: gt_weights[key].cpu() - rec_weights[key].cpu() for key in base_weights}
    if with_weights:
        return err, gt_weights, rec_weights
    
    return err

def get_image(weights, pipe, prompt, ddim_steps, guidance_scale, negative_prompt,latents):
    pipe.unet.load_state_dict(weights, strict = False)
    pipe.to(device)
    image = pipe(prompt, num_inference_steps=ddim_steps, guidance_scale=guidance_scale,  negative_prompt=negative_prompt, latents = latents).images[0]
    return image

def add_error(err, weights):
    return {key: weights[key].cpu() + err[key] for key in err}


#---------------------------------------- norm utils ---------------
def get_group_indices(df, group_columns=["block_type","block_number"], return_names=False, device='cuda'):
    df = df.sort_values(by=group_columns + ["start"])
    group_indices = []
    group_names = []

    for group_vals, group_df in df.groupby(group_columns):
        group_name = ".".join(map(str, group_vals))  # e.g., "attn_proj.lora_ab"
        cols = []
        for _, row in group_df.iterrows():
            cols.extend(range(row['start'], row['end']))
        group_indices.append(torch.tensor(cols, dtype=torch.long, device=device))
        group_names.append(group_name)
    
    if return_names:
        return group_indices, group_names
    
    return group_indices

def get_group_norms(group_indices, W_dec, input_group_norms):
    group_norms = []

    for i, idx in enumerate(group_indices):
        # Efficiently select columns using precomputed index tensor
        group_block = torch.index_select(W_dec, dim=1, index=idx)
        group_len = idx.numel()
        # Normalize by length
        norm = group_block.norm(dim=1) / (group_len ** 0.5)
        # Normalize by input data
        if input_group_norms is not None:
            norm /= input_group_norms[i]
        group_norms.append(norm)

    return torch.stack(group_norms, dim=1)