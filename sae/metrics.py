from copy import deepcopy
import torch
import torchvision.transforms as transforms
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from diffusers import DiffusionPipeline 
from peft import PeftModel
from peft.utils.save_and_load import load_peft_weights

#inference parameters
prompt = "sks person" #"sks person in a hat in a forest"
negative_prompt = "low quality, blurry, unfinished"
guidance_scale = 3.0
ddim_steps = 50

def get_latents(device):
    #random seed generator
    generator = torch.Generator(device=device)
    generator = generator.manual_seed(5)
    latents = torch.randn(
            (1, 4, 512 // 8, 512 // 8),
            generator = generator,
            device = device
        ).half()
    return latents
    
def load_base_pipeline(device="cuda"):
    pipe = DiffusionPipeline.from_pretrained("stablediffusionapi/realistic-vision-v51", 
                                         torch_dtype=torch.float16,safety_checker = None,
                                         requires_safety_checker = False).to(device)
   
    pipe.unet = PeftModel.from_pretrained(pipe.unet, f"base_model/unet", adapter_name="identity1")
    return pipe
    # adapters_weights1 = load_peft_weights(f"{model_path}/unet", device="cuda:0")
    # pipe.unet.load_state_dict(adapters_weights1, strict = False)
    # pipe.to(device)

def generate_images(weights, latents, device="cuda"):
    images = []
    transform = transforms.ToTensor()
    pipe = load_base_pipeline(device)
    for row in weights:
        pipe.unet.load_state_dict(row, strict = False)
        pipe.to(device)
        images.append(transform(pipe(
            prompt, 
            num_inference_steps=ddim_steps, 
            guidance_scale=guidance_scale, 
            negative_prompt=negative_prompt, 
            latents=latents).images[0]))
    return torch.stack(images)
    
    
def calculate_similarity(images0, images1):
    mse = torch.mean((images0 - images1) ** 2)

    lpips = LPIPS(net_type='vgg', normalize=True)
    mlpips = lpips(images0, images1)
    return mse, mlpips


def unflatten_batch(batch):
    batch_unflattened = []
    for row in batch:
        batch_unflattened.append(unflatten(row.unsqueeze(0)))
    return batch_unflattened
    

def unflatten(flattened_weights, weight_dimensions_path="weights2weights/weights_datasets/weight_dimensions.pt"):
    weight_dimensions = torch.load(weight_dimensions_path)
    final_weights0 = {}
    counter = 0
    for key in weight_dimensions.keys():
        final_weights0[key] = flattened_weights[0, counter:counter+weight_dimensions[key][0][0]].unflatten(0, weight_dimensions[key][1])
        counter += weight_dimensions[key][0][0]
    
    #renaming keys to be compatible with Diffusers
    for key in list(final_weights0.keys()):
        final_weights0[key.replace( "lora_unet_", "base_model.model.").replace("A", "down").replace("B", "up").replace( "weight", "identity1.weight").replace("_lora", ".lora").replace("lora_down", "lora_A").replace("lora_up", "lora_B")] = final_weights0.pop(key)

    final_weights0_keys = sorted(final_weights0.keys())

    final_weights = {}
    for key in final_weights0_keys:
        final_weights[key] = final_weights0[key]

    return final_weights

def extract_input_batch(batch, keyword="mid_block"):
    batch_keys, batch_values, batch_shapes = [], [], []
    for row in batch:
        keys, values, shapes = extract_input(row, keyword)
        batch_keys.append(keys)
        batch_values.append(values)
        batch_shapes.append(shapes)
    return batch_keys, torch.stack(batch_values), batch_shapes

def extract_input(base_weights, keyword="mid_block"):
    keys = [k for k in base_weights.keys() if keyword in k]
    values = [base_weights[k].flatten() for k in keys]
    shapes = {k: base_weights[k].shape for k in keys}
    return keys, torch.cat(values), shapes

def update_extracted_batch(base_weights, weights_out, keys, shapes):
    new_weights = []
    for b, w, k, s in zip(base_weights, weights_out, keys, shapes):
        new_weights.append(update_extracted(b, w, k, s))
    return new_weights

def update_extracted(weights, weights_out, keys, shapes):
    new_weights = deepcopy(weights)
    start_idx = 0

    for key in keys:
        shape = shapes[key]
        end_idx = start_idx + torch.prod(torch.tensor(shape)).item()  # Calculate end index
        new_weights[key] = weights_out[start_idx:end_idx].reshape(*shape)
        start_idx = end_idx
    return new_weights