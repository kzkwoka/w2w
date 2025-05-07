import torch

from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader

from sae import Sae

path = "weights2weights/weights_datasets"
device = "cuda"

sae_path = "runs/run_lat5000_bs512_auxk0.03_k256_epoch10_1381189_0/0"
sae = Sae.load_from_disk(sae_path,device=device)
sae.eval()

d_path = f"{path}/full"
dataset = Dataset.load_from_disk(d_path, keep_in_memory=False)
dl = DataLoader(
            dataset,
            batch_size=1024,
            shuffle=False
        )

n_samples = len(dataset)
activations_tensor = torch.empty((n_samples, 5000), dtype=torch.float32)
filenames = [None] * n_samples

offset=0
with torch.no_grad():
    for batch in tqdm(dl):
        data_batch = torch.stack(batch['data'], dim=1).to(device)

        output_batch = sae.pre_acts(data_batch.to(device))
        # #TODO: maybe no need to extend, rather collect differently
        # activations.extend(zip(filenames, output_batch))
        
        bsz = output_batch.size(0)

        activations_tensor[offset:offset + bsz] = output_batch
        filenames[offset:offset + bsz] = batch["filename"]

        offset += bsz

activations = dict(zip(filenames, activations_tensor.cpu()))

torch.save(activations, f"{path}/activations_crosscoder.pt")