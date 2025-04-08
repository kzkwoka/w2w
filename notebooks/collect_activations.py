import torch

from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader

from sae import Sae

path = "weights2weights/weights_datasets"
device = "cuda"

sae_path = "runs/test_feature_logging/0"
sae = Sae.load_from_disk(sae_path,device=device)
sae.eval()

d_path = f"{path}/full"
dataset = Dataset.load_from_disk(d_path, keep_in_memory=False)
dl = DataLoader(
            dataset,
            batch_size=1024,
            shuffle=False
        )

activations = []

with torch.no_grad():
    for batch in tqdm(dl):
        filenames = batch['filename']
        data_batch = torch.stack(batch['data'], dim=1).to(device)

        output_batch = sae.pre_acts(data_batch)
        activations.extend(zip(filenames, output_batch))

activations = dict(activations)
torch.save(activations, f"{path}/activations.pt")