# Setup

## Initial setup on Athena
```bash
module load Miniconda3/23.3.1-0 
eval "$(conda shell.bash hook)"
conda create -p $SCRATCH/w2w -y
conda activate $SCRATCH/w2w
conda install python==3.12.3 -y
cd $SCRATCH
pip install -r requirements.txt

git clone https://github.com/snap-research/weights2weights.git
# git clone https://huggingface.co/snap-research/weights2weights hfweights2weights 

cache_path=$(python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id='snap-research/weights2weights', cache_dir='/net/tscratch/people/plgkingak/.cache')
print(path)
")

cd weights2weights
ln -s $cache_path/files ./files
ln -s $cache_path/weights_datasets ./weights_datasets


```

## Working in interactive mode
### Requesting a node
```bash
srun --account=plggenerativepw2-gpu-a100 -p plgrid-gpu-a100 --nodes=1 --ntasks-per-node=1 --time=01:00:00 --gres gpu:1 --mem 40G --pty bash -i 
```

### On the allocated node
```bash
module load Miniconda3/23.3.1-0 
eval "$(conda shell.bash hook)"
conda activate $SCRATCH/w2w
cd $SCRATCH
```
#### OPTIONAL: Running Jupyter Server
```bash
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
```