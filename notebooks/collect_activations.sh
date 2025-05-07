#!/bin/bash
#SBATCH --account=plgdiffusion-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --output=logs/slurm-%j.out

# Load Miniconda module
module load Miniconda3/23.3.1-0

# Initialize Conda in bash and activate your environment
eval "$(conda shell.bash hook)"
conda activate $SCRATCH/w2w

# Navigate to your working directory
cd $SCRATCH

python notebooks/collect_activations.py