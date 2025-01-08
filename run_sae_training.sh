#!/bin/bash
#SBATCH --account=plggenerativepw2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G

# Load Miniconda module
module load Miniconda3/23.3.1-0

# Initialize Conda in bash and activate your environment
eval "$(conda shell.bash hook)"
conda activate $SCRATCH/w2w

# Navigate to your working directory
cd $SCRATCH

# Print start time
echo "Starting SAE runs at $(date)"

# Run the Python command with different values
# for num_latents in 1000 5000 10000 30000; do
#     for batch_size in 8 32 256; do
#         for auxk_alpha in 0.0 0.03 0.1 0.3; do
#             echo "Running with --num_latents=$num_latents --batch_size=$batch_size --auxk_alpha=$auxk_alpha"
#             python -m sae --num_latents=$num_latents --batch_size=$batch_size --auxk_alpha=$auxk_alpha --run_name=run_${num_latents}_bs${batch_size}_auxk${auxk_alpha}
#             echo "Finished run with --num_latents=$num_latents, batch_size=$batch_size, auxk_alpha=$auxk_alpha at $(date)"
#         done
#     done
# done

for num_latents in 1000 5000 10000; do
    for batch_size in 256 4096; do
        for auxk_alpha in 0.0 0.03 0.1; do
            for k in 128 512; do #skip default 32
                echo "Running with --num_latents=$num_latents --batch_size=$batch_size --auxk_alpha=$auxk_alpha --num_epochs=10 --k=$k"
                python -m sae \
                    --num_latents=$num_latents \
                    --batch_size=$batch_size \
                    --auxk_alpha=$auxk_alpha \
                    --run_name=run_lat${num_latents}_bs${batch_size}_auxk${auxk_alpha} \
                    --k=$k \
                    --num_epochs=10 \
                    --lr_warmup_steps=50 \
                    --dead_feature_threshold=30_000
                echo "Finished run with --num_latents=$num_latents, batch_size=$batch_size, auxk_alpha=$auxk_alpha, num_epochs=10, k=$k at $(date)"
            done
        done
    done
done

# Print end time
echo "All runs completed at $(date)"
