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

# Print start time
echo "Starting SAE runs at $(date)"
i=0

for num_latents in 5000; do
    for batch_size in 512; do
        for auxk_alpha in 0.03; do
            for per_block_alpha in 10; do
                for k in 256; do 
                    for epoch in 10; do
                        for grad in 1; do
                            for norm in 'l1' 'l2'; do
                                for dead in 30000; do
                                    echo "Running with --num_latents=$num_latents --batch_size=$batch_size --auxk_alpha=$auxk_alpha --num_epochs=10 --k=$k"
                                    python -m sae \
                                        --num_latents=$num_latents \
                                        --batch_size=$batch_size \
                                        --auxk_alpha=$auxk_alpha \
                                        --run_name=run_lat${num_latents}_bs$((batch_size*grad))_auxk${auxk_alpha}_k${k}_epoch${epoch}_${SLURM_JOB_ID}_${i} \
                                        --k=$k \
                                        --num_epochs=$epoch \
                                        --grad_acc_steps=$grad \
                                        --lr_warmup_steps=50 \
                                        --dead_feature_threshold=$dead \
                                        --normalize_decoder=False \
                                        --per_block_norm=$norm \
                                        --per_block_alpha=$per_block_alpha \
                                        --input_width=99648 \
                                        --normalized=False \
                                        '/net/tscratch/people/plgkingak/weights2weights/weights_datasets/full'
                                    echo "Finished run with --num_latents=$num_latents, batch_size=$batch_size, auxk_alpha=$auxk_alpha, num_epochs=10, k=$k at $(date)"
                                    ((i++))
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# Print end time
echo "All runs completed at $(date)"
