#!/bin/bash
#SBATCH --job-name=adaptive-ac-single
#SBATCH --account=abatur
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus=4               # CHANGED: Request only 1 GPU for this step
#SBATCH --mem=32G
#SBATCH --time=00:30:00        # Reduced time (benchmarks are fast)

#SBATCH --chdir=/homes/iws/abatur/cse599o/cse599o/project
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --export=ALL

echo "==== Job started at $(date) ===="
echo "Running on host: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Activate your global venv
source ~/.venv/bin/activate
echo "Using Python: $(which python)"

# ---------------------------------------------------------
# COMMAND CHANGED:
# We use 'python' directly because the script is single-GPU.
# ---------------------------------------------------------
# torchrun \
#   --standalone \
#   --nproc_per_node=8 \
#   fsdp-oom-test.py

# torchrun \
#   --standalone \
#   --nproc_per_node=4 \
#   /homes/iws/abatur/cse599o/cse599o/project/fsdp_train_4gpu.py

torchrun --nproc_per_node=4 final-experiment.py

# python single-gpu-ac-test.py

echo "==== Job finished at $(date) ===="