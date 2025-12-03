#!/bin/bash
#SBATCH --job-name=abatur-fsdp
#SBATCH --account=abatur
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # IMPORTANT: only 1 task
#SBATCH --cpus-per-task=4
#SBATCH --gpus=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00

#SBATCH --chdir=/homes/iws/abatur/cse599o/project
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --export=ALL

echo "==== Job started at $(date) ===="
echo "Running on host: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Activate your global venv
source ~/.venv/bin/activate
echo "Using Python: $(which python)"
python --version

echo "Launching torchrun..."

torchrun \
  --standalone \
  --nproc_per_node=2 \
  /homes/iws/abatur/cse599o/project/train-claude.py

echo "==== Job finished at $(date) ===="
