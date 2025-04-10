#!/bin/bash
#SBATCH --job-name=you_job_name         # Change as needed
#SBATCH --time=02:00:00
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --gres=gpu:2                    # Request 2 GPUs
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # Adjust CPU allocation if needed
#SBATCH --output=interactive_job.out    # Output log file
#SBATCH --error=interactive_job.err     # Error log file

CONFIG_FILE=$1
WANDB=$2
NUM_GPUS=$3

conda activate nanofm
export WANDB_API_KEY=c80687eb51acc4024f6907e16bcf29fd0f9862c1 $$ OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 run_training.py --config cfgs/nanoMaskGIT/mnist_d8w512.yaml
