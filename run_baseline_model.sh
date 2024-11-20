#!/bin/bash
#SBATCH --job-name=baseline_model      # Job name
#SBATCH --partition=rtx8000          # Partition name (if GPUs are needed)
#SBATCH --gres=gpu:2                   # Number of GPUs (if needed)
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --mem=32G                      # Memory per node
#SBATCH --time=10:00:00                # Maximum time (hh:mm:ss)
#SBATCH --output=baseline_model.out    # Standard output and error log
#SBATCH --error=baseline_model.err  # Separate file for error logs


# Load any necessary modules
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh

# Activate the environment (if needed)
conda activate tracklab  # replace with your conda environment name

cd /home/zw4603/soccernet/sn-gamestate/sn_gamestate/configs/

export PYTHONPATH=$PYTHONPATH:/Users/kai/GSR/soccernet/tracklab
# Run your Python command
python -m tracklab.main -cn soccernet

