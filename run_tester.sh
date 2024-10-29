#!/bin/bash
#SBATCH --job-name=run_tester      # Job name
#SBATCH --cpus-per-task=1           # Number of CPU cores per task
#SBATCH --mem=16G                      # Memory per node
#SBATCH --time=10:00:00                # Maximum time (hh:mm:ss)
#SBATCH --output=baseline_model.out    # Standard output and error log
#SBATCH --error=baseline_model.err  # Separate file for error logs


# Load any necessary modules
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh

# Activate the environment (if needed)
conda activate tracklab  # replace with your conda environment name

# Navigate to your mmOCR directory
cd /home/zw4603/soccernet/mmocr/mmocr/datasets/

# Run the training command

python test_dataset.py

# cd /home/zw4603/soccernet/
# python push_notification.py
