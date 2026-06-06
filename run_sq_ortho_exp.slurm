#!/bin/bash
#SBATCH --job-name=effort_sq_ortho_exp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm_sq_ortho_exp_%j.out
#SBATCH --error=logs/slurm_sq_ortho_exp_%j.err

# Load tools and environment
source /home/csgrad/vbork001/miniconda3/etc/profile.d/conda.sh
conda activate DeepfakeBenchv2

cd /data/AmitRoyChowdhury/vedant/DeepfakeBench/
mkdir -p logs

# List of configurations to run
CONFIGS=(
    "training/config/detector/ffpp_train_ortho.yaml"
    "training/config/detector/ffpp_train_no_ortho.yaml"
    "training/config/detector/celebdf_train_ortho.yaml"
    "training/config/detector/celebdf_train_no_ortho.yaml"
    "training/config/detector/celebdfv2_train_ortho.yaml"
    "training/config/detector/celebdfv2_train_no_ortho.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
    echo "========================================================"
    echo "Starting Experiment with Config: $CONFIG"
    echo "========================================================"
    
    python training/train.py --detector_path "$CONFIG"
    
    echo "Finished Config: $CONFIG"
done

echo "All 6 experiments completed."
