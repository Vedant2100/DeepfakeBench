#!/bin/bash
#SBATCH --job-name=effort_ortho_exp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=experiments/orthogonal_loss_exp/logs/slurm_%j.out
#SBATCH --error=experiments/orthogonal_loss_exp/logs/slurm_%j.err

# Load tools and environment
source /home/csgrad/vbork001/miniconda3/etc/profile.d/conda.sh
conda activate DeepfakeBenchv2

cd /data/AmitRoyChowdhury/vedant/DeepfakeBench/

# Apply experimental patch and copy files
PATCH_FILE="experiments/orthogonal_loss_exp/ortho_loss.patch"
DETECTOR_FILE="experiments/orthogonal_loss_exp/effort_detector_orthogonal_loss_exp.py"

if [ -f "$DETECTOR_FILE" ]; then
    echo "Applying experimental changes and copying detector..."
    cp "$DETECTOR_FILE" training/detectors/
    echo "from .effort_detector_orthogonal_loss_exp import EffortDetector" >> training/detectors/__init__.py
else
    echo "Warning: Detector file not found!"
fi

# Ensure cleanup on exit
cleanup() {
    echo "Cleaning up: Reverting experimental changes..."
    git checkout training/detectors/__init__.py
    rm -f training/detectors/effort_detector_orthogonal_loss_exp.py
}
trap cleanup EXIT

# List of configurations to run
CONFIGS=(
    "experiments/orthogonal_loss_exp/faceforensicspp_ortho.yaml"
    "experiments/orthogonal_loss_exp/faceforensicspp_no_ortho.yaml"
    "experiments/orthogonal_loss_exp/celebdfv1_ortho.yaml"
    "experiments/orthogonal_loss_exp/celebdfv1_no_ortho.yaml"
    "experiments/orthogonal_loss_exp/celebdfv2_ortho.yaml"
    "experiments/orthogonal_loss_exp/celebdfv2_no_ortho.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
    echo "========================================================"
    echo "Starting Experiment with Config: $CONFIG"
    echo "========================================================"
    
    python training/train.py --detector_path "$CONFIG"
    
    echo "Finished Config: $CONFIG"
done

echo "All 6 experiments completed."
