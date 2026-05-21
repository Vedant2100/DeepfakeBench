#!/bin/bash
#SBATCH --job-name=baseline_effort_cl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm_cl_baseline_%j.out
#SBATCH --error=logs/slurm_cl_baseline_%j.err

set -e

source /home/csgrad/vbork001/miniconda3/etc/profile.d/conda.sh
conda activate DeepfakeBenchv2

cd /data/AmitRoyChowdhury/vedant/DeepfakeBench/

mkdir -p logs

echo "========================================================"
echo "Task 1: Training EFFORT on FaceForensics++ (Baseline)"
echo "========================================================"
python training/train.py --detector_path training/config/detector/ffpp_train.yaml --task_target CL_FFPP

LATEST_FFPP_LOG=$(ls -td logs/training/effort_CL_FFPP_* | head -1)
CKPT_FFPP="$LATEST_FFPP_LOG/test/FaceForensics++/ckpt_best.pth"

if [ ! -f "$CKPT_FFPP" ]; then
    echo "Error: Checkpoint $CKPT_FFPP not found!"
    exit 1
fi

echo "Found FF++ Checkpoint: $CKPT_FFPP"

echo "========================================================"
echo "Task 2: Training EFFORT on Celeb-DF-v1 (Adaptation)"
echo "========================================================"
python training/train.py --detector_path training/config/detector/celebdf_train.yaml --task_target CL_CelebDF --continue_weight "$CKPT_FFPP"

LATEST_CELEBDF_LOG=$(ls -td logs/training/effort_CL_CelebDF_* | head -1)
CKPT_CELEBDF="$LATEST_CELEBDF_LOG/test/Celeb-DF-v1/ckpt_best.pth"

if [ ! -f "$CKPT_CELEBDF" ]; then
    echo "Error: Checkpoint $CKPT_CELEBDF not found!"
    exit 1
fi

echo "Found Celeb-DF-v1 Checkpoint: $CKPT_CELEBDF"

echo "========================================================"
echo "Task 3: Training EFFORT on DFDCP (Adaptation)"
echo "========================================================"
python training/train.py --detector_path training/config/detector/dfdcp_train.yaml --task_target CL_DFDCP --continue_weight "$CKPT_CELEBDF"

echo "========================================================"
echo "Baseline Effort CL experiment completed!"
echo "========================================================"
