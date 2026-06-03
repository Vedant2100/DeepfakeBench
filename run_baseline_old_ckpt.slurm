#!/bin/bash
#SBATCH --job-name=baseline_old_ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm_old_ckpt_%j.out
#SBATCH --error=logs/slurm_old_ckpt_%j.err

set -e

source /home/csgrad/vbork001/miniconda3/etc/profile.d/conda.sh
conda activate DeepfakeBenchv2

cd /data/AmitRoyChowdhury/vedant/DeepfakeBench/

mkdir -p logs

CKPT_FFPP="./training/pretrained/effort_clip_L14_trainOn_FaceForensic.pth"

if [ ! -f "$CKPT_FFPP" ]; then
    echo "Error: Checkpoint $CKPT_FFPP not found!"
    exit 1
fi

echo "Found Old FF++ Checkpoint: $CKPT_FFPP"

echo "========================================================"
echo "Task 2: Training EFFORT on Celeb-DF-v1 (Adaptation from Old Ckpt)"
echo "========================================================"
python training/train.py --detector_path training/config/detector/celebdf_train.yaml --task_target OldCkpt_CelebDF --continue_weight "$CKPT_FFPP" --svd_residual_rank 1

LATEST_CELEBDF_LOG=$(ls -td logs/training/effort_OldCkpt_CelebDF_* | head -1)
CKPT_CELEBDF="$LATEST_CELEBDF_LOG/test/avg/ckpt_best.pth"

if [ ! -f "$CKPT_CELEBDF" ]; then
    echo "Error: Checkpoint $CKPT_CELEBDF not found!"
    exit 1
fi

echo "Found Celeb-DF-v1 Checkpoint: $CKPT_CELEBDF"

echo "========================================================"
echo "Task 3: Training EFFORT on DFDCP (Adaptation from Old Ckpt)"
echo "========================================================"
python training/train.py --detector_path training/config/detector/dfdcp_train.yaml --task_target OldCkpt_DFDCP --continue_weight "$CKPT_CELEBDF" --svd_residual_rank 1

echo "========================================================"
echo "Old Checkpoint Baseline Effort CL experiment completed!"
echo "========================================================"
