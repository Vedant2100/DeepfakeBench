#!/bin/bash
#SBATCH --job-name=baseline_effort_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm_cl_all_%j.out
#SBATCH --error=logs/slurm_cl_all_%j.err

set -e

source /home/csgrad/vbork001/miniconda3/etc/profile.d/conda.sh
conda activate DeepfakeBenchv2

cd /data/AmitRoyChowdhury/vedant/DeepfakeBench/
mkdir -p logs

CONFIG_FFPP="training/config/detector/ffpp_train.yaml"
CONFIG_CELEBDF="training/config/detector/celebdf_train.yaml"
CONFIG_DFDCP="training/config/detector/dfdcp_train.yaml"

run_sequence() {
    local perm_idx=$1
    local ds1=$2
    local cfg1=$3
    local ds2=$4
    local cfg2=$5
    local ds3=$6
    local cfg3=$7
    
    echo "========================================================"
    echo "Running Permutation $perm_idx: $ds1 -> $ds2 -> $ds3"
    echo "========================================================"
    
    local TGT1="CL_P${perm_idx}_${ds1}"
    echo "Task 1: Training EFFORT on $ds1 (Baseline)"
    python training/train.py --detector_path "$cfg1" --task_target "$TGT1" --svd_residual_rank 1
    
    local LOG1=$(ls -td logs/training/effort_${TGT1}_* | head -1)
    local CKPT1="$LOG1/test/avg/ckpt_best.pth"
    if [ ! -f "$CKPT1" ]; then
        echo "Error: Checkpoint $CKPT1 not found!"
        exit 1
    fi
    echo "Found $ds1 Checkpoint: $CKPT1"
    
    local TGT2="CL_P${perm_idx}_${ds2}"
    echo "Task 2: Training EFFORT on $ds2 (Adaptation)"
    python training/train.py --detector_path "$cfg2" --task_target "$TGT2" --continue_weight "$CKPT1" --svd_residual_rank 1
    
    local LOG2=$(ls -td logs/training/effort_${TGT2}_* | head -1)
    local CKPT2="$LOG2/test/avg/ckpt_best.pth"
    if [ ! -f "$CKPT2" ]; then
        echo "Error: Checkpoint $CKPT2 not found!"
        exit 1
    fi
    echo "Found $ds2 Checkpoint: $CKPT2"
    
    local TGT3="CL_P${perm_idx}_${ds3}"
    echo "Task 3: Training EFFORT on $ds3 (Adaptation)"
    python training/train.py --detector_path "$cfg3" --task_target "$TGT3" --continue_weight "$CKPT2" --svd_residual_rank 1
    
    echo "Permutation $perm_idx completed!"
    echo ""
}

# 0: FFPP -> CelebDF -> DFDCP
run_sequence 0 "FFPP" $CONFIG_FFPP "CelebDF" $CONFIG_CELEBDF "DFDCP" $CONFIG_DFDCP

# 1: FFPP -> DFDCP -> CelebDF
run_sequence 1 "FFPP" $CONFIG_FFPP "DFDCP" $CONFIG_DFDCP "CelebDF" $CONFIG_CELEBDF

# 2: CelebDF -> FFPP -> DFDCP
run_sequence 2 "CelebDF" $CONFIG_CELEBDF "FFPP" $CONFIG_FFPP "DFDCP" $CONFIG_DFDCP

# 3: CelebDF -> DFDCP -> FFPP
run_sequence 3 "CelebDF" $CONFIG_CELEBDF "DFDCP" $CONFIG_DFDCP "FFPP" $CONFIG_FFPP

# 4: DFDCP -> FFPP -> CelebDF
run_sequence 4 "DFDCP" $CONFIG_DFDCP "FFPP" $CONFIG_FFPP "CelebDF" $CONFIG_CELEBDF

# 5: DFDCP -> CelebDF -> FFPP
run_sequence 5 "DFDCP" $CONFIG_DFDCP "CelebDF" $CONFIG_CELEBDF "FFPP" $CONFIG_FFPP

echo "All 6 permutations completed successfully!"
