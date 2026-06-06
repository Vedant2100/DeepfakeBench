#!/bin/bash
#SBATCH --job-name=ortho_effort_v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm_ortho_v2_%j.out
#SBATCH --error=logs/slurm_ortho_v2_%j.err

set -e

source /home/csgrad/vbork001/miniconda3/etc/profile.d/conda.sh
conda activate DeepfakeBenchv2

cd /data/AmitRoyChowdhury/vedant/DeepfakeBench/
mkdir -p logs

CONFIG_FFPP="training/config/detector/ffpp_train_ortho_v2.yaml"
CONFIG_CELEBDF="training/config/detector/celebdfv2_train_ortho.yaml"
CONFIG_DFDCP="training/config/detector/dfdcp_train_ortho_v2.yaml"

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
    
    local TGT1="CL_ortho_v2_P${perm_idx}_${ds1}"
    local LOG_DIR1=$(grep "^log_dir:" "$cfg1" | awk '{print $2}')
    local LOG1=$(ls -td ${LOG_DIR1}/effort_${TGT1}_* 2>/dev/null | head -1)
    local CKPT1="$LOG1/test/avg/ckpt_best.pth"
    
    echo "Task 1: Training EFFORT on $ds1 (Baseline)"
    if [ -f "$CKPT1" ]; then
        echo "Found existing checkpoint, skipping training: $CKPT1"
    else
        python training/train.py --detector_path "$cfg1" --task_target "$TGT1" --svd_residual_rank 1
        LOG1=$(ls -td ${LOG_DIR1}/effort_${TGT1}_* | head -1)
        CKPT1="$LOG1/test/avg/ckpt_best.pth"
        if [ ! -f "$CKPT1" ]; then
            echo "Error: Checkpoint $CKPT1 not found!"
            exit 1
        fi
        echo "Saved new checkpoint: $CKPT1"
    fi
    
    local TGT2="CL_ortho_v2_P${perm_idx}_${ds2}"
    local LOG_DIR2=$(grep "^log_dir:" "$cfg2" | awk '{print $2}')
    local LOG2=$(ls -td ${LOG_DIR2}/effort_${TGT2}_* 2>/dev/null | head -1)
    local CKPT2="$LOG2/test/avg/ckpt_best.pth"
    
    echo "Task 2: Training EFFORT on $ds2 (Adaptation)"
    if [ -f "$CKPT2" ]; then
        echo "Found existing checkpoint, skipping training: $CKPT2"
    else
        python training/train.py --detector_path "$cfg2" --task_target "$TGT2" --continue_weight "$CKPT1" --svd_residual_rank 1
        LOG2=$(ls -td ${LOG_DIR2}/effort_${TGT2}_* | head -1)
        CKPT2="$LOG2/test/avg/ckpt_best.pth"
        if [ ! -f "$CKPT2" ]; then
            echo "Error: Checkpoint $CKPT2 not found!"
            exit 1
        fi
        echo "Saved new checkpoint: $CKPT2"
    fi
    
    local TGT3="CL_ortho_v2_P${perm_idx}_${ds3}"
    echo "Task 3: Training EFFORT on $ds3 (Adaptation)"
    python training/train.py --detector_path "$cfg3" --task_target "$TGT3" --continue_weight "$CKPT2" --svd_residual_rank 1
    
    echo "Permutation $perm_idx completed!"
    echo ""
}

# 0: FFPP -> CelebDFv2 -> DFDCP
run_sequence 0 "FFPP" $CONFIG_FFPP "CelebDFv2" $CONFIG_CELEBDF "DFDCP" $CONFIG_DFDCP

# 1: FFPP -> DFDCP -> CelebDFv2
run_sequence 1 "FFPP" $CONFIG_FFPP "DFDCP" $CONFIG_DFDCP "CelebDFv2" $CONFIG_CELEBDF

# 2: CelebDFv2 -> FFPP -> DFDCP
run_sequence 2 "CelebDFv2" $CONFIG_CELEBDF "FFPP" $CONFIG_FFPP "DFDCP" $CONFIG_DFDCP

# 3: CelebDFv2 -> DFDCP -> FFPP
run_sequence 3 "CelebDFv2" $CONFIG_CELEBDF "DFDCP" $CONFIG_DFDCP "FFPP" $CONFIG_FFPP

# 4: DFDCP -> FFPP -> CelebDFv2
run_sequence 4 "DFDCP" $CONFIG_DFDCP "FFPP" $CONFIG_FFPP "CelebDFv2" $CONFIG_CELEBDF

# 5: DFDCP -> CelebDFv2 -> FFPP
run_sequence 5 "DFDCP" $CONFIG_DFDCP "CelebDFv2" $CONFIG_CELEBDF "FFPP" $CONFIG_FFPP

echo "All 6 permutations completed successfully!"
