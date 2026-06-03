import glob
import os
import re

permutations = {
    0: ["FFPP", "CelebDF", "DFDCP"],
    1: ["FFPP", "DFDCP", "CelebDF"],
    2: ["CelebDF", "FFPP", "DFDCP"],
    3: ["CelebDF", "DFDCP", "FFPP"],
    4: ["DFDCP", "FFPP", "CelebDF"],
    5: ["DFDCP", "CelebDF", "FFPP"]
}

def get_final_metrics(log_file):
    metrics = {"FaceForensics++": "-", "Celeb-DF-v1": "-", "DFDCP": "-"}
    if not os.path.exists(log_file): return metrics
    with open(log_file, "r") as f:
        lines = f.readlines()
    
    # Read backwards to find the last metric evaluation
    for line in reversed(lines):
        match = re.search(r"\|\s*([^:]+):\s*auc=([\d\.]+)", line)
        if match:
            ds = match.group(1).strip()
            val = float(match.group(2))
            if metrics[ds] == "-":
                metrics[ds] = f"{val*100:.2f}%"
        if "Each dataset best metric" in line:
            break
    return metrics

# Map simple names to config dataset names
name_map = {
    "FFPP": "FaceForensics++",
    "CelebDF": "Celeb-DF-v1",
    "DFDCP": "DFDCP"
}

for p in range(6):
    seq = permutations[p]
    print(f"### Permutation {p}: " + " -> ".join(seq))
    
    # Column headers are exactly the sequence order
    header = "| Task | " + " | ".join([f"Eval on {d}" for d in seq]) + " |"
    print(header)
    print("|---" + "|---"*len(seq) + "|")
    
    for task_idx, trained_ds in enumerate(seq):
        pattern = f"logs/training/effort_CL_P{p}_{trained_ds}_*"
        dirs = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not dirs: continue
        latest_dir = dirs[-1]
        log_file = os.path.join(latest_dir, "training.log")
        m = get_final_metrics(log_file)
        
        row = f"| Task {task_idx+1} ({trained_ds}) |"
        for col_idx, eval_ds in enumerate(seq):
            if col_idx > task_idx:
                row += " - |"  # Upper triangular is masked out
            else:
                val = m[name_map[eval_ds]]
                row += f" {val} |"
        print(row)
    print()
