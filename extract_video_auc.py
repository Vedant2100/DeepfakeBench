import glob
import os
import pickle

permutations = {
    0: ["FFPP", "CelebDFv2", "DFDCP"],
    1: ["FFPP", "DFDCP", "CelebDFv2"],
    2: ["CelebDFv2", "FFPP", "DFDCP"],
    3: ["CelebDFv2", "DFDCP", "FFPP"],
    4: ["DFDCP", "FFPP", "CelebDFv2"],
    5: ["DFDCP", "CelebDFv2", "FFPP"]
}

# Map simple names to config dataset names
name_map = {
    "FFPP": "FaceForensics++",
    "CelebDFv2": "Celeb-DF-v2",
    "DFDCP": "DFDCP"
}

for p in range(6):
    seq = permutations[p]
    print(f"### Permutation {p}: " + " -> ".join(seq))
    
    header = "| Task | " + " | ".join([f"Eval on {d}" for d in seq]) + " |"
    print(header)
    print("|---" + "|---"*len(seq) + "|")
    
    for task_idx, trained_ds in enumerate(seq):
        pattern = f"logs/training/effort_CL_P{p}_{trained_ds}_*"
        dirs = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not dirs: 
            print(f"| Task {task_idx+1} ({trained_ds}) | DIR NOT FOUND |")
            continue
        latest_dir = dirs[-1]
        
        row = f"| Task {task_idx+1} ({trained_ds}) |"
        for col_idx, eval_ds in enumerate(seq):
            if col_idx > task_idx:
                row += " - |"
            else:
                eval_ds_long = name_map[eval_ds]
                pickle_path = os.path.join(latest_dir, "test", eval_ds_long, "metric_dict_best.pickle")
                
                if os.path.exists(pickle_path):
                    with open(pickle_path, 'rb') as f:
                        metrics = pickle.load(f)
                        val = metrics.get('video_auc', 0)
                        row += f" {val*100:.2f}% |"
                else:
                    row += " N/A |"
        print(row)
    print()
