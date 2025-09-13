import torch, numpy as np, random, csv, json, os
from datetime import datetime

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def accuracy(output, target):
    preds = output.argmax(dim=1)
    return (preds == target).float().mean().item()

class MetricLogger:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "metrics.csv")
        self.jsonl_path = os.path.join(out_dir, "metrics.jsonl")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch","train_loss","val_loss","val_acc","lr","time"])

    def log(self, **kw):
        kw = {"time": datetime.utcnow().isoformat(timespec="seconds"), **kw}

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([kw.get(k) for k in ["epoch","train_loss","val_loss","val_acc","lr","time"]])
            
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(kw) + "\n")
