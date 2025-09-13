import torch
import torch.nn as nn
import torch.optim as optim
import yaml, argparse, os, time
from pathlib import Path

from models.vit import ViT
from datasets.cifar10 import get_cifar10
from utils import set_seed, accuracy, MetricLogger

def save_ckpt(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss_sum / max(len(loader),1), 100.0 * correct / max(total,1)

def train(config, out_dir):
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader, testloader = get_cifar10(batch_size=config["batch_size"])

    model = ViT(**config["model"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0.0))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    logger = MetricLogger(os.path.join(out_dir, "logs"))
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_acc = -1.0
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(len(trainloader),1)
        val_loss, val_acc = evaluate(model, testloader, device, criterion)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        dt = time.time() - t0
        print(f"Epoch {epoch}/{config['epochs']} | train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.2f}% | lr {lr_now:.2e} | {dt:.1f}s")

        # save last
        save_ckpt({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "val_acc": val_acc,
        }, os.path.join(ckpt_dir, "last.pt"))

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            save_ckpt({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "val_acc": val_acc,
            }, os.path.join(ckpt_dir, "best.pt"))

        logger.log(epoch=epoch, train_loss=round(train_loss,6), val_loss=round(val_loss,6),
                   val_acc=round(val_acc,4), lr=lr_now)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vit_cifar10.yaml")
    parser.add_argument("--out", type=str, default="experiments/")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ensure directories
    Path(args.out).mkdir(parents=True, exist_ok=True)
    train(config, args.out)
