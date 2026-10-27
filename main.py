"""
main.py - Main training and evaluation driver for HEp-2 cell classification.
Handles argument parsing, model training with early validation checkpointing,
live terminal and file logging (TeeLogger), and final test evaluation.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import get_dataloaders
from models import build_model, count_parameters
from engine import train_model, evaluate


class TeeLogger:
    """Mirrors console prints to both stdout and a persistent log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.terminal.flush()
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Train HEp-2 Cell Classification Models.")
    parser.add_argument("--arch", type=str, default="resnet34", choices=["resnet34", "alexnet"],
                        help="Model architecture.")
    parser.add_argument("--strategy", type=str, default="partial", choices=["partial", "full"],
                        help="Transfer learning fine-tuning strategy.")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate for Adam optimizer.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training and evaluation.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of DataLoader worker processes.")
    parser.add_argument("--data_dir", type=str, default="/content/dataset",
                        help="Path to extracted dataset directory.")
    parser.add_argument("--csv_path", type=str, default="/content/dataset/gt_training.csv",
                        help="Path to ground truth metadata CSV.")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/ELEC5622_Project2/outputs",
                        help="Base output directory for saving checkpoints and logs.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Create subfolder per architecture and strategy
    exp_output_dir = os.path.join(args.output_dir, f"{args.arch}_{args.strategy}")
    os.makedirs(exp_output_dir, exist_ok=True)

    # Attach TeeLogger to mirror all prints directly to train.log in the output directory
    log_file_path = os.path.join(exp_output_dir, "train.log")
    sys.stdout = TeeLogger(log_file_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Experiment Output Directory: {exp_output_dir}")

    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 2. Build Model
    model = build_model(arch=args.arch, strategy=args.strategy, num_classes=6)
    model = model.to(device)

    trainable_p, total_p = count_parameters(model)
    print(f"[{args.arch.upper()} | {args.strategy.upper()}] Trainable params: "
          f"{trainable_p:,} / {total_p:,} ({trainable_p / total_p * 100:.2f}%)")

    # 3. Optimization Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # 4. Train Model with Validation Loss Checkpointing
    history, best_epoch, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        output_dir=exp_output_dir
    )

    # 5. Final Evaluation on Independent Test Set
    checkpoint_path = os.path.join(exp_output_dir, "best_model.pth")
    print(f"\nLoading best checkpoint (Epoch {best_epoch}) for final test evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("=" * 65)
    print(f"FINAL METRICS: {args.arch.upper()} ({args.strategy.upper()})")
    print(f"- Optimal Epoch:       {best_epoch}")
    print(f"- Optimal Val Loss:    {best_val_loss:.4f}")
    print(f"- Trainable Params:    {trainable_p:,} ({trainable_p / total_p * 100:.2f}%)")
    print(f"- Final Test Loss:     {test_loss:.4f}")
    print(f"- Final Test Accuracy: {test_acc * 100:.2f}%")
    print(f"- Artifacts saved to:  {exp_output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()