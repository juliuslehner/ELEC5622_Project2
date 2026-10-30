import argparse
import os
import torch
import torch.nn as nn
import sys

from dataset import get_dataloaders
from models import build_model, count_parameters
from engine import train_model, evaluate

class TeeLogger:
    """Prints to both the terminal and a log file."""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

def parse_args():
    parser = argparse.ArgumentParser(description="ELEC5622 Project 2 Runner")
    parser.add_argument("--arch", type=str, default="resnet34", choices=["resnet34", "alexnet"],
                        help="Model architecture to train")
    parser.add_argument("--strategy", type=str, default="partial", choices=["partial", "full"],
                        help="Transfer learning strategy: partial or full")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for DataLoaders")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path containing training/validation/test folders")
    parser.add_argument("--csv_path", type=str, default="gt_training.csv",
                        help="Path to gt_training.csv")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Base directory to save model checkpoints and curves")
    return parser.parse_args()


def main():
    args = parse_args()

    # Device configuration (GPU on Colab, CPU locally)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create run-specific folder, e.g. outputs/resnet34_partial
    run_dir = os.path.join(args.output_dir, f"{args.arch}_{args.strategy}")
    os.makedirs(run_dir, exist_ok=True)

    # Log all terminal output to a file while still printing
    log_path = os.path.join(run_dir, "train.log")
    sys.stdout = TeeLogger(log_path)

    # Build DataLoaders, default to two workers on Colab and zero on CPU
    num_workers = 2 if torch.cuda.is_available() else 0
    print(f"Loading data from '{args.data_dir}'...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_workers=num_workers
    )

    # Instantiate model & report parameter counts for Task 4
    model = build_model(arch=args.arch, strategy=args.strategy, num_classes=6)
    model = model.to(device)

    trainable_params, total_params = count_parameters(model)
    pct = (trainable_params / total_params) * 100
    print(f"[{args.arch.upper()} | {args.strategy.upper()}] "
          f"Trainable params: {trainable_params:,} / {total_params:,} ({pct:.2f}%)")

    # Setup loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    # Filter only parameters that have requires_grad=True
    trainable_weights = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_weights, lr=args.lr)

    # Train model across epochs, track validation loss, save best_model.pth
    best_epoch, total_time = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        output_dir=run_dir
    )

    # Evaluate final test performance using the best validation checkpoint
    best_model_path = os.path.join(run_dir, "best_model.pth")
    print(f"\nLoading best checkpoint (Epoch {best_epoch}) for final test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # Print summary block
    print("=" * 65)
    print(f"FINAL METRICS: {args.arch.upper()} ({args.strategy.upper()})")
    print(f"- Optimal Epoch:       {best_epoch}")
    print(f"- Total Training Time: {total_time / 60:.2f} min ({total_time:.1f} s)")
    print(f"- Trainable Params:    {trainable_params:,} ({pct:.2f}%)")
    print(f"- Final Test Loss:     {test_loss:.4f}")
    print(f"- Final Test Accuracy: {test_acc * 100:.2f}%")
    print(f"- Artifacts saved to:  {run_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()