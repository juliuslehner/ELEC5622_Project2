"""
evaluate.py - Standalone evaluation CLI for HEp-2 cell classification.
Evaluates saved model checkpoints on the test set, generates normalized
confusion matrices, outputs per-class classification reports, and logs
all terminal outputs to the target experiment directory.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from dataset import get_dataloaders
from models import build_model


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
    parser = argparse.ArgumentParser(description="Evaluate HEp-2 cell classification checkpoint.")
    parser.add_argument("--arch", type=str, default="resnet34", choices=["resnet34", "alexnet"],
                        help="Model architecture name.")
    parser.add_argument("--strategy", type=str, default="full", choices=["partial", "full"],
                        help="Fine-tuning strategy corresponding to the checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit path to best_model.pth (optional; defaults to output_dir/best_model.pth).")
    parser.add_argument("--data_dir", type=str, default="/content/dataset",
                        help="Root dataset directory containing 'test/' folder.")
    parser.add_argument("--csv_path", type=str, default="/content/dataset/gt_training.csv",
                        help="Path to ground truth metadata CSV.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for test evaluation.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of DataLoader workers.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Experiment output directory where logs and confusion_matrix.png are saved.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Attach dual logger to output directory
    log_path = os.path.join(args.output_dir, "eval.log")
    sys.stdout = TeeLogger(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"EVALUATION SUITE: {args.arch.upper()} [{args.strategy.upper()}]")
    print(f"Using device: {device}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 65)

    # 1. Load Data (deterministic test set)
    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 2. Locate and Load Checkpoint
    checkpoint_path = args.checkpoint or os.path.join(args.output_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading weights from: {checkpoint_path}")
    model = build_model(arch=args.arch, strategy=args.strategy, num_classes=6).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 3. Test Inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    class_names = ["Centromere", "Golgi", "Homogeneous", "NuMem", "Nucleolar", "Speckled"]

    # 4. Classification Metrics
    report_text = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\nPER-CLASS CLASSIFICATION REPORT:")
    print(report_text)

    # 5. Plot and Save Normalized Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = f"{cm[i, j]*100:.1f}%"
            color = "white" if cm[i, j] > 0.5 else "black"
            ax.text(x=j, y=i, s=val, va="center", ha="center", color=color, fontweight="bold", fontsize=10)

    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="left")
    ax.set_yticklabels(class_names)
    plt.xlabel("Predicted Class", fontweight="bold", labelpad=10)
    plt.ylabel("True Class", fontweight="bold")
    overall_acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    plt.title(f"Normalized Confusion Matrix - {args.arch.upper()} {args.strategy.upper()} ({overall_acc:.2f}%)",
              fontweight="bold", pad=20)
    plt.tight_layout()

    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"Confusion matrix plot successfully saved to: {cm_path}")
    print(f"Full terminal log saved to: {log_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()