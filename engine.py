import os
import time
import torch
import matplotlib.pyplot as plt


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs one full pass over the training dataset, computing gradients 
    and updating model weights.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 1. Reset gradients from the previous batch
        optimizer.zero_grad()

        # 2. Forward pass: compute predicted logits
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 3. Backward pass: compute gradients and step optimizer
        loss.backward()
        optimizer.step()

        # 4. Running metrics accumulation
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on validation or test data without computing gradients.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    eval_loss = running_loss / total
    eval_acc = correct / total
    return eval_loss, eval_acc


def plot_history(history, save_path):
    """
    Plots training and validation Loss and Accuracy curves across epochs
    and saves the figure to disk.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Subplot 1: Cross-Entropy Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    plt.title("Loss vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Subplot 2: Classification Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "b-o", label="Train Acc")
    plt.plot(epochs, history["val_acc"], "r-o", label="Val Acc")
    plt.title("Accuracy vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved learning curves to: {save_path}")


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs=25, output_dir="outputs"):
    """
    Manages multi-epoch training, tracks validation loss, saves best_model.pth,
    records wall-clock execution time, and outputs learning curve plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_path = os.path.join(output_dir, "best_model.pth")

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    start_time = time.time()
    print(f"\nTraining on {device} for {num_epochs} epochs...")
    print("-" * 72)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # Record metrics for plotting
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Validation loss checkpointing (hyperparameter selection)
        saved_tag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            saved_tag = " [* Best Model Saved]"

        print(f"Epoch {epoch:02d}/{num_epochs:02d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc * 100:5.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc * 100:5.2f}% | "
              f"Time: {epoch_time:4.1f}s{saved_tag}")

    total_time = time.time() - start_time
    print("-" * 72)
    print(f"Training finished in {total_time / 60:.2f} minutes.")
    print(f"Optimal Epoch chosen by Validation Loss: Epoch {best_epoch} (Loss: {best_val_loss:.4f})")

    # Generate and save plot
    plot_path = os.path.join(output_dir, "learning_curves.png")
    plot_history(history, plot_path)

    return best_epoch, total_time


if __name__ == "__main__":
    print("engine.py is an execution module meant to be imported by main.py.")