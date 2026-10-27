import torch
import torch.nn as nn
from torchvision import models


def build_model(arch="resnet34", strategy="partial", num_classes=6):
    """
    Builds AlexNet or ResNet34 with ImageNet weights and adapts output head.
    """
    if arch == "alexnet":
        # -------------------------------------------------------------
        # TODO: Group partner to implement AlexNet fine-tuning logic:
        # 1. Load pretrained AlexNet weights
        # 2. Replace model.classifier[6] with nn.Linear(4096, num_classes)
        # 3. Apply partial freezing (freeze features) vs full training
        # -------------------------------------------------------------
        raise NotImplementedError("AlexNet is not implemented yet. Group partner to complete.")

    elif arch == "resnet34":
        # 1. Initialize with ImageNet-pretrained weights
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # 2. Replace final 1000-class layer with 6-class classifier
        in_features = model.fc.in_features  # 512
        model.fc = nn.Linear(in_features, num_classes)

        # 3. Apply layer freezing strategy
        if strategy == "partial":
            # Freeze stem and early residual blocks (conv1, bn1, layer1, layer2, layer3)
            for param in model.parameters():
                param.requires_grad = False
            # Train only the final residual block (layer4) and the classification head (fc)
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.fc.parameters():
                param.requires_grad = True

        elif strategy == "full":
            # Train all 34 layers end-to-end
            for param in model.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose 'partial' or 'full'.")

        return model

    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose 'resnet34' or 'alexnet'.")


def count_parameters(model):
    """Returns (trainable_params, total_params) for report metrics."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    print("Testing ResNet34 configurations:")
    print("-" * 65)
    for strat in ["partial", "full"]:
        m = build_model(arch="resnet34", strategy=strat, num_classes=6)
        trainable, total = count_parameters(m)
        pct = (trainable / total) * 100
        print(f"[RESNET34 | {strat.upper():7s}] "
              f"Trainable: {trainable:>10,d} / {total:>10,d} ({pct:5.2f}%)")
    print("-" * 65)

    # Test dummy forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    model = build_model("resnet34", "partial")
    output = model(dummy_input)
    print(f"Forward pass output shape: {output.shape} (Expected: torch.Size([2, 6]))")
    print("models.py is ready for ResNet34!")