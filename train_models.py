import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data import get_datasets
from src.model import CNN
from src.train import get_device, train_epoch_standard, train_epoch_adversarial, evaluate


def run(dataset_name, epochs):
    device = get_device()
    print(f"\nDevice: {device}")

    train_ds, test_ds, _ = get_datasets(dataset_name)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    os.makedirs("models", exist_ok=True)

    # Standard
    print(f"\n[{dataset_name}] Standard training ({epochs} epochs)")
    model = CNN().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, epochs + 1):
        loss = train_epoch_standard(model, train_loader, opt, criterion, device)
        acc = evaluate(model, test_loader, device)
        print(f"  {ep:02d}/{epochs}  loss={loss:.4f}  acc={acc:.4f}")
    path = f"models/{dataset_name}_standard.pt"
    torch.save(model.state_dict(), path)
    print(f"  Saved {path}")

    # Adversarial (PGD-7)
    print(f"\n[{dataset_name}] Adversarial training PGD-7 ({epochs} epochs)")
    model_adv = CNN().to(device)
    opt_adv = optim.Adam(model_adv.parameters(), lr=1e-3)
    for ep in range(1, epochs + 1):
        loss = train_epoch_adversarial(model_adv, train_loader, opt_adv, criterion, device)
        acc = evaluate(model_adv, test_loader, device)
        print(f"  {ep:02d}/{epochs}  loss={loss:.4f}  acc={acc:.4f}")
    path = f"models/{dataset_name}_robust.pt"
    torch.save(model_adv.state_dict(), path)
    print(f"  Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    datasets = ["MNIST", "FashionMNIST"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run(ds, args.epochs)
    print("\nDone.")
