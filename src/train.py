import torch
import torch.nn as nn
from src.attacks import pgd


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch_standard(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_epoch_adversarial(model, loader, optimizer, criterion, device,
                             epsilon=0.3, alpha=0.01, steps=7):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.eval()
        x_adv = pgd(model, x, y, epsilon=epsilon, alpha=alpha, steps=steps, criterion=criterion)
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_adv), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total
