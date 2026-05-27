import torch
import torch.nn as nn


def fgsm(model, x, y, epsilon, criterion=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    x_adv = x.clone().detach().requires_grad_(True)
    loss = criterion(model(x_adv), y)
    loss.backward()
    return (x + epsilon * x_adv.grad.sign()).clamp(0, 1).detach()


def pgd(model, x, y, epsilon, alpha, steps, criterion=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    x_adv = (x + torch.empty_like(x).uniform_(-epsilon, epsilon)).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = criterion(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = (x + (x_adv - x).clamp(-epsilon, epsilon)).clamp(0, 1)
    return x_adv.detach()
