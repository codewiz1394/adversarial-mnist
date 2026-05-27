# Adversarial Robustness Demo — MNIST & Fashion-MNIST

An interactive demonstration of adversarial attacks and defenses in deep learning. Train a CNN on MNIST or Fashion-MNIST, attack it with FGSM or PGD, and compare how a standard model collapses versus one hardened with adversarial training.

Built as a companion to my research on adversarial robustness in deep reinforcement learning.

---

## Demo

```
streamlit run app.py
```

![App screenshot](assets/demo.png)

Pick a dataset, choose an attack, drag the epsilon slider — watch the standard model fail while the robust model holds.

---

## What's inside

| File | Purpose |
|---|---|
| `src/model.py` | 2-layer CNN (32→64 conv, 128 FC head) |
| `src/attacks.py` | FGSM and PGD attack implementations |
| `src/data.py` | MNIST / Fashion-MNIST loaders |
| `src/train.py` | Standard and adversarial training loops |
| `train_models.py` | CLI to train and save all models |
| `app.py` | Streamlit interactive demo |

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. MPS (Apple Silicon) and CUDA are used automatically if available.

---

## Train models

```bash
python train_models.py --dataset both --epochs 10
```

This trains four models and saves them to `models/`:

| File | Description |
|---|---|
| `MNIST_standard.pt` | Standard CNN on MNIST |
| `MNIST_robust.pt` | PGD-7 adversarially trained CNN on MNIST |
| `FashionMNIST_standard.pt` | Standard CNN on Fashion-MNIST |
| `FashionMNIST_robust.pt` | PGD-7 adversarially trained CNN on Fashion-MNIST |

Training takes ~30–45 minutes on CPU. On MPS (Apple Silicon) or CUDA it's significantly faster.

---

## Attacks

**FGSM** (Fast Gradient Sign Method) — single-step attack:

$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(\theta, x, y))$$

**PGD** (Projected Gradient Descent) — iterative, strongest first-order attack:

$$x^{t+1} = \Pi_{x+\mathcal{S}} \left( x^t + \alpha \cdot \text{sign}(\nabla_x \mathcal{L}(\theta, x^t, y)) \right)$$

---

## Results

Trained for 10 epochs on each dataset (MPS device):

| Model | Clean Accuracy |
|---|---|
| MNIST Standard | 99.2% |
| MNIST Robust (AT) | 99.1% |
| Fashion-MNIST Standard | 91.9% |
| Fashion-MNIST Robust (AT) | 83.4% |

The clean accuracy drop on Fashion-MNIST (91.9% → 83.4%) is the classic **robustness-accuracy tradeoff** — the robust model trades some clean performance for resilience under attack.

---

## App features

- **Single-image view** — pick any test image, apply an attack, see predictions from both models side by side
- **Perturbation map** — amplified (×15) visualization of what the attack actually changed
- **Confidence bars** — full class probability breakdown for each model
- **Batch evaluation** — run over 500 test samples and compare attack success rates

---

## Related work

This project is a hands-on companion to:

- *Towards Robust Agents: A Survey of Adversarial Attacks and Defenses in Deep Reinforcement Learning* — IEEE Access 2026
- *Advancing Robustness in Deep Reinforcement Learning with an Ensemble Defense Approach* — ITSC 2025
