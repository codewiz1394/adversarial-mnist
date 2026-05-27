import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from torch.utils.data import DataLoader

from src.attacks import fgsm, pgd
from src.data import get_datasets
from src.model import CNN

st.set_page_config(page_title="Adversarial Robustness Demo", layout="wide")


@st.cache_resource
def load_models(dataset_name):
    result = {}
    for key, filename in [("Standard", f"{dataset_name}_standard.pt"),
                           ("Robust (AT)", f"{dataset_name}_robust.pt")]:
        path = f"models/{filename}"
        if os.path.exists(path):
            m = CNN()
            m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            m.eval()
            result[key] = m
    return result


@st.cache_resource
def load_test_data(dataset_name):
    _, test_ds, classes = get_datasets(dataset_name)
    return test_ds, classes


def predict(model, x_tensor):
    with torch.no_grad():
        probs = torch.softmax(model(x_tensor.unsqueeze(0)), dim=1)[0]
    pred = probs.argmax().item()
    return pred, probs[pred].item(), probs.numpy()


def apply_attack(attack, model, x, y, epsilon, pgd_steps, pgd_alpha):
    y_t = torch.tensor([y])
    if attack == "FGSM":
        return fgsm(model, x.unsqueeze(0), y_t, epsilon).squeeze(0)
    if attack == "PGD":
        return pgd(model, x.unsqueeze(0), y_t, epsilon, pgd_alpha, pgd_steps).squeeze(0)
    return x.clone()


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Controls")
dataset_name = st.sidebar.selectbox("Dataset", ["MNIST", "FashionMNIST"])
attack_type = st.sidebar.selectbox("Attack", ["None", "FGSM", "PGD"])

epsilon = pgd_steps = pgd_alpha = 0
if attack_type != "None":
    epsilon = st.sidebar.slider("Epsilon (ε)", 0.0, 0.5, 0.1, 0.01,
                                help="Perturbation budget — higher = stronger attack")
if attack_type == "PGD":
    pgd_steps = st.sidebar.slider("PGD steps", 1, 40, 10)
    pgd_alpha = st.sidebar.slider("Step size (α)", 0.001, 0.05, 0.01, 0.001)

img_index = st.sidebar.number_input("Image index (0 – 9999)", 0, 9999, 42)

# ── Load ─────────────────────────────────────────────────────────────────────
test_ds, classes = load_test_data(dataset_name)
models = load_models(dataset_name)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Adversarial Robustness Demo")
st.caption(
    "Visualise how FGSM / PGD attacks fool a standard CNN and why adversarially "
    "trained models hold up better."
)

if not models:
    st.error("No trained models found. Run the training script first:")
    st.code("python train_models.py --dataset both --epochs 10")
    st.stop()

# ── Single-image view ─────────────────────────────────────────────────────────
x, y = test_ds[int(img_index)]
ref_model = next(iter(models.values()))
x_adv = apply_attack(attack_type, ref_model, x, y, epsilon, pgd_steps, pgd_alpha)
perturbation = (x_adv - x).abs()

n_models = len(models)
fig, axes = plt.subplots(1, 2 + n_models, figsize=(3.5 * (2 + n_models), 4))
fig.patch.set_facecolor("#0e1117")

def style_ax(ax):
    ax.set_facecolor("#0e1117")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# Original
axes[0].imshow(x.squeeze(), cmap="gray")
axes[0].set_title(f"Original\n{classes[y]}", color="white", fontsize=11)
style_ax(axes[0])

# Perturbation (amplified for visibility)
pert_vis = (perturbation.squeeze() * 15).clamp(0, 1).numpy()
axes[1].imshow(pert_vis, cmap="hot")
label = f"Perturbation ×15\n{attack_type}, ε={epsilon:.2f}" if attack_type != "None" else "Perturbation\n(none)"
axes[1].set_title(label, color="white", fontsize=11)
style_ax(axes[1])

# Model predictions on adversarial image
for i, (name, model) in enumerate(models.items()):
    pred, conf, _ = predict(model, x_adv)
    correct = pred == y
    color = "#4ade80" if correct else "#f87171"
    axes[2 + i].imshow(x_adv.squeeze().numpy(), cmap="gray")
    axes[2 + i].set_title(f"{name}\n{classes[pred]} ({conf:.1%})", color=color, fontsize=11)
    style_ax(axes[2 + i])

plt.tight_layout(pad=0.5)
st.pyplot(fig)
plt.close()

# ── Confidence bars ───────────────────────────────────────────────────────────
st.subheader("Class probabilities under attack")
cols = st.columns(n_models)
for i, (name, model) in enumerate(models.items()):
    pred, _, probs = predict(model, x_adv)
    with cols[i]:
        st.markdown(f"**{name}**")
        for j, (cls, prob) in enumerate(zip(classes, probs)):
            marker = "✓ " if j == y else ""
            st.progress(float(prob), text=f"{marker}{cls}: {prob:.1%}")

# ── Batch evaluation ──────────────────────────────────────────────────────────
st.divider()
st.subheader("Batch attack success rate (500 samples)")

if st.button("Run batch evaluation"):
    with st.spinner("Evaluating…"):
        loader = DataLoader(test_ds, batch_size=500, shuffle=False)
        x_batch, y_batch = next(iter(loader))
        batch_cols = st.columns(n_models)

        for col_idx, (name, model) in enumerate(models.items()):
            with torch.no_grad():
                clean_acc = (model(x_batch).argmax(1) == y_batch).float().mean().item()

            if attack_type != "None" and epsilon > 0:
                if attack_type == "FGSM":
                    x_adv_b = fgsm(model, x_batch, y_batch, epsilon)
                else:
                    x_adv_b = pgd(model, x_batch, y_batch, epsilon, pgd_alpha, pgd_steps)
                with torch.no_grad():
                    adv_acc = (model(x_adv_b).argmax(1) == y_batch).float().mean().item()
            else:
                adv_acc = clean_acc

            with batch_cols[col_idx]:
                st.metric(f"{name} — clean", f"{clean_acc:.1%}")
                if attack_type != "None":
                    st.metric(
                        f"{name} — {attack_type} (ε={epsilon:.2f})",
                        f"{adv_acc:.1%}",
                        delta=f"{adv_acc - clean_acc:.1%}",
                    )
