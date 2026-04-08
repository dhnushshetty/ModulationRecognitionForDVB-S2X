"""
evaluate.py
Evaluate the trained Random Forest classifier.

Outputs:
  - Classification report printed to console
  - results/confusion_matrix.png
  - results/accuracy_per_snr.png

Run from the AMR-DVB-S2X/ directory:
    python src/evaluate.py
"""

import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
MODELS_DIR  = "models"
RESULTS_DIR = "results"

MODEL_PATH  = os.path.join(MODELS_DIR, "random_forest_model.pkl")
IN_X_TEST   = os.path.join(DATA_DIR, "X_test.npy")
IN_Y_TEST   = os.path.join(DATA_DIR, "y_test.npy")
IN_SNRS_TEST = os.path.join(DATA_DIR, "snrs_test.npy")

OUT_CONF_MATRIX = os.path.join(RESULTS_DIR, "confusion_matrix.png")
OUT_ACC_SNR     = os.path.join(RESULTS_DIR, "accuracy_per_snr.png")


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalize

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        vmin=0, vmax=1,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix (Normalized)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved to '{save_path}'.")


def plot_accuracy_per_snr(y_true, y_pred, snrs, save_path):
    unique_snrs = sorted(set(snrs))
    accuracies  = []

    for snr in unique_snrs:
        mask = snrs == snr
        acc  = accuracy_score(y_true[mask], y_pred[mask])
        accuracies.append(acc * 100)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(unique_snrs, accuracies, marker="o", linewidth=2, color="steelblue")
    ax.axhline(y=50, color="red", linestyle="--", linewidth=1, label="Random baseline (50%)")
    ax.axhline(y=90, color="green", linestyle="--", linewidth=1, label="90% target")
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Classification Accuracy vs SNR", fontsize=14)
    ax.set_ylim(0, 105)
    ax.set_xticks(unique_snrs)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Accuracy vs SNR plot saved to '{save_path}'.")

    # Print per-SNR table
    print("\n[INFO] Accuracy per SNR level:")
    print(f"       {'SNR (dB)':<12} {'Accuracy (%)'}")
    print(f"       {'-'*12} {'-'*12}")
    for snr, acc in zip(unique_snrs, accuracies):
        print(f"       {snr:<12} {acc:.2f}%")


def main():
    # ── Load model and test data ───────────────────────────────────────────────
    for path in (MODEL_PATH, IN_X_TEST, IN_Y_TEST, IN_SNRS_TEST):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[ERROR] '{path}' not found. Run train_model.py first."
            )

    print("[INFO] Loading model and test data ...")
    clf    = joblib.load(MODEL_PATH)
    X_test = np.load(IN_X_TEST,    allow_pickle=True)
    y_test = np.load(IN_Y_TEST,    allow_pickle=True)
    snrs   = np.load(IN_SNRS_TEST, allow_pickle=True)

    # ── Predict ────────────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)

    # ── Overall metrics ────────────────────────────────────────────────────────
    overall_acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Overall test accuracy: {overall_acc * 100:.2f}%")
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred))

    # ── Plots ──────────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    classes = sorted(set(y_test))

    plot_confusion_matrix(y_test, y_pred, classes, OUT_CONF_MATRIX)
    plot_accuracy_per_snr(y_test, y_pred, snrs, OUT_ACC_SNR)

    print("\n[DONE] evaluate.py complete. Check the results/ folder for plots.")


if __name__ == "__main__":
    main()
