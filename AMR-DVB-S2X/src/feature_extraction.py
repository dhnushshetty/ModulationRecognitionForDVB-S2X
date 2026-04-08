"""
feature_extraction.py
Extract hand-crafted features from raw I/Q samples and save them for training.

Features extracted per sample (6 total):
  1. Mean Amplitude
  2. Variance of Amplitude
  3. Kurtosis of Amplitude
  4. Mean Phase
  5. Variance of Phase
  6. Max Amplitude

Run from the AMR-DVB-S2X/ directory:
    python src/feature_extraction.py
"""

import numpy as np
from scipy.stats import kurtosis
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR      = "data"
IN_X          = os.path.join(DATA_DIR, "X_raw.npy")
IN_Y          = os.path.join(DATA_DIR, "y_raw.npy")
IN_SNRS       = os.path.join(DATA_DIR, "snrs_raw.npy")
OUT_FEATURES  = os.path.join(DATA_DIR, "X_features.npy")
OUT_LABELS    = os.path.join(DATA_DIR, "y_labels.npy")
OUT_SNRS_OUT  = os.path.join(DATA_DIR, "snrs.npy")

FEATURE_NAMES = [
    "Mean Amplitude",
    "Variance of Amplitude",
    "Kurtosis of Amplitude",
    "Mean Phase",
    "Variance of Phase",
    "Max Amplitude",
]


def extract_features(X: np.ndarray) -> np.ndarray:
    """
    Extract 6 statistical features from each I/Q sample.

    Parameters
    ----------
    X : ndarray, shape (N, 2, 128)
        Raw I/Q samples. Channel 0 = I, Channel 1 = Q.

    Returns
    -------
    features : ndarray, shape (N, 6)
    """
    I = X[:, 0, :]  # In-phase component,    shape (N, 128)
    Q = X[:, 1, :]  # Quadrature component,  shape (N, 128)

    amplitude = np.sqrt(I**2 + Q**2)          # (N, 128)
    phase     = np.arctan2(Q, I)              # (N, 128)

    mean_amp  = amplitude.mean(axis=1)        # (N,)
    var_amp   = amplitude.var(axis=1)         # (N,)
    kurt_amp  = kurtosis(amplitude, axis=1)   # (N,)  — Fisher definition, 0 for normal
    mean_phase = phase.mean(axis=1)           # (N,)
    var_phase  = phase.var(axis=1)            # (N,)
    max_amp    = amplitude.max(axis=1)        # (N,)

    features = np.column_stack([
        mean_amp, var_amp, kurt_amp,
        mean_phase, var_phase, max_amp,
    ])
    return features


def main():
    # ── Load raw arrays ────────────────────────────────────────────────────────
    for path in (IN_X, IN_Y, IN_SNRS):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[ERROR] '{path}' not found. Run load_data.py first."
            )

    print("[INFO] Loading raw I/Q data ...")
    X    = np.load(IN_X,    allow_pickle=True)
    y    = np.load(IN_Y,    allow_pickle=True)
    snrs = np.load(IN_SNRS, allow_pickle=True)
    print(f"       X shape: {X.shape}, samples: {len(y)}")

    # ── Extract features ───────────────────────────────────────────────────────
    print("[INFO] Extracting features ...")
    features = extract_features(X)
    print(f"       Feature matrix shape: {features.shape}")
    print(f"       Features: {FEATURE_NAMES}")

    # Quick sanity check — no NaNs or Infs
    assert not np.isnan(features).any(), "NaN detected in features!"
    assert not np.isinf(features).any(), "Inf detected in features!"
    print("       Sanity check passed (no NaN / Inf).")

    # ── Save ───────────────────────────────────────────────────────────────────
    np.save(OUT_FEATURES, features)
    np.save(OUT_LABELS,   y)
    np.save(OUT_SNRS_OUT, snrs)
    print(f"\n[INFO] Saved feature matrix to:")
    print(f"       {OUT_FEATURES}")
    print(f"       {OUT_LABELS}")
    print(f"       {OUT_SNRS_OUT}")
    print("\n[DONE] feature_extraction.py complete. Run train_model.py next.")


if __name__ == "__main__":
    main()
