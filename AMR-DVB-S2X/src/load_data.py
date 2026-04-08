"""
load_data.py
Load the RadioML 2016.10A dataset, explore its structure,
filter to QPSK and 8PSK only, and save the filtered arrays to data/.

Run from the AMR-DVB-S2X/ directory:
    python src/load_data.py
"""

import pickle
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = "data"
PKL_PATH  = os.path.join(DATA_DIR, "RML2016.10a_dict.pkl")
OUT_X     = os.path.join(DATA_DIR, "X_raw.npy")
OUT_Y     = os.path.join(DATA_DIR, "y_raw.npy")
OUT_SNRS  = os.path.join(DATA_DIR, "snrs_raw.npy")

# Modulations to keep (must match dataset key names exactly)
TARGET_MODS = {"QPSK", "8PSK"}


def load_dataset(pkl_path: str) -> dict:
    """Load the pickled dataset and return the raw dictionary."""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at '{pkl_path}'.\n"
            "Please download RML2016.10a.tar.bz2 from:\n"
            "  http://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2\n"
            "Extract it and place RML2016.10a.pkl inside the data/ folder."
        )
    print(f"[INFO] Loading dataset from '{pkl_path}' ...")
    with open(pkl_path, "rb") as f:
        try:
            data = pickle.load(f, encoding="latin1")  # Python 3 reading Python 2 pickle
        except TypeError:
            data = pickle.load(f)
    return data


def explore_dataset(data: dict) -> None:
    """Print a summary of the dataset structure."""
    keys = list(data.keys())
    print(f"\n[INFO] Dataset contains {len(keys)} (modulation, SNR) entries.")

    mods = sorted({k[0] for k in keys})
    snrs = sorted({k[1] for k in keys})
    sample_shape = data[keys[0]].shape

    print(f"       Modulation types ({len(mods)}): {mods}")
    print(f"       SNR levels       ({len(snrs)}): {snrs} dB")
    print(f"       Sample shape per entry: {sample_shape}  "
          f"-> {sample_shape[0]} samples x {sample_shape[1]} channels x {sample_shape[2]} time steps")


def filter_and_collect(data: dict, target_mods: set):
    """
    Filter dataset to target modulations and collect arrays.

    Returns
    -------
    X    : ndarray, shape (N, 2, 128) — raw I/Q samples
    y    : ndarray, shape (N,)        — string labels
    snrs : ndarray, shape (N,)        — integer SNR values
    """
    X_list, y_list, snr_list = [], [], []

    for (mod, snr), samples in data.items():
        if mod not in target_mods:
            continue
        n = samples.shape[0]
        X_list.append(samples)
        y_list.extend([mod] * n)
        snr_list.extend([snr] * n)

    X    = np.vstack(X_list)
    y    = np.array(y_list)
    snrs = np.array(snr_list, dtype=int)
    return X, y, snrs


def main():
    data = load_dataset(PKL_PATH)
    explore_dataset(data)

    print(f"\n[INFO] Filtering to modulations: {TARGET_MODS} ...")
    X, y, snrs = filter_and_collect(data, TARGET_MODS)

    print(f"[INFO] Filtered dataset — X: {X.shape}, y: {y.shape}, snrs: {snrs.shape}")
    print(f"       Label distribution: { {m: int((y == m).sum()) for m in sorted(set(y))} }")

    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(OUT_X,    X)
    np.save(OUT_Y,    y)
    np.save(OUT_SNRS, snrs)
    print(f"\n[INFO] Saved filtered arrays to:")
    print(f"       {OUT_X}")
    print(f"       {OUT_Y}")
    print(f"       {OUT_SNRS}")
    print("\n[DONE] load_data.py complete. Run feature_extraction.py next.")


if __name__ == "__main__":
    main()
