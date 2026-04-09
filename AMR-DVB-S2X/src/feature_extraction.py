"""
feature_extraction.py
Extract rich hand-crafted features from raw I/Q samples and save them for training.

Features extracted per sample (30 total):
  Amplitude-domain (9): mean, std, variance, skewness, kurtosis, max, min,
                         peak-to-peak, RMS
  Phase-domain      (4): mean, std, variance, phase discontinuity count
  Frequency-domain  (6): spectral centroid, spectral spread, peak FFT mag,
                         spectral energy sub-bands (4 bins)
  I/Q cross-features(2): I-Q cross-correlation at lag 0, I power fraction
  Higher-order stats(9): 2nd/4th/6th order cumulants of amplitude (I, Q, envelope)

Run from the AMR-DVB-S2X/ directory:
    python src/feature_extraction.py
"""

import numpy as np
from scipy.stats import kurtosis, skew
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
    # Amplitude-domain (9)
    "Amp Mean", "Amp Std", "Amp Var", "Amp Skewness", "Amp Kurtosis",
    "Amp Max", "Amp Min", "Amp Peak2Peak", "Amp RMS",
    # Phase-domain (4)
    "Phase Mean", "Phase Std", "Phase Var", "Phase Disc Count",
    # Frequency-domain (7)
    "Spectral Centroid", "Spectral Spread", "Peak FFT Mag",
    "FFT Energy Band0", "FFT Energy Band1", "FFT Energy Band2", "FFT Energy Band3",
    # I/Q cross-features (2)
    "IQ CrossCorr", "I Power Fraction",
    # Higher-order cumulants (9)
    "C20 I", "C40 I", "C60 I",
    "C20 Q", "C40 Q", "C60 Q",
    "C20 Env", "C40 Env", "C60 Env",
    # PSK-specific discriminative features (5)
    "C40 Phase", "Inst Freq Std", "Dist QPSK", "Dist 8PSK", "Dist Ratio",
]


def _cumulants(x: np.ndarray):
    """
    Compute 2nd, 4th, 6th order cumulants of a 2D array along axis=1.
    x shape: (N, T)
    Returns three arrays of shape (N,).
    """
    c2 = np.mean(x ** 2, axis=1)
    c4 = np.mean(x ** 4, axis=1) - 3 * c2 ** 2
    c6 = (np.mean(x ** 6, axis=1)
          - 15 * np.mean(x ** 4, axis=1) * c2
          + 30 * c2 ** 3)
    return c2, c4, c6


def extract_features(X: np.ndarray) -> np.ndarray:
    """
    Extract 30 rich features from each I/Q sample.

    Parameters
    ----------
    X : ndarray, shape (N, 2, 128)
        Raw I/Q samples. Channel 0 = I, Channel 1 = Q.

    Returns
    -------
    features : ndarray, shape (N, 30)
    """
    N = X.shape[0]
    I = X[:, 0, :]   # (N, 128)
    Q = X[:, 1, :]   # (N, 128)

    amplitude = np.sqrt(I ** 2 + Q ** 2)   # (N, 128)
    phase     = np.arctan2(Q, I)           # (N, 128)

    # ── Amplitude-domain features ──────────────────────────────────────────────
    amp_mean   = amplitude.mean(axis=1)
    amp_std    = amplitude.std(axis=1)
    amp_var    = amplitude.var(axis=1)
    amp_skew   = skew(amplitude, axis=1)
    amp_kurt   = kurtosis(amplitude, axis=1)
    amp_max    = amplitude.max(axis=1)
    amp_min    = amplitude.min(axis=1)
    amp_p2p    = amp_max - amp_min
    amp_rms    = np.sqrt(np.mean(amplitude ** 2, axis=1))

    # ── Phase-domain features ──────────────────────────────────────────────────
    phase_mean = phase.mean(axis=1)
    phase_std  = phase.std(axis=1)
    phase_var  = phase.var(axis=1)
    # Phase discontinuity count: jumps larger than π (PSK symbol transitions)
    phase_diff = np.diff(phase, axis=1)          # (N, 127)
    phase_disc = (np.abs(phase_diff) > np.pi).sum(axis=1).astype(float)

    # ── Frequency-domain features ──────────────────────────────────────────────
    # Use complex FFT of I+jQ
    sig_complex = I + 1j * Q
    fft_mag = np.abs(np.fft.fft(sig_complex, axis=1))[:, :64]  # one-sided (N, 64)
    freqs   = np.arange(64, dtype=float)

    total_energy    = fft_mag.sum(axis=1) + 1e-12
    spec_centroid   = (fft_mag * freqs).sum(axis=1) / total_energy
    spec_spread     = np.sqrt(
        (fft_mag * (freqs - spec_centroid[:, None]) ** 2).sum(axis=1) / total_energy
    )
    peak_fft_mag    = fft_mag.max(axis=1)

    # 4 equal sub-band energies (each band = 16 bins)
    band_energy = np.column_stack([
        fft_mag[:, 0:16].sum(axis=1),
        fft_mag[:, 16:32].sum(axis=1),
        fft_mag[:, 32:48].sum(axis=1),
        fft_mag[:, 48:64].sum(axis=1),
    ])  # (N, 4)

    # ── I/Q cross-features ─────────────────────────────────────────────────────
    # Normalised cross-correlation at lag 0
    I_norm   = I - I.mean(axis=1, keepdims=True)
    Q_norm   = Q - Q.mean(axis=1, keepdims=True)
    iq_xcorr = (I_norm * Q_norm).mean(axis=1) / (
        I_norm.std(axis=1) * Q_norm.std(axis=1) + 1e-12
    )
    i_power_frac = np.mean(I ** 2, axis=1) / (
        np.mean(I ** 2, axis=1) + np.mean(Q ** 2, axis=1) + 1e-12
    )

    # ── Higher-order cumulants ─────────────────────────────────────────────────
    c2_I,   c4_I,   c6_I   = _cumulants(I)
    c2_Q,   c4_Q,   c6_Q   = _cumulants(Q)
    c2_env, c4_env, c6_env = _cumulants(amplitude)

    # ── PSK-specific discriminative features ──────────────────────────────────
    # C40 of phase: canonical PSK discriminator (QPSK≈0, 8PSK≠0)
    phase_c2  = np.mean(phase ** 2, axis=1)
    phase_c40 = np.mean(phase ** 4, axis=1) - 3 * phase_c2 ** 2

    # Instantaneous frequency std (d/dt of unwrapped phase)
    phase_unwrap  = np.unwrap(phase, axis=1)
    inst_freq     = np.diff(phase_unwrap, axis=1)           # (N, 127)
    inst_freq_std = inst_freq.std(axis=1)

    # Distance to ideal QPSK and 8PSK constellation points
    qpsk_pts = np.exp(1j * np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]))
    psk8_pts = np.exp(1j * np.arange(8) * np.pi / 4)
    sig      = (I + 1j * Q)                                 # (N, 128) complex
    # Mean nearest-point distance across all 128 timesteps
    dist_qpsk = np.array([
        np.abs(s[:, None] - qpsk_pts).min(axis=1).mean() for s in sig
    ])
    dist_8psk = np.array([
        np.abs(s[:, None] - psk8_pts).min(axis=1).mean() for s in sig
    ])
    dist_ratio = dist_qpsk / (dist_8psk + 1e-12)

    # ── Stack all features ─────────────────────────────────────────────────────
    features = np.column_stack([
        # Amplitude (9)
        amp_mean, amp_std, amp_var, amp_skew, amp_kurt,
        amp_max, amp_min, amp_p2p, amp_rms,
        # Phase (4)
        phase_mean, phase_std, phase_var, phase_disc,
        # Frequency (7: centroid, spread, peak, 4 bands)
        spec_centroid, spec_spread, peak_fft_mag, band_energy,
        # I/Q cross (2)
        iq_xcorr, i_power_frac,
        # Cumulants (9)
        c2_I, c4_I, c6_I,
        c2_Q, c4_Q, c6_Q,
        c2_env, c4_env, c6_env,
        # PSK-specific (5)
        phase_c40, inst_freq_std, dist_qpsk, dist_8psk, dist_ratio,
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
    print("[INFO] Extracting 30 rich features ...")
    features = extract_features(X)
    print(f"       Feature matrix shape: {features.shape}")
    print(f"       Features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

    # Sanity check
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
