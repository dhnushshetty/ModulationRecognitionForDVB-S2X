"""
train_model.py
Train a Random Forest classifier on the extracted features.

Run from the AMR-DVB-S2X/ directory:
    python src/train_model.py
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODELS_DIR = "models"

IN_FEATURES = os.path.join(DATA_DIR, "X_features.npy")
IN_LABELS   = os.path.join(DATA_DIR, "y_labels.npy")
IN_SNRS     = os.path.join(DATA_DIR, "snrs.npy")

MODEL_PATH      = os.path.join(MODELS_DIR, "random_forest_model.pkl")
OUT_X_TEST      = os.path.join(DATA_DIR, "X_test.npy")
OUT_Y_TEST      = os.path.join(DATA_DIR, "y_test.npy")
OUT_SNRS_TEST   = os.path.join(DATA_DIR, "snrs_test.npy")

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_ESTIMATORS = 100
TEST_SIZE    = 0.2
RANDOM_STATE = 42


def main():
    # ── Load features ──────────────────────────────────────────────────────────
    for path in (IN_FEATURES, IN_LABELS, IN_SNRS):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[ERROR] '{path}' not found. Run feature_extraction.py first."
            )

    print("[INFO] Loading features ...")
    X    = np.load(IN_FEATURES, allow_pickle=True)
    y    = np.load(IN_LABELS,   allow_pickle=True)
    snrs = np.load(IN_SNRS,     allow_pickle=True)
    print(f"       X: {X.shape}, classes: {sorted(set(y))}")

    # ── Train/Test split ───────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, snrs_train, snrs_test = train_test_split(
        X, y, snrs,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\n[INFO] Split — train: {len(y_train)}, test: {len(y_test)}")

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"[INFO] Training RandomForest (n_estimators={N_ESTIMATORS}) ...")
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print("[INFO] Training complete.")

    # ── Evaluate on test set ───────────────────────────────────────────────────
    y_pred   = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Overall test accuracy: {accuracy * 100:.2f}%")

    # Feature importance summary
    importances = clf.feature_importances_
    feature_names = [
        "Mean Amplitude", "Variance of Amplitude", "Kurtosis of Amplitude",
        "Mean Phase", "Variance of Phase", "Max Amplitude",
    ]
    print("\n[INFO] Feature importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"       {name:<28} {imp:.4f}")

    # ── Save model and test split ──────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\n[INFO] Model saved to '{MODEL_PATH}'.")

    np.save(OUT_X_TEST,    X_test)
    np.save(OUT_Y_TEST,    y_test)
    np.save(OUT_SNRS_TEST, snrs_test)
    print(f"[INFO] Test split saved to data/ for evaluation.")
    print("\n[DONE] train_model.py complete. Run evaluate.py next.")


if __name__ == "__main__":
    main()
