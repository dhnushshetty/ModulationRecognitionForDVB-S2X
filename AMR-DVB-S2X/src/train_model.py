"""
train_model.py
Train Random Forest and XGBoost classifiers on the extracted features.
Saves the best model (XGBoost) for downstream evaluation.

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

MODEL_PATH      = os.path.join(MODELS_DIR, "best_model.pkl")
RF_MODEL_PATH   = os.path.join(MODELS_DIR, "random_forest_model.pkl")
OUT_X_TEST      = os.path.join(DATA_DIR, "X_test.npy")
OUT_Y_TEST      = os.path.join(DATA_DIR, "y_test.npy")
OUT_SNRS_TEST   = os.path.join(DATA_DIR, "snrs_test.npy")

# ── Hyperparameters ────────────────────────────────────────────────────────────
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

    # ── Model 1: Random Forest ─────────────────────────────────────────────────
    print("\n[INFO] Training RandomForest (n_estimators=200) ...")
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"[RESULT] Random Forest accuracy : {rf_acc * 100:.2f}%")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(rf, RF_MODEL_PATH)
    print(f"[INFO] RF model saved to '{RF_MODEL_PATH}'.")

    # ── Model 2: XGBoost ──────────────────────────────────────────────────────
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc  = le.transform(y_test)

        print("\n[INFO] Training XGBoost (n_estimators=300, max_depth=6) ...")
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        xgb.fit(X_train, y_train_enc,
                eval_set=[(X_test, y_test_enc)],
                verbose=False)
        xgb_acc = accuracy_score(y_test_enc, xgb.predict(X_test))
        print(f"[RESULT] XGBoost accuracy       : {xgb_acc * 100:.2f}%")

        # Save best model
        best_model = xgb if xgb_acc >= rf_acc else rf
        best_name  = "XGBoost" if xgb_acc >= rf_acc else "RandomForest"
        joblib.dump({'model': best_model, 'label_encoder': le if xgb_acc >= rf_acc else None},
                    MODEL_PATH)
        print(f"[INFO] Best model ({best_name}) saved to '{MODEL_PATH}'.")

    except ImportError:
        print("\n[WARN] XGBoost not installed. Skipping. Install with: pip install xgboost")
        print("[INFO] Random Forest will be used as the primary model.")
        joblib.dump({'model': rf, 'label_encoder': None}, MODEL_PATH)

    # ── Save test split ────────────────────────────────────────────────────────
    np.save(OUT_X_TEST,    X_test)
    np.save(OUT_Y_TEST,    y_test)
    np.save(OUT_SNRS_TEST, snrs_test)
    print(f"[INFO] Test split saved to data/ for evaluation.")
    print("\n[DONE] train_model.py complete. Run evaluate.py next.")


if __name__ == "__main__":
    main()
