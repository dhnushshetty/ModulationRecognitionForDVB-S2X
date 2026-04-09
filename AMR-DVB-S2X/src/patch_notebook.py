"""
Patch AMR_DVB_S2X.ipynb:
  - Cell 22: add 5 PSK-specific features to extract_features()
  - Cell 26: replace single-model training with 4-model classical ML comparison
Run from AMR-DVB-S2X/:  python src/patch_notebook.py
"""
import json, py_compile, tempfile, os

NB = os.path.join("notebooks", "AMR_DVB_S2X.ipynb")
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)


# ── Helper ─────────────────────────────────────────────────────────────────────
def check(src, label):
    tmp = tempfile.mktemp(suffix=".py")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(src)
    try:
        py_compile.compile(tmp, doraise=True)
        print(f"  {label}: OK")
    except py_compile.PyCompileError as e:
        print(f"  {label}: ERROR — {e}")
    finally:
        os.unlink(tmp)
        if os.path.exists(tmp + "c"):
            os.unlink(tmp + "c")


# ── Cell 22: feature extraction (Section 3.8) ──────────────────────────────────
CELL22 = """\
# ── Rich Feature Extraction Function ─────────────────────────────────────────
def _cumulants(x):
    c2 = np.mean(x**2, axis=1)
    c4 = np.mean(x**4, axis=1) - 3*c2**2
    c6 = np.mean(x**6, axis=1) - 15*np.mean(x**4, axis=1)*c2 + 30*c2**3
    return c2, c4, c6

def extract_features(X_data):
    I = X_data[:, 0, :]
    Q = X_data[:, 1, :]
    amplitude = np.sqrt(I**2 + Q**2)
    phase     = np.arctan2(Q, I)

    # Amplitude (9)
    amp_mean = amplitude.mean(axis=1);  amp_std  = amplitude.std(axis=1)
    amp_var  = amplitude.var(axis=1);   amp_skew = skew(amplitude, axis=1)
    amp_kurt = kurtosis(amplitude, axis=1)
    amp_max  = amplitude.max(axis=1);   amp_min  = amplitude.min(axis=1)
    amp_p2p  = amp_max - amp_min
    amp_rms  = np.sqrt(np.mean(amplitude**2, axis=1))

    # Phase (4)
    phase_mean = phase.mean(axis=1);  phase_std = phase.std(axis=1)
    phase_var  = phase.var(axis=1)
    phase_disc = (np.abs(np.diff(phase, axis=1)) > np.pi).sum(axis=1).astype(float)

    # Frequency (7)
    fft_mag = np.abs(np.fft.fft(I + 1j*Q, axis=1))[:, :64]
    freqs   = np.arange(64, dtype=float)
    total_e = fft_mag.sum(axis=1) + 1e-12
    spec_centroid = (fft_mag * freqs).sum(axis=1) / total_e
    spec_spread   = np.sqrt((fft_mag * (freqs - spec_centroid[:,None])**2).sum(axis=1) / total_e)
    peak_fft      = fft_mag.max(axis=1)
    band_e = np.column_stack([fft_mag[:,0:16].sum(axis=1), fft_mag[:,16:32].sum(axis=1),
                              fft_mag[:,32:48].sum(axis=1), fft_mag[:,48:64].sum(axis=1)])

    # I/Q cross (2)
    In = I - I.mean(axis=1, keepdims=True)
    Qn = Q - Q.mean(axis=1, keepdims=True)
    iq_xcorr     = (In*Qn).mean(axis=1) / (In.std(axis=1)*Qn.std(axis=1) + 1e-12)
    i_power_frac = np.mean(I**2, axis=1) / (np.mean(I**2, axis=1) + np.mean(Q**2, axis=1) + 1e-12)

    # Higher-order cumulants (9)
    c2I, c4I, c6I = _cumulants(I)
    c2Q, c4Q, c6Q = _cumulants(Q)
    c2E, c4E, c6E = _cumulants(amplitude)

    # PSK-specific discriminative features (5)
    phase_c2  = np.mean(phase**2, axis=1)
    phase_c40 = np.mean(phase**4, axis=1) - 3 * phase_c2**2

    phase_unwrap  = np.unwrap(phase, axis=1)
    inst_freq_std = np.diff(phase_unwrap, axis=1).std(axis=1)

    qpsk_pts = np.exp(1j * np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]))
    psk8_pts = np.exp(1j * np.arange(8) * np.pi / 4)
    sig      = I + 1j * Q
    dist_qpsk = np.array([np.abs(s[:,None] - qpsk_pts).min(axis=1).mean() for s in sig])
    dist_8psk = np.array([np.abs(s[:,None] - psk8_pts).min(axis=1).mean() for s in sig])
    dist_ratio = dist_qpsk / (dist_8psk + 1e-12)

    return np.column_stack([
        amp_mean, amp_std, amp_var, amp_skew, amp_kurt,
        amp_max, amp_min, amp_p2p, amp_rms,
        phase_mean, phase_std, phase_var, phase_disc,
        spec_centroid, spec_spread, peak_fft, band_e,
        iq_xcorr, i_power_frac,
        c2I, c4I, c6I, c2Q, c4Q, c6Q, c2E, c4E, c6E,
        phase_c40, inst_freq_std, dist_qpsk, dist_8psk, dist_ratio,
    ])

FEATURE_NAMES = [
    "Amp Mean","Amp Std","Amp Var","Amp Skewness","Amp Kurtosis",
    "Amp Max","Amp Min","Amp Peak2Peak","Amp RMS",
    "Phase Mean","Phase Std","Phase Var","Phase Disc Count",
    "Spectral Centroid","Spectral Spread","Peak FFT Mag",
    "FFT Band0","FFT Band1","FFT Band2","FFT Band3",
    "IQ CrossCorr","I Power Fraction",
    "C20 I","C40 I","C60 I","C20 Q","C40 Q","C60 Q",
    "C20 Env","C40 Env","C60 Env",
    "C40 Phase","Inst Freq Std","Dist QPSK","Dist 8PSK","Dist Ratio",
]

X_features = extract_features(X)
print(f"Feature matrix shape: {X_features.shape}  ({len(FEATURE_NAMES)} features)")

# Correlation heatmap (sample 3000 rows for speed)
idx = np.random.choice(len(X_features), 3000, replace=False)
corr_df = pd.DataFrame(X_features[idx], columns=FEATURE_NAMES).corr()
fig, ax = plt.subplots(figsize=(16, 13))
sns.heatmap(corr_df, annot=True, fmt=".1f", cmap="coolwarm",
            vmin=-1, vmax=1, square=True, linewidths=0.3, annot_kws={"size":5}, ax=ax)
ax.set_title("Feature Correlation Heatmap (36 features)", fontsize=14)
plt.tight_layout()
plt.show()
print("Note: Highly correlated features (|r| > 0.9) are candidates for removal.")
"""

# ── Cell 26: 4-model comparison (Section 5) ────────────────────────────────────
CELL26 = """\
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

X_train, X_test, y_train, y_test, snrs_train, snrs_test = train_test_split(
    X_features, y, snrs,
    test_size=0.2, random_state=42, stratify=y
)
print(f"Train size : {len(y_train)},  Test size : {len(y_test)}")

models = {
    "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=15,
                               min_samples_leaf=5, random_state=42, n_jobs=-1),
    "Decision Tree":       DecisionTreeClassifier(max_depth=12, min_samples_leaf=10,
                               random_state=42),
    "Logistic Regression": Pipeline([("scaler", StandardScaler()),
                               ("lr", LogisticRegression(C=1.0, max_iter=1000,
                                    random_state=42, n_jobs=-1))]),
    "Naive Bayes":         Pipeline([("scaler", StandardScaler()),
                               ("nb", GaussianNB())]),
}

results = {}
for name, model in models.items():
    print(f"Training {name} ...")
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = acc
    print(f"  {name:<22} : {acc*100:.2f}%")

# XGBoost comparison
if XGBOOST_AVAILABLE:
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_train)
    y_te_enc  = le.transform(y_test)
    clf_xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             use_label_encoder=False, eval_metric="logloss",
                             random_state=42, n_jobs=-1)
    clf_xgb.fit(X_train, y_tr_enc, eval_set=[(X_test, y_te_enc)], verbose=False)
    y_pred_xgb = le.inverse_transform(clf_xgb.predict(X_test))
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    results["XGBoost"] = acc_xgb
    print(f"  {'XGBoost':<22} : {acc_xgb*100:.2f}%")

# Best classical model for downstream evaluation
clf = models["Random Forest"]
y_pred = clf.predict(X_test)
acc_rf = results["Random Forest"]

# Comparison bar chart
baseline = 54.79
fig, ax = plt.subplots(figsize=(11, 5))
model_names = list(results.keys())
model_accs  = [results[n]*100 for n in model_names]
colors = ["#5bc0de","#f0ad4e","#d9534f","#9b59b6","#5cb85c"][:len(model_names)]
bars = ax.bar(model_names, model_accs, color=colors, edgecolor="white", width=0.5)
ax.axhline(baseline, color="gray",  linestyle="--", linewidth=1,
           label=f"Old RF baseline ({baseline}%)")
ax.axhline(90, color="green", linestyle="--", linewidth=1, label="90% target")
for bar, acc in zip(bars, model_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(0, 105)
ax.set_ylabel("Test Accuracy (%)", fontsize=12)
ax.set_title("Classical ML Model Comparison (36 features)", fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()
"""

# ── Cell 27: feature importance (RF only, simpler) ────────────────────────────
CELL27 = """\
# Feature importance — Random Forest
feat_imp_rf = pd.Series(models["Random Forest"].feature_importances_,
                         index=FEATURE_NAMES).sort_values()
fig, ax = plt.subplots(figsize=(9, 10))
feat_imp_rf.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Random Forest — Feature Importances (36 features)", fontsize=13)
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.show()
"""

check(CELL22, "Cell 22")
check(CELL26, "Cell 26")
check(CELL27, "Cell 27")

nb["cells"][22]["source"] = [CELL22]
nb["cells"][22]["outputs"] = []
nb["cells"][22]["execution_count"] = None

nb["cells"][26]["source"] = [CELL26]
nb["cells"][26]["outputs"] = []
nb["cells"][26]["execution_count"] = None

nb["cells"][27]["source"] = [CELL27]
nb["cells"][27]["outputs"] = []
nb["cells"][27]["execution_count"] = None

# Clear downstream outputs (cells 28-31 depend on clf/y_pred)
for i in range(28, min(35, len(nb["cells"]))):
    nb["cells"][i]["outputs"] = []
    nb["cells"][i]["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook patched. Total cells: {len(nb['cells'])}")
