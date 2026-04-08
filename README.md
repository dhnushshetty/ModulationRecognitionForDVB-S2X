# 🛰️ Automatic Modulation Recognition (AMR) for DVB-S2X Waveforms

> **Beginner ML Project** | Classification using Random Forest | Dataset: RadioML 2016.10A

---

## 📄 Abstract

Satellite communication systems operating under the DVB-S2X standard dynamically switch between multiple modulation schemes — including QPSK and 8PSK — based on prevailing channel conditions. Accurate and automatic identification of these modulation schemes at the ground receiver is essential for reliable signal synchronization. Traditional manual and rule-based modulation recognition methods are computationally expensive and error-prone under noisy conditions.

This project presents an Automatic Modulation Recognition (AMR) system that uses a **Random Forest classifier** to identify DVB-S2X modulation schemes from raw In-phase and Quadrature (I/Q) signal samples. The system is trained and evaluated on the **RadioML 2016.10A** dataset — a synthetic dataset generated using GNU Radio, containing 11 modulation types across SNR levels from -20 dB to +18 dB, with 1,000 samples per modulation-SNR pair. Statistical features including amplitude, phase, and kurtosis are extracted from I/Q samples and used for classification. Model performance is evaluated using classification accuracy and confusion matrices across multiple SNR levels.

---

## 📁 Project Structure

```
AMR-DVB-S2X/
│
├── data/
│   └── RML2016.10a.pkl          # Downloaded dataset (place here)
│
├── src/
│   ├── load_data.py             # Load and filter dataset
│   ├── feature_extraction.py   # Extract features from I/Q samples
│   ├── train_model.py          # Train Random Forest classifier
│   └── evaluate.py             # Accuracy + confusion matrix
│
├── models/
│   └── random_forest_model.pkl  # Saved trained model
│
├── results/
│   ├── confusion_matrix.png     # Output plot
│   └── accuracy_per_snr.png    # Output plot
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🗄️ Dataset

| Field | Detail |
|---|---|
| **Name** | RadioML 2016.10A |
| **Source** | DeepSig Inc. (GNU Radio synthetic) |
| **Modulation Types** | 11 total (we use QPSK, 8PSK) |
| **SNR Range** | -20 dB to +18 dB (2 dB steps) |
| **Samples per class/SNR** | 1,000 |
| **Total Samples** | 220,000 |
| **Format** | `.pkl` (Python pickle) |
| **Signal Format** | Raw I/Q (In-phase & Quadrature) samples |

### 📥 Download

```
Direct download:
http://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2
```

After downloading, extract and place `RML2016.10a.pkl` inside the `data/` folder.

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| Data Loading | `pickle`, `numpy` |
| Feature Extraction | `numpy`, `scipy` |
| ML Model | `scikit-learn` (Random Forest) |
| Evaluation | `scikit-learn`, `matplotlib`, `seaborn` |
| Model Saving | `joblib` |

---

## 🚀 Setup & Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/your-username/AMR-DVB-S2X.git
cd AMR-DVB-S2X
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download and place dataset
```bash
# Download
wget http://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2

# Extract
tar -xvjf RML2016.10a.tar.bz2

# Move to data folder
mv RML2016.10a.pkl data/
```

### Step 4 — Run the pipeline
```bash
python src/load_data.py
python src/feature_extraction.py
python src/train_model.py
python src/evaluate.py
```

---

## 🔬 ML Pipeline

```
Raw Dataset (RML2016.10a.pkl)
          ↓
  Load & Filter
  (Keep only QPSK, 8PSK)
          ↓
  Feature Extraction from I/Q
  - Mean Amplitude
  - Variance of Amplitude
  - Kurtosis
  - Mean Phase
  - Variance of Phase
  - Max Amplitude
          ↓
  Train/Test Split (80/20)
          ↓
  Train Random Forest Classifier
          ↓
  Evaluate:
  - Overall Accuracy
  - Accuracy per SNR level
  - Confusion Matrix
          ↓
  Save Model → models/random_forest_model.pkl
```

---

## 📊 Expected Results

| Metric | Expected Value |
|---|---|
| Accuracy at high SNR (≥ 10 dB) | > 90% |
| Accuracy at low SNR (≤ -10 dB) | 50–65% |
| Overall Accuracy | ~75–85% |
| Classes | QPSK, 8PSK |

---

## 📦 requirements.txt

```
numpy
scipy
scikit-learn
matplotlib
seaborn
joblib
pickle5
```

---

## 🤖 Using Claude Code for Further Development

This project is designed to be extended using **Claude Code** — Anthropic's agentic CLI coding tool.

### What is Claude Code?
Claude Code is a command-line tool that understands your codebase and helps you develop faster using natural language. It can read files, write code, run commands, and handle Git workflows autonomously.

### Installation

**macOS / Linux / WSL:**
```bash
curl -fsSL https://claude.ai/install.sh | bash
```

**Windows PowerShell:**
```powershell
irm https://claude.ai/install.ps1 | iex
```

> ⚠️ Requires a Claude Pro, Max, Team, Enterprise, or Console account. The free plan does not include Claude Code access.

After installation, log in:
```bash
claude
# Follow the browser prompts to authenticate
```

### Using Claude Code on This Project

Navigate to the project folder and launch Claude Code:
```bash
cd AMR-DVB-S2X
claude
```

#### 💬 Example Prompts for Further Development

**Explore the dataset:**
```
"Load the RML2016.10a.pkl file and show me the structure,
available modulation types, and sample shapes"
```

**Build feature extraction:**
```
"Write a Python script that extracts amplitude mean, variance,
kurtosis, and phase statistics from I/Q signal samples in the dataset"
```

**Train the model:**
```
"Train a Random Forest classifier on the extracted features,
split 80/20 train/test, and print the accuracy"
```

**Add more modulations:**
```
"Update the pipeline to also include AM-DSB and WBFM
modulations from the dataset alongside QPSK and 8PSK"
```

**Try other models:**
```
"Replace the Random Forest with an SVM classifier using
an RBF kernel and compare accuracy with Random Forest"
```

**Plot results:**
```
"Generate a confusion matrix heatmap and an accuracy vs SNR
line plot, and save them to the results/ folder"
```

**Improve accuracy:**
```
"Add more features like spectral kurtosis and autocorrelation
to the feature extraction script and retrain the model"
```

**Save and load the model:**
```
"Save the trained Random Forest model using joblib to
models/random_forest_model.pkl and write a load function"
```

---

## 📅 Development Roadmap

| Week | Task | Status |
|---|---|---|
| Week 1 | Project planning, problem definition, tooling setup | ✅ Done |
| Week 2 | Functional modeling, use case diagrams, actor identification | ✅ Done |
| Week 3 | Dataset download, exploration, preprocessing | 🔲 Pending |
| Week 4 | Feature extraction from I/Q samples | 🔲 Pending |
| Week 5 | Model training (Random Forest) | 🔲 Pending |
| Week 6 | Evaluation, confusion matrix, accuracy vs SNR | 🔲 Pending |
| Week 7 | Model optimization, compare with SVM/KNN | 🔲 Pending |
| Week 8 | Final report and documentation | 🔲 Pending |

---

## 👥 Team Roles

| Role | Responsibility |
|---|---|
| Project Manager | Planning, tracking, documentation |
| Signal Processing Engineer | Dataset handling, feature extraction |
| ML Engineer | Model training, evaluation, optimization |
| Testing Lead | Validation, results verification |

---

## 📚 References

- T.J. O'Shea, "RadioML2016.10A", DeepSig Inc., 2016
- ETSI EN 302 307-2 — DVB-S2X Standard Specification
- Scikit-learn Documentation: https://scikit-learn.org
- Claude Code Documentation: https://docs.anthropic.com/en/docs/claude-code/overview

---

## 📝 Notes for Beginners

- You do **not** need to understand all the signal processing theory to run this project
- Claude Code can explain any part of the code to you — just ask it in plain English
- Start by running the scripts one by one and reading the output
- If anything breaks, paste the error into Claude Code and ask it to fix it

---

*Project developed as part of academic coursework on Automatic Modulation Recognition for satellite communication systems.*
