# Zindi Financial Health Prediction Challenge – Financial Health Index (FHI)

## 📌 Overview

This repository contains the complete pipeline used for the **data.org Financial Health Prediction Challenge** hosted on Zindi.

The goal of the competition is to predict the **Financial Health Index (FHI)** of small and medium-sized enterprises (SMEs) across Southern Africa using socio-economic and business data.

Each SME must be classified into one of three categories:

* **Low**
* **Medium**
* **High**

The evaluation metric used in the challenge is **Macro F1 Score**.

---

## 🌍 Competition Context

SMEs play a critical role in employment and economic growth but often face financial instability due to:

* Limited access to credit
* Unstable cash flow
* Exposure to economic and climate shocks

This challenge introduces a **data-driven Financial Health Index** based on:

* Savings & assets
* Debt & repayment ability
* Resilience to shocks
* Access to financial services

Participants must build machine learning models using business, demographic, and economic indicators to predict financial well-being.

---

## 🗂️ Repository Structure

```
aki-008-zindi-financial-health-prediction-challenge-submission/
├── README.md
├── scripts/
│   ├── Train.py        # Model training pipeline
│   └── predict.py      # Inference & submission generation
└── Submission/
    └── predictions.csv # Final output file
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd <repo-name>
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

```bash
# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Download Competition Data

1. Go to the Zindi competition page
2. Download:

   * Train dataset
   * Test dataset
   * Sample submission
3. Place them in:

```
data/raw/
```

---

## 🔍 Step-by-Step Usage Guide

### Step 1 — Exploratory Data Analysis

Run the notebook:

```
notebooks/EDA.ipynb
```

This helps understand:

* Class imbalance
* Missing values
* Feature distributions

---

### Step 2 — Feature Engineering

Run:

```
notebooks/feature_engineering.ipynb
```

This step includes:

* Categorical encoding
* Binary mapping
* Interaction features
* Aggregations

Output:

```
data/processed/train_processed.csv
```

---

### Step 3 — Model Training

Run:

```bash
python src/train.py
```

Typical pipeline:

* Train/validation split
* Oversampling (SMOTE/ADASYN optional)
* Feature selection
* RandomForest / XGBoost / LightGBM
* Hyperparameter tuning

Output:

```
models/model.pkl
```

---

### Step 4 — Validation

Evaluate model performance:

* Classification report
* Confusion matrix
* Macro F1 score

Focus on improving recall for:

* Medium
* High

---

### Step 5 — Generate Predictions

```bash
python src/predict.py
```

Output file:

```
submissions/submission.csv
```

Format:

```
ID,Target
ID_XXXX,Low
ID_YYYY,Medium
```

---

## 🧠 Modeling Strategy

Key approaches used:

* Class imbalance handling
* Feature engineering from categorical + numeric interactions
* Ensemble models
* Calibration
* Error analysis using misclassified samples

---

## 📊 Evaluation Metric

**Macro F1 Score**

Why?

* Balanced importance across all classes
* Penalizes poor performance on minority classes

---

## 🚀 Submission Guide

1. Generate predictions
2. Ensure format:

```
ID,Target
```

3. Upload CSV to Zindi

Limits:

* 10 submissions per day
* Public leaderboard = ~30% test data
* Private leaderboard = ~70% test data

---

## 🏆 Tips to Improve Score

* Handle class imbalance carefully
* Engineer domain-inspired features
* Analyze misclassified samples
* Use cross-validation
* Tune hyperparameters

---

## 📚 Learning Resources

Recommended topics:

* Responsible AI
* Financial inclusion datasets
* Feature engineering for tabular ML

---

## 🧾 License

This project follows the competition's **CC-BY-SA 4.0** data usage policy.

---

## 🙌 Acknowledgements

* data.org
* FinMark Trust
* Zindi platform

---

## 📬 Contact

For questions, open an issue in the repository.
