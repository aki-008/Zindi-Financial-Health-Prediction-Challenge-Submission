# Zindi Financial Health Prediction Challenge – Financial Health Index (FHI)

## Overview

This repository contains the complete training and inference pipeline used for the data.org Financial Health Prediction Challenge hosted on Zindi.

GitHub Repository:
[https://github.com/aki-008/Zindi-Financial-Health-Prediction-Challenge-Submission.git](https://github.com/aki-008/Zindi-Financial-Health-Prediction-Challenge-Submission.git)

The objective is to predict the Financial Health Index (FHI) of small and medium-sized enterprises (SMEs) across Southern Africa.

Each SME is classified into one of three categories:

* Low
* Medium
* High

The evaluation metric used in the challenge is Macro F1 Score.

---

## Competition Context

SMEs are critical to economic growth and employment but often face financial vulnerability due to:

* Limited access to credit
* Unstable cash flow
* Exposure to economic shocks

This challenge builds a data-driven Financial Health Index using:

* Savings and assets indicators
* Debt and repayment capacity
* Business resilience factors
* Access to financial services

---

## Repository Structure

```
aki-008-zindi-financial-health-prediction-challenge-submission/
├── README.md
├── environment.yml
├── requirements.txt
├── scripts/
│   ├── Train.py        # Model training pipeline
│   └── predict.py      # Inference and submission generation
└── Submission/
    └── predictions.csv # Final submission output
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/aki-008/Zindi-Financial-Health-Prediction-Challenge-Submission.git
cd Zindi-Financial-Health-Prediction-Challenge-Submission
```

---

### 2. Create Environment

You can use either conda or venv.

Option A — Using Conda

```bash
conda env create -f environment.yml
conda activate zindi-fhi
```

Option B — Using Virtual Environment

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

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Download Competition Data

1. Visit the Zindi competition page.
2. Download:

   * Train dataset
   * Test dataset
   * Sample submission file
3. Place them in the project root (or update file paths inside scripts accordingly).

Example expected location:

```
project_root/
    train.csv
    test.csv
```

---

## Workflow

### Train the Model

Train the model using the training script:

```bash
python scripts/Train.py
```

This script:

* Loads the training data
* Applies preprocessing and feature engineering
* Handles class imbalance (if enabled)
* Trains the classification model
* Evaluates using Macro F1 Score
* Saves model artifacts if configured

### Generate Predictions

Make predictions using the inference script:

```bash
python scripts/predict.py
```

This script:

* Loads the trained model
* Applies identical preprocessing to the test data
* Generates predictions
* Creates the submission file

Output file:

```
Submission/predictions.csv
```

Submission format:

```
ID,Target
ID_XXXX,Low
ID_YYYY,Medium
```

In addition, an accompanying Jupyter Notebook (.ipynb) is included in the repository that demonstrates the full workflow step by step, including preprocessing, model training, evaluation, and prediction generation.

---

## Modeling Approach

Key techniques used:

* Structured preprocessing pipeline
* Feature engineering from numeric and categorical variables
* Handling class imbalance
* Cross-validation for stability
* Error analysis of misclassified samples

Primary objective: maximize Macro F1 Score while maintaining balanced class performance.

---

## Evaluation Metric

Macro F1 Score

Why Macro F1?

* Treats all classes equally
* Prevents bias toward majority class
* Encourages balanced performance across Low, Medium, and High categories

---

## Submission Instructions

1. Run predict.py.
2. Verify CSV format: ID,Target.
3. Upload Submission/predictions.csv to Zindi.

Competition limits:

* Limited daily submissions
* Public leaderboard uses partial test data
* Private leaderboard uses full hidden test data

---

## Suggestions for Improvement

* Advanced feature engineering
* Ensemble stacking
* Probability calibration
* Threshold tuning
* Detailed confusion matrix analysis
* Cross-validation averaging

---

## License

This repository follows the competition data usage policy.

---

## Acknowledgements

* data.org
* FinMark Trust
* Zindi platform

---

## Contact

For questions or collaboration, open an issue on the GitHub repository.
