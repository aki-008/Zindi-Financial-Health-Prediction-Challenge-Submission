
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import randint
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('data\\Train.csv')
test_df = pd.read_csv('data\\Test.csv')

import numpy as np
import pandas as pd

def create_financial_features(df):

    df = df.copy()

    required_cols = [
        'business_turnover', 'business_expenses', 
        'personal_income', 'owner_age', 'business_age_years'
    ]
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
            
    df['profit_margin'] = (
        (df['business_turnover'] - df['business_expenses']) / 
        df['business_turnover'].replace(0, np.nan)
    )
    
    df['expense_ratio'] = (
        df['business_expenses'] / 
        df['personal_income'].replace(0, np.nan)
    )
    
    df['age_per_year_business'] = (
        df['owner_age'] / 
        df['business_age_years'].replace(0, np.nan)
    )
    
    return df

train_df = create_financial_features(train_df)
test_df = create_financial_features(test_df)

def preprocess(train_df, test_df):
    X_train = train_df.drop(['ID', 'Target'], axis=1)

    test_ids = test_df['ID']
    X_test = test_df.drop(['ID'], axis=1)

    num_cols = X_train.select_dtypes(include=['number']).columns
    cat_cols = X_train.select_dtypes(include=['object']).columns

    imp_num = SimpleImputer(strategy='median')
    X_train[num_cols] = imp_num.fit_transform(X_train[num_cols])
    X_test[num_cols] = imp_num.transform(X_test[num_cols])

    imp_cat = SimpleImputer(strategy='most_frequent')
    X_train[cat_cols] = imp_cat.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = imp_cat.transform(X_test[cat_cols])

    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    X_train[cat_cols] = oe.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = oe.transform(X_test[cat_cols])


    X_train["Target"] = train_df['Target']
    X_train["ID"] = train_df['ID']

    X_test["ID"] = test_df['ID']

    return X_train, X_test

train, test = preprocess(train_df, test_df)

train.to_csv("data\\Preprocessed-Train_clean.csv", index=False)
test.to_csv("data\\Preprocessed-Test_clean.csv", index=False)

X = train.drop(['ID', 'Target'], axis=1)

le = LabelEncoder()
y = le.fit_transform(train['Target'])
print("Target Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))


X_temp, X_test_internal, y_temp, y_test_internal = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp
)

print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}")

pipe = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("rf", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        warm_start=False
    ))
])


param_dist = {
    "rf__n_estimators": randint(200, 1000), 
    "rf__max_depth": [8, 10, 12, 14, 20, None], 
    "rf__min_samples_split": randint(2, 25),
    "rf__min_samples_leaf": randint(1, 10),
    "rf__max_features": ["sqrt", "log2", 0.5],
    "rf__class_weight": ["balanced", "balanced_subsample"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rs = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=40,      
    scoring="f1_macro",
    n_jobs=-1,
    cv=cv,
    verbose=2,
    random_state=42,
    refit=True
)

print("--- Searching for best hyperparameters... ---")
rs.fit(X_train, y_train)

print("Best params:", rs.best_params_)
print("Best CV f1_macro:", rs.best_score_)

best_model = rs.best_estimator_
print("--- Training and Calibrating best estimator... ---")
calibrated = CalibratedClassifierCV(best_model, cv=cv, method='sigmoid')
calibrated.fit(X_train, y_train)

y_pred = calibrated.predict(X_val)
y_proba = calibrated.predict_proba(X_val)

print("--- Validation Report (after tuning + calibration) ---")

print(classification_report(y_val, y_pred, digits=4))
print("f1_macro:", f1_score(y_val, y_pred, average="macro"))

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_val, y_pred)
print(cm)
disp =  ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=calibrated.classes_)
disp.plot()
plt.show()

ConfusionMatrixDisplay.from_predictions(
    y_val, 
    y_pred, 
    normalize='true', 
    values_format='.1%',
    display_labels=calibrated.classes_
)
plt.show()

X_submission = test.drop(["ID"], axis=1)

final_preds_encoded = calibrated.predict(X_submission)

import pandas as pd

inv_map = {0: 'High', 1: 'Low', 2: 'Medium'}

final_preds_labels = [inv_map[p] for p in final_preds_encoded]

submission_df = pd.DataFrame({
    'Id':test_df["ID"],
    'Target': final_preds_labels
})

submission_df.to_csv('Submission\\predictions.csv', index=False)

print("File saved successfully as predictions.csv")

import pickle

with open('model\\model.pkl', 'wb') as f:
    pickle.dump(calibrated, f)

print("Model saved successfully as model.pkl")