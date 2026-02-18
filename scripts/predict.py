import pickle
import pandas as pd

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
print("model saved successfully as model.pkl")


test = pd.read_csv("Preprocessed-Test_clean.csv")
X_submission = test.drop(["ID"], axis=1)
final_preds_encoded = model.predict(X_submission)
inv_map = {0: 'High', 1: 'Low', 2: 'Medium'}
final_preds_labels = [inv_map[p] for p in final_preds_encoded]
submission_df = pd.DataFrame({
    'Id':test["ID"],
    'Target': final_preds_labels
})
submission_df.to_csv('predictions.csv', index=False)
print("File saved successfully as predictions.csv")