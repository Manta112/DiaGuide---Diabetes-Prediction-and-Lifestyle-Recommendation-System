"""
DiaGuide - Train Models
Run: python train_model.py
"""
import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib, sklearn

BASE = os.path.dirname(os.path.abspath(__file__))
MD   = os.path.join(BASE, "models_saved")
os.makedirs(MD, exist_ok=True)

print(f"scikit-learn: {sklearn.__version__}")

# Load dataset
csv = os.path.join(BASE, "dataset", "diabetes.csv")
df  = pd.read_csv(csv)
print(f"Dataset: {len(df)} records | Diabetic: {df['Outcome'].sum()}")

FEAT = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
        'Insulin','BMI','DiabetesPedigreeFunction','Age']

# Replace zeros with median
for c in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
    df[c] = df[c].replace(0, df[df[c]>0][c].median())

X = df[FEAT].values
y = df['Outcome'].values

scaler = StandardScaler()
Xs     = scaler.fit_transform(X)

Xtr,Xte,ytr,yte = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=42)

# Clean models - no deprecated params
rf  = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
lr  = LogisticRegression(C=0.8, max_iter=2000, solver='lbfgs', random_state=42)
ens = VotingClassifier(estimators=[('rf',rf),('lr',lr)], voting='soft')

# Cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv  = cross_val_score(ens, Xs, y, cv=skf, scoring='accuracy')
print(f"CV Accuracy : {cv.mean()*100:.2f}% (+/- {cv.std()*100:.2f}%)")

# Train
ens.fit(Xtr,ytr); rf.fit(Xtr,ytr); lr.fit(Xtr,ytr)
yp = ens.predict(Xte); ypr = ens.predict_proba(Xte)[:,1]
print(f"Test Accuracy: {accuracy_score(yte,yp)*100:.2f}%")
print(f"ROC-AUC     : {roc_auc_score(yte,ypr):.4f}")

# Save
joblib.dump(ens,    os.path.join(MD,"ensemble.pkl"))
joblib.dump(rf,     os.path.join(MD,"rf.pkl"))
joblib.dump(lr,     os.path.join(MD,"lr.pkl"))
joblib.dump(scaler, os.path.join(MD,"scaler.pkl"))
json.dump(dict(zip(FEAT, rf.feature_importances_.tolist())),
          open(os.path.join(MD,"feature_importance.json"),"w"), indent=2)

print("\nAll models saved to models_saved/")
print("Now run: streamlit run app.py")
