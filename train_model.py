"""train_model.py
- Generates synthetic patient data and medicines dataset
- Trains a RandomForestClassifier for disease prediction
- Builds TF-IDF vectors for medicines descriptions for content-based recommendations
- Saves artifacts into models/
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 1) Create synthetic patient dataset
patients_csv = os.path.join(DATA_DIR, "patients.csv")
if not os.path.exists(patients_csv):
    np.random.seed(42)
    n = 500
    ages = np.random.randint(18, 90, size=n)
    bp = np.random.randint(80, 180, size=n)          # blood pressure
    glucose = np.random.randint(60, 250, size=n)     # glucose level
    hr = np.random.randint(50, 120, size=n)          # heart rate
    # Synthetic diagnoses: 'healthy', 'diabetes', 'hypertension', 'cardiac'
    diag = []
    for a,b,g,h in zip(ages,bp,glucose,hr):
        if g > 180:
            diag.append('diabetes')
        elif b > 140:
            diag.append('hypertension')
        elif h > 100:
            diag.append('cardiac')
        else:
            diag.append('healthy')
    df = pd.DataFrame({
        "age": ages,
        "blood_pressure": bp,
        "glucose": glucose,
        "heart_rate": hr,
        "diagnosis": diag
    })
    df.to_csv(patients_csv, index=False)
else:
    df = pd.read_csv(patients_csv)

# 2) Train a simple disease model
X = df[["age","blood_pressure","glucose","heart_rate"]]
y = df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)
clf = RandomForestClassifier(n_estimators=100, random_state=1)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(classification_report(y_test, pred))
joblib.dump(clf, os.path.join(MODEL_DIR, "disease_model.joblib"))
print("Saved disease model to models/disease_model.joblib")

# 3) Create a medicines dataset and vectorizer
meds_csv = os.path.join(DATA_DIR, "medicines.csv")
if not os.path.exists(meds_csv):
    meds = [
        {"id":1,"name":"Metformin","desc":"Used to treat high blood sugar in type 2 diabetes."},
        {"id":2,"name":"Insulin","desc":"Hormone therapy for diabetes to control blood glucose levels."},
        {"id":3,"name":"Lisinopril","desc":"Used for high blood pressure and heart failure."},
        {"id":4,"name":"Amlodipine","desc":"Calcium channel blocker for hypertension."},
        {"id":5,"name":"Aspirin","desc":"Pain reliever; low-dose aspirin used for cardiac protection."},
        {"id":6,"name":"Atorvastatin","desc":"Lowers cholesterol, helps prevent cardiovascular disease."},
        {"id":7,"name":"Glimepiride","desc":"Oral diabetes medicine that helps control blood sugar."}
    ]
    pd.DataFrame(meds).to_csv(meds_csv, index=False)

meds_df = pd.read_csv(meds_csv)
vectorizer = TfidfVectorizer()
item_vectors = vectorizer.fit_transform(meds_df["desc"].astype(str))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
joblib.dump(meds_df, os.path.join(MODEL_DIR, "medicines_df.joblib"))
print("Saved TF-IDF vectorizer and medicines dataframe to models/")
