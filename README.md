# Personalized Healthcare Recommendation System (Minimal)

This is a minimal, runnable starter project implementing:
- A simple **disease prediction** model (Random Forest) trained on synthetic data.
- A **content-based medicine recommender** using TF-IDF + cosine similarity.
- A small **Flask API** with JWT-based authentication (signup/login).
- Example frontend (static) that calls the API endpoints.
- Scripts to (re)train models and recreate assets.

## Structure
```
personalized_reco_system/
├─ app.py                 # Flask application (API)
├─ train_model.py         # Generates synthetic data and trains models
├─ requirements.txt
├─ README.md
├─ data/
│  ├─ patients.csv
│  └─ medicines.csv
├─ models/
│  ├─ disease_model.joblib    # (created by train_model.py)
│  └─ tfidf_vectorizer.joblib # (created by train_model.py)
└─ frontend/
   └─ index.html
```

## Quick start (local)
1. Create a virtualenv:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train models (generates models/ files):
   ```bash
   python train_model.py
   ```
   This generates `models/disease_model.joblib` and `models/tfidf_vectorizer.joblib`.
4. Run the Flask app:
   ```bash
   export FLASK_APP=app.py
   export FLASK_ENV=development
   flask run
   ```
   Or: `python app.py` to run directly.

5. Visit the frontend:
   Open `frontend/index.html` in your browser and follow instructions (or use curl/Postman).

## Notes
- This project uses synthetic data for demonstration. Replace `data/patients.csv` with real de-identified clinical data if you have it.
- For production use (real healthcare data), ensure compliance with local regulations (HIPAA, GDPR), use HTTPS, proper authentication, logging, and secure storage.
