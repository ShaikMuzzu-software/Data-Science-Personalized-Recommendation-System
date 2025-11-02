# app.py
"""
Flask demo app for disease prediction and medicine recommendations.

This version:
- Keeps a trimmed recent-history list (LAST_LOG_ENTRIES, default 20).
- Exposes /last_request (most recent) and /last_requests (list).
- Preserves endpoints: /signup, /login, /recommend, /predict, /predict_open, /ping
- Serves login.html and dashboard.html at /login_page and /dashboard.
"""

import os
from pathlib import Path
import sqlite3
import joblib
import datetime
from flask import Flask, request, jsonify, send_from_directory, g, redirect, url_for
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

from recommend_semantic import (
    query_recommendations,
    med_embeddings_available,
    tfidf_available,
    tfidf_query_topk,
)

BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder=str(BASE), static_folder=str(BASE))
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "change_this_for_demo_only")
CORS(app)
jwt = JWTManager(app)

# ---------------- simple sqlite user store ----------------
DB_PATH = BASE / "users.db"

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = sqlite3.connect(str(DB_PATH))
        db.row_factory = sqlite3.Row
        g._database = db
    return db

def init_db():
    db = get_db()
    db.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )
    """
    )
    db.commit()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

def create_user(username, password):
    pw_hash = generate_password_hash(password, method="pbkdf2:sha256", salt_length=8)
    try:
        db = get_db()
        db.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, pw_hash))
        db.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    except Exception as e:
        print("Error creating user:", e)
        return False

def verify_user(username, password):
    row = get_db().execute("SELECT password_hash FROM users WHERE username = ?", (username,)).fetchone()
    if not row:
        return False
    return check_password_hash(row["password_hash"], password)

# ---------------- model loading ----------------
disease_model = None

def load_disease_model():
    global disease_model
    try:
        path = MODELS_DIR / "disease_model.joblib"
        if path.exists():
            disease_model = joblib.load(path)
            print("Loaded disease_model.joblib")
        else:
            disease_model = None
            print("disease_model.joblib not found in models/")
    except Exception as e:
        disease_model = None
        print("Failed to load disease_model:", e)

# ---------------- Recent-history (trimmed) ----------------
LAST_LOG_ENTRIES = 20
# store list of {timestamp, endpoint, request, response}
LAST_LOG = []

def _log_request_response(endpoint: str, req_obj, resp_obj):
    """Append to LAST_LOG, keep only LAST_LOG_ENTRIES most recent."""
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "endpoint": endpoint,
        "request": req_obj,
        "response": resp_obj,
    }
    LAST_LOG.append(entry)
    # trim
    if len(LAST_LOG) > LAST_LOG_ENTRIES:
        del LAST_LOG[0: len(LAST_LOG) - LAST_LOG_ENTRIES]

# ---------------- Serve pages ----------------
@app.route("/")
def root():
    return redirect(url_for("login_page"))

@app.route("/login_page")
def login_page():
    path = BASE / "login.html"
    if path.exists():
        return send_from_directory(str(BASE), "login.html")
    return jsonify({"error": "login.html not found"}), 404

@app.route("/dashboard")
def dashboard():
    path = BASE / "dashboard.html"
    if path.exists():
        return send_from_directory(str(BASE), "dashboard.html")
    return jsonify({"error": "dashboard.html not found"}), 404

@app.route("/demo")
def demo():
    demo_file = "reco_demo_index.html"
    demo_path = BASE / demo_file
    if demo_path.exists():
        return send_from_directory(str(BASE), demo_file)
    return jsonify({"error": "demo HTML not found."}), 404

# ---------------- Auth API ----------------
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"msg": "username and password required"}), 400
    if not create_user(username, password):
        return jsonify({"msg": "user exists or creation failed"}), 400
    token = create_access_token(identity=username)
    return jsonify(access_token=token)

@app.route("/login", methods=["POST"])
def login():
    data = request.json or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"msg": "username and password required"}), 400
    if not verify_user(username, password):
        return jsonify({"msg": "bad username/password"}), 401
    token = create_access_token(identity=username)
    return jsonify(access_token=token)

# ---------------- Predict API ----------------
def _make_cause_map():
    return {
        "hypertension": "Elevated blood pressure — risk factors: high salt intake, obesity, sedentary lifestyle",
        "diabetes": "High blood glucose — risk factors: poor diet, insulin resistance",
        "healthy": "No disease predicted — maintain healthy lifestyle",
        "cardiac": "Possible cardiac issue — refer to cardiology (example)",
    }

@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    global disease_model
    if disease_model is None:
        return jsonify({"error": "Model not available (disease_model.joblib missing)"}), 500

    data = request.json or {}
    features = ["age", "blood_pressure", "glucose", "heart_rate"]
    try:
        X = [float(data.get(f, 0)) for f in features]
    except Exception:
        return jsonify({"error": "invalid input types"}), 400

    pred_raw = disease_model.predict([X])[0]
    pred_label = str(pred_raw).strip().lower()
    cause = _make_cause_map().get(pred_label, "Cause not available. Consult clinician.")

    resp_json = {"prediction": pred_label, "cause": cause}
    _log_request_response("/predict", data, resp_json)
    return jsonify(resp_json)

@app.route("/predict_open", methods=["POST"])
def predict_open():
    global disease_model
    if disease_model is None:
        return "Error: Model not available (place disease_model.joblib in models/)", 500

    data = request.json or {}
    features = ["age", "blood_pressure", "glucose", "heart_rate"]
    try:
        X = [float(data.get(f, 0)) for f in features]
    except Exception:
        return "Error: Invalid input types for features", 400

    try:
        pred_raw = disease_model.predict([X])[0]
    except Exception as e:
        return f"Error: prediction failed: {e}", 500

    pred_label = str(pred_raw).strip().lower()
    cause = _make_cause_map().get(pred_label, "Cause not available. Consult clinician.")
    resp_json = {"prediction": pred_label, "cause": cause}
    _log_request_response("/predict_open", data, resp_json)
    return jsonify(resp_json)

# ---------------- Recommend API ----------------
@app.route("/recommend", methods=["POST"])
@jwt_required()
def recommend():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    top_k = int(data.get("top_k", 5))
    if not query:
        return jsonify({"error": "provide a 'query'"}), 400

    try:
        if med_embeddings_available():
            results = query_recommendations(query, top_k=top_k, use_semantic=True)
            results = [{"id": r["id"], "name": r["name"], "desc": r.get("desc", ""), "semantic_score": r["score"]} for r in results]
        elif tfidf_available():
            results = tfidf_query_topk(query, top_k=top_k)
            results = [{"id": r["id"], "name": r["name"], "desc": r.get("desc", ""), "tfidf_score": r["score"]} for r in results]
        else:
            return jsonify({"error": "No recommendation artifacts available."}), 500

        resp_payload = {"results": results, "disclaimer": "Informational only — not medical advice. Consult a clinician."}
        _log_request_response("/recommend", {"query": query, "top_k": top_k}, resp_payload)
        return jsonify(resp_payload)
    except Exception as e:
        return jsonify({"error": f"Recommendation failed: {e}"}), 500

# ---------------- Small helpers & history endpoints ----------------
@app.route("/ping")
def ping():
    return jsonify({"msg": "pong"})

@app.route("/last_request")
def last_request():
    # return the most recent entry
    if not LAST_LOG:
        return jsonify({})
    return jsonify(LAST_LOG[-1])

@app.route("/last_requests")
def last_requests():
    # return the list (most recent last)
    return jsonify(LAST_LOG)

# ---------------- Main ----------------
if __name__ == "__main__":
    with app.app_context():
        init_db()
    load_disease_model()
    print("Starting Flask app. Open /login_page after server starts.")
    app.run(debug=True, host="0.0.0.0", port=5000)
