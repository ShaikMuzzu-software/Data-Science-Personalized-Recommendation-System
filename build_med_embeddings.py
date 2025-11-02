# build_med_embeddings.py
"""
Batched builder for medicine description embeddings.
Saves models/med_embeddings.joblib -> {"embeddings": np.array, "ids": [..]}

Run:
  python build_med_embeddings.py
"""

from pathlib import Path
import joblib
import numpy as np
import math

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise SystemExit("sentence-transformers not available. Run: pip install sentence-transformers")

BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"
MEDS_FILE = MODELS_DIR / "medicines_df.joblib"
OUT_FILE = MODELS_DIR / "med_embeddings.joblib"

if not MEDS_FILE.exists():
    raise SystemExit(f"Medicines file not found: {MEDS_FILE}\nMake sure models/medicines_df.joblib exists.")

medicines_df = joblib.load(MEDS_FILE)

if "desc" not in medicines_df.columns:
    raise SystemExit("medicines_df must have a 'desc' column with text descriptions.")

texts = medicines_df["desc"].astype(str).tolist()
ids = medicines_df["id"].tolist()
n = len(texts)
print(f"Found {n} descriptions to encode.")

# Model choice: small & fast CPU model
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading SentenceTransformer model:", EMBED_MODEL_NAME)
model = SentenceTransformer(EMBED_MODEL_NAME)

# batching to reduce peak memory usage
batch_size = 32  # lower if you run into memory issues
batches = math.ceil(n / batch_size)
all_embs = []

for i in range(batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, n)
    chunk = texts[start:end]
    print(f"Encoding batch {i+1}/{batches} (items {start}..{end-1})")
    embs = model.encode(chunk, convert_to_numpy=True, show_progress_bar=False)
    all_embs.append(embs)

embeddings = np.vstack(all_embs)
print("Embeddings shape:", embeddings.shape)

# Save a dict with embeddings and ids so we can align later
joblib.dump({"embeddings": embeddings, "ids": ids}, OUT_FILE)
print("Saved med embeddings to:", OUT_FILE)
