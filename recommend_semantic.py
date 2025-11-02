# recommend_semantic.py
"""
Recommender helper used by app.py

Expecting files in ./models:
- medicines_df.joblib      (pandas DataFrame with at least columns: id, name, desc)
- Either:
    - med_embeddings.joblib (dict with keys 'embeddings' (np.array) and 'ids' (list/array))
  OR
    - tfidf_vectorizer.joblib (scikit-learn TfidfVectorizer) + medicines_df

This module provides:
- med_embeddings_available()
- tfidf_available()
- query_recommendations(query, top_k=5, use_semantic=True)
- tfidf_query_topk(query, top_k=5)
"""

from pathlib import Path
import joblib
import numpy as np

# Optional helpers
try:
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim
except Exception:
    sklearn_cosine_sim = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"
MED_EMB_FILE = MODELS_DIR / "med_embeddings.joblib"
MEDS_FILE = MODELS_DIR / "medicines_df.joblib"
TFIDF_FILE = MODELS_DIR / "tfidf_vectorizer.joblib"

# cached globals (lazy loaded)
_med_embeddings = None          # numpy array shape (n, dim)
_med_embedding_ids = None       # list/array of med ids (same order as embeddings)
_med_embeddings_normed = None
_meds_df = None
_tfidf = None
_embed_model = None


def med_embeddings_available():
    """Return True if precomputed med_embeddings and medicines_df are present."""
    return MED_EMB_FILE.exists() and MEDS_FILE.exists()


def tfidf_available():
    """Return True if TF-IDF vectorizer and medicines_df are present."""
    return TFIDF_FILE.exists() and MEDS_FILE.exists()


def _load_meds_df():
    """Lazy-load medicines dataframe from models/medicines_df.joblib"""
    global _meds_df
    if _meds_df is None:
        if MEDS_FILE.exists():
            _meds_df = joblib.load(MEDS_FILE)
        else:
            _meds_df = None
    return _meds_df


def _load_med_embeddings():
    """
    Loads med_embeddings.joblib and normalizes them for cosine similarity.
    Supports two formats:
      - dict with keys {'embeddings': np.array, 'ids': [...]}
      - a plain numpy array (index aligns with medicines_df rows)
    """
    global _med_embeddings, _med_embedding_ids, _med_embeddings_normed
    if _med_embeddings is not None and _med_embeddings_normed is not None:
        return

    if not MED_EMB_FILE.exists():
        return

    emb_obj = joblib.load(MED_EMB_FILE)

    # If a dict with embeddings + ids
    if isinstance(emb_obj, dict) and "embeddings" in emb_obj:
        embs = np.asarray(emb_obj["embeddings"])
        ids = np.asarray(emb_obj.get("ids") if "ids" in emb_obj else np.arange(len(embs)))
    else:
        # could be a numpy array where row index corresponds to meds_df rows
        embs = np.asarray(emb_obj)
        ids = np.arange(len(embs))

    # normalize (avoid division by zero)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    embs_norm = embs / norms

    _med_embeddings = embs
    _med_embedding_ids = ids
    _med_embeddings_normed = embs_norm


def _ensure_tfidf():
    """Lazy-load TF-IDF vectorizer."""
    global _tfidf
    if _tfidf is None and TFIDF_FILE.exists():
        _tfidf = joblib.load(TFIDF_FILE)
    return _tfidf


def _lazy_load_sentence_model(model_name="all-MiniLM-L6-v2"):
    """Lazy-load a SentenceTransformer for on-the-fly encoding if needed."""
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    if SentenceTransformer is None:
        return None
    try:
        _embed_model = SentenceTransformer(model_name)
        return _embed_model
    except Exception:
        _embed_model = None
        return None


def query_recommendations(query, top_k=5, use_semantic=True, embed_model_name="all-MiniLM-L6-v2"):
    """
    Returns list of dicts: [{'id', 'name', 'desc', 'score'}].

    If use_semantic True and med_embeddings available, uses precomputed embeddings +
    sentence-transformers for query encoding and cosine similarity.

    Otherwise falls back to TF-IDF (if available).
    """
    meds = _load_meds_df()
    if meds is None:
        raise RuntimeError("medicines_df.joblib not found in models/")

    # Prefer semantic with precomputed med embeddings
    if use_semantic and MED_EMB_FILE.exists():
        # ensure embeddings loaded and normalized
        _load_med_embeddings()
        if _med_embeddings_normed is None:
            raise RuntimeError("med_embeddings.joblib exists but failed to load properly.")

        # load encoder for query
        model = _lazy_load_sentence_model(embed_model_name)
        if model is None:
            raise RuntimeError("SentenceTransformer not available for on-the-fly encoding. Install sentence-transformers or use TF-IDF artifacts.")

        q_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        q_norm = np.linalg.norm(q_emb) or 1e-9
        q_emb_norm = q_emb / q_norm

        # cosine similarity = dot product with normalized med embeddings
        sims = _med_embeddings_normed.dot(q_emb_norm)
        top_idx = np.argsort(-sims)[:top_k]

        results = []
        for i in top_idx:
            score = float(sims[i])
            med_id = int(_med_embedding_ids[i]) if _med_embedding_ids is not None else int(i)

            # find row by id if possible
            if "id" in meds.columns:
                row = meds[meds["id"] == med_id]
                if not row.empty:
                    name = str(row.iloc[0].get("name", f"id_{med_id}"))
                    desc = str(row.iloc[0].get("desc", ""))
                else:
                    name = f"id_{med_id}"
                    desc = ""
            else:
                # fallback to indexing by position
                row = meds.iloc[[i]]
                name = str(row.iloc[0].get("name", f"id_{med_id}"))
                desc = str(row.iloc[0].get("desc", ""))

            results.append({"id": int(med_id), "name": name, "desc": desc, "score": score})
        return results

    # TF-IDF fallback
    if TFIDF_FILE.exists():
        tfidf = _ensure_tfidf()
        meds = _load_meds_df()
        if tfidf is None or meds is None:
            raise RuntimeError("TF-IDF artifacts not loaded correctly.")

        descs = meds["desc"].astype(str).tolist()
        desc_vecs = tfidf.transform(descs)
        q_vec = tfidf.transform([query])

        if sklearn_cosine_sim is not None:
            sims = sklearn_cosine_sim(q_vec, desc_vecs)[0]
        else:
            sims = (q_vec @ desc_vecs.T).toarray()[0]

        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for i in top_idx:
            results.append({
                "id": int(meds.iloc[i].get("id", i)),
                "name": meds.iloc[i].get("name", f"id_{i}"),
                "desc": meds.iloc[i].get("desc", ""),
                "score": float(sims[i]),
            })
        return results

    raise RuntimeError("No recommendation artifacts available (med_embeddings.joblib or tfidf_vectorizer.joblib missing).")


def tfidf_query_topk(query, top_k=5):
    """Compatibility helper used by app.py to request TF-IDF-only results."""
    return query_recommendations(query, top_k=top_k, use_semantic=False)
