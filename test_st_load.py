# test_st_load.py
from sentence_transformers import SentenceTransformer

print("Loading model: sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(["hello world"], convert_to_numpy=True)
print("Loaded OK. Embedding shape:", emb.shape)
