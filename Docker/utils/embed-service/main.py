from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from sentence_transformers import SentenceTransformer

"""
FastAPI service to embed text properties into vector embeddings using SentenceTransformer.
"""

app = FastAPI()
model = SentenceTransformer("/app/models/all-MiniLM-L6")

class EmbedRequest(BaseModel):
    label: str
    properties: Dict[str, str]

@app.post("/embed")
async def embed_node(req: EmbedRequest):
    # Concatenate all string properties into one text
    text = " ".join([f"{k}: {v}" for k, v in req.properties.items() if isinstance(v, str)])
    embedding = model.encode(text).tolist()
    return {
        "label": req.label,
        "embedding": embedding
    }
