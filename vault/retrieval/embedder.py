from __future__ import annotations
 
from functools import lru_cache
from sentence_transformers import SentenceTransformer
 
from vault.cli.logger import done, log, step
 
DEFAULT_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    return SentenceTransformer(model_name)

def embed_chunks(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    show_log: bool = True,
) -> list[list[float]]:
    if not texts:
        return []
    
    if show_log:
        step("EMBED", f"Loading embedding model: [bold]{model_name}[/bold]")
        log("Model runs fully locally — no data sent to any API")
        log("First run downloads ~80MB model to ~/.cache/huggingface/")
        log(f"Embedding {len(texts)} chunk(s) in batches of {batch_size}")
 
    model = _load_model(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_log,
        convert_to_numpy=True,
    )
 
    if show_log:
        done("EMBED", f"{len(vectors)} vectors computed  (dim={vectors.shape[1]})")
 
    return vectors.tolist()

def embed_query(query: str, model_name: str = DEFAULT_MODEL) -> list[float]:
    model = _load_model(model_name)
    return model.encode([query], convert_to_numpy=True)[0].tolist()