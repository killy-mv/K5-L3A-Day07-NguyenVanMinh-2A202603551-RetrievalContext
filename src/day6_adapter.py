"""Day 7 CP1: read-only reuse of the real Day 6 corpus and pgvector index.

No benchmark labels are loaded here. No schema creation, embedding writes, or
vector-store mutation. Query embeddings match the pinned Day 6 recipe.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DIMENSIONS = 384
MODEL_TAG = f"{MODEL}@{REVISION}"
SOURCE_FILENAME = "starter/results/cp3_chunks_B_paragraph.jsonl"
DAY6_FOLDER = "K5-L3A-Day06-NguyenVanMinh-2A202603551-DataIngestion"


def default_day6_root() -> Path:
    """For the original E:/code/Vin/lab6 + lab7 sibling directories."""
    return Path(os.getenv("DAY6_ROOT") or
                Path(__file__).resolve().parents[3] / "lab6" / DAY6_FOLDER)


def database_url(day6_root: Path) -> str:
    """Use DATABASE_URL or read the *existing* local Day 6 .env (never copy it)."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        from dotenv import dotenv_values
        url = dotenv_values(day6_root / "starter" / ".env").get("DATABASE_URL")
    if not url or not isinstance(url, str):
        raise RuntimeError("Missing DATABASE_URL (set environment variable or Day 6 starter/.env)")
    return url


def _to_day7_chunk(raw: dict[str, Any]) -> dict[str, Any]:
    for name in ("chunk_id", "document_id", "source", "page", "text", "metadata"):
        if name not in raw:
            raise ValueError(f"Day 6 chunk missing {name}")
    if any(not isinstance(raw[k], str) or not raw[k].strip()
           for k in ("chunk_id", "document_id", "source", "text")):
        raise ValueError("Chunk identifier, source and content must be nonempty strings")
    if not isinstance(raw["page"], int) or raw["page"] < 1:
        raise ValueError("Invalid physical PDF page")
    if not isinstance(raw["metadata"], dict) or not raw["metadata"].get("chapter"):
        raise ValueError("Chunk missing verified chapter metadata")
    return {
        "id": raw["chunk_id"],
        "source_id": raw["document_id"],
        "source": raw["source"],
        "page": raw["page"],
        "text": raw["text"],
        "metadata": raw["metadata"],
    }


def load_day6_chunks(day6_root: Path) -> list[dict[str, Any]]:
    src = day6_root / SOURCE_FILENAME
    if not src.is_file():
        raise FileNotFoundError(f"Day 6 chunk export missing: {src}")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    with src.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = _to_day7_chunk(json.loads(line))
            if item["id"] in ids:
                raise ValueError(f"Duplicate chunk at line {line_no}: {item['id']}")
            ids.add(item["id"])
            result.append(item)
    if not result:
        raise ValueError("Empty Day 6 corpus")
    return result


def validate_source(day6_root: Path) -> dict[str, Any]:
    """Validate the original source hash and publishable provenance, not PDF contents."""
    manifest = json.loads((day6_root / "data" / "manifest.json").read_text(encoding="utf-8"))
    pdf = day6_root / manifest["local_path"]
    if not pdf.is_file():
        raise FileNotFoundError(f"Missing licensed Day 6 source file: {pdf}")
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("Original Day 6 source checksum does not match manifest")
    return {
        "corpus_id": manifest["corpus_id"],
        "source_id": manifest["source_id"],
        "title": manifest["title"],
        "author": manifest["author"],
        "source_url": manifest["source_path_or_url"],
        "license": manifest["license_or_permission"],
        "license_url": manifest["license_url"],
        "content_pages": manifest["content_page_count"],
        "source_sha256": digest,
    }


def audit_pgvector(url: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare *every* B-index record with the local export, read-only."""
    import psycopg

    expected = {c["id"]: c for c in chunks}
    with psycopg.connect(url) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        config = conn.execute(
            "SELECT dimensions, model_name FROM day6_embedding_config WHERE id = 1"
        ).fetchone()
        if config != (DIMENSIONS, MODEL_TAG):
            raise ValueError(f"Day 6 index uses unexpected embedding configuration: {config}")
        records = conn.execute(
            """SELECT chunk_id,document_id,source,page,content,metadata
               FROM day6_embeddings
               WHERE strategy='B' AND model_name=%s ORDER BY chunk_id""",
            (MODEL_TAG,),
        ).fetchall()
        for chunk_id, source_id, source, page, content, metadata in records:
            if chunk_id not in expected:
                raise ValueError(f"Unexpected vector-store chunk: {chunk_id}")
            chunk = expected[chunk_id]
            if (source_id, source, page, content, metadata) != (
                    chunk["source_id"], chunk["source"], chunk["page"],
                    chunk["text"], chunk["metadata"]):
                raise ValueError(f"Vector-store content/metadata mismatch for {chunk_id}")
    if len(records) != len(chunks):
        raise ValueError(f"Corpus/index row count mismatch: {len(chunks)} vs {len(records)}")
    return {
        "vector_store": "PostgreSQL 16 + pgvector 0.8.7 (Day 6, read-only)",
        "table": "day6_embeddings",
        "strategy": "B",
        "model": MODEL,
        "revision": REVISION,
        "dimensions": DIMENSIONS,
        "normalized_embeddings": True,
        "metric": "cosine similarity",
        "stored_rows": len(records),
        "content_and_metadata_verified": len(records),
    }


class LocalEmbedder:
    """Exactly the embedding model, revision, CPU and normalized encoding of Day 6."""
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
        if self.model.get_embedding_dimension() != DIMENSIONS:
            raise RuntimeError("Embedding dimension does not match Day 6")

    def embed_query(self, question: str) -> list[float]:
        vector = self.model.encode(
            [question], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )[0].tolist()
        if len(vector) != DIMENSIONS:
            raise RuntimeError("Wrong query embedding dimension")
        return vector


def validate_retrieval_params(query: str, filters: dict[str, str] | None, top_k: int) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Nonempty query required")
    if not isinstance(top_k, int) or not (1 <= top_k <= 100):
        raise ValueError("top_k must be 1..100")
    if filters is not None and (
        not isinstance(filters, dict) or set(filters) != {"chapter"} or
        not isinstance(filters.get("chapter"), str) or not filters["chapter"].strip()
    ):
        raise ValueError("Only a trusted, nonempty chapter filter is supported")


class Day6DenseRetriever:
    """Retriever.search compatible with Lab 7 starter/agentic.py, using real pgvector."""
    def __init__(self, url: str, embedder: Any = None) -> None:
        if not url:
            raise ValueError("DATABASE_URL is required")
        self.url = url
        self.embedder = embedder  # optionally injected for unit tests

    def search(self, query: str, *, filters: dict[str, str] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        validate_retrieval_params(query, filters, top_k)
        if self.embedder is None:
            self.embedder = LocalEmbedder()
        vector = self.embedder.embed_query(query)
        if len(vector) != DIMENSIONS:
            raise ValueError("Query embedding dimension mismatch")
        import psycopg
        vector_literal = '[' + ','.join(str(float(v)) for v in vector) + ']'
        clause = "strategy = 'B' AND model_name = %s"
        args: list[Any] = [MODEL_TAG]
        if filters:
            clause += ' AND chapter = %s'
            args.append(filters['chapter'])
        sql = f'''SELECT chunk_id,document_id,source,page,content,metadata,
                   1 - (embedding <=> %s::vector) AS similarity
                   FROM day6_embeddings WHERE {clause}
                   ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s'''
        with psycopg.connect(self.url) as conn:
            conn.execute('SET TRANSACTION READ ONLY')
            rows = conn.execute(sql, [vector_literal, *args, vector_literal, top_k]).fetchall()
        return [{
            'id': chunk_id, 'source_id': source_id, 'source': source,
            'page': page, 'text': content, 'metadata': metadata, 'score': float(score),
        } for chunk_id, source_id, source, page, content, metadata, score in rows]
