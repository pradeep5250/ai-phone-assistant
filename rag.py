"""
rag.py — MintX Solutions Retrieval-Augmented Generation (RAG) System

High-Performance Architectural Standards:
- 🔎 Embeddings: text-embedding-3-large (3072 dimensions)
- 📚 Vector DB: Qdrant (Collection: mintx_knowledge)
- 🧠 Document Chunking & Hybrid Cosine Similarity Retrieval
- ⚡ Offline-resilient local cache + Lexical fallback
- Standard search_mintx_knowledge tool callable by GPT Realtime & AIAssistant
"""

import os
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()
load_dotenv(dotenv_path=Path(__file__).parent / ".env.local", override=True)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_FILE = DATA_DIR / "mintx_knowledge.json"
CACHE_FILE = DATA_DIR / "knowledge_embeddings_large.json"
QDRANT_STORAGE_DIR = DATA_DIR / "qdrant_storage"

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM = 3072
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mintx_knowledge")


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


class MintXRAG:
    """
    RAG Vector Database for MintX Solutions powered by Qdrant & text-embedding-3-large.
    Provides semantic vector retrieval + hybrid keyword search.
    """

    def __init__(
        self,
        knowledge_path: Path = KNOWLEDGE_FILE,
        cache_path: Path = CACHE_FILE,
        qdrant_path: Path = QDRANT_STORAGE_DIR
    ):
        self.knowledge_path = knowledge_path
        self.cache_path = cache_path
        self.qdrant_path = qdrant_path
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Dict[str, List[float]] = {}
        self._openai_client = None
        self._qdrant_client = None
        self._qdrant_mode = "offline"
        self._initialized = False

        self._load_documents()
        self._init_qdrant()
        self._load_or_compute_embeddings()
        self._sync_qdrant_collection()

    def _get_openai_client(self):
        """Lazy-init OpenAI client."""
        if self._openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    from openai import OpenAI
                    self._openai_client = OpenAI(api_key=api_key)
                except Exception as e:
                    print(f"[RAG] Warning initializing OpenAI client: {e}")
        return self._openai_client

    def _load_documents(self):
        """Load knowledge documents from disk."""
        if not self.knowledge_path.exists():
            print(f"[RAG] Knowledge base file not found at {self.knowledge_path}")
            self.documents = []
            return

        try:
            with open(self.knowledge_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            print(f"[RAG] Loaded {len(self.documents)} knowledge documents for MintX Solutions.")
        except Exception as e:
            print(f"[RAG] Error loading knowledge file: {e}")
            self.documents = []

    def _init_qdrant(self):
        """Initialize Qdrant client with persistent local storage or remote URL."""
        qdrant_url = os.getenv("QDRANT_URL")
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            if qdrant_url:
                api_key = os.getenv("QDRANT_API_KEY")
                self._qdrant_client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=5.0)
                self._qdrant_mode = f"remote ({qdrant_url})"
            else:
                self.qdrant_path.mkdir(parents=True, exist_ok=True)
                try:
                    self._qdrant_client = QdrantClient(path=str(self.qdrant_path))
                    self._qdrant_mode = f"local persistent ({self.qdrant_path.name})"
                except Exception as local_err:
                    print(f"[RAG] Notice: local Qdrant path lock ({local_err}), switching to in-memory Qdrant.")
                    self._qdrant_client = QdrantClient(":memory:")
                    self._qdrant_mode = "in-memory"

            # Create collection if it doesn't exist
            collections = self._qdrant_client.get_collections().collections
            col_names = [c.name for c in collections]
            if QDRANT_COLLECTION not in col_names:
                self._qdrant_client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                )
                print(f"[RAG / Qdrant] Created collection '{QDRANT_COLLECTION}' (dim={EMBEDDING_DIM}, metric=Cosine).")
            else:
                print(f"[RAG / Qdrant] Connected to collection '{QDRANT_COLLECTION}'.")
        except Exception as e:
            print(f"[RAG] Warning initializing Qdrant client: {e}. Falling back to in-memory cosine search.")
            self._qdrant_client = None
            self._qdrant_mode = "fallback in-memory"

    def _load_or_compute_embeddings(self):
        """Load cached 3072-dim embeddings or compute them using OpenAI text-embedding-3-large."""
        cached = {}
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                # Verify vector dimension matches 3072
                if cached:
                    sample_vec = next(iter(cached.values()))
                    if len(sample_vec) != EMBEDDING_DIM:
                        print(f"[RAG] Cached embeddings dimension {len(sample_vec)} != {EMBEDDING_DIM}. Recomputing with {EMBEDDING_MODEL}...")
                        cached = {}
            except Exception as e:
                print(f"[RAG] Could not read embeddings cache: {e}")

        # Check which docs need embedding
        missing_docs = [doc for doc in self.documents if doc["id"] not in cached]

        if missing_docs:
            client = self._get_openai_client()
            if client:
                try:
                    texts_to_embed = [
                        f"{doc.get('title', '')}: {doc.get('content', '')} {' '.join(doc.get('keywords', []))}"
                        for doc in missing_docs
                    ]
                    print(f"[RAG] Computing embeddings for {len(texts_to_embed)} knowledge items with {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})...")
                    response = client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=texts_to_embed
                    )
                    for doc, emb_data in zip(missing_docs, response.data):
                        cached[doc["id"]] = emb_data.embedding

                    # Save to cache file
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.cache_path, "w", encoding="utf-8") as f:
                        json.dump(cached, f)
                    print(f"[RAG] Successfully cached {len(cached)} text-embedding-3-large embeddings to {self.cache_path.name}.")
                except Exception as e:
                    print(f"[RAG] Notice: OpenAI embedding generation skipped ({e}). Keyword search fallback is active.")
            else:
                print(f"[RAG] Notice: OpenAI API key unavailable for embedding generation. Lexical search is active.")

        self.embeddings = cached
        self._initialized = True

    def _sync_qdrant_collection(self):
        """Sync loaded documents and embeddings into Qdrant collection."""
        if not self._qdrant_client or not self.embeddings:
            return

        try:
            from qdrant_client.models import PointStruct

            points = []
            for idx, doc in enumerate(self.documents, 1):
                doc_id = doc["id"]
                if doc_id in self.embeddings:
                    points.append(PointStruct(
                        id=idx,
                        vector=self.embeddings[doc_id],
                        payload={
                            "doc_id": doc_id,
                            "title": doc.get("title", ""),
                            "content": doc.get("content", ""),
                            "category": doc.get("category", ""),
                            "keywords": doc.get("keywords", []),
                            "url": doc.get("url", ""),
                        }
                    ))

            if points:
                self._qdrant_client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=points
                )
                print(f"[RAG / Qdrant] Synced {len(points)} knowledge points to Qdrant collection '{QDRANT_COLLECTION}'.")
        except Exception as e:
            print(f"[RAG / Qdrant] Warning syncing points to Qdrant: {e}")

    def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Compute embedding for user query with text-embedding-3-large."""
        client = self._get_openai_client()
        if not client or not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            res = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[query]
            )
            return res.data[0].embedding
        except Exception:
            return None

    def _keyword_score(self, query: str, doc: Dict[str, Any]) -> float:
        """Compute lexical keyword match score."""
        q_clean = query.lower()
        q_tokens = set(re.findall(r"\w+", q_clean))
        if not q_tokens:
            return 0.0

        score = 0.0
        # Title matches
        title_tokens = set(re.findall(r"\w+", doc.get("title", "").lower()))
        score += len(q_tokens & title_tokens) * 2.5

        # Keywords matches
        doc_kws = {kw.lower() for kw in doc.get("keywords", [])}
        for kw in doc_kws:
            if kw in q_clean:
                score += 3.0
            else:
                kw_parts = set(re.findall(r"\w+", kw))
                score += len(q_tokens & kw_parts) * 1.5

        # Content token matches
        content_tokens = set(re.findall(r"\w+", doc.get("content", "").lower()))
        score += len(q_tokens & content_tokens) * 0.8

        return score

    def search_qdrant(self, query_vec: List[float], top_k: int = 2) -> List[Dict[str, Any]]:
        """Search vector database via Qdrant."""
        if not self._qdrant_client:
            return []

        try:
            # Try query_points (qdrant-client >= 1.10)
            if hasattr(self._qdrant_client, "query_points"):
                response = self._qdrant_client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=query_vec,
                    limit=top_k
                )
                points = response.points if hasattr(response, "points") else response
            else:
                points = self._qdrant_client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=query_vec,
                    limit=top_k
                )

            results = []
            for pt in points:
                payload = pt.payload or {}
                results.append({
                    "doc": {
                        "id": payload.get("doc_id"),
                        "title": payload.get("title"),
                        "content": payload.get("content"),
                        "category": payload.get("category"),
                        "keywords": payload.get("keywords", []),
                        "url": payload.get("url"),
                    },
                    "score": float(pt.score),
                    "vector_score": float(pt.score),
                    "keyword_score": 0.0,
                    "engine": "Qdrant"
                })
            return results
        except Exception as e:
            print(f"[RAG / Qdrant] Search error ({e}), falling back to in-memory cosine.")
            return []

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Search knowledge base using hybrid Qdrant vector search + keyword matching.
        Returns top_k most relevant documents.
        """
        if not self.documents:
            return []

        q_vec = self._get_query_embedding(query)

        # 1. Primary Vector Search via Qdrant
        if q_vec and self._qdrant_client:
            qdrant_results = self.search_qdrant(q_vec, top_k=top_k * 2)
            if qdrant_results:
                # Augment with keyword boosts for exact phrase matches
                for item in qdrant_results:
                    kw_score = self._keyword_score(query, item["doc"])
                    item["keyword_score"] = kw_score
                    # Combined score
                    item["score"] = (item["vector_score"] * 0.75) + (min(kw_score / 10.0, 1.0) * 0.25)

                qdrant_results.sort(key=lambda x: x["score"], reverse=True)
                return qdrant_results[:top_k]

        # 2. Resilient In-Memory / Keyword Fallback
        scored_results = []
        for doc in self.documents:
            doc_id = doc["id"]
            vec_score = 0.0
            if q_vec and doc_id in self.embeddings:
                vec_score = cosine_similarity(q_vec, self.embeddings[doc_id])

            kw_score = self._keyword_score(query, doc)
            combined_score = (vec_score * 0.7) + (min(kw_score / 10.0, 1.0) * 0.3) if q_vec else kw_score

            scored_results.append({
                "doc": doc,
                "score": combined_score,
                "vector_score": vec_score,
                "keyword_score": kw_score,
                "engine": "In-Memory Cosine Fallback" if q_vec else "Lexical Fallback"
            })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    def search_as_context(self, query: str, top_k: int = 2) -> str:
        """
        Return a clean, factual context string suitable for injection into LLM prompt.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return "No specific MintX documentation found for this query."

        # Filter out results with very low relevance score
        relevant = [r for r in results if r["score"] > 0.15 or r["keyword_score"] > 0]
        if not relevant:
            relevant = results[:1]

        context_lines = []
        for idx, item in enumerate(relevant, 1):
            doc = item["doc"]
            context_lines.append(f"[{idx}] {doc.get('title')}:\n{doc.get('content')}")

        return "\n\n".join(context_lines)

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic status of Qdrant and Embedding engine."""
        points_count = len(self.embeddings)
        if self._qdrant_client:
            try:
                info = self._qdrant_client.get_collection(QDRANT_COLLECTION)
                points_count = info.points_count
            except Exception:
                pass

        return {
            "status": "ready" if self._initialized else "initializing",
            "vector_db": "Qdrant",
            "qdrant_mode": self._qdrant_mode,
            "collection": QDRANT_COLLECTION,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIM,
            "document_count": len(self.documents),
            "indexed_points": points_count,
            "cache_file": self.cache_path.name
        }


# Singleton RAG instance
_rag_instance: Optional[MintXRAG] = None


def get_rag_engine() -> MintXRAG:
    """Return the global singleton MintXRAG engine."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = MintXRAG()
    return _rag_instance


def search_mintx_knowledge(query: str, top_k: int = 2) -> str:
    """
    Standard tool function to query MintX RAG knowledge base.
    Used by OpenAI Realtime function tools and AIAssistant.
    """
    engine = get_rag_engine()
    return engine.search_as_context(query, top_k=top_k)
