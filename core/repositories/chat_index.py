# core/repositories/chat_index.py
from __future__ import annotations
import os, datetime as dt, json
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb import HttpClient

try:
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
except Exception:
    SentenceTransformerEmbeddingFunction = None

ISO = "%Y-%m-%dT%H:%M:%S.%fZ"

class ChatIndex:
    def __init__(self, embedding_function=None, client: Optional[object] = None):
        if client is not None:
            self.client = client
        else:
            server_url = os.getenv("CHROMA_SERVER_URL", "").rstrip("/")
            if not server_url:
                raise RuntimeError("CHROMA_SERVER_URL is not set")
            self.client = chromadb.HttpClient(
                host=server_url,
                settings=Settings(anonymized_telemetry=False)
            )

        if SentenceTransformerEmbeddingFunction is not None and embedding_function is None:
            model_name = os.getenv("EMBED_MODEL", "paraphrase-multilingual-mpnet-base-v2")
            embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)

        self.collection_name = os.getenv("CHROMA_CHAT_COLLECTION", "chat_messages")
        self.messages = self.client.get_or_create_collection(
            name="chat_messages",
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def index_message(self, chat_id: int, message_id: int, role: str, content: str,
                      ts: Optional[dt.datetime] = None, extra_meta: Optional[Dict[str, Any]] = None):
        ts = ts or dt.datetime.utcnow()
        meta = {"chat_id": chat_id, "role": role, "ts": ts.strftime(ISO)}

        if extra_meta:
            for k, v in extra_meta.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    meta[k] = v
                else:
                    meta[k] = json.dumps(v, ensure_ascii=False)

        # Uniqe ID
        doc_id = f"chat:{chat_id}:msg:{message_id}"

        self.messages.add(ids=[doc_id], documents=[content], metadatas=[meta])

    def build_history_context(self, chat_id: int, query: str,
                              recent_docs: int = 8, semantic_k: int = 6, max_chars: int = 2000) -> List[str]:
        # Collect chat history
        res_all = self.messages.get(where={"chat_id": chat_id}, include=["documents", "metadatas"])

        docs = res_all.get("documents") or []
        metas = res_all.get("metadatas") or []

        if docs and isinstance(docs[0], list):   docs = docs[0]
        if metas and isinstance(metas[0], list): metas = metas[0]

        rows = []
        for doc, meta in zip(docs, metas):
            ts = (meta or {}).get("ts", "")
            role = (meta or {}).get("role", "user")
            rows.append({"ts": ts, "role": role, "text": doc or ""})

        rows.sort(key=lambda x: x["ts"])
        last_texts = [f"{r['role']}: {r['text']}" for r in rows[-recent_docs:]]

        try:
            got = self.messages.get(where={"chat_id": chat_id}) or {}
            ids = got.get("ids") or []
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            total = len(ids)
        except Exception:
            total = 0

        k = max(1, min(semantic_k, total))

        try:
            sem = self.messages.query(
                query_texts=[query],
                n_results=k,
                where={"chat_id": chat_id},
                include=["documents"]
            )
            sem_docs = (sem.get("documents") or [[]])[0]
            sem_texts = [t for t in sem_docs if t]
        except Exception:
            sem_texts = []

        seen, out = set(), []
        for t in (last_texts + sem_texts):
            key = (t or "").strip()
            if key and key not in seen:
                out.append(key[:max_chars]);
                seen.add(key)

        return out

    def delete_chat(self, chat_id: int) -> int:
        """Удаляет все сообщения чата по ids. Возвращает число удалённых документов."""
        try:
            res = self.messages.get(where={"chat_id": chat_id}) or {}
            ids = res.get("ids") or []
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if not ids:
                return 0
            self.messages.delete(ids=ids)
            return len(ids)
        except Exception:
            # В некоторых версиях сервер ожидает явный $eq
            try:
                res = self.messages.get(where={"chat_id": {"$eq": chat_id}}) or {}
                ids = res.get("ids") or []
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                if not ids:
                    return 0
                self.messages.delete(ids=ids)
                return len(ids)
            except Exception:
                raise
