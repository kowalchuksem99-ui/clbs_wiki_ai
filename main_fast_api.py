import os
from dotenv import load_dotenv

load_dotenv(os.getenv("ENV_PATH", ".env.hot"), override=False)

import sys
import json
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from fastapi import FastAPI, Depends, Request, HTTPException, Path, Header
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from core.utils.env_hot import EnvHot
from fastapi.middleware.cors import CORSMiddleware

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import json_log_formatter

from core.ai_models_2 import AiModel
from core.repositories.chroma_repo import ChromaRepository

from sqlalchemy.orm import Session
from core.repositories.postgresql_repo_fastapi import (
    Base as PGBase, engine as PGEngine, SessionLocal, PostgreSQLRepository
)
from core.repositories.chat_index import ChatIndex


# --------------------- Pydantic Schemas ---------------------
class DocumentSchema(BaseModel):
    id: str = Field(..., min_length=3, max_length=100)
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    parent_id: Optional[str] = ''
    version_id: Optional[int] = 1
    exists: Optional[bool] = True
    versions: Optional[List[Any]] = Field(default_factory=list)
    links: Optional[List[Any]] = Field(default_factory=list)
    views_count: Optional[int] = 0
    slug: Optional[str] = ''
    favorite_id: Optional[int] = 0
    is_public: Optional[bool] = True
    tags: Optional[List[str]] = Field(default_factory=list)
    user_permissions: Optional[List[Any]] = Field(default_factory=list)
    date: Optional[str] = ''
    tags_count: Optional[int] = 0

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "doc-1",
                    "title": "SLA модуля X",
                    "content": "SLA модуля X — 99.95%.",
                    "tags": ["sla", "module-x"],
                }
            ]
        }
    }


class QuerySchema(BaseModel):
    query: str = Field(..., min_length=3)
    model_config = {
        "json_schema_extra": {
            "examples": [{"query": "Какой SLA у модуля X?"}]
        }
    }


class SearchSchema(BaseModel):
    query: str = Field(..., min_length=2)
    model_config = {"json_schema_extra": {"examples": [{"query": "SLA"}]}}


class ChatCreateSchema(BaseModel):
    title: Optional[str] = Field(default="Новый чат", min_length=1, max_length=120)
    model_config = {"json_schema_extra": {"examples": [{"title": "Тестовый чат"}]}}


class ChatRenameSchema(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)
    model_config = {"json_schema_extra": {"examples": [{"title": "Чат про SLA"}]}}


class ChatSendSchema(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)
    model_config = {
        "json_schema_extra": {
            "examples": [{"content": "Какой SLA у модуля X?"}]
        }
    }

# --------------------- Logging (JSON) ---------------------
formatter = json_log_formatter.JSONFormatter()
json_handler = logging.StreamHandler(sys.stdout)
json_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.handlers = [json_handler]


# --------------------- FastAPI ---------------------
limiter = Limiter(key_func=get_remote_address)

tags_metadata = [
    {"name": "System", "description": "Статус и здоровье сервиса"},
    {"name": "Search/QA", "description": "RAG-запросы к базе знаний"},
    {"name": "Documents", "description": "CRUD по документам ChromaDB"},
    {"name": "Chats", "description": "CRUD по чатам пользователя"},
    {"name": "Messages", "description": "Сообщения чата и генерация ответов"},
    {"name": "Admin", "description": "Админ-доступ к чатам пользователей"},
    {"name": "Debug", "description": "Отладочные эндпоинты (служебные)"},
]

app = FastAPI(
    title="Saiga3 + ChromaDB API",
    description=(
        "REST API для RAG (ChromaDB) + чаты (PostgreSQL). "
        "Аутентификация — через заголовок X-API-Key. "
        "Документация: /docs (Swagger) и /redoc."
    ),
    version="1.0.0",
    contact={"name": "Backend Team", "email": "devnull@example.com"},
    license_info={"name": "Proprietary"},
    root_path="/dev",
    openapi_tags=tags_metadata,
)

app.state.limiter = limiter

# CORS
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173" # Сменить под фронт!
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],             # покрывает X-API-Key и X-User
)

# rate-limiter после CORS
app.add_middleware(SlowAPIMiddleware)

from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # Security scheme: X-API-Key в заголовке
    openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})["XApiKeyAuth"] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
    }
    # Требуем ключ на всех методах
    openapi_schema["security"] = [{"XApiKeyAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi



# --------------------- Env & runtime-config ---------------------
env = EnvHot(env_path=os.getenv("ENV_PATH", ".env.hot"), watch=True)
app.state.env = env

LOG_LEVEL        = env.get_str("LOG_LEVEL", "INFO")
MODEL_PATH       = env.get_str("MODEL_PATH", "data/mistral/Mistral-7B-Instruct-v0.3-Q8_0.gguf")
MAX_WORKERS      = env.get_int("MAX_WORKERS", 4)

# Новые ручки
CHROMA_TOP_K     = env.get_int("CHROMA_TOP_K", 5)
CHAT_RECENT_DOCS = env.get_int("CHAT_RECENT_DOCS", 8)
CHAT_SEMANTIC_K  = env.get_int("CHAT_SEMANTIC_K", 6)
CHAT_MAX_CHARS   = env.get_int("CHAT_MAX_CHARS", 1500)

logger.setLevel(LOG_LEVEL)

# --- PostgreSQL (создаём таблицы) ---
PGBase.metadata.create_all(bind=PGEngine)

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

pg_repo = PostgreSQLRepository()

# --------------------- Services ---------------------
repo = ChromaRepository()
ai_model = AiModel(model_path=MODEL_PATH, repo_path=None, env=env)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# --- Chroma-индекс истории чатов ---
chat_index = ChatIndex(
    embedding_function=getattr(repo, "embedding_function", None),
    client=getattr(repo, "client", None)
)


logger.info(
    "FastAPI initialized. Model: %s, ChromaURL: %s",
    MODEL_PATH, os.getenv("CHROMA_SERVER_URL", "")
)

# --------------------- Helpers ---------------------
def require_api_key(request: Request):
    api_key = request.headers.get("X-API-Key", "")
    if not request.app.state.env.is_key_valid(api_key):
        raise HTTPException(status_code=401, detail="Unauthorized")
    request.state.api_key = api_key
    request.state.is_admin = hasattr(request.app.state.env, "admin_keys") and api_key in request.app.state.env.admin_keys()
    return True

def user_header(
    request: Request,
    x_user: str = Header(..., alias="X-User", description="Логин пользователя")
):
    username = x_user.strip()
    if not username:
        raise HTTPException(status_code=400, detail="X-User must be non-empty")
    request.state.username = username
    return True

def resolve_username(request: Request) -> str:
    return request.state.username

def require_admin(request: Request):
    if not getattr(request.state, "is_admin", False):
        raise HTTPException(status_code=403, detail="Admin required")


def success(data, status: int = 200):
    return JSONResponse(content={"success": True, "data": data}, status_code=status)

def error(message, status: int = 400):
    return JSONResponse(content={"success": False, "error": message}, status_code=status)


async def safe_json(request: Request):
    """Safe JSON parsing with fallback encodings"""
    raw_data = await request.body()
    payload = None
    errors = []

    for enc in ["utf-8", "utf-16", "cp1251", "latin-1"]:
        if payload is None:
            try:
                payload = json.loads(raw_data.decode(enc))
            except Exception as e:
                errors.append(f"{enc} fail: {e}")

    if payload is None:
        try:
            payload = await request.json()
        except Exception as e:
            errors.append(f"force-json fail: {e}")

    if payload is None:
        raise HTTPException(status_code=400,
                            detail=f"Unable to decode JSON. Attempts: {errors}")

    return payload


# --------------------- Routes ---------------------
@app.get("/api/status", tags=["System"], summary="Статус сервиса")
@limiter.limit("10/minute")
def api_status(request: Request, dep=Depends(require_api_key)):
    return success({
        "documents_count": repo.count(),
        "model_loaded": True,
        "max_workers": MAX_WORKERS,
        "status": "operational"
    })


@app.post("/api/query", tags=["Search/QA"], summary="Задать вопрос к базе знаний (RAG)")
@limiter.limit("5/minute")
async def handle_query(request: Request, data: QuerySchema, dep=Depends(require_api_key)):
    future = executor.submit(ai_model.generate, data.query)
    try:
        result = future.result(timeout=120)
        return success(result)
    except TimeoutError:
        return error("Processing timed out", 504)
    except Exception as e:
        logger.error("Query failed: %s", e, exc_info=True)
        return error("Processing failed", 500)



@app.post("/api/documents", tags=["Documents"], summary="Создать/обновить документ в ChromaDB")
@limiter.limit("10/minute")
async def create_document(request: Request, doc_data: DocumentSchema, dep=Depends(require_api_key)):
    repo.upsert_document(doc_data.model_dump())
    return success({"id": doc_data.id, "status": "created"}, 201)


@app.get("/api/documents/{doc_id}", tags=["Documents"], summary="Получить документ по ID")
@limiter.limit("15/minute")
def get_document(request: Request, doc_id: str, dep=Depends(require_api_key)):
    doc = repo.get_by_id(doc_id)
    if not doc or not doc.get("ids"):
        return error("Document not found", 404)
    return success({
        "id": doc["ids"][0],
        "content": doc["documents"][0],
        "metadata": doc["metadatas"][0]
    })


@app.put("/api/documents/{doc_id}", tags=["Documents"], summary="Обновить документ по ID")
@limiter.limit("10/minute")
async def update_document(request: Request, doc_id: str, doc_data: DocumentSchema, dep=Depends(require_api_key)):
    if doc_data.id != doc_id:
        return error("Document ID mismatch", 400)
    repo.upsert_document(doc_data.model_dump())
    return success({"id": doc_id, "status": "updated"})



@app.delete("/api/documents/{doc_id}", tags=["Documents"], summary="Удалить документ по ID")
@limiter.limit("5/minute")
def delete_document(request: Request, doc_id: str, dep=Depends(require_api_key)):
    if repo.delete(doc_id):
        return success({"status": "deleted"})
    return error("Document not found or could not be deleted", 404)


@app.get("/healthz", tags=["System"], summary="Проверка здоровья", description="Быстрый self-check: доступ к ChromaDB и вызов LLM.")
def healthz():
    try:
        doc_count = repo.count()
    except Exception as e:
        return error(f"ChromaDB error: {e}", 500)

    try:
        test = ai_model.generate("тестовая проверка")
        model_ok = True if test else False
    except Exception as e:
        return error(f"AI model error: {e}", 500)

    return success({
        "status": "ok",
        "chroma_ok": True,
        "documents_count": doc_count,
        "model_ok": model_ok
    })


# --------------------- Lifecycle ---------------------
@app.on_event("startup")
async def on_startup():
    logger.info("API started")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("API shutting down...")
    try:
        app.state.env.stop()
    finally:
        executor.shutdown(wait=True)


# --------------------- Global Exception Handler ---------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return PlainTextResponse("Internal server error", status_code=500)


# --------------------- Rate-limit handler ---------------------
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return PlainTextResponse("Too Many Requests", status_code=429)

# ---------- CHATS CRUD (PostgreSQL) ----------

@app.post("/api/chats", tags=["Chats"], summary="Создать чат")
@limiter.limit("10/minute")
def create_chat(
    request: Request,
    body: ChatCreateSchema,
    db: Session = Depends(get_db),
    dep_api=Depends(require_api_key),
    dep_user=Depends(user_header),
):
    username = resolve_username(request)
    user = pg_repo.get_or_create_user(db, username)
    chat = pg_repo.create_chat(db, user_id=user.id, title=body.title or "Новый чат")
    return success({
        "id": chat.id, "title": chat.title, "user": user.username,
        "created_at": chat.created_at.isoformat(), "updated_at": chat.updated_at.isoformat()
    }, 201)

@app.get("/api/chats", tags=["Chats"], summary="Список чатов текущего пользователя")
@limiter.limit("30/minute")
def list_chats(
    request: Request,
    db: Session = Depends(get_db),
    dep_api=Depends(require_api_key),
    dep_user=Depends(user_header),
):
    username = resolve_username(request)
    user = pg_repo.get_or_create_user(db, username)
    items = pg_repo.get_user_chats(db, user_id=user.id)
    return success([{
        "id": c.id, "title": c.title,
        "created_at": c.created_at.isoformat(),
        "updated_at": c.updated_at.isoformat()
    } for c in items])

@app.put("/api/chats/{chat_id}", tags=["Chats"], summary="Переименовать чат")
@limiter.limit("10/minute")
def rename_chat(
    request: Request,
    chat_id: int,
    body: ChatRenameSchema,
    db: Session = Depends(get_db),
    dep_api=Depends(require_api_key),
    dep_user=Depends(user_header),
):
    username = resolve_username(request)
    user = pg_repo.get_or_create_user(db, username)
    chat = pg_repo.get_chat(db, chat_id=chat_id, user_id=user.id)
    if not chat:
        return error("Chat not found", 404)
    chat = pg_repo.update_chat_title(db, chat, body.title)
    return success({"status": "renamed", "id": chat.id, "title": chat.title})

# ---------- CHAT MESSAGES ----------

@app.get("/api/chats/{chat_id}/messages", tags=["Messages"], summary="Сообщения чата")
@limiter.limit("60/minute")
def get_chat_messages(
    request: Request,
    chat_id: int,
    db: Session = Depends(get_db),
    dep_api=Depends(require_api_key),
    dep_user=Depends(user_header),
):
    username = resolve_username(request)
    user = pg_repo.get_or_create_user(db, username)
    msgs = pg_repo.get_chat_messages(db, chat_id=chat_id, user_id=user.id)
    if msgs is None:
        return error("Chat not found", 404)
    return success([{
        "id": m.id, "role": m.role, "content": m.content,
        "timestamp": m.timestamp.isoformat()
    } for m in msgs])

@app.post("/api/chats/{chat_id}/messages", tags=["Messages"], summary="Отправить сообщение и получить ответ ассистента")
@limiter.limit("12/minute")
async def send_chat_message(
    request: Request,
    chat_id: int,
    data: ChatSendSchema,
    db: Session = Depends(get_db),
    dep_api=Depends(require_api_key),
    dep_user=Depends(user_header),
):
    username = resolve_username(request)
    user = pg_repo.get_or_create_user(db, username)
    chat = pg_repo.get_chat(db, chat_id=chat_id, user_id=user.id)
    if not chat:
        return error("Chat not found", 404)

    user_msg = pg_repo.add_message(db, chat_id=chat.id, role="user", content=data.content)
    chat_index.index_message(chat_id=chat.id, message_id=user_msg.id, role="user",
                             content=user_msg.content, ts=user_msg.timestamp)

    history_snippets = chat_index.build_history_context(
        chat_id=chat.id,
        query=data.content,
        recent_docs=CHAT_RECENT_DOCS,
        semantic_k=CHAT_SEMANTIC_K,
        max_chars=CHAT_MAX_CHARS
    )

    future = executor.submit(ai_model.generate_with_chat, data.content, history_snippets, CHROMA_TOP_K)

    try:
        result = future.result(timeout=120)
    except TimeoutError:
        return error("Processing timed out", 504)
    except Exception as e:
        logger.error("Chat generation failed: %s", e, exc_info=True)
        return error("Processing failed", 500)

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    assistant_msg = pg_repo.add_message(db, chat_id=chat.id, role="assistant", content=answer)
    chat_index.index_message(chat_id=chat.id, message_id=assistant_msg.id, role="assistant",
                             content=assistant_msg.content, ts=assistant_msg.timestamp,
                             extra_meta={"sources": sources})

    return success({
        "answer": answer,
        "sources": sources,
        "user_message": {
            "id": user_msg.id, "role": user_msg.role, "content": user_msg.content,
            "timestamp": user_msg.timestamp.isoformat()
        },
        "assistant_message": {
            "id": assistant_msg.id, "role": assistant_msg.role, "content": assistant_msg.content,
            "timestamp": assistant_msg.timestamp.isoformat()
        }
    }, 201)

@app.get("/api/users/{username}/chats", tags=["Admin"], summary="(Admin) Список чатов пользователя")
@limiter.limit("30/minute")
def admin_get_user_chats(request: Request, username: str, db: Session = Depends(get_db), dep=Depends(require_api_key)):
    require_admin(request)
    user = pg_repo.get_user_by_username(db, username)
    if not user:
        return error("User not found", 404)
    chats = pg_repo.get_user_chats(db, user_id=user.id)
    return success([{
        "id": c.id, "title": c.title,
        "created_at": c.created_at.isoformat(),
        "updated_at": c.updated_at.isoformat()
    } for c in chats])

@app.get("/api/users/{username}/chats/{chat_id}/messages", tags=["Admin"], summary="(Admin) Сообщения чата пользователя")
@limiter.limit("60/minute")
def admin_get_user_chat_messages(request: Request, username: str, chat_id: int,
                                 db: Session = Depends(get_db), dep=Depends(require_api_key)):
    require_admin(request)
    user = pg_repo.get_user_by_username(db, username)
    if not user:
        return error("User not found", 404)
    msgs = pg_repo.get_chat_messages(db, chat_id=chat_id, user_id=user.id)
    if msgs is None:
        return error("Chat not found", 404)
    return success([{
        "id": m.id, "role": m.role, "content": m.content,
        "timestamp": m.timestamp.isoformat()
    } for m in msgs])

CHAT_COLLECTION = getattr(chat_index, "collection_name", "chat_messages")

@app.delete("/debug/chroma/chat_messages", tags=["Debug"], summary="Сбросить коллекцию истории чатов")
def debug_clear_chat_index(dep=Depends(require_api_key)):
    try:
        # Удаляем коллекцию, если есть
        try:
            chat_index.client.delete_collection(name=CHAT_COLLECTION)
        except Exception:
            pass  # если её нет — не критично

        # Пересоздаём её на том же клиенте и с тем же embedding_function
        chat_index.messages = chat_index.client.get_or_create_collection(
            name=CHAT_COLLECTION,
            embedding_function=getattr(repo, "embedding_function", None),
            metadata={"hnsw:space": "cosine"},
        )
        return success({"status": "recreated", "collection": CHAT_COLLECTION})
    except Exception as e:
        logger.error("Failed to recreate chat collection: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to recreate chat collection")


@app.delete("/debug/chroma/chat_messages/{chat_id}", tags=["Debug"], summary="Очистить историю конкретного чата")
def debug_clear_chat(dep=Depends(require_api_key), chat_id: int = Path(..., ge=1)):
    try:
        removed = chat_index.delete_chat(chat_id)
        return success({"status": "cleared", "chat_id": chat_id, "removed": removed})
    except Exception as e:
        logger.error("Failed to clear chat history for %s: %s", chat_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear chat history")

@app.delete("/api/chats/{chat_id}", tags=["Chats"], summary="Удалить чат")
@limiter.limit("5/minute")
def remove_chat(
    request: Request,
    chat_id: int,
    db: Session = Depends(get_db),
    dep_api=Depends(require_api_key),
    dep_user=Depends(user_header),
):
    username = resolve_username(request)
    user = pg_repo.get_or_create_user(db, username)

    ok = pg_repo.delete_chat(db, chat_id=chat_id, user_id=user.id)
    try:
        chat_index.delete_chat(chat_id)  # ← заменили прямой where-delete
    except Exception:
        pass

    return success({"status": "deleted"}) if ok else error("Chat not found", 404)

