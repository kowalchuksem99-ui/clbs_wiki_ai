from core.repositories.chroma_repo import ChromaRepository
import threading, logging, os
from typing import List, Dict, Any, Optional
import httpx
import openai

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

from core.utils.env_hot import EnvHot

logger = logging.getLogger(__name__)

class _YandexClient:
    """Клиент для Яндекс API"""
    def __init__(self, api_key: str = None, model: str = None):
        self.model = model or os.getenv("YANDEX_CLOUD_MODEL")
        self.api_key = api_key or os.getenv("YANDEX_CLOUD_API_KEY")

        openai.api_key = self.api_key  # Настройка ключа API для OpenAI

    def chat(self, messages: str, max_tokens: int = 1000, temperature: float = 0.3) -> str:
        YANDEX_CLOUD_MODEL = os.getenv("YANDEX_CLOUD_MODEL")
        YANDEX_CLOUD_FOLDER = os.getenv("YANDEX_CLOUD_FOLDER")

        client = openai.OpenAI(
            api_key=os.getenv("YANDEX_CLOUD_API_KEY"),
            base_url=os.getenv("YANDEX_LLM_BASE_URL"),
            project=YANDEX_CLOUD_FOLDER
        )

        response = client.responses.create(
            model=f"gpt://{YANDEX_CLOUD_FOLDER}/{YANDEX_CLOUD_MODEL}",
            input=messages,
            temperature=temperature,
            max_output_tokens=max_tokens )
        return response.output[0].content[0].text

class _OpenAICompatClient:
    """Мини-клиент для OpenAI-совместимого Chat Completions API."""
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60, extra_headers: Dict[str,str]|None=None):
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        self.client = httpx.Client(base_url=base_url, timeout=timeout, headers=headers)

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.3) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        r = self.client.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()


class AiModel:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path: str, repo_path: str | None, env: EnvHot):
        if hasattr(self, 'initialized'):
            return
        try:
            self.repo = ChromaRepository()
            self.model_lock = threading.Lock()
            self.env = env

            backend = os.getenv("LLM_BACKEND", "local").lower()
            self.backend = backend

            if backend == "openai_compat":
                base_url    = os.getenv("LLM_BASE_URL", "https://api.groq.com")
                api_key     = os.getenv("LLM_API_KEY", "")
                model_name  = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
                timeout_sec = int(os.getenv("LLM_TIMEOUT", "60"))
                if not api_key:
                    raise RuntimeError("LLM_API_KEY is required for openai_compat backend")
                extra_headers = None
                if "openrouter.ai" in base_url:
                    extra_headers = {
                        "HTTP-Referer": "http://localhost",
                        "X-Title": "My FastAPI Service"
                    }

                self.remote = _OpenAICompatClient(
                    base_url, api_key, model_name, timeout=timeout_sec,
                    extra_headers=extra_headers
                )
                logger.info("LLM backend: openai_compat (%s, model=%s)", base_url, model_name)

            elif backend == "yandex":
                api_key = os.getenv("YANDEX_CLOUD_API_KEY", "")
                model_name = os.getenv("YANDEX_CLOUD_MODEL", "yandexgpt-lite")

                if not api_key:
                    raise RuntimeError("YANDEX_CLOUD_API_KEY is required for Yandex backend")

                self.remote = _YandexClient(
                    api_key=api_key, model=model_name
                )
                logger.info("LLM backend: Yandex (model=%s)", model_name)

            else:
                if Llama is None:
                    raise RuntimeError("llama_cpp not available, install it or use openai_compat backend")
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=16000,
                    n_gpu_layers=0,
                    verbose=False,
                    n_threads=4
                )
                logger.info("LLM backend: local (llama_cpp): %s", model_path)

            self.initialized = True
        except Exception as e:
            logger.exception("Failed to initialize AiModel: %s", e)
            raise

    def generate(self, query: str) -> dict:
        if len(query) < 3 or len(query.split()) < 2:
            return {'answer': 'Уточните или переформулируйте вопрос', 'sources': []}

        try:
            with self.model_lock:
                top_k = int(os.getenv("CHROMA_TOP_K", "5"))
                docs = self.repo.find_similar(
                    query=query.lower(),
                    include=['documents', 'metadatas'],
                    n_results=top_k
                )
                if not docs or not (docs.get("documents") and (docs["documents"][0] or [])):
                    return {'answer': 'Информация не найдена', 'sources': []}

                filtered_docs   = docs['documents'][0]
                filtered_ids = (docs.get('ids') or [[]])[0]
                filtered_titles = [m.get('title', 'Без названия') for m in docs['metadatas'][0]]

                prompt = self._build_prompt(filtered_docs, query)
                messages = [{"role": "user", "content": prompt}]

                max_tokens  = int(os.getenv("LLM_MAX_TOKENS", "512"))
                temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

                if self.backend == "openai_compat":
                    result_text = self.remote.chat(messages, max_tokens=max_tokens, temperature=temperature)
                elif self.backend == "yandex":
                    result_text = self.remote.chat(messages, max_tokens=max_tokens, temperature=temperature)
                else:
                    result = self.llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    result_text = result['choices'][0]['message']['content'].strip()

                sources = self._generate_sources(filtered_ids, filtered_titles)
                answer_with_links = self._inject_sources_into_answer(result_text, sources)
                hide_sources = os.getenv("HIDE_SOURCES_FIELD", "0").lower() in ("1", "true", "yes", "on")
                return {"answer": answer_with_links, "sources": [] if hide_sources else sources}

        except httpx.HTTPError as e:
            logger.error("Remote LLM HTTP error: %s", e, exc_info=True)
            return {'answer': 'Ошибка обращения к сервисной модели.', 'sources': []}
        except Exception as e:
            logger.error("Error generating response: %s", e, exc_info=True)
            return {'answer': 'Произошла ошибка при обработке запроса.', 'sources': []}

    def generate_with_chat(self, query: str, chat_history_snippets: Optional[List[str]] = None,
                           n_results: int = 5) -> dict:
        history = chat_history_snippets or []

        try:
            top_k = int(os.getenv("CHROMA_TOP_K", str(n_results)))
            docs = self.repo.find_similar(
                query=query.lower(),
                include=['documents', 'metadatas'],
                n_results=top_k
            )
            kb_docs = []
            if docs and docs.get("documents"):
                kb_docs = docs["documents"][0] or []
            ids = (docs.get("ids") or [[]])[0] if isinstance(docs.get("ids"), list) else []
            titles = [m.get('title', 'Без названия') for m in (docs.get("metadatas") or [[]])[0]] if docs.get(
                "metadatas") else []
            sources = self._generate_sources(ids, titles) if ids else [
                {"id": f"doc-{i + 1}", "title": titles[i] if i < len(titles) else "Документ", "url": ""} for i in
                range(len(kb_docs))]
        except Exception as e:
            logger.error("Error finding similar documents for chat: %s", e, exc_info=True)
            kb_docs = []
            sources = []

        parts = []
        if history:
            parts.extend(history)
        if kb_docs:
            parts.extend(kb_docs)

        merged_context = "\n\n".join(parts)

        # 4) если совсем нет контекста — "Информация не найдена"
        prompt = self._build_prompt(merged_context, query)
        messages = [{"role": "user", "content": prompt}]

        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "512"))
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

        try:
            if self.backend == "openai_compat":
                result_text = self.remote.chat(messages, max_tokens=max_tokens, temperature=temperature)
            if self.backend == "yandex":
                result_text = self.remote.chat(messages, max_tokens=max_tokens, temperature=temperature)   
            else:
                result = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                result_text = result['choices'][0]['message']['content'].strip()

            answer_with_links = self._inject_sources_into_answer(result_text, sources)
            return {"answer": answer_with_links, "sources": sources}
        except Exception as e:
            logger.error("Error generating chat response: %s", e, exc_info=True)
            return {"answer": "Произошла ошибка при обработке запроса.", "sources": []}

    def _generate_sources(self, ids: list, titles: list) -> list:
        return [{"id": i, "title": t, "url": f"https://wiki.clbs.digital/{i}"} for i, t in zip(ids, titles)]

    def _inject_sources_into_answer(self, text: str, sources: list) -> str:
        """
        Вшивает список источников в конец текста ответа.
        Управляется флагом INLINE_SOURCES_IN_ANSWER (1/true/on).
        """
        show = os.getenv("INLINE_SOURCES_IN_ANSWER", "1").lower() in ("1", "true", "yes", "on")
        if not show or not sources:
            return text

        lines = []
        for s in sources:
            title = s.get("title", "Документ")
            url = (s.get("url") or "").strip()
            if url:
                lines.append(f"- [{title}]({url})")
            else:
                lines.append(f"- {title}")

        return text.rstrip() + "\n\nИсточники:\n" + "\n".join(lines)

    def _build_prompt(self, docs, query: str) -> str:
        max_doc_len = 2000
        if isinstance(docs, str):
            context = docs[: max_doc_len * 4]
        else:
            context = "\n\n".join((d or "")[:max_doc_len] for d in docs)
        template = self.env.prompt_text()
        return template.format(context=context, query=query)

    # Опционально: смена модели для remote/local
    def reload_model(self, new_model_path_or_name: str):
        if self.backend == "openai_compat":
            self.remote.model = new_model_path_or_name
        else:
            if Llama is None:
                raise RuntimeError("llama_cpp not available")
            with self.model_lock:
                self.llm = Llama(
                    model_path=new_model_path_or_name,
                    n_ctx=16000,
                    n_gpu_layers=0,
                    verbose=False,
                    n_threads=4
                )
