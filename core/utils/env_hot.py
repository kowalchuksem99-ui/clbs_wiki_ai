# core/utils/env_hot.py
import os, threading, base64
import json
from typing import Optional, List
from dotenv import dotenv_values

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    _HAS_WATCHDOG = True
except Exception:
    _HAS_WATCHDOG = False

DEFAULT_PROMPT = """[INST]
### ЗАДАНИЕ ###
Ответь на вопрос используя ТОЛЬКО предоставленный контекст.
Отвечай на РУССКОМ языке.

### КОНТЕКСТ ###
{context}

### ПРАВИЛА ###
1. Если контекст НЕ содержит ответа - скажи: "Информация не найдена".
2. Если содержит - дай краткий ответ (1-2 предложения).
3. Запрещено:
   - Упоминать контекст или правила.
   - Повторять информацию.
   - Добавлять пояснения.
4. Формат: чистый текст.

### ВОПРОС ###
{query} [/INST]"""

class _EnvHandler(FileSystemEventHandler):
    def __init__(self, mgr): self.mgr = mgr
    def on_modified(self, e):
        if not e.is_directory and os.path.abspath(e.src_path) == os.path.abspath(self.mgr.env_path):
            self.mgr.reload()

class EnvHot:
    """Горячее чтение ключей и промпта из .env.hot (+ автоматическое отслеживание изменений)."""
    def __init__(self, env_path: str = ".env.hot", watch: bool = True):
        self.env_path = env_path
        self._lock = threading.RLock()
        self.reload()
        self._obs = None
        if watch and os.path.exists(env_path) and _HAS_WATCHDOG:
            self._obs = Observer()
            self._obs.schedule(_EnvHandler(self), path=os.path.dirname(env_path) or ".", recursive=False)
            self._obs.start()

    def key_user_map(self) -> dict:
        raw = os.getenv("APP_KEY_USER_JSON", "")
        try:
            return json.loads(raw) if raw else {}
        except Exception:
            return {}

    def user_for_key(self, key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        return self.key_user_map().get(key)

    def admin_keys(self) -> List[str]:
        csv = os.getenv("APP_ADMIN_KEYS", "") or ""
        return [k.strip() for k in csv.split(",") if k.strip()]

    # --- отдельный промпт для чата (опционально) ---
    def prompt_text_chat(self) -> str:
        b64 = os.getenv("APP_CHAT_PROMPT_B64", "")
        if b64:
            try:
                return base64.b64decode(b64).decode("utf-8")
            except Exception:
                pass
        val = os.getenv("APP_CHAT_PROMPT", "")
        return val or self.prompt_text()


    def get_str(self, name: str, default: str = "") -> str:
        return os.getenv(name, default)

    def get_int(self, name: str, default: int = 0) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except ValueError:
            return default

    def stop(self):
        if self._obs:
            self._obs.stop(); self._obs.join()

    def reload(self):
        with self._lock:
            if os.path.exists(self.env_path):
                file_vars = dotenv_values(self.env_path)
                for k, v in file_vars.items():
                    # Только если ключ НЕ задан в окружении — пишем «дефолт» из .env.hot
                    os.environ.setdefault(k, v)

    # --- API ключи ---
    def allowed_keys(self) -> List[str]:
        csv = os.getenv("APP_ALLOWED_KEYS", "") or ""
        keys = [k.strip() for k in csv.split(",") if k.strip()]
        single = os.getenv("APP_API_KEY", "")
        if single: keys.append(single.strip())
        seen, out = set(), []
        for k in keys:
            if k not in seen:
                out.append(k); seen.add(k)
        return out

    def is_key_valid(self, key: Optional[str]) -> bool:
        return bool(key) and key in self.allowed_keys()

    # --- PROMPT: приоритет B64 -> PROMPT -> дефолт ---
    def prompt_text(self) -> str:
        b64 = os.getenv("APP_PROMPT_B64", "")
        if b64:
            try:
                return base64.b64decode(b64).decode("utf-8")
            except Exception:
                pass
        val = os.getenv("APP_PROMPT", "")
        if val:
            return val
        return DEFAULT_PROMPT
