"""
llm_client.py — доступ к языковой модели за общим интерфейсом.

ПРАКТИКА 4. Что делает студент:
  1. GigaChatClient.generate() — вызов Сбер GigaChat через официальный SDK
  2. OllamaClient.generate()   — вызов локальной модели через HTTP API Ollama
  3. count_tokens()            — оценка длины промпта до отправки

Готово и трогать не нужно: build_client() и выбор бэкенда по LLM_BACKEND.

Критерии приёмки:
  - смена LLM_BACKEND с gigachat на ollama не требует правки orchestrator.py
  - длина промпта считается до отправки и логируется
  - оба клиента принимают один и тот же список messages и возвращают строку

НЕ МЕНЯТЬ: сигнатуру generate(messages, temperature) — она объявлена в contracts.py
и на неё опирается оркестратор.

Зачем абстракция. Учебные квоты кончаются, сеть в аудитории падает, а занятие
должно продолжаться. Ollama — фолбэк, который работает офлайн. Заодно это
материал для сравнения авторегрессионной модели и reasoning-модели из Л4:
у DeepSeek-R1 в ответе виден блок рассуждений, у GigaChat — нет.

Третий бэкенд — fake_llm.FakeClient, офлайн-заглушка без сети и ключей. Она
нужна для отладки, когда недоступны оба клиента, и ваши реализации не заменяет:
работа, сданная на LLM_BACKEND=fake, не принимается.
"""

import logging
import math
import re

import requests
import config
from contracts import LLMClient

log = logging.getLogger(__name__)
UNAVAILABLE = "Языковая модель недоступна. Проверьте подключение и настройки выбранного провайдера."


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 3)


class GigaChatClient:
    def __init__(self):
        from gigachat import GigaChat
        self.model = config.GIGACHAT_MODEL
        self._client = GigaChat(
            credentials=config.GIGACHAT_CREDENTIALS,
            model=self.model,
            verify_ssl_certs=config.GIGACHAT_VERIFY_SSL,
            timeout=60,
        )

    def generate(self, messages: list[dict], temperature: float = 0.3) -> str:
        try:
            response = self._client.chat({
                "messages": messages, "temperature": max(0.01, temperature),
            })
            return response.choices[0].message.content
        except Exception as error:
            # Исключение SDK может содержать HTTP-заголовки: выводим только тип.
            log.warning("GigaChat недоступен (%s)", type(error).__name__)
            return UNAVAILABLE

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return self._client.tokens_count([text])[0].tokens
        except Exception:
            return estimate_tokens(text)


class OllamaClient:
    def __init__(self):
        self.model = config.OLLAMA_MODEL
        self.host = config.OLLAMA_HOST.rstrip("/")

    def generate(self, messages: list[dict], temperature: float = 0.3) -> str:
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False,
                      "options": {"temperature": max(0.01, temperature), "num_ctx": 8192, "num_predict": 512}},
                timeout=(10, 180),
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
            # Служебное рассуждение не является ответом с подтверждёнными условиями.
            return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
        except (requests.RequestException, ValueError, KeyError, TypeError):
            return UNAVAILABLE

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)


def build_client() -> LLMClient:
    """Выбрать бэкенд по конфигурации. Готово, менять не нужно."""
    if config.LLM_BACKEND == "ollama":
        return OllamaClient()
    if config.LLM_BACKEND == "gigachat":
        return GigaChatClient()
    if config.LLM_BACKEND == "fake":
        # Импорт внутри ветки: без него fake_llm тянулся бы при любом бэкенде
        from fake_llm import FakeClient

        return FakeClient()
    raise ValueError(
        f"Неизвестный LLM_BACKEND: {config.LLM_BACKEND!r}. "
        f"Допустимо: gigachat | ollama | fake"
    )
