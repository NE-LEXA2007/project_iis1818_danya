"""
memory.py — сессионная память на SQLite.

ПРАКТИКА 5 (рубежный контроль бэкенда). Что делает студент:
  1. create_schema() — CREATE TABLE messages по схеме ниже
  2. save()          — вставка реплики с привязкой к session_id
  3. window()        — выборка последних N сообщений в хронологическом порядке

Критерии приёмки:
  - БД создаётся автоматически при первом запуске
  - window(sid, 4) на истории из 10 сообщений возвращает ровно 4 последних,
    именно в хронологическом порядке (старое сверху), а не в обратном
  - история переживает перезапуск процесса
  - две разные сессии не видят реплик друг друга
  - диалог из трёх реплик демонстрирует память: третий вопрос опирается на первый ответ

НЕ МЕНЯТЬ: сигнатуры save() и window(). На них завязаны orchestrator.py,
mcp_server.py и test_suite.py.

Почему заглушки здесь не падают, а молчат. Ядро вызывает память начиная с П3,
а пишете вы её на П5. Поэтому до реализации save() ничего не делает, а window()
возвращает пустой список: система работает без памяти, просто забывчивая.
Как только вы уберёте заглушки, память включится сама.
"""

import sqlite3
from datetime import datetime, timezone
from contextlib import closing

import config
from contracts import Message

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    timestamp  TEXT    NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    """Соединение с БД. Готово, менять не нужно."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(config.DB_PATH)


def create_schema() -> None:
    """Создать таблицу messages, если её ещё нет.

    Подсказка: SCHEMA выше уже написана — выполните её и не забудьте commit().
    """
    with closing(connect()) as connection, connection:
        connection.executescript(SCHEMA)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)")


def save(session_id: str, role: str, content: str) -> None:
    """Сохранить одну реплику диалога.

    timestamp пишите в ISO-формате: datetime.now().isoformat(timespec="seconds").
    Пользуйтесь параметрами запроса (?, ?, ?), а не конкатенацией строк —
    иначе получите SQL-инъекцию в дополнение к prompt-инъекции из П7.
    """
    if role not in {"user", "assistant", "system"}:
        raise ValueError("Неизвестная роль сообщения")
    create_schema()
    with closing(connect()) as connection, connection:
        connection.execute(
            "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )


def window(session_id: str, n: int = config.MEMORY_WINDOW) -> list[Message]:
    """Последние n сообщений сессии в хронологическом порядке.

    Ловушка: ORDER BY id DESC LIMIT n даёт последние n, но в обратном порядке.
    Разверните результат перед возвратом, иначе LLM прочитает диалог задом наперёд.
    """
    if n < 0:
        raise ValueError("Размер окна не может быть отрицательным")
    if n == 0:
        return []
    create_schema()
    with closing(connect()) as connection:
        rows = connection.execute(
            "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, n),
        ).fetchall()
    return [Message(*row) for row in reversed(rows)]


def clear(session_id: str) -> None:
    """Удалить историю сессии. Нужна кнопке очистки в app.py (П6)."""
    create_schema()
    with closing(connect()) as connection, connection:
        connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
