"""Сквозная проверка MCP→поиск→LLM→память на реальном индексе."""
import logging
import sys
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import memory
from llm_client import UNAVAILABLE
from orchestrator import Orchestrator


def main():
    logging.basicConfig(level=logging.INFO)
    if config.get_collection().count() == 0:
        raise SystemExit('Индекс пуст: сначала подготовьте корпус и выполните ingest.py.')
    session = f'smoke:{uuid.uuid4()}'
    core = Orchestrator()
    try:
        for i, question in enumerate(['Какой минимальный возраст указан в преддоговорном информировании о потребительском кредите?',
                         'А какой максимальный возраст на момент окончания договора?',
                         'Повтори возрастные ограничения из первого и второго ответов одним предложением.']):
            answer = core.handle(session, question)
            print(question, '\n', answer)
            if not answer.strip() or answer == UNAVAILABLE:
                raise SystemExit('Реальный ответ не получен.')
            if config.LLM_BACKEND != 'fake':
                required = ('21',) if i == 0 else ('67',) if i == 1 else ('21', '67')
                assert all(value in answer for value in required), answer
                assert 'raiffeisen_cash_' in answer, 'Нет ссылки на источник: ' + answer
        assert len(memory.window(session, 10)) == 6
        assert len(core.usage) == 3
        print('Конвейер завершил три запроса; шесть сообщений сохранены.')
        if config.LLM_BACKEND == 'fake':
            print('ВАЖНО: это отладка fake, не проверка качества языковой модели.')
    finally:
        core.reset(session)
        core.close()


if __name__ == '__main__':
    main()
