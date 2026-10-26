"""Проверка подлинности и извлечения корпуса; ничего не скачивает без URL."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from ingest import READERS


def main():
    manifest = json.loads((config.DATA_DIR / 'sources.json').read_text())
    documents = manifest['documents']
    if len(documents) < 2:
        raise SystemExit('Нужно минимум два настоящих документа в data/sources.json; корпус не готов.')
    total = 0
    for doc in documents:
        path = config.RAW_DIR / doc['filename']
        if path.parent.resolve() != config.RAW_DIR.resolve():
            raise SystemExit('Имя документа не должно выходить за data/raw')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != doc['sha256']:
            raise SystemExit(f'Контрольная сумма не совпала: {path.name}')
        pages = READERS[path.suffix.lower()](path)
        nonempty = sum(bool(text.strip()) for _, text in pages)
        if not nonempty:
            raise SystemExit(f'Текст не извлекается: {path.name}')
        if path.suffix.lower() == '.pdf':
            if len(pages) != doc['pages']:
                raise SystemExit(f'Не совпало число страниц: {path.name}')
            total += len(pages)
        print(f'{path.name}: {nonempty}/{len(pages)} текстовых страниц; SHA-256 совпал')
        print(repr(pages[0][1][:300]))
    if not 20 <= total <= 50:
        raise SystemExit(f'Проверьте объём корпуса: {total} PDF-страниц; цель 20–50.')


if __name__ == '__main__':
    main()
