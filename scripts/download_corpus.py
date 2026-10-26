"""Восстановление фиксированных редакций из официальных PDF с проверкой SHA-256."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path
import requests
from pypdf import PdfReader, PdfWriter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def main():
    manifest = json.loads((config.DATA_DIR / 'sources.json').read_text())
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for doc in manifest['documents']:
        target = config.RAW_DIR / doc['filename']
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() == doc['sha256']:
                print(target.name, 'уже проверен')
                continue
            raise SystemExit(f'Не перезаписываю изменённый файл: {target}')
        response = requests.get(doc['url'], timeout=(15, 90))
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != doc['original_sha256']:
            raise SystemExit(f'Источник изменил редакцию: {doc["url"]}. Требуется отдельное обновление корпуса и gold, проверка суммы сохранена.')
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'source.pdf'
            source.write_bytes(response.content)
            pages = doc['original_pages']
            reader = PdfReader(source)
            if pages == list(range(1, len(reader.pages) + 1)):
                result = source.read_bytes()
            else:
                writer = PdfWriter()
                for number in pages:
                    writer.add_page(reader.pages[number - 1])
                extracted = Path(temp) / 'extracted.pdf'
                writer.write(str(extracted))
                result = extracted.read_bytes()
        if hashlib.sha256(result).hexdigest() != doc['sha256']:
            raise SystemExit(f'Контрольная сумма извлечения не совпала: {target.name}')
        target.write_bytes(result)
        print(target.name, 'скачан и проверен')


if __name__ == '__main__':
    main()
