"""Три настоящих прогона. Требует проверенный корпус и заполненный gold."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import ingest
from check_corpus import main as check_corpus


def run():
    check_corpus()
    dataset = json.loads((config.BASE_DIR / 'gold_dataset.json').read_text())
    if any(not c['expected_sources'] or not c['expected'] for c in dataset if c['type'] != 'out_of_scope'):
        raise SystemExit('Сначала заполните проверенные expected_sources и expected.')
    original = config.CHUNK_SIZE, config.CHUNK_OVERLAP
    rows = []
    try:
        for size, overlap in [(512, 64), (768, 64), (768, 128)]:
            config.CHUNK_SIZE, config.CHUNK_OVERLAP = size, overlap
            # У функции шаблона параметры по умолчанию связываются при импорте.
            ingest.chunk_text.__defaults__ = (size, overlap)
            count = ingest.ingest()
            rows.append(dict(size=size, overlap=overlap, chunks=count,
                             hit_rate_5=ingest.evaluate_retrieval(5),
                             hit_rate_1=ingest.evaluate_retrieval(1)))
        output = config.BASE_DIR / 'reports' / 'retrieval_measurements.json'
        output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
    finally:
        config.CHUNK_SIZE, config.CHUNK_OVERLAP = original
        ingest.chunk_text.__defaults__ = original
        ingest.ingest()


if __name__ == '__main__':
    run()
