"""Настоящие Chroma/E5/MCP на СИНТЕТИЧЕСКИХ данных, изолированно от корпуса."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    with tempfile.TemporaryDirectory(prefix='iis-integration-') as temp:
        os.environ['IIS_DATA_DIR'] = temp
        import config
        import memory
        from mcp_client import MCPTools
        memory.create_schema()
        collection = config.get_collection()
        collection.upsert(
            ids=['synthetic-general', 'synthetic-early', 'synthetic-penalty'],
            documents=['СИНТЕТИЧЕСКИЙ ТЕСТ. Общие условия учебного кредита.',
                       'СИНТЕТИЧЕСКИЙ ТЕСТ. Досрочное погашение учебного кредита.',
                       'СИНТЕТИЧЕСКИЙ ТЕСТ. Неустойка при просрочке учебного кредита.'],
            metadatas=[dict(source='synthetic.txt', page=0, category=c)
                       for c in ['general', 'early_repayment', 'penalties']],
        )
        memory.save('integration', 'user', 'MCP test')
        # SDK MCP запускает subprocess с ограниченным окружением. Тестовый
        # сервер-обёртка передаёт только путь временного корпуса, без секретов.
        server = Path(temp) / "server.py"
        server.write_text(
            "import os, runpy, sys\n"
            + f"os.environ['IIS_DATA_DIR'] = {temp!r}\n"
            + f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
            + f"runpy.run_path({str(Path(__file__).resolve().parents[1] / 'mcp_server.py')!r}, run_name='__main__')\n"
        )
        client = MCPTools(server_script=str(server), timeout=120)
        try:
            found = client.search('досрочное погашение', k=1)
            assert found[0].category == 'early_repayment', found
            filtered = client.search_filtered('кредит', {'category': 'penalties'}, k=1)
            assert len(filtered) == 1 and filtered[0].category == 'penalties'
            assert client.get_history('integration')[0].content == 'MCP test'
            print(json.dumps({'synthetic_data_only': True, 'real_embeddings': config.EMBEDDING_MODEL,
                              'mcp_search': 'passed', 'mcp_category_filter': 'passed',
                              'mcp_history': 'passed'}, ensure_ascii=False, indent=2))
        finally:
            client.close()


if __name__ == '__main__':
    main()
