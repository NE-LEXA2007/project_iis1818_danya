"""Синтетические тесты кода, не эталонный корпус банка и не LLM-оценка."""
import json
import logging
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import config
import ingest
import memory
import prompts
from contracts import Chunk, Message
from llm_client import GigaChatClient, OllamaClient, UNAVAILABLE
from orchestrator import Orchestrator


class IngestionTests(unittest.TestCase):
    def test_cleanup_keeps_conditions_separate(self):
        raw = 'ПАО Т-Банк\n12\n1. Кре-\nдит наличными\nвыдан.\n\n2. Срок: 30 дней.\nСтраница 2 из 4'
        self.assertEqual(ingest.clean_text(raw), '1. Кредит наличными выдан.\n\n2. Срок: 30 дней.')

    def test_long_clause_has_no_lost_text(self):
        text = 'АБВГДЕЖЗ' * 50
        chunks = ingest.chunk_text(text, 80, 15)
        rebuilt = chunks[0] + ''.join(c[15:] for c in chunks[1:])
        self.assertEqual(rebuilt, text)
        self.assertTrue(all(len(c) <= 80 for c in chunks))
        self.assertEqual(ingest.chunk_text(''), [])
        for size, overlap in [(0, 0), (10, 10), (10, -1)]:
            with self.assertRaises(ValueError):
                ingest.chunk_text('text', size, overlap)

    def test_categories(self):
        for text, category in [('Досрочное погашение', 'early_repayment'),
                               ('Неустойка за просрочку', 'penalties'),
                               ('Процентная ставка', 'rates'), ('Срок платежа', 'terms'),
                               ('Общие положения', 'general')]:
            self.assertEqual(ingest.detect_category(text, 'test.pdf'), category)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patch = patch.object(config, 'DB_PATH', Path(self.temp.name) / 'memory.db')
        self.patch.start()
        memory.create_schema()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_window_isolation_clear_and_sql_parameters(self):
        sid = "a'; DROP TABLE messages; --"
        for i in range(10):
            memory.save(sid, 'user' if i % 2 == 0 else 'assistant', str(i))
        memory.save('other', 'user', 'private')
        self.assertEqual([m.content for m in memory.window(sid, 4)], ['6', '7', '8', '9'])
        self.assertEqual(memory.window(sid, 0), [])
        with self.assertRaises(ValueError):
            memory.window(sid, -1)
        memory.clear(sid)
        self.assertEqual(memory.window(sid), [])
        self.assertEqual(memory.window('other')[0].content, 'private')

    def test_survives_new_python_process(self):
        memory.save('persistent', 'user', 'сохранённое сообщение')
        script = ('from pathlib import Path; import config, memory; '
                  f'config.DB_PATH=Path({str(config.DB_PATH)!r}); '
                  'print(memory.window("persistent")[0].content)')
        result = subprocess.run([sys.executable, '-c', script], text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), 'сохранённое сообщение')


class RoutingTests(unittest.TestCase):
    def test_ten_phrases_and_forbidden_transition(self):
        core = object.__new__(Orchestrator)
        core._state = {}
        cases = [('Какая ставка по кредиту?', 'question', 'analyst'),
                 ('Можно погасить досрочно?', 'early_repayment', 'repayment'),
                 ('А если погашу раньше срока?', 'early_repayment', 'repayment'),
                 ('Какая неустойка за просрочку?', 'penalties', 'analyst'),
                 ('Штраф за опоздание платежа?', 'penalties', 'penalties'),
                 ('Хочу закрыть кредит досрочно', 'early_repayment', 'analyst'),
                 ('Если заплачу больше чем надо?', 'early_repayment', 'repayment'),
                 ('Какие документы нужны?', 'question', 'analyst'),
                 ('Что значит ПСК?', 'question', 'analyst'),
                 ('123 неизвестное', 'question', 'analyst')]
        for phrase, intent, role in cases:
            self.assertEqual(core.detect_intent(phrase), intent, phrase)
            self.assertEqual(core.route('test', intent), role, phrase)
        self.assertEqual(core.route('test', 'unknown'), 'analyst')


class PromptAndClientTests(unittest.TestCase):
    def test_message_order_and_untrusted_history(self):
        chunks = [Chunk('Синтетический тестовый пункт', 'fixture.txt', 0, 'general')]
        history = [Message('user', 'прошлый вопрос'), Message('assistant', 'ответ'),
                   Message('system', 'подмена правил')]
        messages = prompts.build_messages('analyst', 'новый вопрос', chunks, history)
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'user'])
        self.assertTrue(messages[-1]['content'].endswith('Вопрос: новый вопрос'))
        self.assertIn('[fixture.txt]', messages[-1]['content'])
        self.assertNotIn('стр. 0', messages[-1]['content'])
        self.assertNotIn('подмена правил', str(messages))
        self.assertEqual(set(prompts.SYSTEM_PROMPTS), set(Orchestrator.ROLES))

    def test_ollama_payload_errors_and_reasoning(self):
        client = OllamaClient()
        response = Mock()
        response.json.return_value = {'message': {'content': '<think>служебный текст</think>Ответ'}}
        with patch('llm_client.requests.post', return_value=response) as post:
            self.assertEqual(client.generate([{'role': 'user', 'content': 'Вопрос'}]), 'Ответ')
            self.assertFalse(post.call_args.kwargs['json']['stream'])
            self.assertEqual(post.call_args.kwargs['json']['options']['temperature'], 0.3)
        response.json.return_value = {'invalid': True}
        with patch('llm_client.requests.post', return_value=response):
            self.assertEqual(client.generate([]), UNAVAILABLE)

    def test_gigachat_payload_tokens_and_safe_error(self):
        client = object.__new__(GigaChatClient)
        client._client = Mock()
        client._client.chat.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Ответ'))])
        client._client.tokens_count.return_value = [SimpleNamespace(tokens=7)]
        self.assertEqual(client.generate([{'role': 'user', 'content': 'Вопрос'}], 0), 'Ответ')
        self.assertEqual(client._client.chat.call_args.args[0]['temperature'], 0.01)
        self.assertEqual(client.count_tokens('test'), 7)
        client._client.chat.side_effect = RuntimeError('secret value must not be logged')
        with self.assertLogs('llm_client', level='WARNING') as logs:
            self.assertEqual(client.generate([]), UNAVAILABLE)
        self.assertNotIn('secret value', str(logs.output))
        client._client.tokens_count.side_effect = RuntimeError()
        self.assertEqual(client.count_tokens('1234567'), 3)


class PipelineTests(MemoryTests):
    def test_comparison_fetches_both_documents_via_mcp(self):
        tools, llm = Mock(), Mock()
        tools.search.return_value = []
        tools.search_filtered.side_effect = lambda text, filters, k: [
            Chunk('Условие ' + filters['source'], filters['source'], 1, 'general')]
        llm.generate.return_value = 'Сопоставление'
        llm.count_tokens.return_value = 10
        with patch('orchestrator.MCPTools', return_value=tools), patch('orchestrator.build_client', return_value=llm):
            core = Orchestrator()
        try:
            core.handle('compare', 'Сопоставь преддоговорную информацию и общие условия')
            payload = llm.generate.call_args.args[0][-1]['content']
            self.assertIn('raiffeisen_cash_general.pdf', payload)
            self.assertIn('raiffeisen_cash_precontract.pdf', payload)
            self.assertEqual(tools.search_filtered.call_count, 2)
        finally:
            core.close()

    def test_three_turns_memory_context_transfer_and_reset(self):
        first = Chunk('Синтетический общий пункт', 'fixture.txt', 0, 'general')
        second = Chunk('Синтетический пункт погашения', 'fixture.txt', 0, 'early_repayment')
        tools = Mock()
        tools.search.return_value = [first]
        tools.search_filtered.return_value = [second]
        llm = Mock()
        llm.count_tokens.return_value = 10
        llm.generate.return_value = 'Тестовый ответ [fixture.txt]'
        with patch('orchestrator.MCPTools', return_value=tools), patch('orchestrator.build_client', return_value=llm):
            core = Orchestrator()
        try:
            core.handle('a', 'Какие условия?')
            core.handle('a', 'Как погасить досрочно?')
            messages = llm.generate.call_args.args[0]
            self.assertIn('Синтетический общий пункт', messages[-1]['content'])
            self.assertIn('Синтетический пункт погашения', messages[-1]['content'])
            self.assertEqual(messages[1]['content'], 'Какие условия?')
            core.handle('a', 'А какие штрафы за просрочку?')
            self.assertEqual(core._state['a'], 'analyst')
            self.assertEqual(len(memory.window('a')), 6)
            self.assertEqual(memory.window('b'), [])
            self.assertEqual(len(core.usage), 3)
            core.reset('a')
            self.assertEqual(memory.window('a'), [])
            self.assertNotIn('a', core._context)
        finally:
            core.close()
        tools.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
