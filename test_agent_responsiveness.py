"""Bounded offline regressions; HTTP/session dependencies are replaced, never called."""
import asyncio
import copy
import json
import time
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import httpx
from starlette.requests import Request
from offline_test_support import install_network_deny
install_network_deny()

import mimoly as m
import mimoly_rag as rag
from mimoly_interceptor import ActionEvaluator, Verdict
from mimoly_reliability import PooledAccount, RetryBudget

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}}]

def request():
    async def receive():
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}
    return Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []}, receive)

class Response:
    status_code = 200
    def __init__(self, script):
        self.script = script
        self.closed = False
    async def __aenter__(self):
        return self
    async def __aexit__(self, *exc):
        self.closed = True
    async def aiter_lines(self):
        for item in self.script:
            if isinstance(item, float):
                await asyncio.sleep(item)
            elif isinstance(item, Exception):
                raise item
            else:
                yield 'data: ' + json.dumps({"type": "answer", "content": item})

class Client:
    def __init__(self, scripts):
        self.scripts = iter(scripts)
        self.responses = []
        self.queries = []
    def stream(self, *args, **kwargs):
        self.queries.append(kwargs['json']['query'])
        response = Response(next(self.scripts))
        self.responses.append(response)
        return response

class ProtocolTests(unittest.TestCase):
    def test_developer_goal_and_raw_observation_preserved(self):
        observation = '{"old_string": "say \\"hello\\"", "code": "x = \\"value\\""}'
        messages = [{"role": "system", "content": "SYSTEM_SENTINEL"}, {"role": "developer", "content": "DEVELOPER_SENTINEL"}, {"role": "user", "content": "GOAL_SENTINEL"}] + [{"role": "tool", "name": "read_file", "content": observation}] * 6
        for tools in (TOOLS, None):
            with self.subTest(tools=bool(tools)):
                prompt = m.build_agent_prompt(messages, tools)
                for text in ('SYSTEM_SENTINEL', 'DEVELOPER_SENTINEL', 'GOAL_SENTINEL'):
                    self.assertIn(text, prompt)
                if tools:
                    self.assertIn(observation, prompt)

    def test_instruction_roles_remain_distinguishable(self):
        messages = [{'role': 'system', 'content': 'system rule'}, {'role': 'developer', 'content': 'developer rule'}, {'role': 'user', 'content': 'goal'}]
        for tools in (TOOLS, None):
            prompt = m.build_agent_prompt(messages, tools)
            self.assertIn('[System]', prompt)
            self.assertIn('[Developer]', prompt)

    def test_read_pagination_preserved(self):
        self.assertEqual(m.normalize_tool_args('read_file', {'path': 'a.py', 'offset': 51, 'limit': 25}, available_tools=TOOLS), {'path': 'a.py', 'offset': 51, 'limit': 25})

    def test_emitted_protocols_parse(self):
        fixtures = [
            {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "a.py", "offset": 51}},
            {"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"a.py","offset":51}'}}]},
        ]
        for fixture in fixtures:
            for wrap in (lambda s: s, lambda s: '<tool_call>' + s + '</tool_call>', lambda s: '```json\n' + s + '\n```'):
                text = wrap(json.dumps(fixture))
                with self.subTest(text=text):
                    calls, _ = m.smart_extract_tool_calls(text, 'Read a.py', TOOLS)
                    self.assertEqual(calls, [{'name': 'read_file', 'arguments': {'path': 'a.py', 'offset': 51}}])

    def test_multiple_json_calls_preserved(self):
        text = ''.join('<tool_call>' + json.dumps({'name': 'read_file', 'arguments': {'path': name}}) + '</tool_call>' for name in ('a.py', 'b.py'))
        calls, clean = m.smart_extract_tool_calls(text, 'Read files', TOOLS)
        self.assertEqual([c['arguments']['path'] for c in calls], ['a.py', 'b.py'])
        self.assertEqual(clean, '')

    def test_unknown_and_illustrative_calls_not_executed(self):
        fixtures = [
            '<tool_call><function=unknown>{"path":"a.py"}</function></tool_call>',
            'For example:\n```xml\n<tool_call><function=read_file>{"path":"a.py"}</function></tool_call>\n```',
            'Example usage:\n```json\n{"name":"read_file","arguments":{"path":"a.py"}}\n```',
        ]
        for text in fixtures:
            with self.subTest(text=text):
                self.assertEqual(m.smart_extract_tool_calls(text, 'Explain tool syntax', TOOLS)[0], [])

    def test_action_prose_is_not_an_example_introduction(self):
        invocation = '```json\n{"name":"read_file","arguments":{"path":"a.py"}}\n```'
        for prose in ('I will inspect the syntax error now.',
                      'I will inspect the sample file now.',
                      'The usage count changed. I will inspect a.py now.'):
            with self.subTest(prose=prose):
                calls, _ = m.smart_extract_tool_calls(prose + '\n' + invocation, 'Read a.py', TOOLS)
                self.assertEqual(calls, [{'name': 'read_file', 'arguments': {'path': 'a.py'}}])

    def test_clear_example_introductions_stay_non_executable(self):
        invocation = '```json\n{"name":"read_file","arguments":{"path":"a.py"}}\n```'
        for prose in ('For example:', 'Example usage:', 'Here is an example:',
                      'Sample invocation:', 'Syntax:', 'Usage:', 'Contoh:', 'e.g.:'):
            with self.subTest(prose=prose):
                self.assertEqual(m.smart_extract_tool_calls(prose + '\n' + invocation, '', TOOLS)[0], [])

    def test_mixed_protocol_blocks_keep_source_order_without_duplicates(self):
        json_call = '<tool_call>{"name":"read_file","arguments":{"path":"json.py"}}</tool_call>'
        xml_call = '<tool_call><function=read_file>{"path":"xml.py"}</function></tool_call>'
        for first, second, paths in ((json_call, xml_call, ['json.py', 'xml.py']),
                                     (xml_call, json_call, ['xml.py', 'json.py']),
                                     ('```json\n{"name":"read_file","arguments":{"path":"json.py"}}\n```', xml_call, ['json.py', 'xml.py']),
                                     (xml_call, '```json\n{"name":"read_file","arguments":{"path":"json.py"}}\n```', ['xml.py', 'json.py'])):
            with self.subTest(paths=paths, first=first):
                calls, clean = m.smart_extract_tool_calls(first + '\n' + second, '', TOOLS)
                self.assertEqual([c['arguments']['path'] for c in calls], paths)
                self.assertEqual(clean, '')

    def test_invalid_blocks_do_not_hide_later_valid_calls(self):
        valid_json = '```json\n{"name":"read_file","arguments":{"path":"a.py"}}\n```'
        valid_xml = '<tool_call><function=read_file>{"path":"a.py"}</function></tool_call>'
        invalids = ('<tool_call>{"name":"unknown","arguments":{}}</tool_call>',
                    '<tool_call><function=unknown>{}</function></tool_call>',
                    '```json\n{"name":"unknown","arguments":{}}\n```',
                    '```json\nnot JSON\n```',
                    '<tool_call>{"name":"read_file","arguments":42}</tool_call>',
                    '<tool_call>not JSON',
                    '<tool_call><function=unknown>{}</function>')
        for invalid in invalids:
            for valid in (valid_json, valid_xml):
                with self.subTest(invalid=invalid, valid=valid):
                    calls, _ = m.smart_extract_tool_calls(invalid + '\n' + valid, '', TOOLS)
                    self.assertEqual(calls, [{'name': 'read_file', 'arguments': {'path': 'a.py'}}])

    def test_wrapped_json_fence_is_one_call(self):
        text = '<tool_call>```json\n{"name":"read_file","arguments":{"path":"a.py"}}\n```</tool_call>'
        calls, clean = m.smart_extract_tool_calls(text, '', TOOLS)
        self.assertEqual(calls, [{'name': 'read_file', 'arguments': {'path': 'a.py'}}])
        self.assertEqual(clean, '')

    def test_network_deny_is_fail_closed(self):
        from offline_test_support import ATTEMPTS, NetworkForbidden
        import socket
        before = len(ATTEMPTS)
        with self.assertRaises(NetworkForbidden):
            socket.create_connection(('offline.invalid', 443))
        self.assertEqual(len(ATTEMPTS), before + 1)

    def test_conservative_interceptor(self):
        fixtures = [('Write a poem about rain', 'Rain falls.'), ('Hitung 2 + 2', '4'), ('Read the text below; do not use tools', 'I will read it now.'), ('Create an implementation plan only; do not execute', 'I will create a plan.'), ('Create a file', 'Created the file successfully.')]
        for goal, reply in fixtures:
            with self.subTest(goal=goal):
                self.assertEqual(ActionEvaluator.evaluate_turn([{'role': 'user', 'content': goal}], True, reply), Verdict.PASS)
        self.assertEqual(ActionEvaluator.evaluate_turn([{'role': 'user', 'content': 'Read a.py'}], True, 'I will read it now.'), Verdict.NUDGE_PROSE)

    def test_cache_fingerprints_description_and_schema(self):
        a = copy.deepcopy(TOOLS)
        b = copy.deepcopy(TOOLS)
        a[0]['function']['description'] = 'alpha'
        b[0]['function']['description'] = 'omega'
        self.assertIn('alpha', rag._get_or_build_index(a).postings)
        self.assertIn('omega', rag._get_or_build_index(b).postings)
        c = copy.deepcopy(b)
        c[0]['function']['parameters']['properties']['special_field'] = {'type': 'string'}
        self.assertIn('special', rag._get_or_build_index(c).postings)

    def test_retry_backoff_never_exceeds_remaining_or_final_attempt(self):
        budget = RetryBudget(max_attempts=2, deadline=0.01, min_delay=9, base_delay=10)
        budget.start()
        self.assertLessEqual(budget.next_delay(0, 'rate_limit'), 0.01)
        self.assertEqual(budget.next_delay(1), 0)

class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.account = PooledAccount('offline', {})
        self.upstream = self.stack.enter_context(patch.object(
            m, 'get_shared_client', side_effect=AssertionError('Unmocked upstream client')))
        self.releases = []
        self.stack.enter_context(patch.object(m, 'acquire_account', return_value=({}, self.account)))
        self.stack.enter_context(patch.object(m, 'release_account', side_effect=lambda acct, success, error_class=None: self.releases.append((acct, success))))
        self.stack.enter_context(patch.object(m, 'can_rotate_account', return_value=False))
        async def conversation(*args, **kwargs):
            return 'offline'
        async def throttle(*args, **kwargs):
            pass
        self.stack.enter_context(patch.object(m, 'ensure_new_conversation', conversation))
        self.stack.enter_context(patch.object(PooledAccount, 'wait_throttle', throttle))
        # Keep each request unique to bypass any prior exact response cache.
        self.body = {'messages': [{'role': 'user', 'content': 'Say hello ' + str(id(self))}], 'stream': False}
        self.stack.enter_context(patch('builtins.open', unittest.mock.mock_open()))

    def client(self, scripts):
        client = Client(scripts)
        self.stack.enter_context(patch.object(m, 'get_shared_client', return_value=client))
        return client

    async def collect(self, response):
        if not hasattr(response, 'body_iterator'):
            return json.dumps(response) if isinstance(response, dict) else response.body.decode()
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return ''.join(chunks)

    async def test_idle_heartbeat_does_not_cancel_late_frame(self):
        self.body['stream'] = True
        self.client([[0.04, 'LATE_ANSWER']])
        # Scale only wait_for's heartbeat timeout; leaves the pending read delayed.
        original = asyncio.wait_for
        async def fast_wait(awaitable, timeout=None):
            return await original(awaitable, timeout=0.01 if timeout == 10.0 else timeout)
        self.stack.enter_context(patch.object(m.asyncio, 'wait_for', fast_wait))
        # New helper's heartbeat interval is also independently injectable.
        self.stack.enter_context(patch.object(m, '_HEARTBEAT_INTERVAL', 0.01, create=True))
        output = await self.collect(await m.handle_chat_completion(self.body, request()))
        self.assertIn('keep-alive', output)
        self.assertIn('LATE_ANSWER', output)
        self.assertFalse(self.account.get_lock().locked())

    async def test_partial_failed_attempt_does_not_contaminate_retry(self):
        client = self.client([['STALE_PARTIAL', httpx.ReadError('offline')], ['CLEAN_ANSWER']])
        # No real retry sleep; exercise real retry and response assembly.
        async def no_sleep(*args, **kwargs):
            pass
        self.stack.enter_context(patch.object(RetryBudget, 'sleep', no_sleep))
        output = await self.collect(await m.handle_chat_completion(self.body, request()))
        self.assertNotIn('STALE_PARTIAL', output)
        self.assertIn('CLEAN_ANSWER', output)
        self.assertEqual(len(client.queries), 2)

    async def test_streamed_answer_is_not_retried(self):
        self.body['stream'] = True
        client = self.client([['FIRST_ANSWER', httpx.ReadError('offline')], ['SHOULD_NOT_RUN']])
        output = await self.collect(await m.handle_chat_completion(self.body, request()))
        self.assertIn('FIRST_ANSWER', output)
        self.assertNotIn('SHOULD_NOT_RUN', output)
        self.assertEqual(len(client.queries), 1)

    async def test_streamed_reasoning_is_not_mixed_with_retry(self):
        self.body['stream'] = True
        client = self.client([['<think>FIRST_REASONING', httpx.ReadError('offline')], ['SHOULD_NOT_RUN']])
        async def no_sleep(*args, **kwargs):
            pass
        self.stack.enter_context(patch.object(RetryBudget, 'sleep', no_sleep))
        output = await self.collect(await m.handle_chat_completion(self.body, request()))
        self.assertIn('FIRST_REASONING', output)
        self.assertNotIn('SHOULD_NOT_RUN', output)
        self.assertEqual(len(client.queries), 1)

    async def test_programming_error_is_not_retried(self):
        client = self.client([[ValueError('BUG')], ['SHOULD_NOT_RUN']])
        async def no_sleep(*args, **kwargs):
            pass
        self.stack.enter_context(patch.object(RetryBudget, 'sleep', no_sleep))
        output = await self.collect(await m.handle_chat_completion(self.body, request()))
        self.assertIn('BUG', output)
        self.assertNotIn('SHOULD_NOT_RUN', output)
        self.assertEqual(len(client.queries), 1)

    async def test_deadline_includes_lock_wait(self):
        lock = self.account.get_lock()
        await lock.acquire()
        self.client([['unused']])
        self.stack.enter_context(patch.object(m, '_request_budget_seconds', return_value=0.03, create=True))
        started = time.monotonic()
        response = await asyncio.wait_for(m.handle_chat_completion(self.body, request()), timeout=0.15)
        self.assertEqual(response.status_code, 504)
        self.assertLess(time.monotonic() - started, 0.15)
        self.assertTrue(lock.locked())
        lock.release()
        self.assertEqual(len(self.releases), 1)

    async def test_deadline_includes_preprocessing(self):
        self.stack.enter_context(patch.object(m, 'acquire_account', return_value=(
            {'xiaomichatbot_ph': 'offline-dummy'}, self.account)))
        async def preprocessing(*args, **kwargs):
            await asyncio.sleep(0.2)
        preprocess = self.stack.enter_context(patch.object(
            m, 'prepare_multimedias_for_request', side_effect=preprocessing))
        self.stack.enter_context(patch.object(m, '_request_budget_seconds', return_value=0.03, create=True))
        response = await asyncio.wait_for(m.handle_chat_completion(self.body, request()), timeout=0.15)
        preprocess.assert_awaited_once()
        self.upstream.assert_not_called()
        self.assertEqual(response.status_code, 504)
        self.assertEqual(len(self.releases), 1)

    async def test_global_deadline_bounds_busy_and_nudge_backoff(self):
        self.stack.enter_context(patch.object(m, '_request_budget_seconds', return_value=0.04, create=True))
        self.body['tools'] = TOOLS
        self.body['messages'] = [{'role': 'user', 'content': 'Read a.py'}]
        for reply in ('服务器繁忙', 'I will read it now.'):
            client = Client([[reply]])
            with patch.object(m, 'get_shared_client', return_value=client):
                response = await asyncio.wait_for(m.handle_chat_completion(self.body, request()), timeout=0.2)
            self.assertEqual(response.status_code, 504)
            self.assertEqual(len(client.queries), 1)
            self.assertFalse(self.account.get_lock().locked())

    async def test_stream_deadline_cancels_pending_read(self):
        self.body['stream'] = True
        client = self.client([[1.0, 'SHOULD_NOT_EMIT']])
        self.stack.enter_context(patch.object(m, '_request_budget_seconds', return_value=0.03, create=True))
        response = await m.handle_chat_completion(self.body, request())
        output = await asyncio.wait_for(self.collect(response), timeout=0.2)
        self.assertIn('Request deadline exceeded', output)
        self.assertNotIn('SHOULD_NOT_EMIT', output)
        self.assertTrue(client.responses[0].closed)
        self.assertFalse(self.account.get_lock().locked())
        self.assertEqual(len(self.releases), 1)

    async def test_compact_retry_preserves_goal_developer_and_observations(self):
        self.body['tools'] = TOOLS
        self.body['messages'] = [
            {'role': 'system', 'content': 'SYSTEM_SENTINEL'},
            {'role': 'developer', 'content': 'DEVELOPER_SENTINEL'},
            {'role': 'user', 'content': 'Read a.py GOAL_SENTINEL'},
        ] + [{'role': 'tool', 'name': 'read_file', 'content': 'RECENT_OBSERVATION "exact"'}] * 7
        client = self.client([['I will read it now.'], ['<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>']])
        original_sleep = asyncio.sleep
        async def short_sleep(delay):
            await original_sleep(0)
        with patch.object(m.asyncio, 'sleep', short_sleep):
            await m.handle_chat_completion(self.body, request())
        self.assertEqual(len(client.queries), 2)
        for sentinel in ('SYSTEM_SENTINEL', 'DEVELOPER_SENTINEL', 'GOAL_SENTINEL', 'RECENT_OBSERVATION "exact"'):
            self.assertIn(sentinel, client.queries[1])

    async def test_cancel_queued_releases_account_not_foreign_lock(self):
        lock = self.account.get_lock()
        await lock.acquire()
        self.client([['unused']])
        before = m._ADMISSION.in_flight
        task = asyncio.create_task(m.handle_chat_completion(self.body, request()))
        await asyncio.sleep(0.02)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(lock.locked())
        lock.release()
        self.assertEqual(len(self.releases), 1)
        self.assertEqual(m._ADMISSION.in_flight, before)

    async def test_explicit_stream_close_releases_resources(self):
        self.body['stream'] = True
        client = self.client([[1.0, 'unused']])
        before = m._ADMISSION.in_flight
        response = await m.handle_chat_completion(self.body, request())
        await response.body_iterator.__anext__()
        await response.body_iterator.aclose()
        self.assertFalse(self.account.get_lock().locked())
        self.assertEqual(len(self.releases), 1)
        self.assertTrue(client.responses[0].closed)
        self.assertEqual(m._ADMISSION.in_flight, before)

    async def test_cancel_midstream_closes_read_and_releases_resources(self):
        self.body['stream'] = True
        client = self.client([[1.0, 'unused']])
        before = m._ADMISSION.in_flight
        response = await m.handle_chat_completion(self.body, request())
        task = asyncio.create_task(self.collect(response))
        await asyncio.sleep(0.02)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(client.responses[0].closed)
        self.assertFalse(self.account.get_lock().locked())
        self.assertEqual(len(self.releases), 1)
        self.assertEqual(m._ADMISSION.in_flight, before)

if __name__ == '__main__':
    unittest.main(verbosity=2)
