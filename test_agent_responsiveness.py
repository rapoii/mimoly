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

REAL_ACQUIRE_ACCOUNT = m.acquire_account
REAL_RELEASE_ACCOUNT = m.release_account

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

    def test_tool_catalogue_explains_external_execution_without_fake_results(self):
        for framework in m.AGENT_PROFILES:
            with self.subTest(framework=framework):
                prompt = m.build_agent_prompt([{'role': 'user', 'content': 'Read a.py'}], TOOLS, framework=framework)
                self.assertIn('external client executor', prompt)
                self.assertIn('not native Xiaomi tools', prompt)
                self.assertIn('Do not invent tool results', prompt)
                self.assertIn('explicit no-tool requests', prompt)
        plain = m.build_agent_prompt([{'role': 'user', 'content': 'Write a poem without tools'}], None)
        self.assertNotIn('external client executor', plain)

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

    def test_json_string_protocol_markers_are_content(self):
        tools = TOOLS + [{'type': 'function', 'function': {
            'name': 'write_file', 'parameters': {'type': 'object',
                'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}},
                'required': ['path', 'content']}}}]
        content = 'Before\n```python\nprint("quoted")\n```\nAfter <tool_call> <function=read_file>'
        call = {'name': 'write_file', 'arguments': {'path': 'README.md', 'content': content}}
        payload = json.dumps(call)
        for text in (payload, '<tool_call>' + payload + '</tool_call>',
                     '```json\n' + payload + '\n```',
                     '<tool_call>```json\n' + payload + '\n```</tool_call>'):
            with self.subTest(text=text):
                calls, clean = m.smart_extract_tool_calls(text, '', tools)
                self.assertEqual(calls, [call])
                self.assertEqual(clean, '')

    def test_json_string_invocations_are_not_executed(self):
        nested = '<tool_call>{"name":"read_file","arguments":{"path":"hidden.py"}}</tool_call>'
        payload = json.dumps({'name': 'unknown', 'arguments': {'content': '```json\n' + nested + '\n```'}})
        for text in (payload, '<tool_call>' + payload + '</tool_call>',
                     '```json\n' + payload + '\n```'):
            with self.subTest(text=text):
                self.assertEqual(m.smart_extract_tool_calls(text, '', TOOLS), ([], text))

    def test_malformed_xml_does_not_swallow_next_invocation(self):
        invalids = ('<function=unknown>oops', '<invoke name="unknown">oops',
                    '<function=read_file>oops', '<tool_call><function=unknown>oops')
        valids = ('<function=read_file>{"path":"xml.py"}</function>',
                  '<invoke name="read_file">{"path":"xml.py"}</invoke>',
                  '<function=read_file>{"path":"xml.py"}',
                  '<tool_call>{"name":"read_file","arguments":{"path":"xml.py"}}</tool_call>',
                  '```json\n{"name":"read_file","arguments":{"path":"xml.py"}}\n```')
        for invalid in invalids:
            for valid in valids:
                with self.subTest(invalid=invalid, valid=valid):
                    calls, clean = m.smart_extract_tool_calls(invalid + '\n' + valid, '', TOOLS)
                    self.assertEqual(calls, [{'name': 'read_file', 'arguments': {'path': 'xml.py'}}])
                    self.assertEqual(clean, invalid)

    def test_malformed_xml_alone_is_not_an_invocation(self):
        for text in ('<function=unknown>oops', '<invoke name="unknown">oops',
                     '<function=read_file>oops', '<tool_call><function=unknown>oops'):
            with self.subTest(text=text):
                self.assertEqual(m.smart_extract_tool_calls(text, '', TOOLS), ([], text))

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

class RealPoolLifecycleTests(unittest.IsolatedAsyncioTestCase):
    """Integration coverage must not mock the account-release ownership boundary."""
    client = LifecycleTests.client
    collect = LifecycleTests.collect

    def setUp(self):
        LifecycleTests.setUp(self)
        from mimoly_reliability import AccountPool, TTLCache
        self.pool = AccountPool([{'id': 'offline', 'cookies': {}}])
        self.account = self.pool.accounts[0]
        self.stack.enter_context(patch.object(m, '_ACCOUNT_POOL', self.pool))
        self.stack.enter_context(patch.object(m, 'acquire_account', REAL_ACQUIRE_ACCOUNT))
        self.stack.enter_context(patch.object(m, 'release_account', REAL_RELEASE_ACCOUNT))
        self.stack.enter_context(patch.object(m, '_RESPONSE_CACHE', TTLCache(maxsize=16, ttl=60)))

    async def test_real_release_success_and_cache_hit(self):
        client = self.client([['42']])
        before = m._ADMISSION.in_flight
        first = await m.handle_chat_completion(self.body, request())
        lock = self.account.get_lock()
        await lock.acquire()  # cache hit must neither wait for nor release another owner
        try:
            second = await asyncio.wait_for(m.handle_chat_completion(self.body, request()), 0.2)
            self.assertTrue(lock.locked())
        finally:
            lock.release()
        self.assertEqual(first.status_code, 200)
        self.assertEqual(json.loads(first.body)['choices'][0]['message']['content'], '42')
        self.assertEqual(json.loads(first.body), json.loads(second.body))
        self.assertEqual(len(client.queries), 1)
        self.assertEqual(self.account.in_flight, 0)
        self.assertEqual(self.account.successes, 1)
        self.assertFalse(self.account.get_lock().locked())
        self.assertEqual(m._ADMISSION.in_flight, before)

    async def test_real_release_http_early_return(self):
        client = self.client([[]])
        original_stream = client.stream
        def stream(*args, **kwargs):
            response = original_stream(*args, **kwargs)
            response.status_code = 400
            async def aread():
                return b'offline bad request'
            response.aread = aread
            return response
        client.stream = stream
        result = await m.handle_chat_completion(self.body, request())
        self.assertEqual(result.status_code, 400)
        self.assertEqual(self.account.in_flight, 0)
        self.assertFalse(self.account.get_lock().locked())

    async def test_real_release_queued_timeout_keeps_owner_lock(self):
        lock = self.account.get_lock()
        await lock.acquire()
        self.addCleanup(lambda: lock.release() if lock.locked() else None)
        self.stack.enter_context(patch.object(m, '_request_budget_seconds', return_value=0.02))
        result = await m.handle_chat_completion(self.body, request())
        self.assertEqual(result.status_code, 504)
        self.assertTrue(lock.locked())
        self.assertEqual(self.account.in_flight, 0)
        self.upstream.assert_not_called()

    async def test_real_release_queued_cancel_keeps_owner_lock(self):
        lock = self.account.get_lock()
        await lock.acquire()
        self.addCleanup(lambda: lock.release() if lock.locked() else None)
        entered = asyncio.Event()
        original = m._RequestLifecycle.acquire_lock
        async def acquire(lifecycle, candidate):
            entered.set()
            await original(lifecycle, candidate)
        with patch.object(m._RequestLifecycle, 'acquire_lock', acquire):
            task = asyncio.create_task(m.handle_chat_completion(self.body, request()))
            await asyncio.wait_for(entered.wait(), 0.2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(lock.locked())
        self.assertEqual(self.account.in_flight, 0)

    async def test_real_release_does_not_unlock_owned_lock_before_lifecycle(self):
        owner = m._RequestLifecycle(1)
        _, account = owner.acquire()
        lock = account.get_lock()
        await owner.acquire_lock(lock)
        owner.release(account, success=True)
        self.assertTrue(lock.locked(), 'account bookkeeping cannot release lifecycle lock')
        owner.close()
        await lock.acquire()  # another request now owns it
        try:
            owner.close()
            self.assertTrue(lock.locked(), 'repeated close cannot release next owner')
            self.assertEqual(account.successes, 1)
        finally:
            lock.release()

    async def test_real_release_close_twice_is_idempotent(self):
        lifecycle = m._RequestLifecycle(1)
        _, account = lifecycle.acquire()
        await lifecycle.acquire_lock(account.get_lock())
        with patch('builtins.print') as log:
            lifecycle.close()
            lifecycle.close()
        self.assertEqual(account.in_flight, 0)
        self.assertEqual(account.successes, 1)
        self.assertEqual(log.call_count, 1)

    async def test_real_release_stream_normal_close(self):
        self.body['stream'] = True
        self.client([['42']])
        output = await self.collect(await m.handle_chat_completion(self.body, request()))
        self.assertIn('42', output)
        self.assertEqual(self.account.in_flight, 0)
        self.assertFalse(self.account.get_lock().locked())


class NonstreamFrameProbeTests(unittest.IsolatedAsyncioTestCase):
    """Deterministic wire-shape probes, not assertions about unseen live frames."""
    collect = LifecycleTests.collect

    def setUp(self):
        LifecycleTests.setUp(self)
        self.body['thinking'] = {'type': 'disabled'}
        self.body['tools'] = TOOLS
        self.body['messages'] = [{'role': 'user', 'content': 'Read a.py'}]
        from mimoly_reliability import TTLCache
        self.stack.enter_context(patch.object(m, '_RESPONSE_CACHE', TTLCache(maxsize=16, ttl=60)))
        self.stack.enter_context(patch.object(m, '_request_budget_seconds', return_value=0.04))

    def wire_client(self, script):
        class WireResponse(Response):
            async def aiter_lines(self):
                for frame in self.script:
                    if isinstance(frame, float):
                        await asyncio.sleep(frame)
                    elif isinstance(frame, dict):
                        yield 'data: ' + json.dumps(frame)
                    else:
                        yield frame
        client = Client([])
        client.payloads = []
        def stream(*args, **kwargs):
            client.payloads.append(copy.deepcopy(kwargs['json']))
            client.queries.append(kwargs['json']['query'])
            response = WireResponse(script)
            client.responses.append(response)
            return response
        client.stream = stream
        self.stack.enter_context(patch.object(m, 'get_shared_client', return_value=client))
        return client

    async def test_standard_text_tool_call_with_done_is_accepted(self):
        invocation = '<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>'
        client = self.wire_client([{'type': 'text', 'content': invocation}, 'data: [DONE]'])
        result = await m.handle_chat_completion(self.body, request())
        data = json.loads(result.body)
        self.assertEqual(data['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(data['choices'][0]['message']['tool_calls'][0]['function']['name'], 'read_file')
        self.assertFalse(client.payloads[0]['modelConfig']['enableThinking'])
        self.assertTrue(client.responses[0].closed)

    async def test_native_tool_call_with_eof_is_accepted(self):
        self.wire_client([{'toolCalls': [{'name': 'read_file', 'arguments': '{"path":"a.py"}'}]}])
        result = await m.handle_chat_completion(self.body, request())
        self.assertEqual(json.loads(result.body)['choices'][0]['finish_reason'], 'tool_calls')

    async def test_normal_answer_with_done_is_not_a_reasoning_stall(self):
        self.wire_client([{'type': 'text', 'content': '42'}, 'data: [DONE]'])
        result = await m.handle_chat_completion(self.body, request())
        self.assertEqual(json.loads(result.body)['choices'][0]['message']['content'], '42')

    async def test_tool_frame_then_stalled_read_expires_without_returning_tool(self):
        client = self.wire_client([{'toolCalls': [{'name': 'read_file', 'arguments': '{"path":"a.py"}'}]}, 1.0])
        result = await asyncio.wait_for(m.handle_chat_completion(self.body, request()), 0.2)
        self.assertEqual(result.status_code, 504)
        self.assertEqual(len(client.queries), 1)
        self.assertTrue(client.responses[0].closed)
        self.assertFalse(self.account.get_lock().locked())

    async def test_error_frames_abort_before_stall_without_fabricated_success(self):
        for streaming in (False, True):
            for frame in ({'error': {'message': 'PRIVATE_SENTINEL'}},
                          {'error': {}}, {'type': 'error', 'message': 'PRIVATE_SENTINEL'}):
                with self.subTest(stream=streaming, frame=frame):
                    self.body['stream'] = streaming
                    client = self.wire_client([frame, 1.0])
                    result = await asyncio.wait_for(m.handle_chat_completion(self.body, request()), 0.2)
                    output = await asyncio.wait_for(self.collect(result), 0.2)
                    if not streaming:
                        self.assertEqual(result.status_code, 502)
                    self.assertIn('upstream_error', output)
                    self.assertIn('gateway_error', output)
                    self.assertNotIn('PRIVATE_SENTINEL', output)
                    self.assertNotIn('tool_calls', output)
                    self.assertNotIn('deadline exceeded', output)
                    self.assertEqual(len(client.queries), 1)
                    self.assertTrue(client.responses[0].closed)
                    self.assertFalse(self.account.get_lock().locked())

    async def test_named_error_event_aborts_and_status_classifies(self):
        for streaming in (False, True):
            with self.subTest(stream=streaming):
                self.body['stream'] = streaming
                client = self.wire_client(['event: error', {'message': 'PRIVATE_SENTINEL', 'status': 400}, 1.0])
                result = await m.handle_chat_completion(self.body, request())
                output = await self.collect(result)
                if not streaming:
                    self.assertEqual(result.status_code, 400)
                self.assertIn('client_error', output)
                self.assertNotIn('PRIVATE_SENTINEL', output)
                self.assertEqual(len(client.queries), 1)

    async def test_explicit_retryable_status_is_terminal_without_verified_replay_contract(self):
        for streaming in (False, True):
            with self.subTest(stream=streaming):
                self.body['stream'] = streaming
                client = self.wire_client([{'error': {'status': 503, 'message': 'PRIVATE_SENTINEL'}}, 1.0])
                result = await m.handle_chat_completion(self.body, request())
                output = await self.collect(result)
                if not streaming:
                    self.assertEqual(result.status_code, 503)
                self.assertIn('server_error', output)
                self.assertNotIn('PRIVATE_SENTINEL', output)
                self.assertEqual(len(client.queries), 1)
                self.assertTrue(client.responses[0].closed)

    async def test_error_after_visible_answer_does_not_retry(self):
        self.body.pop('tools')
        self.body['stream'] = True
        client = self.wire_client([{'type': 'text', 'content': 'VISIBLE_PREFIX'}, '',
                                  {'error': {'status': 503}}, 1.0])
        output = await self.collect(await m.handle_chat_completion(self.body, request()))
        self.assertIn('VISIBLE_PREFIX', output)
        self.assertIn('upstream_error', output)
        self.assertNotIn('"finish_reason": "stop"', output)
        self.assertEqual(len(client.queries), 1)
        self.assertTrue(client.responses[0].closed)

    async def test_native_tool_then_stall_is_bounded_in_both_modes(self):
        for streaming in (False, True):
            with self.subTest(stream=streaming):
                self.body['stream'] = streaming
                client = self.wire_client([{'toolCalls': [{'name': 'read_file', 'arguments': '{"path":"a.py"}'}]}, 1.0])
                result = await m.handle_chat_completion(self.body, request())
                output = await self.collect(result)
                self.assertIn('deadline exceeded', output)
                self.assertNotIn('tool_calls', output)
                self.assertTrue(client.responses[0].closed)
                self.assertFalse(self.account.get_lock().locked())

    async def test_captured_terminal_call_followed_by_upstream_error_is_returned_not_executed(self):
        # Contract change 2026-10-16 (XiaomiMiMo/MiMo#103): a fully-buffered
        # valid tool call survives a trailing busy error frame. The proxy
        # RETURNS the call for the client to execute -- it never executes
        # inline (no stop answer, no execution output) and never 502s.
        self.body['tools'] = [{'type': 'function', 'function': {
            'name': 'terminal', 'parameters': {'type': 'object',
            'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}]
        for streaming in (False, True):
            with self.subTest(stream=streaming):
                self.body['stream'] = streaming
                client = self.wire_client(['event: message', {'toolCalls': [{
                    'id': 'call_synthetic', 'type': 'function', 'name': 'terminal',
                    'arguments': '{"command": "python -c \\"print(19+23)\\""}{}'}]}, '',
                    'event: error', {'type': 'text', 'content': '服务器繁忙，请稍后再试'}, ''])
                result = await m.handle_chat_completion(self.body, request())
                output = await self.collect(result)
                if not streaming:
                    self.assertEqual(result.status_code, 200)
                    data = json.loads(result.body)
                    self.assertEqual(data['choices'][0]['finish_reason'], 'tool_calls')
                    msg = data['choices'][0]['message']
                    self.assertIsNone(msg['content'])
                    fn = msg['tool_calls'][0]['function']
                    self.assertEqual(fn['name'], 'terminal')
                    self.assertEqual(json.loads(fn['arguments']),
                                     {'command': 'python -c "print(19+23)"'})
                else:
                    self.assertIn('tool_calls', output)
                    self.assertNotIn('upstream_error', output)
                self.assertNotIn('服务器繁忙', output)
                self.assertEqual(len(client.queries), 1)
                self.assertTrue(client.responses[0].closed)

    async def test_error_after_buffered_valid_tool_returns_tool(self):
        # Contract change 2026-10-16: a valid buffered tool call is
        # preserved across a trailing error frame, not discarded.
        for streaming in (False, True):
            with self.subTest(stream=streaming):
                self.body['stream'] = streaming
                client = self.wire_client([{'toolCalls': [{'name': 'read_file', 'arguments': '{"path":"a.py"}'}]},
                                  {'error': 'PRIVATE_SENTINEL'}, 1.0])
                result = await m.handle_chat_completion(self.body, request())
                output = await self.collect(result)
                self.assertIn('tool_calls', output)
                self.assertNotIn('upstream_error', output)
                self.assertNotIn('PRIVATE_SENTINEL', output)
                self.assertEqual(len(client.queries), 1)

    async def test_native_tool_then_busy_error_returns_tool_in_both_modes(self):
        # Live 2026-10-16 (XiaomiMiMo/MiMo#103): a single toolCalls frame whose
        # arguments carry a trailing '{}' suffix, followed by event:error with
        # 服务器繁忙. A fully-buffered valid tool call survives the trailing
        # error frame: tool_calls emitted, no 502, no error-text leak, 1 attempt.
        self.body['tools'] = [{'type': 'function', 'function': {
            'name': 'terminal', 'parameters': {'type': 'object',
            'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}]
        script = ['event: message',
                  {'messageType': 'tool', 'toolCalls': [{
                      'name': 'terminal',
                      'arguments': '{"command": "python -c \\"print(19+23)\\""}{}'}]},
                  'event: error',
                  {'type': 'text', 'content': '服务器繁忙，请稍后再试'}, '']
        for streaming in (False, True):
            with self.subTest(stream=streaming):
                self.body['stream'] = streaming
                client = self.wire_client(script)
                result = await m.handle_chat_completion(self.body, request())
                output = await self.collect(result)
                if not streaming:
                    self.assertEqual(result.status_code, 200)
                    data = json.loads(result.body)
                    self.assertEqual(data['choices'][0]['finish_reason'], 'tool_calls')
                    fn = data['choices'][0]['message']['tool_calls'][0]['function']
                    self.assertEqual(fn['name'], 'terminal')
                    self.assertEqual(json.loads(fn['arguments']),
                                     {'command': 'python -c "print(19+23)"'})
                else:
                    self.assertIn('tool_calls', output)
                    self.assertIn('terminal', output)
                    self.assertIn('print(19+23)', output)
                    self.assertNotIn('upstream_error', output)
                    self.assertNotIn('Stream connection error', output)
                self.assertNotIn('服务器繁忙', output)
                self.assertEqual(len(client.queries), 1)
                self.assertTrue(client.responses[0].closed)

    async def test_upstream_refusal_text_is_surfaced_not_retried_to_exhaustion(self):
        # Live 2026-10-16: plain conversational prompt (no tools) got the
        # upstream refusal "Hello, I cannot answer this question at the
        # moment. Let's talk about something else instead." Current code
        # treats those phrases as busy (UPSTREAM_BUSY_PHRASES) and retries 4x
        # with growing sleeps (~25s of dead waiting) before 502. A refusal is
        # a FINAL answer, not a capacity signal: surface it immediately,
        # single attempt.
        self.body.pop('tools', None)
        refusal = ("Hello, I cannot answer this question at the moment. "
                   "Let's talk about something else instead.")
        for streaming in (False, True):
            with self.subTest(stream=streaming):
                self.body['stream'] = streaming
                self.body['messages'] = [{'role': 'user', 'content': 'PROMPT_SENTINEL'}]
                client = self.wire_client([{'type': 'text', 'content': refusal},
                                           'data: [DONE]'])
                result = await m.handle_chat_completion(self.body, request())
                output = await self.collect(result)
                if not streaming:
                    self.assertEqual(result.status_code, 200)
                self.assertNotIn('upstream_error', output)
                self.assertNotIn('gateway_error', output)
                self.assertIn("cannot answer this question", output)
                self.assertEqual(len(client.queries), 1)
                self.assertTrue(client.responses[0].closed)

    async def test_done_substring_in_json_is_not_a_terminator(self):
        self.body.pop('tools')
        for streaming in (False, True):
            with self.subTest(stream=streaming):
                self.body['stream'] = streaming
                self.wire_client([{'type': 'text', 'content': 'literal [DONE] text'}, 'data: [DONE]'])
                output = await self.collect(await m.handle_chat_completion(self.body, request()))
                self.assertIn('literal [DONE] text', output)

    async def test_captured_untyped_numeric_message_survives_in_both_modes(self):
        # Live 2026-10-16: event: message, keys content/messageType, no type.
        # The dialog ID arrived in its own event; numeric answer was exactly 42.
        self.body.pop('tools')
        self.body['messages'] = [{'role': 'user', 'content': 'Berapa 19 + 23? Jawab angkanya saja, tanpa tool.'}]
        for streaming in (False, True):
            for answer in ('42', '123456', '0'):
                with self.subTest(stream=streaming, answer=answer):
                    self.body['messages'][0]['content'] = f'Jawab {answer} saja, tanpa tool.'
                    self.body['stream'] = streaming
                    self.wire_client(['event: dialogId', {'content': '987654'}, '',
                                      'event: message', {'reasoningContent': 'Synthetic reasoning'}, '',
                                      'event: message', {'content': answer}, '',
                                      'event: finish', {'content': '[DONE]'}, ''])
                    result = await m.handle_chat_completion(self.body, request())
                    output = await self.collect(result)
                    if streaming:
                        chunks = [json.loads(line[5:]) for line in output.splitlines() if line.startswith('data:') and line[5:].strip() != '[DONE]']
                        text = ''.join(c['choices'][0]['delta'].get('content', '') or '' for c in chunks if 'choices' in c)
                    else:
                        text = json.loads(output)['choices'][0]['message']['content']
                    self.assertEqual(text, answer)
                    self.assertNotIn('987654', output)
                    self.assertNotIn('Synthetic reasoning', output)

    async def test_dialog_id_event_resets_at_blank_boundary(self):
        self.body.pop('tools')
        for streaming in (False, True):
            with self.subTest(stream=streaming):
                self.body['stream'] = streaming
                client = self.wire_client(['event: dialogId', {'content': '12345'}, '',
                                          ': comment', '',
                                          {'type': 'text', 'content': 'BOUNDARY_ANSWER'}, 'data: [DONE]'])
                output = await self.collect(await m.handle_chat_completion(self.body, request()))
                self.assertIn('BOUNDARY_ANSWER', output)
                self.assertNotIn('12345', output)
                self.assertEqual(len(client.queries), 1)
                self.assertTrue(client.responses[0].closed)
                self.assertFalse(self.account.get_lock().locked())

    async def test_dialog_id_event_still_applies_until_blank_boundary(self):
        self.body.pop('tools')
        for streaming in (False, True):
            with self.subTest(stream=streaming):
                self.body['stream'] = streaming
                self.wire_client(['event: dialogId', {'type': 'text', 'content': 'HIDDEN_ID'},
                                  ': comment', {'type': 'text', 'content': 'ALSO_HIDDEN'}, '',
                                  {'type': 'text', 'content': 'VISIBLE'}, 'data: [DONE]'])
                output = await self.collect(await m.handle_chat_completion(self.body, request()))
                self.assertIn('VISIBLE', output)
                self.assertNotIn('HIDDEN', output)


class RequestTimeoutPolicyTests(unittest.IsolatedAsyncioTestCase):
    setUp = LifecycleTests.setUp
    client = LifecycleTests.client
    collect = LifecycleTests.collect

    def short_retry_window(self):
        def budget(**kwargs):
            kwargs['deadline'] = 0.005
            return RetryBudget(**kwargs)
        self.stack.enter_context(patch.object(m, 'RetryBudget', side_effect=budget))

    def test_positive_finite_env_and_default(self):
        with patch.dict(m.os.environ, {}, clear=True):
            for model in ('mimo-v2.6-pro', 'mimo-v2.5-pro-ultraspeed'):
                self.assertEqual(m._request_budget_seconds(model), 180.0)
        for raw in ('0', '-1', 'nan', 'NaN', 'inf', '-inf', 'Infinity', '', 'garbage'):
            with self.subTest(raw=raw), patch.dict(m.os.environ, {'MIMOLY_REQUEST_TIMEOUT_SECONDS': raw}):
                with self.assertRaisesRegex(ValueError, 'MIMOLY_REQUEST_TIMEOUT_SECONDS'):
                    m._request_budget_seconds('mimo-v2.6-pro')
        with patch.dict(m.os.environ, {'MIMOLY_REQUEST_TIMEOUT_SECONDS': '12.5'}):
            self.assertEqual(m._request_budget_seconds('mimo-v2.6-pro'), 12.5)

    async def test_invalid_env_is_explicit_server_configuration_error(self):
        before = m._ADMISSION.in_flight
        with patch.dict(m.os.environ, {'MIMOLY_REQUEST_TIMEOUT_SECONDS': 'nan'}):
            result = await m.handle_chat_completion(self.body, request())
        self.assertEqual(result.status_code, 500)
        self.assertIn('configuration_error', result.body.decode())
        self.upstream.assert_not_called()
        self.assertEqual(m._ADMISSION.in_flight, before)

    async def test_successful_inference_can_outlive_retry_window(self):
        self.short_retry_window()
        for streaming in (False, True):
            with self.subTest(stream=streaming):
                self.body['stream'] = streaming
                client = self.client([[0.02, 'LONG_SUCCESS']])
                with patch.dict(m.os.environ, {'MIMOLY_REQUEST_TIMEOUT_SECONDS': '0.2'}):
                    result = await m.handle_chat_completion(self.body, request())
                    output = await self.collect(result)
                self.assertIn('LONG_SUCCESS', output)
                self.assertNotIn('error', output)
                self.assertEqual(len(client.queries), 1)
                self.assertTrue(client.responses[0].closed)

    async def test_first_attempt_allowed_after_queue_exceeds_retry_window(self):
        self.short_retry_window()
        for streaming in (False, True):
            with self.subTest(stream=streaming):
                self.body['stream'] = streaming
                lock = self.account.get_lock()
                await lock.acquire()
                client = self.client([['QUEUED_SUCCESS']])
                async def unlock():
                    await asyncio.sleep(0.02)
                    lock.release()
                unlocker = asyncio.create_task(unlock())
                try:
                    with patch.dict(m.os.environ, {'MIMOLY_REQUEST_TIMEOUT_SECONDS': '0.2'}):
                        output = await self.collect(await m.handle_chat_completion(self.body, request()))
                    self.assertIn('QUEUED_SUCCESS', output)
                    self.assertEqual(len(client.queries), 1)
                finally:
                    await unlocker

    async def test_retry_window_exhaustion_is_explicit_for_busy_and_timeout(self):
        self.short_retry_window()
        for streaming in (False, True):
            for failure in ('服务器繁忙', httpx.ReadTimeout('offline timeout')):
                with self.subTest(stream=streaming, failure=str(failure)):
                    self.body['stream'] = streaming
                    client = self.client([[0.02, failure]])
                    with patch.dict(m.os.environ, {'MIMOLY_REQUEST_TIMEOUT_SECONDS': '0.2'}):
                        result = await m.handle_chat_completion(self.body, request())
                        output = await self.collect(result)
                    if not streaming:
                        self.assertIn(result.status_code, (502, 503, 504))
                    self.assertIn('error', output)
                    self.assertNotIn('"finish_reason": "stop"', output)
                    self.assertEqual(len(client.queries), 1)
                    self.assertTrue(client.responses[0].closed)
                    self.assertFalse(self.account.get_lock().locked())

    async def test_cancel_configured_request_releases_owned_resources(self):
        for streaming in (False, True):
            with self.subTest(stream=streaming):
                self.body['stream'] = streaming
                client = self.client([[1.0, 'NEVER_RETURN']])
                with patch.dict(m.os.environ, {'MIMOLY_REQUEST_TIMEOUT_SECONDS': '0.2'}):
                    async def consume():
                        return await self.collect(await m.handle_chat_completion(self.body, request()))
                    task = asyncio.create_task(consume())
                    await asyncio.sleep(0.01)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                self.assertTrue(client.responses[0].closed)
                self.assertFalse(self.account.get_lock().locked())
                self.assertEqual(m._ADMISSION.in_flight, 0)

    async def test_total_env_deadline_still_bounds_read_and_cancellation(self):
        for streaming in (False, True):
            with self.subTest(stream=streaming):
                self.body['stream'] = streaming
                client = self.client([[1.0, 'NEVER_RETURN']])
                with patch.dict(m.os.environ, {'MIMOLY_REQUEST_TIMEOUT_SECONDS': '0.02'}):
                    result = await m.handle_chat_completion(self.body, request())
                    output = await asyncio.wait_for(self.collect(result), 0.2)
                if not streaming:
                    self.assertEqual(result.status_code, 504)
                self.assertIn('deadline exceeded', output)
                self.assertNotIn('NEVER_RETURN', output)
                self.assertTrue(client.responses[0].closed)
                self.assertFalse(self.account.get_lock().locked())


if __name__ == '__main__':
    unittest.main(verbosity=2)
