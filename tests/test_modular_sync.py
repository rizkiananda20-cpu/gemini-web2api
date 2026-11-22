import http.client
import base64
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
from urllib.parse import parse_qs

from gemini_web2api.config import CONFIG, DEFAULT_CONFIG
from gemini_web2api.gemini import _build_headers, _build_payload
from gemini_web2api.models import MODELS, build_model_ticket, resolve_model
from gemini_web2api.server import GeminiHandler, ThreadedServer
from gemini_web2api.tools import google_contents_to_prompt, messages_to_prompt, parse_tool_calls

_SINGLE_SPEC = importlib.util.spec_from_file_location(
    "gemini_web2api_single",
    Path(__file__).resolve().parent.parent / "gemini_web2api.py",
)
single_file = importlib.util.module_from_spec(_SINGLE_SPEC)
_SINGLE_SPEC.loader.exec_module(single_file)


def _decode_payload(payload):
    outer = json.loads(parse_qs(payload)["f.req"][0])
    return json.loads(outer[1])


def _decode_sse(body):
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        event_type = next(
            (line[len("event: "):] for line in lines if line.startswith("event: ")),
            None,
        )
        data = next(
            (line[len("data: "):] for line in lines if line.startswith("data: ")),
            None,
        )
        if event_type and data:
            events.append((event_type, json.loads(data)))
    return events


class PayloadPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(CONFIG)

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def test_temporary_chats_default_to_disabled(self):
        self.assertIs(DEFAULT_CONFIG["temporary_chats"], False)

    def test_persistent_chat_payload(self):
        CONFIG["temporary_chats"] = False

        inner = _decode_payload(_build_payload("hello", 1, 4))

        self.assertEqual(inner[41], [2])
        self.assertIsNone(inner[45])

    def test_temporary_chat_payload(self):
        CONFIG["temporary_chats"] = True

        inner = _decode_payload(_build_payload("hello", 1, 4))

        self.assertEqual(inner[41], [1])
        self.assertEqual(inner[45], 1)

    def test_payload_includes_uploaded_image_refs(self):
        inner = _decode_payload(_build_payload("describe", 1, 4, ["/uploaded/image-ref"]))

        self.assertEqual(inner[0][0], "describe")
        self.assertEqual(inner[0][3], [[None, None, "/uploaded/image-ref"]])

    def test_ticketed_payload_sets_turn_zero_and_thinking_wire(self):
        _, mode, think, _, extra = resolve_model("gemini-4-neon-high")
        inner = _decode_payload(_build_payload("solve", mode, think, None, extra))

        self.assertEqual(inner[17], [[0]])
        self.assertEqual(inner[41], [1])
        self.assertEqual(inner[79], 3)
        self.assertEqual(inner[80], 2)
        self.assertEqual(inner[96], 1)

    def test_mode_8_carbon_payload_populates_required_slots(self):
        _, mode, think, _, extra = resolve_model("gemini-4-carbon")
        inner = _decode_payload(_build_payload("solve", mode, think, None, extra))

        self.assertEqual(inner[17], [[0]])
        self.assertEqual(inner[49], 40)
        self.assertEqual(inner[68], 2)
        self.assertEqual(inner[79], 8)
        self.assertEqual(inner[83], 1)
        self.assertEqual(inner[91], 1)
        self.assertEqual(inner[96], 1)
        self.assertEqual(inner[98], 1)


class MessageParsingTests(unittest.TestCase):
    def test_messages_to_prompt_extracts_openai_image_url_data_url(self):
        image_data = base64.b64encode(b"fake png").decode()

        prompt, images = messages_to_prompt([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
            ],
        }])

        self.assertEqual(prompt, "Describe [Image attached]")
        self.assertEqual(images, [(b"fake png", "image/png")])

    def test_messages_to_prompt_extracts_responses_input_image_url(self):
        prompt, images = messages_to_prompt([{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe"},
                {"type": "input_image", "image_url": "https://example.com/image.png"},
            ],
        }])

        self.assertEqual(prompt, "Describe [Image attached]")
        self.assertEqual(images, [("https://example.com/image.png", "image/png")])

    def test_messages_to_prompt_ignores_malformed_image_data_url(self):
        prompt, images = messages_to_prompt([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%"}},
            ],
        }])

        self.assertEqual(prompt, "Describe")
        self.assertEqual(images, [])

    def test_google_contents_to_prompt_extracts_inline_image_data(self):
        image_data = base64.b64encode(b"fake png").decode()

        prompt, images = google_contents_to_prompt({
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "Describe"},
                    {"inlineData": {"mimeType": "image/png", "data": image_data}},
                ],
            }],
        })

        self.assertEqual(prompt, "Describe\n[Image attached]")
        self.assertEqual(images, [(b"fake png", "image/png")])

    def test_google_contents_to_prompt_ignores_malformed_inline_image_data(self):
        prompt, images = google_contents_to_prompt({
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "Describe"},
                    {"inlineData": {"mimeType": "image/png", "data": "%%%"}},
                ],
            }],
        })

        self.assertEqual(prompt, "Describe")
        self.assertEqual(images, [])


class StreamingEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadedServer(("127.0.0.1", 0), GeminiHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.original_config = dict(CONFIG)
        CONFIG["api_keys"] = []
        CONFIG["log_requests"] = False

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def post_json(self, path, payload, headers=None):
        req_headers = {"Content-Type": "application/json"}
        if headers:
            req_headers.update(headers)
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(
            "POST",
            path,
            body=json.dumps(payload),
            headers=req_headers,
        )
        response = connection.getresponse()
        body = response.read().decode()
        resp_headers = dict(response.getheaders())
        connection.close()
        return response.status, resp_headers, body

    def post_chunked_json(self, path, payload):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(
            "POST",
            path,
            body=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            encode_chunked=True,
        )
        response = connection.getresponse()
        body = response.read().decode()
        headers = dict(response.getheaders())
        connection.close()
        return response.status, headers, body

    @mock.patch("gemini_web2api.server.generate_stream")
    def test_chat_stream_starts_with_assistant_role(self, generate_stream):
        generate_stream.return_value = iter(["hel", "lo"])

        status, headers, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        chunks = [
            json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: {")
        ]
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": "hel"})
        self.assertEqual(chunks[2]["choices"][0]["delta"], {"content": "lo"})
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    @mock.patch("gemini_web2api.server.generate", return_value="chunked ok")
    def test_chat_accepts_chunked_body(self, _generate):
        status, _, body = self.post_chunked_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "chunked ok")

    @mock.patch("gemini_web2api.server.upload_image", return_value="/uploaded/image-ref")
    @mock.patch("gemini_web2api.server.generate", return_value="looks good")
    def test_chat_accepts_openai_image_url_data_url(self, generate, upload_image):
        image_data = base64.b64encode(b"fake png").decode()

        status, _, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            },
                        },
                    ],
                }],
            },
        )

        self.assertEqual(status, 200)
        upload_image.assert_called_once_with(b"fake png", "image.png", "image/png")
        self.assertEqual(generate.call_args.args[3], ["/uploaded/image-ref"])
        self.assertIn("[Image attached]", generate.call_args.args[0])
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "looks good")

    @mock.patch("gemini_web2api.server.fetch_image_bytes", return_value=b"\xff\xd8\xffremote jpeg")
    @mock.patch("gemini_web2api.server.upload_image", return_value="/uploaded/remote-ref")
    @mock.patch("gemini_web2api.server.generate", return_value="remote ok")
    def test_responses_accepts_input_image_url(self, generate, upload_image, fetch_image_bytes):
        status, _, _ = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": [{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is shown?"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/image.jpg",
                        },
                    ],
                }],
            },
        )

        self.assertEqual(status, 200)
        fetch_image_bytes.assert_called_once_with("https://example.com/image.jpg")
        upload_image.assert_called_once_with(b"\xff\xd8\xffremote jpeg", "image.png", "image/jpeg")
        self.assertEqual(generate.call_args.args[3], ["/uploaded/remote-ref"])
        self.assertIn("[Image attached]", generate.call_args.args[0])

    @mock.patch("gemini_web2api.server.upload_image", return_value="/uploaded/image-ref")
    @mock.patch("gemini_web2api.server.generate", return_value="top-level image ok")
    def test_responses_accepts_top_level_input_image(self, generate, upload_image):
        image_data = base64.b64encode(b"fake png").decode()

        status, _, _ = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": [
                    {"type": "input_text", "text": "What is shown?"},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_data}",
                    },
                ],
            },
        )

        self.assertEqual(status, 200)
        upload_image.assert_called_once_with(b"fake png", "image.png", "image/png")
        self.assertEqual(generate.call_args.args[3], ["/uploaded/image-ref"])
        self.assertIn("What is shown?", generate.call_args.args[0])
        self.assertIn("[Image attached]", generate.call_args.args[0])

    @mock.patch("gemini_web2api.server.upload_image", side_effect=RuntimeError("upload denied"))
    def test_google_image_upload_failure_returns_502(self, _upload_image):
        image_data = base64.b64encode(b"fake png").decode()

        status, _, body = self.post_json(
            "/v1beta/models/gemini-3.6-flash:generateContent",
            {
                "contents": [{
                    "role": "user",
                    "parts": [{
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": image_data,
                        },
                    }],
                }],
            },
        )

        self.assertEqual(status, 502)
        self.assertIn("image upload failed: upload denied", json.loads(body)["error"]["message"])

    @mock.patch("gemini_web2api.server.generate_stream", return_value=iter(["streamed"]))
    def test_google_stream_generate_content_uses_sse(self, _generate_stream):
        status, headers, body = self.post_json(
            "/v1beta/models/gemini-3.6-flash:streamGenerateContent",
            {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": "Stream this"}],
                }],
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        self.assertIn('"text": "streamed"', body)

    @mock.patch("gemini_web2api.server.generate", return_value="hello")
    def test_responses_text_stream_has_complete_event_sequence(self, _generate):
        status, headers, body = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": "hello",
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        events = _decode_sse(body)
        self.assertEqual(
            [event_type for event_type, _ in events],
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        self.assertEqual(
            [event["sequence_number"] for _, event in events],
            list(range(1, len(events) + 1)),
        )
        self.assertEqual(events[4][1]["delta"], "hello")
        self.assertEqual(events[-1][1]["response"]["status"], "completed")
        self.assertEqual(events[-1][1]["response"]["output"][0]["content"][0]["text"], "hello")

    @mock.patch("gemini_web2api.server.parse_tool_calls")
    @mock.patch("gemini_web2api.server.generate", return_value="tool output")
    def test_responses_function_call_stream_has_complete_event_sequence(
        self, _generate, parse_tool_calls
    ):
        parse_tool_calls.return_value = (
            "",
            [
                {
                    "id": "call_test",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Shanghai"}'},
                }
            ],
        )

        status, _, body = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": "weather",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    }
                ],
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        events = _decode_sse(body)
        self.assertEqual(
            [event_type for event_type, _ in events],
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        self.assertEqual(
            [event["sequence_number"] for _, event in events],
            list(range(1, len(events) + 1)),
        )
        self.assertEqual(events[2][1]["output_index"], 0)
        self.assertEqual(events[3][1]["delta"], '{"city":"Shanghai"}')
        self.assertEqual(events[4][1]["arguments"], '{"city":"Shanghai"}')
        self.assertEqual(events[-1][1]["response"]["output"][0]["name"], "get_weather")

    @mock.patch("gemini_web2api.server.generate", return_value="argon answer")
    def test_chat_completions_forwards_argon_reasoning_effort_and_ticket(self, generate):
        status, _, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-4-argon",
                "reasoning_effort": "high",
                "messages": [{"role": "user", "content": "solve this"}],
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(generate.call_args.args[1], 8)
        self.assertEqual(generate.call_args.args[2], 0)
        ticket = json.loads(generate.call_args.args[4]["_ticket"])
        self.assertEqual(ticket[4], "95a4c0abcba24364")
        self.assertEqual(ticket[14], 8)
        self.assertEqual(ticket[15], 2)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "argon answer")

    def test_v1_cookies_updates_memory_preserves_metadata_and_blocks_csrf(self):
        from gemini_web2api.gemini import load_cookie, _cookie_cache

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(
                {
                    "cookie": "SID=old; SAPISID=sapi_old",
                    "sapisid": "sapi_old",
                    "xsrf_token": "xsrf_keep",
                    "gemini_bl": "boq_keep_p0",
                    "auth_user": "2",
                    "model_tickets": {"gemini-3.1-pro": "e6fa609c3fa255c0"},
                },
                tf,
            )
            temp_path = tf.name

        sync_headers = {
            "X-Gemini-Cookie-Sync": "1",
            "Origin": "chrome-extension://abcdefghijklmnop",
        }
        try:
            with mock.patch.dict(CONFIG, {"cookie_file": temp_path, "api_keys": []}):
                _cookie_cache.update({"str": "", "sapisid": None, "mtime": 0})

                csrf_status, _, _ = self.post_json(
                    "/v1/cookies",
                    {"cookie": "SID=evil; SAPISID=evil"},
                )
                self.assertEqual(csrf_status, 403)

                web_origin_status, _, _ = self.post_json(
                    "/v1/cookies",
                    {"cookie": "SID=evil; SAPISID=evil"},
                    headers={"X-Gemini-Cookie-Sync": "1", "Origin": "http://localhost:3000"},
                )
                self.assertEqual(web_origin_status, 403)

                rebind_status, _, _ = self.post_json(
                    "/v1/cookies",
                    {"cookie": "SID=evil; SAPISID=evil"},
                    headers={"X-Gemini-Cookie-Sync": "1", "Host": "evil.example:8081"},
                )
                self.assertEqual(rebind_status, 403)

                status, _, body = self.post_json(
                    "/v1/cookies",
                    {"cookie": "SID=new_sid; __Secure-1PSIDTS=ts_new; SAPISID=sapi_new", "auth_user": 1},
                    headers=sync_headers,
                )
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)["status"], "ok")

                cookie_str, sapisid = load_cookie()
                self.assertEqual(cookie_str, "SID=new_sid; __Secure-1PSIDTS=ts_new; SAPISID=sapi_new")
                self.assertEqual(sapisid, "sapi_new")
                self.assertEqual(CONFIG["auth_user"], "1")

                with open(temp_path, "r", encoding="utf-8") as rf:
                    saved = json.load(rf)
                self.assertEqual(saved["cookie"], "SID=new_sid; __Secure-1PSIDTS=ts_new; SAPISID=sapi_new")
                self.assertEqual(saved["sapisid"], "sapi_new")
                self.assertEqual(saved["xsrf_token"], "xsrf_keep")
                self.assertEqual(saved["gemini_bl"], "boq_keep_p0")
                self.assertEqual(saved["auth_user"], "1")
                self.assertEqual(saved["model_tickets"], {"gemini-3.1-pro": "e6fa609c3fa255c0"})

                status2, _, _ = self.post_json(
                    "/v1/cookies",
                    {
                        "cookie": "SID=new_sid; __Secure-1PSIDTS=ts_new; SAPISID=sapi_new",
                        "model_tickets": {
                            "gemini-4-argon": "95A4C0ABCBA24364",
                            "bad key!": "95a4c0abcba24364",
                            "gemini-auto": "not-16-hex",
                        },
                    },
                    headers=sync_headers,
                )
                self.assertEqual(status2, 200)
                self.assertEqual(
                    CONFIG["model_tickets"],
                    {
                        "gemini-3.1-pro": "e6fa609c3fa255c0",
                        "gemini-4-argon": "95a4c0abcba24364",
                    },
                )

                crlf_status, _, _ = self.post_json(
                    "/v1/cookies",
                    {"cookie": "SID=bad\r\nX-Injected: 1"},
                    headers=sync_headers,
                )
                self.assertEqual(crlf_status, 400)
        finally:
            _cookie_cache.update({"str": "", "sapisid": None, "mtime": 0})
            if os.path.exists(temp_path):
                os.remove(temp_path)


class ArgonAndModelRoutingTests(unittest.TestCase):
    def setUp(self):
        from gemini_web2api.gemini import _cookie_cache
        self.original_config = dict(CONFIG)
        self.original_cache = dict(_cookie_cache)
        CONFIG["cookie_file"] = None
        _cookie_cache.update({"str": "", "sapisid": None, "mtime": 0})

    def tearDown(self):
        from gemini_web2api.gemini import _cookie_cache
        CONFIG.clear()
        CONFIG.update(self.original_config)
        _cookie_cache.clear()
        _cookie_cache.update(self.original_cache)

    def test_modular_single_file_and_worker_models_in_sync(self):
        from gemini_web2api.models import REASONING_EFFORT_MAP, THINK_TO_VARIANT
        self.assertEqual(set(MODELS.keys()), set(single_file.MODELS.keys()))
        self.assertEqual(REASONING_EFFORT_MAP, single_file.REASONING_EFFORT_MAP)
        self.assertEqual(THINK_TO_VARIANT, single_file.THINK_TO_VARIANT)
        for name in MODELS:
            self.assertEqual(MODELS[name]["mode"], single_file.MODELS[name]["mode"])
            self.assertEqual(MODELS[name]["think"], single_file.MODELS[name]["think"])
            self.assertEqual(MODELS[name].get("variant"), single_file.MODELS[name].get("variant"))
            self.assertEqual(MODELS[name].get("ticket"), single_file.MODELS[name].get("ticket"))

        worker_path = Path(__file__).resolve().parents[1] / "cloudflare" / "worker.js"
        worker_src = worker_path.read_text(encoding="utf-8")
        for base_key in (
            "gemini-4-hydrogen",
            "gemini-4-neon",
            "gemini-4-carbon",
            "gemini-4-argon",
            "gemini-3.8-flash",
            "gemini-3.1-pro",
            "gemini-flash-lite",
            "gemini-3.5-flash-lite",
        ):
            self.assertIn(f"'{base_key}'", worker_src)

    def test_hydrogen_neon_carbon_argon_variants_and_precedence(self):
        for model_name, expected_mode, expected_ticket in (
            ("gemini-4-hydrogen", 1, "56fdd199312815e2"),
            ("gemini-4-neon", 3, "e6fa609c3fa255c0"),
            ("gemini-4-carbon", 8, "95a4c0abcba24364"),
            ("gemini-4-argon", 8, "95a4c0abcba24364"),
        ):
            name, mode, think, err, extra = resolve_model(model_name, reasoning_effort="low")
            self.assertIsNone(err)
            self.assertEqual((name, mode, think), (model_name, expected_mode, 4))
            self.assertEqual(json.loads(extra["_ticket"])[15], 1)

            name, mode, think, err, extra = resolve_model(model_name, reasoning_effort="medium")
            self.assertIsNone(err)
            self.assertEqual((name, mode, think), (model_name, expected_mode, 2))
            self.assertEqual(json.loads(extra["_ticket"])[15], 4)

            name, mode, think, err, extra = resolve_model(model_name, reasoning_effort="high")
            self.assertIsNone(err)
            self.assertEqual((name, mode, think), (model_name, expected_mode, 0))
            ticket = json.loads(extra["_ticket"])
            self.assertEqual(len(ticket), 17)
            self.assertEqual(ticket[4], expected_ticket)
            self.assertEqual(ticket[14], expected_mode)
            self.assertEqual(ticket[15], 2)
            headers = _build_headers(extra)
            self.assertIn("X-Goog-Ext-525001261-Jspb", headers)

        # Explicit suffix (-low) takes precedence over body reasoning_effort ("high")
        name, mode, think, err, extra = resolve_model("gemini-4-argon-low", reasoning_effort="high")
        self.assertEqual((name, mode, think), ("gemini-4-argon-low", 8, 4))
        self.assertEqual(json.loads(extra["_ticket"])[15], 1)

        # Explicit @think=0 takes precedence over suffix (-low) and reasoning_effort
        name, mode, think, err, extra = resolve_model("gemini-4-argon-low@think=0", reasoning_effort="low")
        self.assertEqual((name, mode, think), ("gemini-4-argon-low", 8, 0))
        self.assertEqual(json.loads(extra["_ticket"])[15], 2)

        # Legacy models without levels ignore reasoning_effort and do not emit _ticket
        name, mode, think, err, extra = resolve_model("gemini-3.5-flash-thinking", reasoning_effort="low")
        self.assertEqual((name, mode, think), ("gemini-3.5-flash-thinking", 2, 0))
        self.assertNotIn("_ticket", extra)

    def test_login_redirect_does_not_overwrite_xsrf_token(self):
        from gemini_web2api.gemini import refresh_bl_and_xsrf
        CONFIG["xsrf_token"] = "valid_gemini_xsrf_token"
        CONFIG["gemini_bl"] = "boq_gemini-web-uiserver_20261119.05_p0"

        calls = []

        def fake_fetch_html(cookie_str="", sapisid=None, timeout=15):
            calls.append(bool(cookie_str))
            if cookie_str:
                return (
                    '{"SNlM0e":"login_page_xsrf","cfb2h":"boq_identityfrontendauthuiserver_20261119.04_p0"}',
                    "https://accounts.google.com/v3/signin/accountchooser",
                )
            return (
                '{"FdrFJe":"-7103","cfb2h":"boq_gemini-web-uiserver_20261122.06_p0"}',
                "https://gemini.google.com/app?hl=en",
            )

        with mock.patch("gemini_web2api.gemini.load_cookie", return_value=("SID=x; SAPISID=y", "y")), \
             mock.patch("gemini_web2api.gemini._fetch_gemini_page_html", side_effect=fake_fetch_html), \
             mock.patch("gemini_web2api.gemini.save_xsrf_token"):
            changed = refresh_bl_and_xsrf()

        self.assertTrue(changed)
        self.assertEqual(calls, [True, False])
        self.assertEqual(CONFIG["xsrf_token"], "valid_gemini_xsrf_token")
        self.assertEqual(CONFIG["gemini_bl"], "boq_gemini-web-uiserver_20261122.06_p0")

    def test_ticket_canned_error_falls_back_without_header_even_when_retry_attempts_is_one(self):
        from gemini_web2api.gemini import generate, _is_canned_error
        seen_headers = []

        def make_frame(msg: str) -> bytes:
            inner = json.dumps([None, None, None, None, [["rc_1", [msg]]]])
            line = json.dumps([["wrb.fr", None, inner + (" " * 220)]]) + "\n"
            return line.encode("utf-8")

        class FakeResp:
            def __init__(self, payload: bytes):
                self._payload = payload

            def read(self):
                return self._payload

        def fake_urlopen(req, context=None, timeout=None):
            seen_headers.append(dict(req.headers))
            if len(seen_headers) == 1:
                return FakeResp(make_frame("I encountered an error doing what you asked. Please try again."))
            return FakeResp(make_frame("fallback model answer"))

        _, mode, think, _, extra = resolve_model("gemini-4-argon")
        with mock.patch.dict(CONFIG, {"retry_attempts": 1}), \
             mock.patch("gemini_web2api.gemini.urllib.request.urlopen", side_effect=fake_urlopen):
            out = generate("hello", mode, think, None, extra)
        self.assertEqual(out, "fallback model answer")
        self.assertEqual(len(seen_headers), 2)
        self.assertTrue(any(k.lower() == "x-goog-ext-525001261-jspb" for k in seen_headers[0]))
        self.assertFalse(any(k.lower() == "x-goog-ext-525001261-jspb" for k in seen_headers[1]))
        self.assertFalse(_is_canned_error("Sorry, something went wrong in your regex:\nuse \\d+ instead."))
        self.assertTrue(_is_canned_error("I encountered an error doing what you asked."))


if __name__ == "__main__":
    unittest.main()
