from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import urllib.error

from agent_core import (
    CapabilityRequest,
    ErrorCode,
    ModelProvider,
    PermissionLevel,
    ResultStatus,
    RiskLevel,
    SideEffect,
)
from agent_sdk import (
    ExtensionManifest,
    load_extension_instance,
    validate_extension_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Import the capability module under its own name so coverage's source=["main"]
# resolves to this file; the SDK loader re-executes the same file under an
# isolated module name at runtime and coverage aggregates both by path.
sys.path.insert(0, str(PROJECT_ROOT))
import main  # noqa: E402,F401  (imported for coverage source resolution)

_ENV_KEYS = (
    "CORAX_LLM_BASE_URL",
    "CORAX_LLM_MODEL",
    "CORAX_LLM_API_KEY",
    "CORAX_LLM_MODALITIES",
    "CORAX_LLM_ENABLE_IMAGE",
    "CORAX_LLM_ENABLE_VIDEO",
)


def request(payload: dict) -> CapabilityRequest:
    return CapabilityRequest(task_id="task-1", session_id="session-1", input=payload)


def _scrub_env() -> None:
    for key in _ENV_KEYS:
        os.environ.pop(key, None)


# --------------------------------------------------------------------------- #
# An in-memory stand-in for ``urllib`` so the real ``_post_chat_completion``
# body runs (request build, json encode/decode, response read) without opening
# a socket -- fast and deterministic.
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class _FakeSSE:
    """Stand-in for an urlopen() streaming response: a context manager that
    yields raw SSE byte lines on iteration."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self):
        return iter(self._lines)

    def __exit__(self, *exc: object) -> bool:
        return False


async def _collect(agen) -> list[str]:
    out: list[str] = []
    async for piece in agen:
        out.append(piece)
    return out


class ManifestTests(unittest.TestCase):
    def test_manifest_is_sdk_valid(self) -> None:
        manifest = ExtensionManifest.load(PROJECT_ROOT)
        result = validate_extension_manifest(manifest, core_version="0.2.0")

        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.warnings, [])
        self.assertEqual(manifest.id, "llm.local")
        self.assertEqual(manifest.name, "LLM Local Connector")
        self.assertEqual(manifest.permission_level, PermissionLevel.CONFIRM)
        self.assertEqual(manifest.risk_level, RiskLevel.MEDIUM)
        self.assertEqual(
            manifest.side_effects,
            (SideEffect.NETWORK_REQUEST, SideEffect.MODEL_INFERENCE),
        )
        self.assertEqual(manifest.entrypoint, "main:LocalModelProvider")
        self.assertEqual(manifest.kind.value, "model_provider")

    def test_manifest_json_identity(self) -> None:
        data = json.loads((PROJECT_ROOT / "extension.json").read_text())

        self.assertEqual(data["id"], "llm.local")
        self.assertEqual(data["name"], "LLM Local Connector")
        self.assertEqual(data["version"], "1.1.4")
        self.assertEqual(data["author"], "Corax")
        self.assertEqual(data["license"], "MIT")
        self.assertEqual(data["kind"], "model_provider")
        self.assertEqual(data["compatibility"]["min_core_version"], "0.2.0")
        self.assertEqual(data["compatibility"]["sdk_version"], "0.2.0")
        self.assertEqual(
            data["security"]["required_scopes"],
            ["model.inference", "network.private"],
        )


class LoaderTests(unittest.TestCase):
    def test_capability_loads_through_sdk_loader(self) -> None:
        manifest = ExtensionManifest.load(PROJECT_ROOT)
        cap = load_extension_instance(
            manifest, PROJECT_ROOT, core_version="0.2.0"
        )

        self.assertIsInstance(cap, ModelProvider)
        self.assertEqual(cap.id, "llm.local")
        self.assertEqual(
            cap.required_scopes,
            {"model.inference", "network.private"},
        )
        self.assertEqual(
            cap.side_effects,
            {SideEffect.NETWORK_REQUEST, SideEffect.MODEL_INFERENCE},
        )


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        _scrub_env()
        manifest = ExtensionManifest.load(PROJECT_ROOT)
        self.cap = load_extension_instance(
            manifest, PROJECT_ROOT, core_version="0.2.0"
        )

    async def asyncTearDown(self) -> None:
        _scrub_env()

    # -- generate (offline via mock_response) --------------------------- #
    async def test_generate_with_mock(self) -> None:
        result = await self.cap.execute(
            request({"prompt": "hello", "system": "be brief", "mock_response": "hi"})
        )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.payload["text"], "hi")
        self.assertEqual(result.payload["enabled_modalities"], ["text"])
        self.assertEqual(result.payload["used_modalities"], ["text"])
        self.assertEqual(result.payload["endpoint"], "mock")

    async def test_generate_with_prebuilt_messages(self) -> None:
        result = await self.cap.execute(
            request(
                {
                    "messages": [{"role": "user", "content": "hey"}],
                    "model": "gemma-4",
                    "mock_response": "ok",
                }
            )
        )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.payload["model"], "gemma-4")

    async def test_messages_with_media_is_rejected(self) -> None:
        os.environ["CORAX_LLM_ENABLE_IMAGE"] = "true"
        result = await self.cap.execute(
            request(
                {
                    "messages": [{"role": "user", "content": "hey"}],
                    "images": ["https://example.com/a.png"],
                    "mock_response": "ok",
                }
            )
        )
        self.assertEqual(result.status, ResultStatus.ERROR)
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)
        self.assertIn("prompt", result.error.message)

    async def test_missing_prompt_and_messages(self) -> None:
        result = await self.cap.execute(request({}))
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)

    async def test_blank_prompt(self) -> None:
        result = await self.cap.execute(request({"prompt": "   "}))
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)

    # -- modality gating ------------------------------------------------- #
    async def test_image_input_disabled_by_default(self) -> None:
        result = await self.cap.execute(
            request({"prompt": "look", "images": ["https://h/a.png"]})
        )
        self.assertEqual(result.status, ResultStatus.POLICY_DENIED)
        self.assertEqual(result.error.code, ErrorCode.POLICY_DENIED)
        self.assertIn("image input is disabled", result.error.message)

    async def test_video_input_disabled_by_default(self) -> None:
        result = await self.cap.execute(
            request({"prompt": "watch", "videos": ["https://h/a.mp4"]})
        )
        self.assertEqual(result.error.code, ErrorCode.POLICY_DENIED)
        self.assertIn("video input is disabled", result.error.message)

    async def test_image_enabled_via_env(self) -> None:
        os.environ["CORAX_LLM_ENABLE_IMAGE"] = "true"
        result = await self.cap.execute(
            request(
                {
                    "prompt": "describe",
                    "images": ["https://h/a.png", "data:image/png;base64,AAA"],
                    "mock_response": "seen",
                }
            )
        )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.payload["used_modalities"], ["text", "image"])
        self.assertIn("image", result.payload["enabled_modalities"])

    async def test_video_enabled_via_env(self) -> None:
        os.environ["CORAX_LLM_ENABLE_VIDEO"] = "true"
        result = await self.cap.execute(
            request(
                {
                    "prompt": "summarize",
                    "videos": ["https://h/a.mp4"],
                    "mock_response": "seen",
                }
            )
        )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.payload["used_modalities"], ["text", "video"])

    # -- media security ------------------------------------------------- #
    async def test_media_scheme_rejected(self) -> None:
        result = await self.cap.execute(
            request({"prompt": "x", "images": ["ftp://host/a.png"]})
        )
        self.assertEqual(result.error.code, ErrorCode.POLICY_DENIED)
        self.assertIn("http(s) or data", result.error.message)

    async def test_media_secret_marker_rejected(self) -> None:
        result = await self.cap.execute(
            request({"prompt": "x", "images": ["https://host/.env"]})
        )
        self.assertEqual(result.error.code, ErrorCode.POLICY_DENIED)
        self.assertIn("secret material", result.error.message)

    async def test_media_non_string_rejected(self) -> None:
        result = await self.cap.execute(request({"prompt": "x", "images": [123]}))
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)

    async def test_media_empty_string_rejected(self) -> None:
        result = await self.cap.execute(request({"prompt": "x", "images": [""]}))
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)

    async def test_too_many_media_items(self) -> None:
        result = await self.cap.execute(
            request(
                {"prompt": "x", "images": [f"https://h/{i}.png" for i in range(17)]}
            )
        )
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)
        self.assertIn("at most", result.error.message)

    # -- schema / dispatch ---------------------------------------------- #
    async def test_schema_type_error(self) -> None:
        result = await self.cap.execute(
            request({"prompt": "x", "temperature": "hot"})
        )
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)

    async def test_out_of_range_max_tokens_rejected(self) -> None:
        for max_tokens in (0, -1, 32769):
            with self.subTest(max_tokens=max_tokens):
                result = await self.cap.execute(
                    request(
                        {
                            "prompt": "x",
                            "max_tokens": max_tokens,
                            "base_url": "http://127.0.0.1:9/v1",
                        }
                    )
                )
                self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)
                self.assertIn("1 to 32768", result.error.message)

    async def test_unsupported_operation(self) -> None:
        result = await self.cap.execute(request({"operation": "frobnicate"}))
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)
        self.assertIn("unsupported operation", result.error.message)

    # -- real network path (urllib stubbed in-memory) ------------------- #
    async def test_generate_against_local_endpoint(self) -> None:
        os.environ["CORAX_LLM_API_KEY"] = "secret-token"
        base_url = "http://192.168.10.5:8000/v1"
        fake = _FakeResponse(
            {
                "model": "qwen3.6",
                "choices": [{"message": {"role": "assistant", "content": "pong"}}],
            }
        )
        with patch("urllib.request.urlopen", return_value=fake) as urlopen:
            result = await self.cap.execute(
                request(
                    {
                        "prompt": "ping",
                        "base_url": base_url,
                        "temperature": 0.2,
                        "max_tokens": 64,
                    }
                )
            )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.payload["text"], "pong")
        self.assertEqual(result.payload["model"], "qwen3.6")
        self.assertEqual(result.payload["endpoint"], base_url)
        # endpoint, headers (incl. the bearer token) are built but never leaked.
        self.assertEqual(urlopen.call_count, 1)
        sent = urlopen.call_args.args[0]
        self.assertEqual(sent.full_url, f"{base_url}/chat/completions")
        self.assertEqual(sent.headers["Authorization"], "Bearer secret-token")
        self.assertEqual(json.loads(sent.data)["max_tokens"], 64)
        self.assertNotIn("secret-token", json.dumps(result.payload))

    async def test_generate_loopback_text_choice(self) -> None:
        fake = MagicMock(return_value={"choices": [{"text": "via-text"}]})
        with patch.object(self.cap, "_post_chat_completion", fake):
            result = await self.cap.execute(
                request({"prompt": "p", "base_url": "http://127.0.0.1:9/v1"})
            )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.payload["text"], "via-text")
        self.assertEqual(result.payload["model"], "google/gemma-4-12B-it")  # default model
        self.assertEqual(fake.call_args.kwargs["payload"]["max_tokens"], 4096)

    async def test_localhost_endpoint_allowed(self) -> None:
        fake = MagicMock(return_value={"choices": []})
        with patch.object(self.cap, "_post_chat_completion", fake):
            result = await self.cap.execute(
                request({"prompt": "p", "base_url": "http://localhost:8000/v1"})
            )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.payload["text"], "")  # empty choices

    async def test_extract_text_first_not_dict(self) -> None:
        fake = MagicMock(return_value={"choices": ["nope"]})
        with patch.object(self.cap, "_post_chat_completion", fake):
            result = await self.cap.execute(
                request({"prompt": "p", "base_url": "http://127.0.0.1:9/v1"})
            )
        self.assertEqual(result.payload["text"], "")

    async def test_extract_text_non_string_content(self) -> None:
        fake = MagicMock(return_value={"choices": [{"message": {"content": 5}}]})
        with patch.object(self.cap, "_post_chat_completion", fake):
            result = await self.cap.execute(
                request({"prompt": "p", "base_url": "http://127.0.0.1:9/v1"})
            )
        self.assertEqual(result.payload["text"], "")

    # -- endpoint security ---------------------------------------------- #
    async def test_endpoint_without_host(self) -> None:
        result = await self.cap.execute(
            request({"prompt": "p", "base_url": "http:///nohost"})
        )
        self.assertEqual(result.error.code, ErrorCode.POLICY_DENIED)
        self.assertIn("no host", result.error.message)

    async def test_endpoint_non_ip_host_rejected(self) -> None:
        result = await self.cap.execute(
            request({"prompt": "p", "base_url": "http://example.com/v1"})
        )
        self.assertEqual(result.error.code, ErrorCode.POLICY_DENIED)

    async def test_endpoint_public_ip_rejected(self) -> None:
        result = await self.cap.execute(
            request({"prompt": "p", "base_url": "http://8.8.8.8/v1"})
        )
        self.assertEqual(result.error.code, ErrorCode.POLICY_DENIED)

    async def test_invalid_timeout(self) -> None:
        result = await self.cap.execute(
            request(
                {"prompt": "p", "base_url": "http://127.0.0.1:9/v1", "timeout_seconds": -1}
            )
        )
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)
        self.assertIn("timeout_seconds", result.error.message)

    # -- network error handling ----------------------------------------- #
    async def test_http_error(self) -> None:
        exc = urllib.error.HTTPError("http://x", 500, "boom", {}, None)
        with patch.object(self.cap, "_post_chat_completion", MagicMock(side_effect=exc)):
            result = await self.cap.execute(
                request({"prompt": "p", "base_url": "http://127.0.0.1:9/v1"})
            )
        self.assertEqual(result.error.code, ErrorCode.CAPABILITY_FAILED)
        self.assertTrue(result.error.retryable)
        self.assertEqual(result.error.details["status"], 500)

    async def test_url_error(self) -> None:
        exc = urllib.error.URLError("down")
        with patch.object(self.cap, "_post_chat_completion", MagicMock(side_effect=exc)):
            result = await self.cap.execute(
                request({"prompt": "p", "base_url": "http://127.0.0.1:9/v1"})
            )
        self.assertEqual(result.error.code, ErrorCode.CAPABILITY_FAILED)
        self.assertIn("request failed", result.error.message)

    async def test_os_error(self) -> None:
        with patch.object(
            self.cap, "_post_chat_completion", MagicMock(side_effect=OSError("io"))
        ):
            result = await self.cap.execute(
                request({"prompt": "p", "base_url": "http://127.0.0.1:9/v1"})
            )
        self.assertEqual(result.error.code, ErrorCode.CAPABILITY_FAILED)

    async def test_json_decode_error(self) -> None:
        exc = json.JSONDecodeError("bad", "", 0)
        with patch.object(self.cap, "_post_chat_completion", MagicMock(side_effect=exc)):
            result = await self.cap.execute(
                request({"prompt": "p", "base_url": "http://127.0.0.1:9/v1"})
            )
        self.assertEqual(result.error.code, ErrorCode.CAPABILITY_FAILED)
        self.assertIn("invalid JSON", result.error.message)

    async def test_no_raw_exceptions_leak(self) -> None:
        with patch.object(
            self.cap,
            "_generate",
            AsyncMock(side_effect=RuntimeError("raw secret failure")),
        ):
            result = await self.cap.execute(request({"prompt": "p"}))
        self.assertEqual(result.error.code, ErrorCode.CAPABILITY_FAILED)
        self.assertNotIn("raw secret failure", result.error.message)

    # -- configure ------------------------------------------------------ #
    async def test_configure_with_modalities(self) -> None:
        result = await self.cap.execute(
            request({"operation": "configure", "modalities": ["text", " Image "]})
        )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.payload["enabled_modalities"], ["text", "image"])
        env = result.payload["env_assignments"]
        self.assertEqual(env["CORAX_LLM_ENABLE_IMAGE"], "true")
        self.assertEqual(env["CORAX_LLM_ENABLE_VIDEO"], "false")
        self.assertEqual(env["CORAX_LLM_MODALITIES"], "text,image")

    async def test_configure_modality_not_string(self) -> None:
        result = await self.cap.execute(
            request({"operation": "configure", "modalities": [5]})
        )
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)

    async def test_configure_unsupported_modality(self) -> None:
        result = await self.cap.execute(
            request({"operation": "configure", "modalities": ["audio"]})
        )
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)
        self.assertIn("unsupported modality", result.error.message)

    async def test_configure_toggles(self) -> None:
        result = await self.cap.execute(
            request(
                {"operation": "configure", "enable_image": True, "enable_video": False}
            )
        )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertIn("image", result.payload["enabled_modalities"])
        self.assertNotIn("video", result.payload["enabled_modalities"])

    async def test_configure_reflects_env(self) -> None:
        os.environ["CORAX_LLM_MODALITIES"] = "text,video,bogus"
        os.environ["CORAX_LLM_ENABLE_IMAGE"] = "false"
        result = await self.cap.execute(request({"operation": "configure"}))
        self.assertEqual(result.payload["enabled_modalities"], ["text", "video"])

    async def test_configure_env_disables_video(self) -> None:
        os.environ["CORAX_LLM_MODALITIES"] = "text,video"
        os.environ["CORAX_LLM_ENABLE_VIDEO"] = "false"
        result = await self.cap.execute(request({"operation": "configure"}))
        self.assertEqual(result.payload["enabled_modalities"], ["text"])

    # -- describe ------------------------------------------------------- #
    async def test_describe(self) -> None:
        os.environ["CORAX_LLM_ENABLE_VIDEO"] = "true"
        result = await self.cap.execute(request({"operation": "describe"}))
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.payload["supported_modalities"], ["text", "image", "video"])
        menu = result.payload["setup_menu"]
        self.assertEqual(len(menu["options"]), 3)
        self.assertEqual(menu["persist_via"], "environment")
        text_option = next(o for o in menu["options"] if o["id"] == "text")
        self.assertTrue(text_option["locked"])

    # -- validate ------------------------------------------------------- #
    async def test_validate_operation(self) -> None:
        result = await self.cap.execute(
            request({"operation": "validate", "prompt": "check"})
        )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertTrue(result.payload["safe"])
        self.assertEqual(result.payload["used_modalities"], ["text"])

    async def test_health_check(self) -> None:
        status = await self.cap.health_check()
        self.assertEqual(status.value, "healthy")

    # -- tool calling --------------------------------------------------- #
    async def test_generate_mock_with_tool_calls(self) -> None:
        tcs = [{"id": "c1", "type": "function", "function": {"name": "list_files", "arguments": "{}"}}]
        result = await self.cap.execute(
            request({"prompt": "list", "mock_response": "", "mock_tool_calls": tcs})
        )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.payload["tool_calls"], tcs)
        self.assertEqual(result.payload["finish_reason"], "tool_calls")

    async def test_generate_mock_without_tools_finish_stop(self) -> None:
        result = await self.cap.execute(request({"prompt": "hi", "mock_response": "yo"}))
        self.assertEqual(result.payload["tool_calls"], [])
        self.assertEqual(result.payload["finish_reason"], "stop")

    async def test_generate_real_returns_tool_calls(self) -> None:
        os.environ["CORAX_LLM_API_KEY"] = "T"
        tc = {"id": "c1", "type": "function", "function": {"name": "list_files", "arguments": "{\"path\":\".\"}"}}
        raw = {"model": "gemma", "choices": [{"message": {"content": None, "tool_calls": [tc]}, "finish_reason": "tool_calls"}]}
        captured = {}

        def fake(*, base_url, api_key, payload, timeout):
            captured["payload"] = payload
            return raw

        with patch.object(self.cap, "_post_chat_completion", fake):
            result = await self.cap.execute(
                request(
                    {
                        "prompt": "list files",
                        "base_url": "http://127.0.0.1:9/v1",
                        "tools": [{"type": "function", "function": {"name": "list_files"}}],
                        "tool_choice": "auto",
                    }
                )
            )
        self.assertEqual(result.payload["tool_calls"], [tc])
        self.assertEqual(result.payload["finish_reason"], "tool_calls")
        self.assertEqual(result.payload["text"], "")  # content was null
        self.assertIn("tools", captured["payload"])
        self.assertEqual(captured["payload"]["tool_choice"], "auto")

    # -- state_key echo (kernel/gateway round-trip) --------------------- #
    async def test_state_key_echoes_payload(self) -> None:
        result = await self.cap.execute(
            request({"prompt": "hi", "mock_response": "yo", "state_key": "out"})
        )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.state_patch, {"out": result.payload})

    async def test_state_key_ignored_on_failure(self) -> None:
        result = await self.cap.execute(request({"prompt": "", "state_key": "out"}))
        self.assertEqual(result.status, ResultStatus.ERROR)
        self.assertEqual(result.state_patch, {})

    async def test_empty_state_key_no_patch(self) -> None:
        result = await self.cap.execute(
            request({"prompt": "hi", "mock_response": "yo", "state_key": ""})
        )
        self.assertEqual(result.state_patch, {})


class StreamHelperTests(unittest.TestCase):
    def test_chunk_text(self) -> None:
        self.assertEqual(main._chunk_text("abcdef", size=2), ["ab", "cd", "ef"])
        self.assertEqual(main._chunk_text(""), [""])

    def test_sse_delta_variants(self) -> None:
        self.assertIsNone(main._sse_delta("event: ping"))            # not a data line
        self.assertIsNone(main._sse_delta("data: [DONE]"))           # end marker
        self.assertIsNone(main._sse_delta("data:   "))               # empty payload
        self.assertIsNone(main._sse_delta("data: {bad json"))        # invalid json
        self.assertEqual(main._sse_delta('data: {"choices":[]}'), "")  # no choices
        self.assertEqual(main._sse_delta('data: {"choices":["x"]}'), "")  # first not dict
        self.assertEqual(main._sse_delta('data: {"choices":[{"delta":{}}]}'), "")  # no content
        self.assertEqual(
            main._sse_delta('data: {"choices":[{"delta":{"content":"hi"}}]}'), "hi"
        )

    def test_extract_tool_calls_variants(self) -> None:
        self.assertEqual(main._extract_tool_calls({}), [])
        self.assertEqual(main._extract_tool_calls({"choices": []}), [])
        self.assertEqual(main._extract_tool_calls({"choices": ["x"]}), [])
        self.assertEqual(main._extract_tool_calls({"choices": [{"message": {}}]}), [])
        tcs = [{"id": "1"}]
        self.assertEqual(
            main._extract_tool_calls({"choices": [{"message": {"tool_calls": tcs}}]}), tcs
        )

    def test_finish_reason_variants(self) -> None:
        self.assertIsNone(main._finish_reason({}))
        self.assertIsNone(main._finish_reason({"choices": [{}]}))
        self.assertEqual(main._finish_reason({"choices": [{"finish_reason": "stop"}]}), "stop")

    def test_iter_sse_content(self) -> None:
        lines = [
            b'data: {"choices":[{"delta":{"content":"He"}}]}',
            b"",  # blank line skipped
            b'data: {"choices":[{"delta":{"content":"llo"}}]}',
            b"data: [DONE]",
        ]
        self.assertEqual(list(main._iter_sse_content(iter(lines))), ["He", "llo"])

    def test_iter_sse_events_with_tool_call_deltas(self) -> None:
        lines = [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"filesystem","arguments":"{\\"operation\\":"}}]}}]}',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"read\\"}"}}]},"finish_reason":"tool_calls"}]}',
            b"data: [DONE]",
        ]
        events = list(main._iter_sse_events(iter(lines)))
        self.assertEqual(events[0]["tool_calls"][0]["function"]["name"], "filesystem")
        self.assertEqual(events[1]["finish_reason"], "tool_calls")

    def test_iter_sse_events_preserves_reasoning_variants(self) -> None:
        lines = [
            b'data: {"choices":[{"delta":{"reasoning":"think"}}]}',
            b'data: {"choices":[{"delta":{"reasoning_content":" more"}}]}',
        ]
        events = list(main._iter_sse_events(iter(lines)))
        self.assertEqual(
            [event["reasoning"] for event in events],
            ["think", " more"],
        )

    def test_iter_sse_events_emits_provider_prompt_context_usage(self) -> None:
        lines = [
            b'data: {"choices":[],"usage":{"prompt_tokens":321,"completion_tokens":7,"total_tokens":400,"prompt_tokens_details":{"cached_tokens":111},"completion_tokens_details":{"reasoning_tokens":5}}}',
        ]
        self.assertEqual(
            list(main._iter_sse_events(iter(lines))),
            [{
                "type": "context",
                "used": 321,
                "unit": "tokens",
                "scope": "prompt",
                "source": "provider",
            }],
        )

    def test_sse_event_derives_prompt_from_total_minus_completion(self) -> None:
        self.assertEqual(
            main._sse_event(
                'data: {"choices":[],"usage":{"completion_tokens":4,"total_tokens":24}}'
            ),
            {
                "type": "context",
                "used": 20,
                "unit": "tokens",
                "scope": "prompt",
                "source": "provider",
            },
        )

    def test_sse_event_rejects_invalid_token_usage(self) -> None:
        self.assertIsNone(
            main._sse_event(
                'data: {"choices":[],"usage":{"prompt_tokens":-1}}'
            )
        )
        self.assertIsNone(
            main._sse_event(
                'data: {"choices":[],"usage":{"completion_tokens":20,"total_tokens":4}}'
            )
        )


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        _scrub_env()
        manifest = ExtensionManifest.load(PROJECT_ROOT)
        self.cap = load_extension_instance(
            manifest, PROJECT_ROOT, core_version="0.2.0"
        )

    async def asyncTearDown(self) -> None:
        _scrub_env()

    async def test_stream_mock(self) -> None:
        chunks = await _collect(
            self.cap.stream_generate(request({"prompt": "x", "mock_response": "hello world"}))
        )
        self.assertEqual("".join(chunks), "hello world")

    async def test_stream_real_via_urlopen(self) -> None:
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            b'data: {"choices":[{"delta":{"content":"lo"}}]}',
            b"data: [DONE]",
        ]
        with patch("urllib.request.urlopen", return_value=_FakeSSE(lines)) as urlopen:
            chunks = await _collect(
                self.cap.stream_generate(
                    request({"prompt": "hi", "base_url": "http://192.168.0.5:8000/v1", "temperature": 0.1, "max_tokens": 32})
                )
            )
        self.assertEqual("".join(chunks), "Hello")
        self.assertEqual(
            json.loads(urlopen.call_args.args[0].data)["max_tokens"],
            32,
        )

    async def test_stream_events_real_with_tool_calls(self) -> None:
        lines = [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"filesystem","arguments":"{\\"path\\":"}}]}}]}',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"notes.txt\\"}"}}]},"finish_reason":"tool_calls"}]}',
            b"data: [DONE]",
        ]
        with patch("urllib.request.urlopen", return_value=_FakeSSE(lines)):
            events = await _collect(
                self.cap.stream_generate_events(
                    request({
                        "messages": [{"role": "user", "content": "read notes"}],
                        "tools": [{"type": "function", "function": {"name": "filesystem", "parameters": {}}}],
                        "tool_choice": "auto",
                        "base_url": "http://192.168.0.5:8000/v1",
                    })
                )
            )
        done = events[-1]
        self.assertEqual(done["finish_reason"], "tool_calls")
        self.assertEqual(done["tool_calls"][0]["id"], "call_1")
        self.assertEqual(done["tool_calls"][0]["function"]["name"], "filesystem")
        self.assertEqual(done["tool_calls"][0]["function"]["arguments"], '{"path":"notes.txt"}')

    async def test_stream_events_emits_reasoning_before_content(self) -> None:
        lines = [
            b'data: {"choices":[{"delta":{"reasoning":"thinking"}}]}',
            b'data: {"choices":[{"delta":{"content":"answer"}}]}',
            b'data: {"choices":[],"usage":{"prompt_tokens":123,"completion_tokens":4,"total_tokens":127}}',
            b"data: [DONE]",
        ]
        with patch("urllib.request.urlopen", return_value=_FakeSSE(lines)) as urlopen:
            events = await _collect(
                self.cap.stream_generate_events(
                    request({
                        "prompt": "hi",
                        "base_url": "http://192.168.0.5:8000/v1",
                    })
                )
            )
        self.assertEqual(events[0], {"type": "reasoning", "content": "thinking"})
        self.assertEqual(events[1], {"type": "delta", "content": "answer"})
        self.assertEqual(
            events[2],
            {
                "type": "context",
                "used": 123,
                "unit": "tokens",
                "scope": "prompt",
                "source": "provider",
            },
        )
        sent = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(sent["stream_options"], {"include_usage": True})
        self.assertEqual(sent["max_tokens"], 4096)

    async def test_stream_events_retries_without_unsupported_stream_options(self) -> None:
        for status in (400, 422):
            with self.subTest(status=status):
                error = urllib.error.HTTPError(
                    "http://192.168.0.5:8000/v1/chat/completions",
                    status,
                    "unsupported stream_options",
                    {},
                    None,
                )
                lines = [
                    b'data: {"choices":[{"delta":{"content":"answer"}}]}',
                    b"data: [DONE]",
                ]
                with patch(
                    "urllib.request.urlopen",
                    side_effect=[error, _FakeSSE(lines)],
                ) as urlopen:
                    events = await _collect(
                        self.cap.stream_generate_events(
                            request({
                                "prompt": "hi",
                                "base_url": "http://192.168.0.5:8000/v1",
                            })
                        )
                    )
                self.assertEqual(events[0], {"type": "delta", "content": "answer"})
                self.assertEqual(urlopen.call_count, 2)
                first = json.loads(urlopen.call_args_list[0].args[0].data)
                second = json.loads(urlopen.call_args_list[1].args[0].data)
                self.assertEqual(first["stream_options"], {"include_usage": True})
                self.assertNotIn("stream_options", second)

    async def test_stream_events_does_not_retry_other_http_errors(self) -> None:
        error = urllib.error.HTTPError(
            "http://192.168.0.5:8000/v1/chat/completions",
            500,
            "server error",
            {},
            None,
        )
        with patch("urllib.request.urlopen", side_effect=error) as urlopen:
            with self.assertRaises(urllib.error.HTTPError):
                await _collect(
                    self.cap.stream_generate_events(
                        request({
                            "prompt": "hi",
                            "base_url": "http://192.168.0.5:8000/v1",
                        })
                    )
                )
        self.assertEqual(urlopen.call_count, 1)

    async def test_stream_events_propagates_failed_compatibility_retry(self) -> None:
        errors = [
            urllib.error.HTTPError("http://x", status, "error", {}, None)
            for status in (400, 422)
        ]
        with patch("urllib.request.urlopen", side_effect=errors) as urlopen:
            with self.assertRaises(urllib.error.HTTPError) as raised:
                await _collect(
                    self.cap.stream_generate_events(
                        request({
                            "prompt": "hi",
                            "base_url": "http://192.168.0.5:8000/v1",
                        })
                    )
                )
        self.assertEqual(raised.exception.code, 422)
        self.assertEqual(urlopen.call_count, 2)

    async def test_stream_helper_does_not_retry_without_stream_options(self) -> None:
        error = urllib.error.HTTPError("http://x", 400, "bad request", {}, None)
        with patch("urllib.request.urlopen", side_effect=error) as urlopen:
            with self.assertRaises(urllib.error.HTTPError):
                list(
                    self.cap._post_chat_completion_stream_events(
                        base_url="http://192.168.0.5:8000/v1",
                        api_key="local",
                        payload={"model": "qwen", "messages": [], "stream": True},
                        timeout=1,
                    )
                )
        self.assertEqual(urlopen.call_count, 1)

    async def test_stream_schema_error_raises(self) -> None:
        with self.assertRaises(ValueError):
            await _collect(
                self.cap.stream_generate(request({"prompt": "hi", "temperature": "hot"}))
            )

    async def test_stream_out_of_range_max_tokens_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "1 to 32768"):
            await _collect(
                self.cap.stream_generate(
                    request(
                        {
                            "prompt": "hi",
                            "max_tokens": 0,
                            "base_url": "http://127.0.0.1:9/v1",
                        }
                    )
                )
            )

    async def test_stream_plan_error_raises(self) -> None:
        with self.assertRaises(ValueError):
            await _collect(self.cap.stream_generate(request({"prompt": ""})))

    async def test_stream_error_propagates(self) -> None:
        with patch.object(
            self.cap, "_post_chat_completion_stream", MagicMock(side_effect=RuntimeError("boom"))
        ):
            with self.assertRaises(RuntimeError):
                await _collect(
                    self.cap.stream_generate(
                        request({"prompt": "hi", "base_url": "http://127.0.0.1:9/v1"})
                    )
                )


if __name__ == "__main__":
    unittest.main()
