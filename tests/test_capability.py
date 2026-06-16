from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import urllib.error

from agent_core import (
    Capability,
    CapabilityRequest,
    ErrorCode,
    PermissionLevel,
    ResultStatus,
    RiskLevel,
    SideEffect,
)
from agent_sdk import CapabilityManifest, load_instance, validate_manifest

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


class ManifestTests(unittest.TestCase):
    def test_manifest_is_sdk_valid(self) -> None:
        manifest = CapabilityManifest.load(PROJECT_ROOT)
        result = validate_manifest(manifest, core_version="0.1.0")

        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.warnings, [])
        self.assertEqual(manifest.id, "llm.local")
        self.assertEqual(manifest.name, "LLM Local Connector")
        self.assertEqual(manifest.permission_level, PermissionLevel.CONFIRM)
        self.assertEqual(manifest.risk_level, RiskLevel.MEDIUM)
        self.assertEqual(manifest.side_effects, (SideEffect.NETWORK_REQUEST,))
        self.assertEqual(manifest.entrypoint, "main:LLMLocalConnector")
        self.assertEqual(manifest.capability_type.value, "connector")

    def test_manifest_json_identity(self) -> None:
        data = json.loads((PROJECT_ROOT / "capability.json").read_text())

        self.assertEqual(data["id"], "llm.local")
        self.assertEqual(data["name"], "LLM Local Connector")
        self.assertEqual(data["version"], "1.0.0")
        self.assertEqual(data["author"], "Corax")
        self.assertEqual(data["license"], "MIT")
        self.assertEqual(data["capability_type"], "connector")
        self.assertEqual(data["min_core_version"], "0.1.0")
        self.assertEqual(data["sdk_version"], "0.1.0")
        self.assertNotIn("required_scopes", data)


class LoaderTests(unittest.TestCase):
    def test_capability_loads_through_sdk_loader(self) -> None:
        manifest = CapabilityManifest.load(PROJECT_ROOT)
        cap = load_instance(manifest, PROJECT_ROOT, core_version="0.1.0")

        self.assertIsInstance(cap, Capability)
        self.assertEqual(cap.id, "llm.local")
        self.assertEqual(cap.required_scopes, set())
        self.assertEqual(cap.side_effects, {SideEffect.NETWORK_REQUEST})


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        _scrub_env()
        manifest = CapabilityManifest.load(PROJECT_ROOT)
        self.cap = load_instance(manifest, PROJECT_ROOT, core_version="0.1.0")

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
        self.assertNotIn("secret-token", json.dumps(result.payload))

    async def test_generate_loopback_text_choice(self) -> None:
        fake = MagicMock(return_value={"choices": [{"text": "via-text"}]})
        with patch.object(self.cap, "_post_chat_completion", fake):
            result = await self.cap.execute(
                request({"prompt": "p", "base_url": "http://127.0.0.1:9/v1"})
            )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.payload["text"], "via-text")
        self.assertEqual(result.payload["model"], "qwen3.6")  # default model

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


if __name__ == "__main__":
    unittest.main()
