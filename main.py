"""Corax LLM Local Connector capability.

This package contains only a capability implementation. It speaks to a local
LLM running on the Spark device (e.g. a DGX Spark on the ``192.168.10.0/24``
network) through an OpenAI-compatible ``/chat/completions`` endpoint.

Two things make it useful inside Corax without ever touching the agent's code:

* It implements the public ``agent_core.ModelProvider`` contract and carries an
  ``agent_sdk`` manifest, so the SDK loader can install and run it as-is.
* It is *multimodal-aware*. Which input modalities (text, image, video) the
  local model accepts is a setup choice. The choice is read from environment
  variables that the runtime menu writes during primary or secondary setup, and
  the capability self-describes that menu through its ``describe`` /
  ``configure`` operations -- so the agent can render the modality picker
  generically, with no capability-specific code.

The capability never reads ``.env``, ``~/.ssh`` or any local file; it only reads
a small set of ``CORAX_LLM_*`` environment variables and forwards media supplied
as ``http(s)`` / ``data:`` URIs. It will only talk to a private/loopback
endpoint, and it never echoes secrets back to the caller.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, AsyncIterator, Iterator

from agent_core import (
    CapabilityRequest,
    CoreError,
    ErrorCode,
    HealthStatus,
    ModelProvider,
    ModelRequest,
    PermissionLevel,
    Result,
    ResultStatus,
    RiskLevel,
    SideEffect,
)
from agent_core import schema as core_schema
from agent_sdk import model_provider

CAPABILITY_ID = "llm.local"
CAPABILITY_NAME = "LLM Local Connector"

# Spark host on the local network (vLLM OpenAI-compatible server on the GB10).
# Fully overridable per request (``base_url``) or via the ``CORAX_LLM_BASE_URL``
# environment variable set during setup.
DEFAULT_BASE_URL = "http://192.168.0.10:8000/v1"
DEFAULT_MODEL = "google/gemma-4-12B-it"
DEFAULT_LOCAL_API_KEY = "local"

DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_TOKENS = 4096
MAX_MAX_TOKENS = 32768
MAX_MEDIA_ITEMS = 16
MAX_OUTPUT_CHARS = 20000

SUPPORTED_MODALITIES = ("text", "image", "video")
TOGGLEABLE_MODALITIES = ("image", "video")
DEFAULT_MODALITIES = ("text",)

_ALLOWED_MEDIA_SCHEMES = ("http", "https", "data")
# Substrings that mean "this reference points at secret material" -- refused
# regardless of which modality is enabled.
_SECRET_MARKERS = (".ssh", ".env", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519")

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"type": "string"},
        "prompt": {"type": "string"},
        "system": {"type": "string"},
        "messages": {"type": "array"},
        "model": {"type": "string"},
        "temperature": {"type": "number"},
        "max_tokens": {"type": "integer"},
        "base_url": {"type": "string"},
        "images": {"type": "array"},
        "videos": {"type": "array"},
        "modalities": {"type": "array"},
        "enable_image": {"type": "boolean"},
        "enable_video": {"type": "boolean"},
        "timeout_seconds": {"type": "number"},
        "mock_response": {"type": "string"},
        "stream": {"type": "boolean"},
        "state_key": {"type": "string"},
        "tools": {"type": "array"},
        "tool_choice": {"type": "string"},
        "mock_tool_calls": {"type": "array"},
    },
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"type": "string"},
        "text": {"type": "string"},
        "model": {"type": "string"},
        "endpoint": {"type": "string"},
        "raw": {"type": "object"},
        "safe": {"type": "boolean"},
        "supported_modalities": {"type": "array"},
        "enabled_modalities": {"type": "array"},
        "used_modalities": {"type": "array"},
        "setup_menu": {"type": "object"},
        "env_assignments": {"type": "object"},
        "tool_calls": {"type": "array"},
        "finish_reason": {},
    },
    "required": ["operation"],
}


class _SecurityError(Exception):
    """A request was refused by the capability's security rules."""


# --------------------------------------------------------------------------- #
# Result helpers
# --------------------------------------------------------------------------- #
def _fail(
    request: CapabilityRequest,
    code: ErrorCode,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    retryable: bool = False,
    status: ResultStatus = ResultStatus.ERROR,
) -> Result:
    return Result.fail(
        CoreError(
            code=code,
            message=message,
            details=details or {},
            retryable=retryable,
        ),
        session_id=request.session_id,
        task_id=request.task_id,
        status=status,
    )


def _ok(request: CapabilityRequest, payload: dict[str, Any]) -> Result:
    return Result.ok(payload, session_id=request.session_id, task_id=request.task_id)


# --------------------------------------------------------------------------- #
# Environment-backed setup (what the runtime menu writes)
# --------------------------------------------------------------------------- #
def _env_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on", "y")


def _configured_modalities() -> set[str]:
    """The currently enabled input modalities, resolved from the environment."""
    enabled: set[str] = set(DEFAULT_MODALITIES)

    raw = os.getenv("CORAX_LLM_MODALITIES")
    if raw is not None:
        enabled = {"text"}
        for item in raw.split(","):
            normalized = item.strip().lower()
            if normalized in SUPPORTED_MODALITIES:
                enabled.add(normalized)

    image = _env_bool("CORAX_LLM_ENABLE_IMAGE")
    if image is True:
        enabled.add("image")
    elif image is False:
        enabled.discard("image")

    video = _env_bool("CORAX_LLM_ENABLE_VIDEO")
    if video is True:
        enabled.add("video")
    elif video is False:
        enabled.discard("video")

    enabled.add("text")  # text input is always available
    return enabled


def _ordered(modalities: set[str]) -> list[str]:
    return [m for m in SUPPORTED_MODALITIES if m in modalities]


def _selection_from_request(data: dict[str, Any]) -> set[str]:
    """Compute the desired modality selection from a ``configure`` request."""
    enabled = set(_configured_modalities())

    if "modalities" in data:
        enabled = {"text"}
        for item in data["modalities"]:
            if not isinstance(item, str):
                raise ValueError("modalities entries must be strings")
            normalized = item.strip().lower()
            if normalized not in SUPPORTED_MODALITIES:
                raise ValueError(f"unsupported modality {item!r}")
            enabled.add(normalized)

    for key, modality in (("enable_image", "image"), ("enable_video", "video")):
        if key in data:
            if data[key]:
                enabled.add(modality)
            else:
                enabled.discard(modality)

    enabled.add("text")
    return enabled


def _env_assignments(enabled: set[str]) -> dict[str, str]:
    """The environment the runtime menu should persist for this selection."""
    return {
        "CORAX_LLM_MODALITIES": ",".join(_ordered(enabled)),
        "CORAX_LLM_ENABLE_IMAGE": "true" if "image" in enabled else "false",
        "CORAX_LLM_ENABLE_VIDEO": "true" if "video" in enabled else "false",
    }


def _setup_menu(enabled: set[str]) -> dict[str, Any]:
    """A render-ready descriptor of the modality picker for the runtime menu."""
    descriptions = {
        "text": "Always-on text prompts.",
        "image": "Accept images (http(s) or data: URI) as model input.",
        "video": "Accept video (http(s) or data: URI) as model input.",
    }
    env_for = {
        "image": "CORAX_LLM_ENABLE_IMAGE",
        "video": "CORAX_LLM_ENABLE_VIDEO",
    }
    options = []
    for modality in SUPPORTED_MODALITIES:
        options.append(
            {
                "id": modality,
                "label": modality.capitalize() + (" input" if modality != "text" else ""),
                "selected": modality in enabled,
                "locked": modality not in TOGGLEABLE_MODALITIES,
                "description": descriptions[modality],
                "env": env_for.get(modality),
            }
        )
    return {
        "title": "LLM Local Connector — input modalities",
        "description": "Choose which input modalities the local Spark model accepts.",
        "stage": "primary_or_secondary_setup",
        "persist_via": "environment",
        "options": options,
    }


# --------------------------------------------------------------------------- #
# Request resolution
# --------------------------------------------------------------------------- #
def _resolve_base_url(data: dict[str, Any]) -> str:
    return str(data.get("base_url") or os.getenv("CORAX_LLM_BASE_URL") or DEFAULT_BASE_URL)


def _resolve_model(data: dict[str, Any]) -> str:
    return str(data.get("model") or os.getenv("CORAX_LLM_MODEL") or DEFAULT_MODEL)


def _resolve_timeout(data: dict[str, Any]) -> float:
    timeout = float(data.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    if timeout <= 0 or timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError(
            f"timeout_seconds must be > 0 and <= {MAX_TIMEOUT_SECONDS:g}"
        )
    return timeout


def _resolve_max_tokens(data: dict[str, Any]) -> int:
    max_tokens = data.get("max_tokens", DEFAULT_MAX_TOKENS)
    if (
        type(max_tokens) is not int
        or max_tokens <= 0
        or max_tokens > MAX_MAX_TOKENS
    ):
        raise ValueError(
            f"max_tokens must be an integer from 1 to {MAX_MAX_TOKENS}"
        )
    return max_tokens


def _ensure_local_endpoint(base_url: str) -> None:
    """Refuse anything that is not a loopback/private/link-local host."""
    host = urllib.parse.urlparse(base_url).hostname
    if not host:
        raise _SecurityError("endpoint URL has no host")
    if host == "localhost":
        return
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise _SecurityError(
            "endpoint host must be a local/private address (this is a local connector)"
        ) from None
    if not (address.is_private or address.is_loopback or address.is_link_local):
        raise _SecurityError(
            "endpoint host must be a local/private address (this is a local connector)"
        )


def _validate_media(items: Any, field: str) -> list[str]:
    """Normalise and security-check a list of media references."""
    refs: list[str] = []
    if not items:
        return refs
    if len(items) > MAX_MEDIA_ITEMS:
        raise ValueError(f"{field} accepts at most {MAX_MEDIA_ITEMS} items")
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field} entries must be non-empty strings")
        ref = item.strip()
        scheme = urllib.parse.urlparse(ref).scheme.lower()
        if scheme not in _ALLOWED_MEDIA_SCHEMES:
            raise _SecurityError(f"{field} entries must be http(s) or data: URIs")
        if any(marker in ref.lower() for marker in _SECRET_MARKERS):
            raise _SecurityError(f"{field} entry references forbidden secret material")
        refs.append(ref)
    return refs


def _build_messages(
    data: dict[str, Any],
    image_refs: list[str],
    video_refs: list[str],
) -> list[dict[str, Any]]:
    raw_messages = data.get("messages")
    if isinstance(raw_messages, list) and raw_messages:
        if image_refs or video_refs:
            raise ValueError(
                "supply images/videos together with 'prompt', not pre-built 'messages'"
            )
        return raw_messages

    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("input must include a non-empty 'prompt' or 'messages'")

    messages: list[dict[str, Any]] = []
    system = data.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})

    if image_refs or video_refs:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for ref in image_refs:
            content.append({"type": "image_url", "image_url": {"url": ref}})
        for ref in video_refs:
            content.append({"type": "video_url", "video_url": {"url": ref}})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})
    return messages


def _used_modalities(image_refs: list[str], video_refs: list[str]) -> list[str]:
    used = ["text"]
    if image_refs:
        used.append("image")
    if video_refs:
        used.append("video")
    return used


def _chunk_text(text: str, size: int = 24) -> list[str]:
    """Split text into small pieces to simulate token streaming (mock/offline)."""
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def _sse_delta(line: str) -> str | None:
    """Extract the incremental content from one OpenAI-style SSE ``data:`` line.

    Returns the delta text, ``""`` for a non-content data line, or ``None`` when
    the line is not a data line or signals end-of-stream (``[DONE]``).
    """
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
        return delta["content"]
    return ""


def _sse_event(line: str) -> dict[str, Any] | None:
    """Parse one OpenAI-style SSE line into a compact streaming event."""
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if not payload:
        return None
    if payload == "[DONE]":
        return {"type": "done"}
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None
    usage = obj.get("usage")
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if type(prompt_tokens) is int and prompt_tokens >= 0:
            used = prompt_tokens
        elif (
            type(total_tokens) is int
            and total_tokens >= 0
            and type(completion_tokens) is int
            and completion_tokens >= 0
            and total_tokens >= completion_tokens
        ):
            used = total_tokens - completion_tokens
        else:
            used = None
        if used is not None:
            return {
                "type": "context",
                "used": used,
                "unit": "tokens",
                "scope": "prompt",
                "source": "provider",
            }
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    delta = first.get("delta") if isinstance(first.get("delta"), dict) else {}
    event: dict[str, Any] = {"type": "delta"}
    if isinstance(delta.get("content"), str):
        event["content"] = delta["content"]
    reasoning = delta.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = delta.get("reasoning_content")
    if isinstance(reasoning, str):
        event["reasoning"] = reasoning
    if isinstance(delta.get("tool_calls"), list):
        event["tool_calls"] = delta["tool_calls"]
    finish_reason = first.get("finish_reason")
    if isinstance(finish_reason, str):
        event["finish_reason"] = finish_reason
    return event


def _iter_sse_content(response: Iterator[bytes]) -> Iterator[str]:
    """Yield non-empty content deltas from an SSE byte-line stream."""
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        delta = _sse_delta(line)
        if delta:
            yield delta


def _iter_sse_events(response: Iterator[bytes]) -> Iterator[dict[str, Any]]:
    """Yield compact delta/done events from an SSE byte-line stream."""
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        event = _sse_event(line)
        if event is not None:
            yield event


def _merge_tool_call_delta(calls: list[dict[str, Any]], delta: dict[str, Any]) -> None:
    index = delta.get("index")
    if not isinstance(index, int):
        index = len(calls)
    while len(calls) <= index:
        calls.append({"function": {"arguments": ""}})
    target = calls[index]
    if isinstance(delta.get("id"), str):
        target["id"] = delta["id"]
    if isinstance(delta.get("type"), str):
        target["type"] = delta["type"]
    function = delta.get("function")
    if isinstance(function, dict):
        target_function = target.setdefault("function", {})
        if isinstance(function.get("name"), str):
            target_function["name"] = function["name"]
        if isinstance(function.get("arguments"), str):
            target_function["arguments"] = (
                str(target_function.get("arguments") or "") + function["arguments"]
            )


def _extract_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content[:MAX_OUTPUT_CHARS]
    text = first.get("text")
    return text[:MAX_OUTPUT_CHARS] if isinstance(text, str) else ""


def _extract_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    first = choices[0]
    if not isinstance(first, dict):
        return []
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("tool_calls"), list):
        return message["tool_calls"]
    return []


def _finish_reason(response: dict[str, Any]) -> str | None:
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        if isinstance(reason, str):
            return reason
    return None


# --------------------------------------------------------------------------- #
# Capability
# --------------------------------------------------------------------------- #
@model_provider(
    id=CAPABILITY_ID,
    name=CAPABILITY_NAME,
    description=(
        "Connect Corax to a local LLM on the Spark device through an "
        "OpenAI-compatible endpoint, with text plus optional image/video input "
        "selectable during agent setup."
    ),
    version="1.1.4",
    author="Corax",
    license="MIT",
    tags=["llm", "local", "model-provider", "spark", "multimodal", "vision"],
    permission_level=PermissionLevel.CONFIRM,
    required_scopes=["network.private", "model.inference"],
    risk_level=RiskLevel.MEDIUM,
    side_effects=[SideEffect.NETWORK_REQUEST, SideEffect.MODEL_INFERENCE],
    secrets=["CORAX_LLM_API_KEY"],
    input_schema=INPUT_SCHEMA,
    output_schema=OUTPUT_SCHEMA,
    entrypoint="main:LocalModelProvider",
    min_core_version="0.2.0",
)
class LocalModelProvider(ModelProvider):
    """Call a local Spark LLM and manage its input-modality setup."""

    async def generate(self, request: ModelRequest) -> Result:
        """Generate through the model-provider contract."""
        return await self.execute(_model_capability_request(request))

    async def execute(self, request: CapabilityRequest) -> Result:
        try:
            data = request.input
            errors = core_schema.validate(data, INPUT_SCHEMA)
            if errors:
                raise ValueError("; ".join(errors))

            operation = data.get("operation", "generate")
            if operation == "generate":
                result = await self._generate(request, data)
            elif operation == "configure":
                result = self._configure(request, data)
            elif operation == "describe":
                result = self._describe(request)
            elif operation == "validate":
                result = self._validate(request, data)
            else:
                raise ValueError(f"unsupported operation {operation!r}")
            # Optional: mirror the payload into the core session state so a
            # kernel-driven caller (e.g. the agent gateway) can read it back via
            # the StateManager. Off unless a non-empty ``state_key`` is given.
            state_key = data.get("state_key")
            if isinstance(state_key, str) and state_key and result.is_success:
                result.state_patch = {state_key: result.payload}
            return result
        except ValueError as exc:
            return _fail(request, ErrorCode.INVALID_INPUT, str(exc))
        except _SecurityError as exc:
            return _fail(
                request,
                ErrorCode.POLICY_DENIED,
                f"request rejected by capability security rules: {exc}",
                status=ResultStatus.POLICY_DENIED,
            )
        except Exception:
            return _fail(
                request,
                ErrorCode.CAPABILITY_FAILED,
                "llm local connector failed before completing the request",
            )

    # -- operations ------------------------------------------------------ #
    def _plan(self, data: dict[str, Any]) -> dict[str, Any]:
        """Shared pre-flight for ``generate`` and ``validate`` (no network)."""
        enabled = _configured_modalities()
        image_refs = _validate_media(data.get("images"), "images")
        video_refs = _validate_media(data.get("videos"), "videos")
        if image_refs and "image" not in enabled:
            raise _SecurityError("image input is disabled; enable it during setup")
        if video_refs and "video" not in enabled:
            raise _SecurityError("video input is disabled; enable it during setup")
        messages = _build_messages(data, image_refs, video_refs)
        return {
            "enabled": enabled,
            "image_refs": image_refs,
            "video_refs": video_refs,
            "messages": messages,
        }

    async def _generate(self, request: CapabilityRequest, data: dict[str, Any]) -> Result:
        plan = self._plan(data)
        model = _resolve_model(data)
        used = _used_modalities(plan["image_refs"], plan["video_refs"])

        mock_response = data.get("mock_response")
        if isinstance(mock_response, str):
            mock_tool_calls = data.get("mock_tool_calls") or []
            return _ok(
                request,
                {
                    "operation": "generate",
                    "text": mock_response,
                    "model": model,
                    "endpoint": "mock",
                    "enabled_modalities": _ordered(plan["enabled"]),
                    "used_modalities": used,
                    "tool_calls": mock_tool_calls,
                    "finish_reason": "tool_calls" if mock_tool_calls else "stop",
                    "raw": {"choices": [{"message": {"content": mock_response}}]},
                },
            )

        base_url = _resolve_base_url(data)
        _ensure_local_endpoint(base_url)
        timeout = _resolve_timeout(data)

        payload: dict[str, Any] = {
            "model": model,
            "messages": plan["messages"],
            "max_tokens": _resolve_max_tokens(data),
        }
        if "temperature" in data:
            payload["temperature"] = data["temperature"]
        if "tools" in data:
            payload["tools"] = data["tools"]
        if "tool_choice" in data:
            payload["tool_choice"] = data["tool_choice"]

        api_key = os.getenv("CORAX_LLM_API_KEY") or DEFAULT_LOCAL_API_KEY

        try:
            raw = await asyncio.to_thread(
                self._post_chat_completion,
                base_url=base_url,
                api_key=api_key,
                payload=payload,
                timeout=timeout,
            )
        except urllib.error.HTTPError as exc:
            return _fail(
                request,
                ErrorCode.CAPABILITY_FAILED,
                "local LLM endpoint returned an HTTP error",
                details={"status": exc.code},
                retryable=True,
            )
        except (urllib.error.URLError, TimeoutError, OSError):
            return _fail(
                request,
                ErrorCode.CAPABILITY_FAILED,
                "local LLM endpoint request failed",
                details={"endpoint": base_url},
                retryable=True,
            )
        except json.JSONDecodeError:
            return _fail(
                request,
                ErrorCode.CAPABILITY_FAILED,
                "local LLM endpoint returned invalid JSON",
                retryable=True,
            )

        return _ok(
            request,
            {
                "operation": "generate",
                "text": _extract_text(raw),
                "model": str(raw.get("model") or model),
                "endpoint": base_url,
                "enabled_modalities": _ordered(plan["enabled"]),
                "used_modalities": used,
                "tool_calls": _extract_tool_calls(raw),
                "finish_reason": _finish_reason(raw),
                "raw": raw,
            },
        )

    def _configure(self, request: CapabilityRequest, data: dict[str, Any]) -> Result:
        enabled = _selection_from_request(data)
        return _ok(
            request,
            {
                "operation": "configure",
                "supported_modalities": list(SUPPORTED_MODALITIES),
                "enabled_modalities": _ordered(enabled),
                "env_assignments": _env_assignments(enabled),
                "setup_menu": _setup_menu(enabled),
                "safe": True,
            },
        )

    def _describe(self, request: CapabilityRequest) -> Result:
        enabled = _configured_modalities()
        return _ok(
            request,
            {
                "operation": "describe",
                "model": _resolve_model({}),
                "endpoint": _resolve_base_url({}),
                "supported_modalities": list(SUPPORTED_MODALITIES),
                "enabled_modalities": _ordered(enabled),
                "env_assignments": _env_assignments(enabled),
                "setup_menu": _setup_menu(enabled),
                "safe": True,
            },
        )

    def _validate(self, request: CapabilityRequest, data: dict[str, Any]) -> Result:
        plan = self._plan(data)
        return _ok(
            request,
            {
                "operation": "validate",
                "safe": True,
                "enabled_modalities": _ordered(plan["enabled"]),
                "used_modalities": _used_modalities(
                    plan["image_refs"], plan["video_refs"]
                ),
            },
        )

    # -- network --------------------------------------------------------- #
    def _post_chat_completion(
        self,
        *,
        base_url: str,
        api_key: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    # -- streaming ------------------------------------------------------- #
    async def stream_generate(
        self,
        request: ModelRequest | CapabilityRequest,
    ) -> AsyncIterator[str]:
        """Yield the model's reply incrementally as text deltas.

        Used directly by a streaming caller (e.g. the agent gateway); the
        single-shot ``execute`` contract returns the whole text instead. Input
        and modalities are validated exactly like ``generate``. Raises (rather
        than returning a Result) so the caller drives error handling.
        """
        request = _model_capability_request(request)
        data = request.input
        errors = core_schema.validate(data, INPUT_SCHEMA)
        if errors:
            raise ValueError("; ".join(errors))
        plan = self._plan(data)

        mock_response = data.get("mock_response")
        if isinstance(mock_response, str):
            for piece in _chunk_text(mock_response):
                yield piece
            return

        base_url = _resolve_base_url(data)
        _ensure_local_endpoint(base_url)
        timeout = _resolve_timeout(data)
        payload: dict[str, Any] = {
            "model": _resolve_model(data),
            "messages": plan["messages"],
            "stream": True,
            "max_tokens": _resolve_max_tokens(data),
        }
        if "temperature" in data:
            payload["temperature"] = data["temperature"]
        api_key = os.getenv("CORAX_LLM_API_KEY") or DEFAULT_LOCAL_API_KEY

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def worker() -> None:
            try:
                for piece in self._post_chat_completion_stream(
                    base_url=base_url, api_key=api_key, payload=payload, timeout=timeout
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, ("data", piece))
            except Exception as exc:  # noqa: BLE001 - surfaced to the awaiting caller
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        threading.Thread(target=worker, daemon=True).start()
        while True:
            kind, value = await queue.get()
            if kind == "data":
                yield value
            elif kind == "error":
                raise value
            else:
                return

    async def stream_generate_events(
        self,
        request: ModelRequest | CapabilityRequest,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield streaming text deltas and final tool calls from one LLM request."""
        request = _model_capability_request(request)
        data = request.input
        errors = core_schema.validate(data, INPUT_SCHEMA)
        if errors:
            raise ValueError("; ".join(errors))
        plan = self._plan(data)

        mock_response = data.get("mock_response")
        if isinstance(mock_response, str):
            for piece in _chunk_text(mock_response):
                yield {"type": "delta", "content": piece}
            mock_tool_calls = data.get("mock_tool_calls")
            yield {
                "type": "done",
                "tool_calls": mock_tool_calls if isinstance(mock_tool_calls, list) else [],
                "finish_reason": "tool_calls" if mock_tool_calls else "stop",
            }
            return

        base_url = _resolve_base_url(data)
        _ensure_local_endpoint(base_url)
        timeout = _resolve_timeout(data)
        payload: dict[str, Any] = {
            "model": _resolve_model(data),
            "messages": plan["messages"],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": _resolve_max_tokens(data),
        }
        if "temperature" in data:
            payload["temperature"] = data["temperature"]
        if "tools" in data:
            payload["tools"] = data["tools"]
        if "tool_choice" in data:
            payload["tool_choice"] = data["tool_choice"]
        api_key = os.getenv("CORAX_LLM_API_KEY") or DEFAULT_LOCAL_API_KEY

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def worker() -> None:
            try:
                for event in self._post_chat_completion_stream_events(
                    base_url=base_url, api_key=api_key, payload=payload, timeout=timeout
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, ("data", event))
            except Exception as exc:  # noqa: BLE001 - surfaced to the awaiting caller
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        threading.Thread(target=worker, daemon=True).start()
        tool_calls: list[dict[str, Any]] = []
        finish_reason: str | None = None
        while True:
            kind, value = await queue.get()
            if kind == "error":
                raise value
            if kind == "done":
                yield {
                    "type": "done",
                    "tool_calls": tool_calls,
                    "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop"),
                }
                return

            event = value
            if not isinstance(event, dict):
                continue
            if event.get("type") == "context":
                yield event
                continue
            if isinstance(event.get("finish_reason"), str):
                finish_reason = event["finish_reason"]
            for tool_delta in event.get("tool_calls") or []:
                if isinstance(tool_delta, dict):
                    _merge_tool_call_delta(tool_calls, tool_delta)
            reasoning = event.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                yield {"type": "reasoning", "content": reasoning}
            content = event.get("content")
            if isinstance(content, str) and content:
                yield {"type": "delta", "content": content}

    def _post_chat_completion_stream(
        self,
        *,
        base_url: str,
        api_key: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> Iterator[str]:
        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        )
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            yield from _iter_sse_content(response)

    def _post_chat_completion_stream_events(
        self,
        *,
        base_url: str,
        api_key: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> Iterator[dict[str, Any]]:
        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        fallback = dict(payload)
        fallback.pop("stream_options", None)
        attempts = (payload, fallback) if fallback != payload else (payload,)
        for index, attempt in enumerate(attempts):
            body = json.dumps(attempt).encode("utf-8")
            http_request = urllib.request.Request(
                endpoint,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
            )
            try:
                response = urllib.request.urlopen(http_request, timeout=timeout)
            except urllib.error.HTTPError as exc:
                if len(attempts) == 1 or index or exc.code not in {400, 422}:
                    raise
                exc.close()
                continue
            with response as stream:
                yield from _iter_sse_events(stream)
            return

    async def health_check(self) -> HealthStatus:
        return HealthStatus.HEALTHY


def _model_capability_request(
    request: ModelRequest | CapabilityRequest,
) -> CapabilityRequest:
    """Translate the role request for the legacy internal operation pipeline."""
    if isinstance(request, CapabilityRequest):
        return request
    data = dict(request.parameters)
    data["operation"] = "generate"
    if request.prompt:
        data["prompt"] = request.prompt
    if request.messages:
        data["messages"] = list(request.messages)
    if request.model:
        data["model"] = request.model
    if request.modalities:
        data["modalities"] = list(request.modalities)
    return CapabilityRequest(
        task_id=f"model-{id(request)}",
        session_id=request.session_id,
        input=data,
        metadata=dict(request.metadata),
    )


# Import compatibility for 0.1 callers. The canonical entrypoint is above.
LLMLocalConnector = LocalModelProvider
