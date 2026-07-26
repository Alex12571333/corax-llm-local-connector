# Corax LLM Local Connector

A standalone [Corax](https://github.com/Alex12571333/corax-agent)
`model_provider` extension that
connects the agent to a **local LLM running on the Spark device** (e.g. a DGX
Spark on the `192.168.10.0/24` network) through an OpenAI-compatible
`/chat/completions` endpoint — with **text plus optional image / video input**,
selectable during primary or secondary agent setup.

It is a pure extension package. It does not modify, vendor, or depend on the
internals of `agent-core`, `agent-sdk`, or `corax-agent`; it only uses their
public contracts (`agent_core.ModelProvider` / `ModelRequest`, the `agent_sdk`
manifest + loader). The agent can install it without any code change — just
point an `extensions.available` entry at this directory.

| | |
|---|---|
| id | `llm.local` |
| version | `1.1.3` |
| entrypoint | `main:LocalModelProvider` |
| kind | `model_provider` |
| permission level | `confirm` |
| risk level | `medium` |
| side effects | `network_request` |

## Operations

The operation is selected with `input.operation` (default `generate`).

### `generate`
Run a completion against the local model.

```jsonc
{
  "operation": "generate",
  "prompt": "Summarise this image.",
  "model": "qwen3.6",
  "base_url": "http://192.168.10.1:8000/v1",
  "images": ["https://host/frame.png"]   // only if image input is enabled
}
```
Returns `{ text, model, endpoint, enabled_modalities, used_modalities, raw }`.
You may instead pass a pre-built OpenAI-style `messages` array (without
`images`/`videos`).

### Streaming events

`stream_generate_events()` keeps reasoning separate from the final answer and
emits events in provider order:

```jsonc
{"type": "reasoning", "content": "I should inspect the available tools..."}
{"type": "delta", "content": "Here is the result"}
{"type": "context", "used": 1234, "unit": "tokens", "scope": "prompt", "source": "provider"}
{"type": "done", "tool_calls": [], "finish_reason": "stop"}
```

OpenAI-compatible servers use both `delta.reasoning` and
`delta.reasoning_content` in practice; version 1.1 accepts either and maps both
to the same `reasoning` event. Answer text remains a `delta` event. Incremental
tool-call fragments are assembled and returned on the terminal `done` event,
so the runtime can preserve correct tool IDs and arguments without mixing them
into visible model text. The connector requests OpenAI-compatible streaming
usage and emits exact provider `prompt_tokens` as the occupied-context event.
If a server omits that field but supplies valid `total_tokens` and
`completion_tokens`, the prompt count is derived by subtraction. Completion
and reasoning usage are deliberately excluded from the context-window meter;
consumers may track them separately as cumulative usage. Incomplete usage is ignored
instead of being labelled exact. Servers that reject
`stream_options` with HTTP 400/422 are retried once without that optional
field, preserving streaming compatibility.

### `describe`
Read-only. Returns the supported and currently-enabled modalities plus a
render-ready `setup_menu` — this is what the runtime setup screen shows.

### `configure`
Compute a modality selection and the environment the runtime should persist.

```jsonc
{ "operation": "configure", "enable_image": true, "enable_video": false }
```
Returns `enabled_modalities`, an `env_assignments` map, and the `setup_menu`.
The capability is pure — it returns *what to persist*, it does not write the
environment itself.

### `validate`
Run all pre-flight checks (modality gating, media security, message building)
without making a network call.

## Multimodality selection (runtime menu, primary/secondary setup)

Which input modalities the local model accepts is a **setup choice**, resolved
from environment variables that the runtime menu writes:

| Env var | Meaning |
|---|---|
| `CORAX_LLM_MODALITIES` | comma list, e.g. `text,image,video` |
| `CORAX_LLM_ENABLE_IMAGE` | `true` / `false` |
| `CORAX_LLM_ENABLE_VIDEO` | `true` / `false` |

`text` is always enabled. Because the capability self-describes its picker via
`describe` and accepts changes via `configure`, the agent renders the modality
selector **generically** — no capability-specific code in the agent. A request
that supplies a disabled modality is refused with a `POLICY_DENIED` result.

## Endpoint configuration

| Env var | Default | Meaning |
|---|---|---|
| `CORAX_LLM_BASE_URL` | `http://192.168.0.10:8000/v1` | Spark vLLM endpoint (GB10, OpenAI-compatible) |
| `CORAX_LLM_MODEL` | `google/gemma-4-12B-it` | default served model (Gemma 4, vision-capable) |
| `CORAX_LLM_API_KEY` | `local` | bearer token; never echoed back |

Per-request `base_url` / `model` override the environment.

## Security

- Only talks to **loopback / private / link-local** endpoints (it's a *local*
  connector); public hosts are refused.
- Media references must be `http(s)` or `data:` URIs; anything pointing at
  secret material (`.ssh`, `.env`, private keys) is refused. The capability
  never opens local files.
- Never reads `.env` or `~/.ssh`; reads only a few `CORAX_LLM_*` variables and
  never dumps the environment.
- The API key is never placed in any result payload.
- No raw exception ever leaks: failures come back as a structured `Result.fail`.

## Install into a Corax Agent (no agent code change)

Drop this package next to the other capability packages and add it to the
agent's config:

```yaml
extensions:
  active:
    model_provider: [llm.local]
  bindings:
    primary_model: llm.local
  available:
    llm.local:
      enabled: true
      kind: model_provider
      description: Local Spark model provider
      path: ../corax-llm-local-connector
```

The agent's `ExtensionLoader` loads it by manifest + entrypoint via the SDK.

## Tests

```bash
python -m unittest discover -s tests -v
# coverage (100%)
python -m coverage run -m unittest discover -s tests && python -m coverage report -m
```
