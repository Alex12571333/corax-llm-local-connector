# Corax LLM Local Connector

A standalone [Corax](https://github.com/Alex12571333/corax-agent) capability that
connects the agent to a **local LLM running on the Spark device** (e.g. a DGX
Spark on the `192.168.10.0/24` network) through an OpenAI-compatible
`/chat/completions` endpoint — with **text plus optional image / video input**,
selectable during primary or secondary agent setup.

It is a pure capability package. It does not modify, vendor, or depend on the
internals of `corax-core`, `corax-sdk`, or `corax-agent`; it only uses their
public contracts (`agent_core.Capability` / `Result`, the `agent_sdk` manifest +
loader). The agent can install it without any code change — just point a
`capabilities.available` entry at this directory.

| | |
|---|---|
| id | `llm.local` |
| entrypoint | `main:LLMLocalConnector` |
| type | `connector` |
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
capabilities:
  enabled: [echo, filesystem, editor, shell, llm.local]
  available:
    llm.local:
      enabled: true
      type: connector
      description: Local Spark LLM connector
      path: ../corax-llm-local-connector
```

The agent's `CapabilityLoader` loads it by manifest + entrypoint via the SDK.

## Tests

```bash
python -m unittest discover -s tests -v
# coverage (100%)
python -m coverage run -m unittest discover -s tests && python -m coverage report -m
```
