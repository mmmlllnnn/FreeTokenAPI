# Changelog

## 2.0.1 — 2026-09-10

### Fixed

- Preserve DeepSeek file-parser error codes and retryability. Provider-busy failures such as `50404` no longer appear as invalid-request HTTP 400 errors.
- Retry an explicitly retryable temporary parser failure at most once, with the original attachment and request settings and a fresh proof. Permanent failures and content/audit rejection are not retried.
- Separate the 300-second attachment-preparation deadline from the HTTP timeout; the single budget includes uploads and retries. Poll at the web UI's three-second interval.
- Never generate from a failed file reference; add regression coverage across all three API formats.
- Clarify the lightweight architecture and required runtime dependencies in both READMEs.

## 2.0.0 — 2026-09-10

### Breaking changes

- Removed every previous DeepSeek model ID without compatibility aliases. Clients must select `deepseek-web` or `deepseek-web-thinking`. Both target the web backend's enabled, switchable `default` model, with different default thinking flags.
- Removed fixed Flash/Pro/Vision attachment rules and updated the Pi extension and configuration examples.

### Added

- Official `model_configs` discovery and a per-account, in-memory TTL cache with coalesced refreshes and bounded failure cooldown. Expired capabilities are never used after refresh failure.
- Dynamic model advertising with input modalities and capability metadata; capability-aware routing for thinking, search, images, file formats and upload limits.
- Regression coverage for heterogeneous accounts, disabled defaults, schema errors, cache refresh, old-ID rejection and attachment routing across all three protocols.

The published 1.0.0 history remains unchanged.

## 1.0.0 — 2026-09-09

Initial public release.

### Included

- Local API adapters for DeepSeek and Qwen web accounts.
- OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages compatibility, including streaming and client tool calls.
- Recommended Pi Agent configuration and a native-file attachment extension for supported Qwen and DeepSeek Flash models.
- Native backend web-search switches and explicit attachment/model limitations.
- Multi-step agent tool handling for Pi and Claude Code, including tool-schema refresh, compatible call notation, and correct JSON/CDATA string arguments.
- DeepSeek file-parsing readiness and just-in-time proof-of-work generation.
- Explicit failure reporting for incomplete upstream responses, without dispatching buffered partial tool calls.
- Windows, macOS, and Linux launchers; English and Simplified Chinese documentation.
- Offline regression tests and credential-free continuous integration.

### Release boundaries

- Unofficial, personal, loopback-only adapter; no client authentication or multi-user isolation.
- Provider quotas, verification challenges, availability, and supported file formats still apply. No unlimited-operation guarantee is made.
- Local client tools retain their normal permissions. Native document snapshots are not live-synchronized with edited source files.
- No interactive API documentation endpoints are exposed.
- Live verification coverage and remaining provider-limited cases are documented in the READMEs.

This release preserves the existing MIT copyright and license notice.
