# Changelog

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
