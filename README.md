# FreeTokenAPI

**Version 2.0.1** · [简体中文](README.zh-CN.md) · [Changelog](CHANGELOG.md)

FreeTokenAPI is a **lightweight local API adapter** for your own **DeepSeek and Qwen web accounts**. It provides OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages compatibility, with streaming, client tool calls, attachments, and the web providers' native search.

**Recommended client: [Pi Agent](https://github.com/earendil-works/pi), optionally configured through [CC Switch](https://github.com/farion1231/cc-switch).**

> This is an unofficial web-service adapter, not an official OpenAI, Anthropic, DeepSeek, or Qwen API. Use your own accounts and follow the providers' terms. Availability, quotas, model capabilities, and rate limits still depend on the upstream service. “Free” does not mean unlimited or guaranteed access.

## Lightweight by design

- One local Python API service; no separate database server, Docker deployment, or browser-automation service is required.
- No local model weights or GPU inference stack: generation and native search run on the web provider, while client tools run in your agent client.
- The optional Pi attachment extension uses Node.js built-ins and does not modify Pi's installed package.

Lightweight does not mean dependency-free or zero-overhead: DeepSeek still needs its Node.js/WASM proof-of-work runtime, and uploads/caches consume resources according to file size.

## Quick start

### 1. Requirements

- **Python 3.10+**, preferably Python 3.12.
- **Node.js LTS** when using DeepSeek, for its proof-of-work challenge. Qwen-only operation does not require Node.js.
- An account that can chat on [DeepSeek](https://chat.deepseek.com) or [Qwen international](https://chat.qwen.ai).

Check your installation with `python --version` (or `python3 --version`) and `node --version`.

### 2. Configure your web login tokens

Copy [.env.example](.env.example) to `.env` **only if `.env` does not already exist**, then fill in at least one provider:

```dotenv
DEEPSEEK_TOKENS=
QWEN_TOKENS=
FREETOKENAPI_HOST=127.0.0.1
FREETOKENAPI_PORT=8000
FREETOKENAPI_TIMEOUT=60
FREETOKENAPI_SEARCH_ENABLED=1
```

Use comma-separated tokens for multiple accounts. Leave the unused provider empty.

To obtain a token, sign into the web app, open browser developer tools → **Network**, send a message, and inspect that request:

- **DeepSeek:** copy only the value after `Bearer ` in the `Authorization` header.
- **Qwen:** use the `token` cookie for `chat.qwen.ai`, or the corresponding Bearer token.
- Do not paste the entire header, a whole Cookie string, or an official developer-platform API key.

**Keep these login tokens in your local `.env`, not in Pi, screenshots, or a repository.** Environment variables take precedence over `.env`. Restart the service after configuration changes.

### 3. Start the service

**Windows:** double-click [start.bat](start.bat), or run:

```powershell
.\start.bat
```

**macOS / Linux:** run the portable launcher:

```sh
sh ./start.sh
```

Both launchers create or reuse a project-local `.venv`, install missing runtime dependencies, and start the app. If `.env` is missing, the app creates a template and asks you to fill it in. An unusable or foreign-platform virtual environment is not deleted automatically. On Linux, creating a virtual environment may require your distribution's `python3-venv` package.

Keep the terminal open. Stop the server with **Ctrl+C**. Do not start another copy while port 8000 is occupied. Do not copy `.venv` between Windows, macOS, and Linux.

Manual startup, if you prefer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.

### 4. Check the local service

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). The root should report `FreeTokenAPI`, version `1.0.0`, and `status: "ok"`. Its `adapter_features` should include `chat_completions_developer`, `qwen_chat_attachments`, and `native_web_search`; missing entries indicate an older process is still running.

[Health](http://127.0.0.1:8000/health) reports account-pool status; [models](http://127.0.0.1:8000/v1/models) lists model IDs. A successful health or model-list request **does not prove that every model can generate**.

## Recommended: Pi Agent

### Configure through CC Switch

In CC Switch, select **Pi**, add a provider, and use one of these formats:

| Pi API format | Pi API identifier | Base URL |
| --- | --- | --- |
| **OpenAI Chat Completions — recommended starting point** | `openai-completions` | `http://127.0.0.1:8000/v1` |
| OpenAI Responses | `openai-responses` | `http://127.0.0.1:8000/v1` |
| Anthropic / Claude Messages | `anthropic-messages` | `http://127.0.0.1:8000` |

Use **`local`** for the client API-key field. It is a placeholder, **not an access password**. Do not add `/chat/completions`, `/responses`, or `/messages` to the Base URL, and do not duplicate `/v1`.

Suggested models:

| Model ID | Suggested Pi input declaration | Notes |
| --- | --- | --- |
| `qwen3.8-max` | `text`, `image` | Recommended for Qwen text and image conversations; account availability still applies |
| `qwen3.7-plus` | `text`, `image` | Use only if available to your account |
| `deepseek-web` | `text`, `image` when advertised | Unified DeepSeek web model; thinking defaults off |
| `deepseek-web-thinking` | `text`, `image` when advertised | Same web model; thinking defaults on; listed only when an account supports it |

Mark image-capable entries as accepting **both text and images** in Pi. Otherwise Pi may omit the image before it reaches this API.

Keep hosted **Tool Search**, official hosted `web_search`, remote compaction, and “1M context” capability declarations disabled. Native web search below is a different capability. Keep Pi's normal tool-execution permission checks enabled.

### Dynamic DeepSeek model discovery

Version 2 removes all old DeepSeek model IDs; there are **no compatibility aliases**. Update Pi, CC Switch and other clients to `deepseek-web` or `deepseek-web-thinking`. These are this adapter's aliases, not model IDs from the paid DeepSeek API. Both send `model_type: "default"` to the web backend; an explicit request thinking flag overrides the alias default.

Each DeepSeek account fetches `/api/v0/client/settings` with `scope=model` and its own client identity/authentication. Only an **enabled, switchable `default` entry** is usable. The adapter reads its thinking, search, file and vision features, permitted file extensions, upload limits, and attachment/search conflict flag. It never falls back to another web mode.

- Capabilities are cached **in memory per account** for 300 seconds (`FREETOKENAPI_MODEL_CONFIG_TTL_SECONDS`; `0` refreshes on every access). Concurrent refreshes are coalesced. No metadata credentials or settings JWTs are stored on disk.
- Each refresh has a 10-second deadline. A failed refresh does not reuse expired capabilities. Failures have a cooldown of up to 30 seconds; retry later or restart after fixing connectivity.
- `/v1/models` advertises only the usable unified aliases. Its extra `input_modalities`, `thinking_enabled` and `capabilities` fields describe capabilities available on at least one eligible account. The thinking alias includes only thinking-capable accounts.
- Each request must fit **one account's complete capability set**. Session affinity and a free account cannot override these checks. If a different account is needed, a new web session is built from the supplied history.
- Unconfigured or disabled DeepSeek accounts do not produce placeholder models. Failed DeepSeek discovery does not hide available Qwen models; when no models can be listed and discovery failed, the endpoint returns an explicit error.

The Pi example below enables images because the current public unified configuration advertises vision. Use `input_modalities` from your own `/v1/models` response when configuring accounts with different capabilities. Metadata availability alone does not prove a login token can generate a response.

### Configure Pi directly

Merge the following provider into `~/.pi/agent/models.json` (on Windows: `%USERPROFILE%\.pi\agent\models.json`). **Do not overwrite your other providers.**

```json
{
  "providers": {
    "free-token-api": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [
        {
          "id": "qwen3.8-max",
          "name": "Qwen3.8-Max (Web)",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 65536,
          "maxTokens": 8192
        },
        {
          "id": "deepseek-web",
          "name": "DeepSeek Web (Unified)",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 65536,
          "maxTokens": 8192
        },
        {
          "id": "deepseek-web-thinking",
          "name": "DeepSeek Web Thinking (Unified)",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 65536,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

These context/output values are **conservative client settings, not measured provider limits or a capacity guarantee**. Reduce them if you encounter context-limit errors. To switch protocols, change `api` and use the matching Base URL in the table above.

No `supportsDeveloperRole: false` workaround is required: Chat Completions accepts Pi's `developer` messages and normalizes them without reordering history. Common OpenAI, DeepSeek, and Qwen thinking switches are accepted. Thinking strength is still fundamentally **on/off**, not a guaranteed low/medium/high compute budget.

### Native PDF / document attachments in Pi

The bundled [Pi native-files extension](pi-extension/index.ts) connects Pi's file input to FreeTokenAPI's native attachment support. It is tested with **Pi 0.85.1**, uses only Node.js built-ins, and does not modify Pi's installed package.

Install from this project directory:

```sh
node ./pi-extension/install.mjs
```

The installer adds this project's extension path to your user-level Pi settings, preserving existing providers and settings. **Restart Pi** (and restart FreeTokenAPI after updating its code), then select a **Qwen or DeepSeek Web model under `free-token-api`** (`deepseek-web` / `deepseek-web-thinking`). Start a fresh conversation for the first test: old PDF text already sent in a previous conversation is not rewritten.

In Pi's editor, send an explicit attachment reference with your question:

```text
Summarize the main conclusions in this PDF:
@"C:\Users\Administrator\Desktop\report.pdf"
```

Replace the example with an existing file. Attach or detach documents while Pi is idle; wait for the current reply to finish or stop it first. Quoted Windows paths, spaces, Unicode filenames, multiple `@` references, and standalone absolute paths produced by terminal drag-and-drop are supported. Paths merely mentioned in prose or code examples are not automatically uploaded. Clipboard file objects themselves are not intercepted; paste a file path or use `/attach`.

| Pi command | Effect |
| --- | --- |
| `/attach "path/to/report.pdf"` | Queue a native attachment, then send your question; attaching alone does not call the model |
| `/attach "first.pdf" "second.pdf"` | Queue multiple documents |
| `/attachments` | List active documents and their IDs |
| `/attachments remove <id-prefix>` | Detach one document from future requests |
| `/attachments clear` | Detach all native documents |

CLI entry points also work (the following are PowerShell examples, not commands to type inside Pi's editor):

```powershell
pi --provider free-token-api --model qwen3.8-max '@C:\Users\Administrator\Desktop\report.pdf' 'Summarize this PDF'
pi --provider free-token-api --model qwen3.8-max --attach 'C:\Users\Administrator\Desktop\report.pdf' 'Summarize this PDF'
pi --provider free-token-api --model deepseek-web-thinking '@C:\Users\Administrator\Desktop\report.pdf' 'Summarize this PDF'
```

`--attach` accepts one file and avoids Pi's initial `@file` text expansion entirely. For ordinary CLI `@PDF`, the extension replaces Pi's exact expanded file block **before it is stored in the conversation or sent to the model**. It does not extract PDF text or depend on a `read` tool call. PDF bytes become `file`, `input_file`, or `document` parts for the selected protocol.

Automatic references cover PDF, DOC/DOCX, XLS/XLSX, PPT/PPTX, ODT/ODS/ODP and RTF. `/attach` also accepts TXT, Markdown, CSV and JSON as explicit documents. **PDF transport has been live-tested with the Qwen and DeepSeek web backends; version 2 model discovery and capability routing have offline regression coverage.** Ordinary image attachments and text/code references keep Pi's original behavior. DeepSeek image/document availability comes from the selected account's official model configuration, not a fixed Flash/Pro/Vision rule. Other providers are not enabled by this extension.

If the model nevertheless tries the local `read` tool on an already attached binary document, the extension blocks that decoding attempt and points it back to the native attachment. Normal text/image reads are unchanged.

The extension keeps immutable local snapshots in `~/.pi/agent/cache/freetokenapi-native-files` (or under `PI_CODING_AGENT_DIR`). Pi session entries contain only attachment metadata/markers, not decoded PDF text or Base64 blobs. Active documents remain available during follow-ups, tool turns, compaction and session restoration. The client caps active documents at 50 files and 100 MiB combined, with a 100 MiB individual-file cap; upstream limits may be lower.

**Snapshots contain the original document contents.** Keep that directory private. Detaching files or uninstalling the extension does not delete snapshots or files already uploaded to the web provider. A missing/corrupt snapshot stops the request with a clear error instead of silently sending an incomplete prompt. Reattach the document or clear the attachments to continue.

Check or remove the registration without deleting document data:

```sh
node ./pi-extension/install.mjs --check
node ./pi-extension/install.mjs --uninstall
```

Before moving this project directory, unregister the extension. If it has already moved, remove the old extension path from Pi settings before installing from the new location. Test the extension with `node --test tests/pi-native-files.test.mjs`; these checks are also included in the Python regression suite.

## Images and other attachments

Qwen attachments are uploaded to the same web backend used by the browser: temporary upload grant → object storage → document parsing where needed → file references in the chat request. The adapter does not merely insert a filename into the prompt.

DeepSeek Web uses its web file-upload endpoint with PoW, waits for file parsing when necessary, and passes `ref_file_ids` to generation. File types, upload limits, vision support, and attachment/search combinations are checked against the account's cached official capabilities before upload.

DeepSeek parsing is asynchronous. The adapter polls at the web UI's three-second cadence and uses a separate **300-second budget for the entire attachment batch**, including uploads, parsing and retries (`FREETOKENAPI_FILE_PARSE_TIMEOUT_SECONDS`). The HTTP timeout remains independent. A `FAILED` state is not necessarily a bad document: the official web client labels parser code `50404` and related codes as **server busy**. Such failures return HTTP 503 with the original parser code/retry flag, rather than a misleading HTTP 400. Only an explicitly retryable temporary `FAILED` state is re-uploaded, at most once, with unchanged bytes/model/thinking and a fresh PoW. Content rejection, cancellation, empty/oversized content and known permanent parsing errors are never automatically retried. The original deadline is not reset, and generation never receives a failed file reference. If the provider remains busy, retry later; no client-side fix can guarantee upstream availability.

Supported inline request forms:

| API | Image content part | File content part |
| --- | --- | --- |
| Chat Completions | `image_url` containing a base64 data URI | `file` with `file.filename` and `file.file_data` |
| Responses | `input_image` with a base64 `image_url` | `input_file` with `filename` and `file_data` |
| Messages | `image` with a base64 `source` | `document` with a base64 source, or a `text/plain` text source |

For example, a Chat Completions file part looks like:

```json
{
  "type": "file",
  "file": {
    "filename": "notes.txt",
    "file_data": "data:text/plain;base64,SGVsbG8="
  }
}
```

- Images, text documents, PDFs, and other file types remain subject to the selected web model's support and upload/parse limits. A Qwen model being multimodal does not guarantee support for every file format.
- Inline base64 data is required. Remote image/file URLs, official provider `file_id` values, and local filesystem paths are not fetched or read by the server.
- The adapter caps requests at 50 files and 100 MiB per file; the upstream may impose lower limits.
- Repeated Qwen uploads of identical content can reuse a short-lived, account-local file reference. Upload credentials are never sent to clients or stored in conversation records.
- Pi's standard model input declaration remains `text` / `image`. Install the native-files extension above for PDF and document attachments; do not add an unsupported `file` input type to Pi's model configuration.
- Qwen chat attachments are separate from the `/v1/images/generations` image-creation endpoint.

## Native web search

To make search available to Pi without adding a custom request field on every turn, set this in `.env` and restart:

```dotenv
FREETOKENAPI_SEARCH_ENABLED=1
```

Ask naturally, for example: “Search the official Python website for the current stable release and cite the source.”

The adapter forwards the web providers' native switches (`search_enabled` for DeepSeek and `auto_search` for Qwen), and adds a short capability hint so native search is not confused with client function tools. **Search runs on the provider's backend; this project does not run a search engine or execute a hosted search tool.**

When enabled, native search has standing authorization: the model may use it proactively whenever it helps with the task, including fact-checking, documentation, comparisons, background research, and current information. It does not need an explicit search request or per-search approval, and may refine queries or search again as needed. Explicit no-search instructions and privacy constraints still apply; client-side tool permissions are unchanged.

- A request's explicit `search: true` or `search: false` overrides the service default.
- The configuration fallback is off when the variable is absent. The example above opts in.
- Enabling search allows the backend to use it; it does not force a fresh search on every request, particularly when relevant results already exist in the web conversation.
- Search support, attachment/search combinations, quotas, and permissions depend on the model and account. Explicit search requests are not silently disabled based on a local model alias.
- This is **not** OpenAI/Anthropic hosted-tool protocol support. No official hosted-search events, exact citation schema, or exhaustive source coverage are promised.
- For sensitive conversations that must not use web search, disable the setting or send `search: false`.

## Agent tool execution and completion

Pi and Claude Code execute local tools themselves; the web backend cannot modify your local files merely by describing an action. FreeTokenAPI converts model output into structured client tool calls and forwards the actual tool results on the next request.

- Each request refreshes the **current tool schemas and tool choice**, including when reusing a web conversation. A tool result is an intermediate step, not an instruction to end the whole task.
- Reused conversations receive only the newest tool-result batch, rather than repeatedly replaying all prior results. Historical calls use a parseable format instead of prose the model might copy as a new call.
- Known bracketed call forms seen in Claude Code are recognized only with declared tool names and complete JSON arguments. Unknown names, quoted examples and truncated arguments are not guessed into executable calls.
- Interrupted or failed upstream streams produce an explicit error, not a successful `stop` / `end_turn`. Buffered tool calls from an incomplete response are not dispatched.
- There is **no unlimited automatic ‘continue’ loop**, fabricated tool execution, or permission bypass. The model should continue the requested work and verification, but must still stop for genuine blockers, denied permissions, or an explicit user stop. Native attachments and native search remain separate from local file tools.

After updating, **restart FreeTokenAPI and start a fresh client conversation** if the old one contains repeated fake tool-call prose or premature completion claims. Reinstalling the Pi attachment extension is not required for this server-side fix. The root endpoint advertises `agent_tool_continuation` when the updated process is running.

## API reference

| Method | Path | Purpose |
| --- | --- | --- |
| GET / HEAD | `/`, `/v1`, `/v1/` | Service identity and reachability |
| GET | `/v1/models` | Model list |
| POST | `/v1/chat/completions` | OpenAI Chat Completions |
| POST | `/v1/responses` | OpenAI Responses |
| GET / DELETE | `/v1/responses/{response_id}` | Retrieve/delete an in-memory response |
| POST | `/v1/messages` | Anthropic Messages |
| POST | `/v1/messages/count_tokens` | Approximate input-token count |
| POST | `/v1/images/generations` | Qwen image generation |
| GET | `/health`, `/v1/usage` | Account status and optional server usage statistics |
| POST | `/v1/tokens` | Validate/add accounts and update the local token configuration |

Client tools execute **in the client**, not on this server. Function/custom tools and Responses namespace groups are adapted through prompts and output parsing; strict schemas, reasoning signatures, and all official tool semantics are not guaranteed.

Responses history is process-local and can expire or be evicted. Restarting invalidates old `previous_response_id` values. Web-conversation reuse is a separate, content-based mechanism; interleaved system/developer instructions intentionally disable reuse to preserve history ordering.

## Configuration, privacy, and PoW

See [config.py](freetokenapi/config.py) for all settings. Common options:

| Variable | Fallback | Purpose |
| --- | --- | --- |
| `FREETOKENAPI_HOST` | `127.0.0.1` | Listen address; keep it on loopback |
| `FREETOKENAPI_PORT` | `8000` | Listen port |
| `FREETOKENAPI_TIMEOUT` | `60` | Upstream HTTP timeout in seconds, not a whole-generation deadline |
| `FREETOKENAPI_FILE_PARSE_TIMEOUT_SECONDS` | `300` | Total DeepSeek attachment-preparation budget, including bounded retries |
| `FREETOKENAPI_MODEL_CONFIG_TTL_SECONDS` | `300` | Per-account DeepSeek capability cache TTL; `0` disables successful-result caching |
| `FREETOKENAPI_SEARCH_ENABLED` | `0` | Default native-search switch; explicit request values win |
| `FREETOKENAPI_CACHE_DIR` | System temp directory / `freetokenapi` | Persistent session/context and usage files |
| `FREETOKENAPI_CACHE_DISABLED` | `0` | Set to `1` to disable persistent session caching |
| `FREETOKENAPI_USAGE_ENABLED` | `1` | Set to `0` to disable the server's separate usage ledger |
| `FREETOKENAPI_LOG_LEVEL` | `INFO` | Log level |
| `FREETOKENAPI_LOG_FILE` | Empty | Optional log file; console-only by default |

To minimize disk persistence:

```dotenv
FREETOKENAPI_CACHE_DISABLED=1
FREETOKENAPI_USAGE_ENABLED=0
```

This does not remove existing files or disable the in-memory Responses store. Caches may contain conversation content. Do not share `.env`, caches, or unchecked logs.

**There is no client API-key authentication or multi-user isolation**, including on the token-management endpoint. Run only as a personal service on a trusted local machine. Do not expose it to your LAN or the internet.

**PoW is not token accounting.** DeepSeek requires a proof-of-work result before accepting certain requests. [pow.py](freetokenapi/pow.py), the JavaScript solver, and the WASM resource are runtime components, not disposable caches. The C solver is an optional acceleration path; normal operation uses Node/WASM without compiling C.

PoW is generated immediately before each DeepSeek upload/completion, not prefetched and cached across user turns: an unused proof can expire while Pi is idle. An upload rejected specifically for an invalid proof is retried once with a fresh proof.

Pi's usage display reads the API's `usage` fields. Those fields and necessary token estimates remain available even when the separate server ledger is disabled. Estimated usage is not an official tokenizer, billing record, or context-capacity measurement.

## Troubleshooting

- **DeepSeek attachment `FAILED`:** read the parser error code and retry flag. Codes such as `50404` indicate provider-side load, not necessarily an invalid PDF. Restart after updating and check for `deepseek_file_parse_recovery` at the root endpoint. Persistent server-busy errors require waiting; increase the separate preparation timeout only for legitimately slow parsing.
- **Pi / Claude Code stops after one tool or prints a call without executing it:** restart the updated server and check for `agent_tool_continuation` at the root endpoint. Start a fresh conversation, keep the needed client tools enabled, and inspect permission denials or upstream errors rather than disabling safeguards.
- **Pi reports `422 status code (no body)`:** restart an old server process after updating. This version accepts Pi's developer role and common thinking formats and returns OpenAI-shaped validation errors. Read the resulting error message rather than disabling validation.
- **Images disappear:** verify the Pi model entry includes `image`, select a compatible model, and send a valid supported image. Tiny, corrupt, or unsupported files can still be rejected upstream.
- **Search is unavailable:** check the service setting, restart after editing `.env`, and distinguish backend native search from client/hosted tools. A model's answer alone is not proof that a fresh search occurred.
- **No valid credentials / 401:** verify the account still works in the web app and refresh the web login token.
- **Node.js not found:** install Node.js LTS and reopen the terminal before using DeepSeek.
- **403 / 429 / timeout:** check upstream availability, permissions, network/proxy settings, concurrency, and rate limits. Do not fix a loopback/proxy problem by exposing the service on `0.0.0.0`.
- **Context-limit or old response-ID errors:** start a new client conversation or provide the full appropriate history. Do not advertise 1M capacity or enable unsupported remote compaction.

## Development

The project runs from source; `pyproject.toml` contains test/lint configuration, not a complete packaging definition. Runtime dependencies are in [requirements.txt](requirements.txt).

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m ruff check .
.venv/bin/python -B -m pytest -q -p no:cacheprovider
```

Use `.venv\Scripts\python.exe` on Windows. Most tests mock the upstream and never need real login tokens. Protocol tests cover Pi-style requests, streaming, tool results, attachment upload/parsing, and search switches. A passing suite is not a guarantee for every provider account, future web API change, or client extension.

CI runs the offline suite on Windows, macOS, and Linux with Node.js 24, including a Python 3.10 compatibility job. It does not receive web login tokens or run the opt-in live smoke scripts. `.env`, virtual environments, local agent settings, caches, and logs are excluded from Git; only the blank `.env.example` is published.

### Verification scope

Version 2.0.1 reproduced a retryable parser-busy error with a generated PDF, then verified successful parsing of a generated 12-page PDF and a real native-file question returning a marker contained only in the document. No user document was uploaded for those checks. The bounded retry and non-retry conditions have offline coverage through all three API formats; provider availability and document-specific parser support still apply.

Version 2 adds account-specific model-catalog, TTL/failure recovery, legacy-ID rejection and capability-routing tests. The parser/cache were also checked with one anonymous, read-only request to the current official model configuration. Actual Pi CLI PDF/tool round-trips passed through all three protocols against a mock backend using the new model ID. Earlier live CLI results below describe the transport/tool layer; they are not a claim that version 2 was live-tested with a configured account.

Pi 0.85.1's actual adapters were exercised against real DeepSeek and Qwen web backends through all three protocols for image questions and native web search. Qwen PDF uploads through all three protocols and text-file upload/parsing were also checked. Tool round-trips, error envelopes, and usage parsing have SDK-level mocked checks and Python regressions.

Agent-loop regression tests additionally cover multiple consecutive tool calls through all three protocols, schema refresh, disabled tools, legacy call notation, JSON/CDATA file contents, and interrupted streams. Live file-editing tasks passed with Claude Code 2.1.266 (Messages, DeepSeek and Qwen) and Pi 0.85.1 (Chat Completions and Responses, both providers), without a manual “continue”. The final Pi Messages live runs were blocked by DeepSeek rate limiting and Qwen web verification; those are not recorded as live passes. Its offline protocol-loop tests passed. Live tests use isolated client profiles and generated files, but still consume web-provider quotas; stop testing when a provider requests verification.

The macOS/Linux launcher has POSIX-shell syntax and isolated setup-path tests. Native macOS/Linux end-to-end startup still needs verification on those operating systems.

## License

[MIT](LICENSE). Preserve the existing copyright and permission notice when redistributing source or derivatives.
