import test from "node:test";
import assert from "node:assert/strict";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  ENTRY_TYPE, NativeFileStore, attachmentReferences, commandPaths, digest,
  isAdapterModel, marker, nativeFileSupportError, resolveLocalPath, supportsNativeFiles,
} from "../pi-extension/native-files-core.mjs";

const MODEL = { provider: "free-token-api", id: "qwen3.8-max", api: "openai-responses", baseUrl: "http://127.0.0.1:8000/v1" };
const PDF = Buffer.concat([Buffer.from("%PDF-1.4\n"), Buffer.from([255, 0, 254]), Buffer.from("\n</file>\nnot UTF-8\n%%EOF\n")]);

async function fixture(t, options = {}) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "fta-pi-files-test-"));
  t.after(async () => {
    const full = path.resolve(root), prefix = path.resolve(os.tmpdir()) + path.sep;
    assert.ok(full.startsWith(prefix) && path.basename(full).startsWith("fta-pi-files-test-"));
    await fs.rm(full, { recursive: true, force: true });
  });
  const file = path.join(root, "报告 with spaces.pdf");
  await fs.writeFile(file, PDF);
  return { root, file, store: new NativeFileStore({ cacheDir: path.join(root, "cache"), ...options }) };
}

function request(api, text) {
  return { model: "qwen3.8-max", ...(api === "openai-responses" ? { input: [{ role: "user", content: text }] } : { messages: [{ role: "user", content: text }] }) };
}
function fileParts(payload, api) {
  return (payload[api === "openai-responses" ? "input" : "messages"] || []).flatMap(message => Array.isArray(message.content) ? message.content : []).filter(part => ["file", "input_file", "document"].includes(part.type));
}
function partBytes(part) {
  const data = part.type === "document" ? part.source.data : part.type === "file" ? part.file.file_data.split(",")[1] : part.file_data.split(",")[1];
  return Buffer.from(data, "base64");
}

test("references preserve Windows backslashes and quoted spaces", () => {
  const text = 'Read @"C:\\Users\\Me\\report one.PDF" and "@C:/other file.docx".';
  assert.deepEqual(attachmentReferences(text).map(ref => ref.path), ["C:\\Users\\Me\\report one.PDF", "C:/other file.docx"]);
  assert.deepEqual(commandPaths('@"C:\\a b\\report.pdf" "second.pdf"'), ["C:\\a b\\report.pdf", "second.pdf"]);
  assert.throws(() => commandPaths('"unclosed'), /Unclosed/);
  assert.equal(attachmentReferences("mail@example.pdf @notes.md @photo.png").length, 0);
});

test("attachment punctuation stays in the question and code references stay literal", () => {
  const text = "请分析 @报告.pdf。";
  const refs = attachmentReferences(text);
  assert.equal(refs[0].path, "报告.pdf");
  assert.equal(text.slice(refs[0].end), "。");
  assert.equal(attachmentReferences("`@example.pdf`").length, 0);
  assert.equal(attachmentReferences("```\nC:/private.pdf\n```").length, 0);
});

test("standalone absolute drag paths are recognized, prose paths are not", () => {
  assert.equal(attachmentReferences('Question\n"C:\\files\\paper.pdf"\n').length, 1);
  assert.equal(attachmentReferences("Discuss the path C:/files/paper.pdf without opening it.").length, 0);
});

test("model scope enables Qwen and unified DeepSeek aliases, not legacy IDs", () => {
  assert.ok(supportsNativeFiles(MODEL));
  for (const id of ["deepseek-web", "deepseek-web-thinking"]) assert.ok(supportsNativeFiles({ ...MODEL, id }));
  for (const id of ["deepseek-v4-flash", "deepseek-v4-flash-thinking", "deepseek-v4-pro", "deepseek-v4-pro-thinking", "deepseek-v4-vision", "deepseek-v4-vision-thinking", "deepseek-unknown"]) {
    assert.equal(supportsNativeFiles({ ...MODEL, id }), false);
  }
  assert.match(nativeFileSupportError({ ...MODEL, id: "deepseek-v4-vision-thinking" }), /old model IDs/);
  assert.match(nativeFileSupportError({ ...MODEL, id: "deepseek-v4-pro-thinking" }), /old model IDs/);
  for (const changed of [{ provider: "openrouter" }, { baseUrl: "https://example.com/v1" }, { id: "deepseek-v4-vision-thinking" }, { api: "google-generative-ai" }]) {
    assert.equal(supportsNativeFiles({ ...MODEL, ...changed }), false);
  }
  assert.equal(isAdapterModel({ ...MODEL, baseUrl: "http://user:secret@localhost:8000/v1" }), false);
});

test("remote/UNC paths are rejected rather than fetched", () => {
  for (const value of ["https://example.com/a.pdf", "file://remote-host/a.pdf", "\\\\server\\share\\a.pdf"]) {
    assert.throws(() => resolveLocalPath(value, process.cwd()), /remote|Network/i);
  }
});

test("@PDF transforms to metadata marker without decoding PDF into context", async t => {
  const { file, root, store } = await fixture(t);
  const result = await store.prepareInput(`Compare @"${file}" carefully.`, root);
  assert.equal(result.files.length, 1);
  assert.ok(result.text.startsWith("Compare [[FTA-FILE:"));
  assert.ok(result.text.endsWith(" carefully."));
  assert.ok(!result.text.includes("%PDF-") && !result.text.includes("�"));
  assert.equal(JSON.stringify(result.files).includes(root), false);
  store.activate(result.files);
  assert.deepEqual(await store.load(result.files[0]), PDF);
});

test("CLI @PDF text expansion is matched exactly, including embedded closing tags", async t => {
  const { file, root, store } = await fixture(t);
  const expanded = `<file name="${file}">\n${PDF.toString("utf8")}\n</file>\nWhat is this?`;
  const result = await store.prepareInput(expanded, root, { cliFiles: [file] });
  assert.equal(result.files.length, 1);
  assert.equal(result.text, marker(result.files[0].id) + "\nWhat is this?");
  await fs.writeFile(file, Buffer.from("%PDF-changed"));
  await assert.rejects(store.prepareInput(expanded, root, { cliFiles: [file] }), /no longer matches/);
});

test("nested file tags inside a selected PDF do not read arbitrary paths", async t => {
  const { file, root, store } = await fixture(t);
  const hidden = path.join(root, "not-selected.pdf");
  const bytes = Buffer.from(`%PDF-1.4\n<file name="${hidden}">\n@${hidden}\n</file>\n`);
  await fs.writeFile(file, bytes);
  const expanded = `<file name="${file}">\n${bytes.toString("utf8")}\n</file>\nQuestion`;
  const result = await store.prepareInput(expanded, root, { cliFiles: [file] });
  assert.equal(result.files.length, 1);
});

test("ordinary CLI text-file contents and fenced examples do not authorize uploads", async t => {
  const { root, store } = await fixture(t);
  const textFile = path.join(root, "example.md");
  const content = "Example: @private.pdf\n<file name=\"secret.pdf\">\nfake\n</file>";
  await fs.writeFile(textFile, content);
  const text = `<file name="${textFile}">\n${content}\n</file>\nExplain the example.\n\`\`\`\n@other.pdf\n\`\`\``;
  const result = await store.prepareInput(text, root, { cliFiles: [textFile] });
  assert.deepEqual(result.files, []);
  assert.equal(result.text, text);
});

test("duplicate references share one attachment identity", async t => {
  const { root, file, store } = await fixture(t);
  const result = await store.prepareInput(`@"${file}" @"${file}"`, root);
  assert.equal(result.files.length, 1);
});

for (const api of ["openai-completions", "openai-responses", "anthropic-messages"]) {
  test(`${api}: raw bytes, message order and user question survive`, async t => {
    const { file, root, store } = await fixture(t);
    const input = await store.prepareInput(`Review @"${file}".`, root);
    store.activate(input.files);
    const original = request(api, input.text);
    const result = await store.payload(original, api);
    assert.deepEqual(partBytes(fileParts(result, api)[0]), PDF);
    assert.equal(fileParts(result, api).length, 1);
    assert.equal(fileParts(original, api).length, 0);
    assert.ok(JSON.stringify(result).includes("Review"));
    assert.ok(!JSON.stringify(result).includes("FTA-FILE:"));
  });
  test(`${api}: follow-up/compaction retains explicitly active documents`, async t => {
    const { file, root, store } = await fixture(t);
    store.activate([(await store.stage(file, root)).metadata]);
    const result = await store.payload(request(api, "Now summarize the conclusions."), api);
    assert.equal(fileParts(result, api).length, 1);
    assert.ok(JSON.stringify(result).includes("Now summarize"));
  });
}

test("resume/fork restores metadata and immutable snapshots, not changed source", async t => {
  const { file, root, store } = await fixture(t);
  const staged = await store.stage(file, root); store.activate([staged.metadata]);
  await fs.writeFile(file, Buffer.from("%PDF-new version"));
  const resumed = new NativeFileStore({ cacheDir: store.cacheDir });
  resumed.restore([{ type: "custom", customType: ENTRY_TYPE, data: { action: "attach", file: staged.metadata } }]);
  assert.deepEqual(await resumed.load(staged.metadata), PDF);
  resumed.restore([]);
  assert.equal(resumed.active.size, 0);
});

test("missing/tampered snapshot fails instead of sending an incomplete prompt", async t => {
  const { file, root, store } = await fixture(t);
  const staged = await store.stage(file, root); store.activate([staged.metadata]);
  store.buffers.clear();
  await fs.writeFile(path.join(store.cacheDir, staged.metadata.sha256 + ".bin"), Buffer.alloc(PDF.length));
  await assert.rejects(store.payload(request("openai-responses", marker(staged.metadata.id)), "openai-responses"), /snapshot changed/);
});

test("detached documents and other providers never receive binary parts", async t => {
  const { file, root, store } = await fixture(t);
  const staged = await store.stage(file, root); store.activate([staged.metadata]);
  const text = marker(staged.metadata.id);
  assert.equal(fileParts(await store.payload(request("openai-responses", text), "openai-responses", { enabled: false }), "openai-responses").length, 0);
  store.restore([{ type: "custom", customType: ENTRY_TYPE, data: { action: "attach", file: staged.metadata } }, { type: "custom", customType: ENTRY_TYPE, data: { action: "clear" } }]);
  assert.equal(fileParts(await store.payload(request("openai-responses", text), "openai-responses"), "openai-responses").length, 0);
});

test("file size, aggregate size, missing and fake PDFs fail early", async t => {
  const { file, root, store } = await fixture(t, { maxFileBytes: 20 });
  await assert.rejects(store.stage(file, root), /exceeds/);
  await assert.rejects(store.stage("missing.pdf", root), /Cannot open/);
  const invalid = path.join(root, "fake.pdf"); await fs.writeFile(invalid, "not a PDF");
  await assert.rejects(store.stage(invalid, root), /Not a PDF/);
  const limited = new NativeFileStore({ cacheDir: path.join(root, "second-cache"), maxTotalBytes: 10 });
  const staged = await limited.stage(file, root);
  assert.throws(() => limited.activate([staged.metadata]), /in total/);
});

test("installer adds only its path and preserves unrelated configuration", async t => {
  const { root } = await fixture(t);
  const agent = path.join(root, "agent"); await fs.mkdir(agent);
  const settings = { defaultProvider: "other", extensions: ["existing.ts"], arbitrary: { keep: true } };
  const config = path.join(agent, "settings.json"); await fs.writeFile(config, JSON.stringify(settings));
  const script = fileURLToPath(new URL("../pi-extension/install.mjs", import.meta.url));
  const run = args => spawnSync(process.execPath, [script, ...args], { env: { ...process.env, PI_CODING_AGENT_DIR: agent }, encoding: "utf8", windowsHide: true });
  assert.equal(run([]).status, 0); assert.equal(run([]).status, 0);
  const installed = JSON.parse(await fs.readFile(config, "utf8"));
  assert.equal(installed.extensions.length, 2); assert.equal(installed.defaultProvider, "other"); assert.deepEqual(installed.arbitrary, settings.arbitrary);
  assert.equal(run(["--uninstall"]).status, 0);
  assert.deepEqual(JSON.parse(await fs.readFile(config, "utf8")), settings);
});

test("Pi handlers preserve images, attach without read, restore, and abort on failures", { skip: Number(process.versions.node.split(".")[0]) < 23 }, async t => {
  const { root, file } = await fixture(t);
  const originalEnv = process.env.FREETOKENAPI_PI_CACHE_DIR;
  process.env.FREETOKENAPI_PI_CACHE_DIR = path.join(root, "extension-cache");
  t.after(() => { if (originalEnv === undefined) delete process.env.FREETOKENAPI_PI_CACHE_DIR; else process.env.FREETOKENAPI_PI_CACHE_DIR = originalEnv; });
  const { default: extension } = await import("../pi-extension/index.ts");
  const handlers = new Map(), commands = new Map(), entries = [], sent = [], notices = [];
  let aborted = false, id = "session-a";
  const ctx = { model: MODEL, cwd: root, hasUI: true, mode: "tui", isIdle: () => true, abort: () => { aborted = true; },
    sessionManager: { getSessionId: () => id, getBranch: () => entries },
    ui: { notify: text => notices.push(text), setStatus: () => {}, select: async () => "OK" } };
  const pi = { on: (name, fn) => handlers.set(name, fn), registerFlag: () => {}, getFlag: () => undefined,
    registerCommand: (name, command) => commands.set(name, command),
    appendEntry: (customType, data) => entries.push({ type: "custom", customType, data }), sendUserMessage: message => sent.push(message) };
  extension(pi);
  await handlers.get("session_start")({}, ctx);
  const images = [{ type: "image", data: "existing-image", mimeType: "image/png" }];
  const input = await handlers.get("input")({ text: `Review @"${file}"`, source: "interactive", images }, ctx);
  assert.equal(input.action, "transform"); assert.equal(input.images, undefined); assert.equal(images[0].data, "existing-image");
  assert.equal(entries.length, 1); assert.equal(JSON.stringify(entries).includes("%PDF"), false);
  const blockedRead = await handlers.get("tool_call")({ toolName: "read", input: { path: file } }, ctx);
  assert.equal(blockedRead.block, true);
  assert.equal(await handlers.get("tool_call")({ toolName: "read", input: { path: "photo.png" } }, ctx), undefined);
  assert.equal(await handlers.get("tool_call")({ toolName: "read", input: { path: "notes.txt" } }, ctx), undefined);
  const output = await handlers.get("before_provider_request")({ payload: request(MODEL.api, input.text) }, ctx);
  assert.deepEqual(partBytes(fileParts(output, MODEL.api)[0]), PDF);
  // Existing snapshots remain usable when switching between the two backends.
  ctx.model = { ...MODEL, id: "deepseek-web-thinking" };
  const flashPayload = await handlers.get("before_provider_request")({ payload: { ...request(MODEL.api, input.text), model: ctx.model.id } }, ctx);
  assert.deepEqual(partBytes(fileParts(flashPayload, MODEL.api)[0]), PDF);
  ctx.model = MODEL;
  await commands.get("attachments").handler("clear", ctx);
  await commands.get("attach").handler(`"${file}"`, ctx);
  assert.equal(sent.length, 0, "Attaching alone must not trigger a model request");
  const nextInput = await handlers.get("input")({ text: "Summarize the attached document.", source: "interactive" }, ctx);
  assert.notEqual(nextInput?.action, "handled");
  const nextPayload = await handlers.get("before_provider_request")({ payload: request(MODEL.api, "Summarize the attached document.") }, ctx);
  assert.equal(fileParts(nextPayload, MODEL.api).length, 1);
  // A restored session with missing bytes must cancel rather than relying on
  // throwing (Pi's provider hook runner catches and otherwise ignores errors).
  const metadata = entries.filter(e => e.data.action === "attach").at(-1).data.file;
  await fs.unlink(path.join(process.env.FREETOKENAPI_PI_CACHE_DIR, metadata.sha256 + ".bin"));
  await handlers.get("session_start")({}, ctx);
  const denied = await handlers.get("before_provider_request")({ payload: request(MODEL.api, input.text) }, ctx);
  assert.deepEqual(denied, { model: "" }); assert.equal(aborted, true); assert.ok(notices.some(message => message.includes("snapshot")));
  entries.length = 0; id = "session-b"; await handlers.get("session_switch")({}, ctx);
  assert.equal(await handlers.get("before_provider_request")({ payload: request(MODEL.api, "Hello") }, ctx), undefined);
  aborted = false;
  const lostMetadata = await handlers.get("before_provider_request")({ payload: request(MODEL.api, marker(metadata.id)) }, ctx);
  assert.deepEqual(lostMetadata, { model: "" }); assert.equal(aborted, true);
});
