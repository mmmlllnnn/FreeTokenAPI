import { createHash, randomUUID } from "node:crypto";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const ENTRY_TYPE = "freetokenapi-native-files-v1";
export const NATIVE_DOCUMENT_NOTICE = "These documents are supplied as native file attachments. Inspect their attached contents directly; do not use the local read tool to decode these binary documents as text.";
export const MARKER_RE = /\[\[FTA-FILE:([0-9a-f-]{36})\]\]/g;
export const DOCUMENT_TYPES = Object.freeze({
  ".pdf": "application/pdf",
  ".doc": "application/msword",
  ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  ".xls": "application/vnd.ms-excel",
  ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  ".ppt": "application/vnd.ms-powerpoint",
  ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  ".odt": "application/vnd.oasis.opendocument.text",
  ".ods": "application/vnd.oasis.opendocument.spreadsheet",
  ".odp": "application/vnd.oasis.opendocument.presentation",
  ".rtf": "application/rtf",
});
const TEXT_TYPES = Object.freeze({ ".txt": "text/plain", ".md": "text/markdown", ".csv": "text/csv", ".json": "application/json" });
const APIS = new Set(["openai-completions", "openai-responses", "anthropic-messages"]);
const MIME_TYPES = new Set([...Object.values(DOCUMENT_TYPES), ...Object.values(TEXT_TYPES)]);
export const digest = bytes => createHash("sha256").update(bytes).digest("hex");
export const marker = id => `[[FTA-FILE:${id}]]`;
export const isNativeDocument = name => Object.hasOwn(DOCUMENT_TYPES, path.extname(name).toLowerCase());

export function defaultCacheDir() {
  const agent = process.env.PI_CODING_AGENT_DIR || path.join(os.homedir(), ".pi", "agent");
  return process.env.FREETOKENAPI_PI_CACHE_DIR || path.join(agent, "cache", "freetokenapi-native-files");
}

export function isAdapterModel(model) {
  if (!model || String(model.provider).toLowerCase().replace(/[^a-z0-9]/g, "") !== "freetokenapi") return false;
  try {
    const url = new URL(model.baseUrl);
    return ["http:", "https:"].includes(url.protocol) && ["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)
      && !url.username && !url.password && APIS.has(model.api);
  } catch { return false; }
}

export function supportsNativeFiles(model) {
  return isAdapterModel(model) && (
    /^qwen/i.test(model.id) || /^deepseek-v4-flash(?:-thinking)?$/.test(model.id)
  );
}

export function nativeFileSupportError(model) {
  if (!isAdapterModel(model)) return "Select a Qwen or DeepSeek Flash model under the local free-token-api provider to attach documents.";
  if (/^deepseek-v4-vision(?:-thinking)?$/.test(model.id)) {
    return "DeepSeek Vision accepts images, not PDF/documents. Use DeepSeek Flash or Qwen for documents; clear active documents before switching to Vision for images.";
  }
  if (/^deepseek-v4-pro(?:-thinking)?$/.test(model.id)) {
    return "DeepSeek Pro file attachments are not supported by this adapter. Use DeepSeek Flash or Qwen for documents.";
  }
  return "Native documents are supported by Qwen and DeepSeek Flash in this adapter. Select a supported model.";
}

export function resolveLocalPath(value, cwd) {
  let name = value.trim();
  if ((name.startsWith('"') && name.endsWith('"')) || (name.startsWith("'") && name.endsWith("'"))) name = name.slice(1, -1);
  if (/^file:/i.test(name)) {
    const url = new URL(name);
    if (url.hostname && url.hostname !== "localhost") throw new Error("Network file URLs are not supported.");
    name = fileURLToPath(url);
  } else if (/^[a-z][a-z0-9+.-]*:\/\//i.test(name)) throw new Error("Attach a local file, not a remote URL.");
  if (/^(?:\\\\|\/\/)/.test(name)) throw new Error("Network-share paths are not supported.");
  if (name === "~" || /^[~][\\/]/.test(name)) name = path.join(os.homedir(), name.slice(2));
  if (/^[a-z]:[\\/]/i.test(name) && process.platform !== "win32") throw new Error("Use a local path for the OS running Pi (for WSL, use /mnt/c/...).");
  return path.resolve(cwd, name);
}

// Preserve Windows backslashes; this is deliberately not a shell evaluator.
export function commandPaths(text) {
  const result = [];
  let i = 0;
  while (i < text.length) {
    while (/\s/.test(text[i] || "") && i < text.length) i++;
    if (i >= text.length) break;
    let token = "";
    if (text[i] === "@") i++;
    const quote = text[i] === '"' || text[i] === "'" ? text[i++] : null;
    if (quote) {
      const end = text.indexOf(quote, i);
      if (end < 0) throw new Error("Unclosed quote in file path.");
      token = text.slice(i, end); i = end + 1;
    } else {
      const start = i;
      while (i < text.length && !/\s/.test(text[i])) i++;
      token = text.slice(start, i);
    }
    if (token.startsWith("@")) token = token.slice(1);
    if (token) result.push(token);
  }
  return result;
}

export function attachmentReferences(text) {
  const refs = [];
  // @path, @"path with spaces", and "@path with spaces".
  const expression = /(^|[\s(])(?:@"([^"\r\n]+)"|@'([^'\r\n]+)'|"@([^"\r\n]+)"|'@([^'\r\n]+)'|@([^\s"']+))/gm;
  for (const match of text.matchAll(expression)) {
    let name = match.slice(2).find(Boolean);
    let end = match.index + match[0].length;
    if (match[6] && name && !isNativeDocument(name)) {
      const trimmed = name.replace(/[.,;:!?，。；：！？)\]}]+$/u, "");
      if (isNativeDocument(trimmed)) { end -= name.length - trimmed.length; name = trimmed; }
    }
    if (name && isNativeDocument(name)) refs.push({ start: match.index + match[1].length, end, path: name });
  }
  // A terminal drop is usually a standalone absolute path, not an image/file object.
  let offset = 0;
  for (const line of text.split(/(?<=\n)/)) {
    const trimmed = line.trim();
    const unquoted = trimmed.replace(/^(["'])(.*)\1$/, "$2");
    if (isNativeDocument(unquoted) && (/^[a-z]:[\\/]/i.test(unquoted) || unquoted.startsWith("/") || unquoted.startsWith("~/") || unquoted.startsWith("file:"))) {
      const start = offset + line.indexOf(trimmed), end = start + trimmed.length;
      if (!refs.some(ref => start < ref.end && end > ref.start)) refs.push({ start, end, path: unquoted });
    }
    offset += line.length;
  }
  const code = [...text.matchAll(/```[\s\S]*?```|`[^`\n]*`/g)].map(match => ({ start: match.index, end: match.index + match[0].length }));
  return refs.filter(ref => !code.some(range => ref.start < range.end && ref.end > range.start)).sort((a, b) => a.start - b.start);
}

function metadataValid(meta, maxFileBytes) {
  return meta && typeof meta === "object" && /^[0-9a-f-]{36}$/.test(meta.id)
    && /^[0-9a-f]{64}$/.test(meta.sha256) && MIME_TYPES.has(meta.mime)
    && Number.isSafeInteger(meta.size) && meta.size > 0 && meta.size <= maxFileBytes
    && typeof meta.filename === "string" && meta.filename.length > 0
    && !/[\\/\x00-\x1f]/.test(meta.filename) && ![".", ".."].includes(meta.filename);
}

export class NativeFileStore {
  constructor({ cacheDir = defaultCacheDir(), maxFileBytes = 100 * 1024 * 1024, maxTotalBytes = 100 * 1024 * 1024, maxFiles = 50 } = {}) {
    this.cacheDir = path.resolve(cacheDir);
    this.maxFileBytes = maxFileBytes; this.maxTotalBytes = maxTotalBytes; this.maxFiles = maxFiles;
    this.known = new Map(); this.active = new Map(); this.buffers = new Map(); this.ignoreMissingMarkers = false;
  }

  restore(entries) {
    this.known.clear(); this.active.clear(); this.buffers.clear(); this.ignoreMissingMarkers = false;
    for (const entry of entries) {
      if (entry.type !== "custom" || entry.customType !== ENTRY_TYPE) continue;
      const data = entry.data;
      if (data?.action === "clear") { this.active.clear(); this.ignoreMissingMarkers = true; }
      else if (data?.action === "remove") this.active.delete(data.id);
      else if (data?.action === "attach" && metadataValid(data.file, this.maxFileBytes)) {
        this.known.set(data.file.id, { ...data.file }); this.active.set(data.file.id, { ...data.file });
      }
    }
  }

  async stage(value, cwd, explicit = false) {
    const absolute = resolveLocalPath(value, cwd);
    const ext = path.extname(absolute).toLowerCase();
    const mime = DOCUMENT_TYPES[ext] || (explicit ? TEXT_TYPES[ext] : undefined);
    if (!mime) throw new Error("Use native image attachment for pictures; /attach supports PDF, Office documents and text documents.");
    const filename = path.basename(absolute);
    if (/[\x00-\x1f]/.test(filename)) throw new Error("Invalid attachment filename.");
    let handle;
    try { handle = await fs.open(absolute, "r"); }
    catch { throw new Error(`Cannot open attachment: ${filename}. Check that the file exists and is readable.`); }
    let bytes;
    try {
      const stat = await handle.stat();
      if (!stat.isFile() || stat.size === 0) throw new Error(`Attachment is empty or not a regular file: ${filename}`);
      if (stat.size > this.maxFileBytes) throw new Error(`Attachment exceeds ${this.maxFileBytes / 1024 / 1024} MiB: ${filename}`);
      bytes = await handle.readFile();
      if (bytes.length > this.maxFileBytes) throw new Error(`Attachment grew beyond the size limit: ${filename}`);
    } finally { await handle.close(); }
    if (ext === ".pdf" && !bytes.subarray(0, 1024).includes(Buffer.from("%PDF-"))) throw new Error(`Not a PDF file: ${filename}`);
    const sha256 = digest(bytes);
    const existing = [...this.known.values()].find(file => file.sha256 === sha256 && file.filename === filename && file.mime === mime);
    const metadata = existing || { id: randomUUID(), filename, mime, size: bytes.length, sha256 };
    await fs.mkdir(this.cacheDir, { recursive: true, mode: 0o700 });
    const snapshot = path.join(this.cacheDir, sha256 + ".bin");
    const temporary = path.join(this.cacheDir, sha256 + "." + randomUUID() + ".tmp");
    try {
      await fs.writeFile(temporary, bytes, { flag: "wx", mode: 0o600 });
      await fs.rename(temporary, snapshot);
    } finally { await fs.unlink(temporary).catch(() => {}); }
    this.buffers.set(sha256, bytes);
    this.known.set(metadata.id, metadata);
    return { metadata, bytes };
  }

  validateBatch(files) {
    const combined = new Map(this.active);
    for (const file of files) combined.set(file.id, file);
    if (combined.size > this.maxFiles) throw new Error(`At most ${this.maxFiles} native documents can be active. Use /attachments clear first.`);
    const total = [...combined.values()].reduce((sum, file) => sum + file.size, 0);
    if (total > this.maxTotalBytes) throw new Error(`Active documents exceed ${this.maxTotalBytes / 1024 / 1024} MiB in total. Use /attachments clear first.`);
  }

  activate(files) {
    this.validateBatch(files);
    for (const file of files) { this.known.set(file.id, file); this.active.set(file.id, file); }
  }

  async load(file) {
    if (this.buffers.has(file.sha256)) return this.buffers.get(file.sha256);
    const snapshot = path.join(this.cacheDir, file.sha256 + ".bin");
    let bytes;
    try {
      const stat = await fs.lstat(snapshot);
      if (!stat.isFile() || stat.isSymbolicLink() || stat.size !== file.size) throw new Error("Invalid snapshot");
      bytes = await fs.readFile(snapshot);
    } catch { throw new Error(`Attachment snapshot is unavailable: ${file.filename}. Reattach it or use /attachments clear.`); }
    if (digest(bytes) !== file.sha256) throw new Error(`Attachment snapshot changed: ${file.filename}. Reattach it.`);
    this.buffers.set(file.sha256, bytes);
    return bytes;
  }

  pruneBuffers() {
    const keep = new Set([...this.active.values()].map(file => file.sha256));
    for (const hash of this.buffers.keys()) if (!keep.has(hash)) this.buffers.delete(hash);
  }

  async preflight() {
    this.validateBatch([]);
    for (const file of this.active.values()) await this.load(file);
  }

  async prepareInput(text, cwd, { cliFiles = [] } = {}) {
    const edits = [], staged = [], protectedRanges = [];
    const allowed = new Set(cliFiles.map(value => resolveLocalPath(value, cwd)));
    // References inside code are not user attachment instructions.
    for (const match of text.matchAll(/```[\s\S]*?```|`[^`\n]*`/g)) protectedRanges.push({ start: match.index, end: match.index + match[0].length, kind: "code" });
    for (const absolute of allowed) {
      if (isNativeDocument(absolute)) continue;
      const prefix = `<file name="${absolute}">`;
      const start = text.indexOf(prefix);
      if (start < 0) continue;
      const empty = prefix + "</file>\n";
      if (text.startsWith(empty, start)) { protectedRanges.push({ start, end: start + empty.length }); continue; }
      try {
        const bytes = await fs.readFile(absolute);
        const image = bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
          || bytes[0] === 255 && bytes[1] === 216 || bytes.subarray(0, 3).toString() === "GIF"
          || bytes.subarray(0, 4).toString() === "RIFF";
        if (image) {
          const end = text.indexOf("</file>\n", start + prefix.length);
          protectedRanges.push({ start, end: end < 0 ? text.length : end + "</file>\n".length });
          continue;
        }
        const wrapper = prefix + "\n" + bytes.toString("utf8").replace(/^\uFEFF/, "") + "\n</file>\n";
        protectedRanges.push({ start, end: text.startsWith(wrapper, start) ? start + wrapper.length : text.length });
      } catch { protectedRanges.push({ start, end: text.length }); }
    }
    // Pi's CLI expands @non-image files before input hooks. Match the exact
    // expansion against the selected file, then discard it BEFORE session storage.
    // No approximate regex stripping of PDF/binary content is performed.
    for (const match of text.matchAll(/<file name="([^"\r\n]+)">\r?\n/g)) {
      if (!isNativeDocument(match[1]) || !allowed.has(resolveLocalPath(match[1], cwd))) continue;
      if ([...edits, ...protectedRanges.filter(range => range.kind !== "code")].some(range => match.index >= range.start && match.index < range.end)) continue;
      const file = await this.stage(match[1], cwd);
      const decoded = file.bytes.toString("utf8").replace(/^\uFEFF/, "");
      const wrapper = `<file name="${match[1]}">\n${decoded}\n</file>\n`;
      if (!text.startsWith(wrapper, match.index)) throw new Error(`Pi's expanded file no longer matches ${file.metadata.filename}. Use /attach instead.`);
      edits.push({ start: match.index, end: match.index + wrapper.length, replacement: marker(file.metadata.id) + "\n" });
      staged.push(file.metadata);
    }
    for (const ref of attachmentReferences(text)) {
      if ([...edits, ...protectedRanges].some(edit => ref.start < edit.end && ref.end > edit.start)) continue;
      const file = await this.stage(ref.path, cwd);
      edits.push({ ...ref, replacement: marker(file.metadata.id) }); staged.push(file.metadata);
    }
    const unique = [...new Map(staged.map(file => [file.id, file])).values()];
    this.validateBatch(unique);
    let rewritten = text;
    for (const edit of edits.sort((a, b) => b.start - a.start)) rewritten = rewritten.slice(0, edit.start) + edit.replacement + rewritten.slice(edit.end);
    return { text: rewritten, files: unique };
  }

  async payload(payload, api, { enabled = true } = {}) {
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) throw new Error("Unexpected Pi provider request shape.");
    const result = structuredClone(payload);
    const key = api === "openai-responses" ? "input" : "messages";
    if (api === "openai-responses" && typeof result.input === "string") result.input = [{ role: "user", content: result.input }];
    if (!Array.isArray(result[key])) throw new Error("Unexpected Pi message array.");
    if (enabled) await this.preflight();
    const emitted = new Set();
    let firstUser;
    let noticeAdded = false;
    const notice = () => ({ type: api === "openai-responses" ? "input_text" : "text", text: NATIVE_DOCUMENT_NOTICE });
    const makePart = async file => {
      const encoded = (await this.load(file)).toString("base64");
      if (api === "openai-completions") return { type: "file", file: { filename: file.filename, file_data: `data:${file.mime};base64,${encoded}` } };
      if (api === "openai-responses") return { type: "input_file", filename: file.filename, file_data: `data:${file.mime};base64,${encoded}` };
      if (api === "anthropic-messages") return { type: "document", title: file.filename, source: { type: "base64", media_type: file.mime, data: encoded } };
      throw new Error("Unsupported Pi API for native documents.");
    };
    for (const message of result[key]) {
      if (message?.role !== "user") continue;
      if (!firstUser) firstUser = message;
      const parts = typeof message.content === "string" ? [{ type: api === "openai-responses" ? "input_text" : "text", text: message.content }] : message.content;
      if (!Array.isArray(parts)) continue;
      const add = [];
      for (const part of parts) {
        if (typeof part?.text !== "string") continue;
        const ids = [];
        part.text = part.text.replace(MARKER_RE, (_whole, id) => {
          const file = this.known.get(id);
          if (!file) {
            if (!enabled || this.ignoreMissingMarkers) return "[Native attachment unavailable]";
            throw new Error("Attachment metadata is missing from this session. Reattach the file or use /attachments clear.");
          }
          if (enabled && this.active.has(id)) { ids.push(id); return `[Attached document: ${file.filename}]`; }
          return `[Document not attached: ${file.filename}]`;
        });
        for (const id of ids) if (!emitted.has(id)) { add.push(await makePart(this.active.get(id))); emitted.add(id); }
      }
      if (add.length && !noticeAdded) { parts.push(notice()); noticeAdded = true; }
      message.content = [...parts, ...add];
    }
    // Compaction may remove textual markers. Keep explicitly active documents
    // available until the user detaches them, without changing question order.
    if (enabled && [...this.active.keys()].some(id => !emitted.has(id))) {
      if (!firstUser) {
        firstUser = { role: "user", content: [{ type: api === "openai-responses" ? "input_text" : "text", text: "Previously attached documents are available below." }] };
        result[key].unshift(firstUser);
      }
      if (!Array.isArray(firstUser.content)) firstUser.content = [{ type: api === "openai-responses" ? "input_text" : "text", text: firstUser.content || "" }];
      if (!noticeAdded) { firstUser.content.push(notice()); noticeAdded = true; }
      for (const file of this.active.values()) if (!emitted.has(file.id)) firstUser.content.push(await makePart(file));
    }
    return result;
  }
}
