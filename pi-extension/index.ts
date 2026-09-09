/** Pi 0.85+ native documents for the local FreeTokenAPI provider. */
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import {
  ENTRY_TYPE, MARKER_RE, NativeFileStore, attachmentReferences, commandPaths,
  isAdapterModel, isNativeDocument, marker, nativeFileSupportError, supportsNativeFiles,
} from "./native-files-core.mjs";

export default function nativeFiles(pi: ExtensionAPI) {
  const store = new NativeFileStore();
  let sessionId: string | undefined;
  let preparing = false;
  let startupFlagConsumed = false;
  let initialInputSeen = false;
  const cliFiles = process.argv.slice(2).filter(value => value.startsWith("@")).map(value => value.slice(1));

  function notify(ctx: ExtensionContext, text: string, error = false) {
    if (ctx.hasUI) ctx.ui.notify(text, error ? "error" : "info");
    else if (error) {
      console.error(`Native attachments: ${text}`);
      if (ctx.mode === "print" || ctx.mode === "json") process.exitCode = 1;
    }
  }

  function status(ctx: ExtensionContext) {
    ctx.ui.setStatus("freetokenapi-native-files", store.active.size ? `Native documents: ${store.active.size}` : undefined);
  }

  function restore(ctx: ExtensionContext) {
    sessionId = ctx.sessionManager.getSessionId();
    store.restore(ctx.sessionManager.getBranch());
    status(ctx);
  }

  function ensureSession(ctx: ExtensionContext) {
    if (sessionId !== ctx.sessionManager.getSessionId()) restore(ctx);
  }

  function requireTarget(ctx: ExtensionContext) {
    if (!supportsNativeFiles(ctx.model)) throw new Error(nativeFileSupportError(ctx.model));
  }

  function commit(files: any[], ctx: ExtensionContext) {
    const added = files.filter(file => !store.active.has(file.id));
    store.activate(files);
    for (const file of added) pi.appendEntry(ENTRY_TYPE, { action: "attach", file });
    store.pruneBuffers();
    status(ctx);
  }

  function failure(ctx: ExtensionContext, error: unknown) {
    const message = error instanceof Error ? error.message : "Native attachment processing failed.";
    notify(ctx, message, true);
    store.pruneBuffers();
  }

  pi.registerFlag("attach", { type: "string", description: "Attach one native document without Pi's @file text expansion; also supply a prompt." });

  pi.on("session_start", async (_event, ctx) => restore(ctx));
  pi.on("session_switch", async (_event, ctx) => restore(ctx));
  pi.on("session_fork", async (_event, ctx) => restore(ctx));
  pi.on("session_tree", async (_event, ctx) => restore(ctx));

  pi.on("input", async (event, ctx) => {
    ensureSession(ctx);
    if (!isAdapterModel(ctx.model)) return;
    const flag = !startupFlagConsumed && event.source !== "extension" ? pi.getFlag("attach") : undefined;
    if (flag !== undefined) startupFlagConsumed = true;
    const allowedCliFiles = !initialInputSeen && event.source !== "extension" ? cliFiles : [];
    if (event.source !== "extension") initialInputSeen = true;
    const legacy = allowedCliFiles.length > 0 && [...event.text.matchAll(/<file name="([^"\r\n]+)">\r?\n/g)].some(match => isNativeDocument(match[1]));
    const hasNewFiles = typeof flag === "string" || (event.source !== "extension" && (legacy || attachmentReferences(event.text).length > 0));
    if (!hasNewFiles && store.active.size === 0 && !MARKER_RE.test(event.text)) return;
    MARKER_RE.lastIndex = 0;
    if (preparing) { notify(ctx, "Wait for the current attachment preparation to finish.", true); return { action: "handled" }; }
    preparing = true;
    try {
      requireTarget(ctx);
      if (hasNewFiles && !ctx.isIdle()) throw new Error("Wait for the current reply to finish (or stop it) before attaching documents.");
      let result = event.source === "extension" ? { text: event.text, files: [] } : await store.prepareInput(event.text, ctx.cwd, { cliFiles: allowedCliFiles });
      if (typeof flag === "string") {
        const staged = await store.stage(flag, ctx.cwd, true);
        result = { text: `${marker(staged.metadata.id)}\n${result.text}`, files: [...result.files, staged.metadata] };
      }
      for (const match of result.text.matchAll(MARKER_RE)) {
        if (!store.known.has(match[1]) && !store.ignoreMissingMarkers) throw new Error("Attachment metadata is missing. Reattach the file or use /attachments clear.");
      }
      // Validate cached documents before accepting a new user turn. The provider
      // hook repeats validation and aborts if a snapshot disappears on resume.
      store.validateBatch(result.files);
      await store.preflight();
      commit(result.files, ctx);
      if (result.files.length) notify(ctx, `Attached ${result.files.map(file => file.filename).join(", ")} as native documents.`);
      if (result.text !== event.text) return { action: "transform", text: result.text };
    } catch (error) {
      failure(ctx, error);
      return { action: "handled" };
    } finally { preparing = false; }
  });

  function payloadHasMarkers(value: unknown) {
    if (!value || typeof value !== "object") return false;
    const body = value as any;
    const pattern = new RegExp(MARKER_RE.source);
    if (typeof body.input === "string" && pattern.test(body.input)) return true;
    const messages = Array.isArray(body.input) ? body.input : body.messages;
    return Array.isArray(messages) && messages.some((message: any) => message?.role === "user" && (
      typeof message.content === "string" ? pattern.test(message.content)
        : Array.isArray(message.content) && message.content.some((part: any) => typeof part?.text === "string" && pattern.test(part.text))
    ));
  }

  pi.on("before_provider_request", async (event, ctx) => {
    ensureSession(ctx);
    if (store.known.size === 0 && !store.ignoreMissingMarkers && !payloadHasMarkers(event.payload)) return;
    const api = ctx.model?.api;
    if (!["openai-completions", "openai-responses", "anthropic-messages"].includes(api || "")) return;
    try {
      if (isAdapterModel(ctx.model) && store.active.size && !supportsNativeFiles(ctx.model)) requireTarget(ctx);
      return await store.payload(event.payload, api, { enabled: supportsNativeFiles(ctx.model) });
    } catch (error) {
      // Pi catches hook exceptions and would otherwise send the old payload.
      // Explicit cancellation + an invalid empty request prevent that fallback.
      failure(ctx, error);
      try { ctx.abort(); } catch { /* An already-aborted run is safe. */ }
      return { model: "" };
    }
  });

  pi.on("tool_call", async (event, ctx) => {
    ensureSession(ctx);
    if (event.toolName !== "read" || !supportsNativeFiles(ctx.model)) return;
    const value = (event.input as { path?: unknown }).path;
    if (typeof value !== "string" || !isNativeDocument(value)) return;
    const filename = value.replaceAll("\\", "/").split("/").at(-1)?.replace(/^@/, "");
    const normalize = (name: string) => process.platform === "win32" ? name.toLowerCase() : name;
    if (filename && [...store.active.values()].some(file => normalize(file.filename) === normalize(filename))) {
      return { block: true, reason: "This document is already provided as a native file attachment. Inspect its attached contents directly; the local read tool would incorrectly decode binary data as text." };
    }
  });

  pi.registerCommand("attach", {
    description: 'Attach native documents: /attach "C:/path/report.pdf" (multiple quoted paths allowed)',
    handler: async (args, ctx) => {
      ensureSession(ctx);
      if (preparing) { notify(ctx, "Attachment preparation is already running.", true); return; }
      preparing = true;
      try {
        requireTarget(ctx);
        if (!ctx.isIdle()) throw new Error("Wait for the current reply to finish before attaching documents.");
        const paths = commandPaths(args);
        if (!paths.length) throw new Error('Usage: /attach "path/to/report.pdf"');
        const staged = [];
        for (const name of paths) staged.push((await store.stage(name, ctx.cwd, true)).metadata);
        const files = [...new Map(staged.map(file => [file.id, file])).values()];
        await store.preflight();
        commit(files, ctx);
        notify(ctx, `Attached ${files.map(file => file.filename).join(", ")}. Send your question when ready.`);
      } catch (error) { failure(ctx, error); }
      finally { preparing = false; }
    },
  });

  pi.registerCommand("attachments", {
    description: "List native documents, /attachments remove <id-prefix>, or /attachments clear",
    handler: async (args, ctx) => {
      ensureSession(ctx);
      const value = args.trim();
      try {
        if (value === "clear" || value.startsWith("remove ")) {
          if (!ctx.isIdle() || preparing) throw new Error("Wait for the current reply/preparation to finish before detaching files.");
          if (value === "clear") {
            pi.appendEntry(ENTRY_TYPE, { action: "clear" });
            store.active.clear(); store.ignoreMissingMarkers = true;
          } else {
            const prefix = value.slice(7).trim();
            const found = [...store.active.values()].filter(file => prefix && file.id.startsWith(prefix));
            if (found.length !== 1) throw new Error("Use a unique attachment ID prefix from /attachments.");
            pi.appendEntry(ENTRY_TYPE, { action: "remove", id: found[0].id });
            store.active.delete(found[0].id);
          }
          store.pruneBuffers(); status(ctx);
          notify(ctx, "Detached. Local snapshots and files already uploaded to the web provider are not deleted.");
          return;
        }
        if (value) throw new Error("Usage: /attachments, /attachments remove <id-prefix>, or /attachments clear");
        const list = [...store.active.values()].map(file => `${file.id.slice(0, 8)}  ${file.filename}  (${Math.ceil(file.size / 1024)} KiB)`);
        const text = list.length ? list.join("\n") : "No active native documents.";
        if (ctx.hasUI) await ctx.ui.select(text, ["OK"]);
        else console.error(text);
      } catch (error) { failure(ctx, error); }
    },
  });
}
