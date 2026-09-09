// Opt-in CLI regression: only generated files, isolated client profiles, loopback API.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";

const option = name => { const i = process.argv.indexOf(name); return i < 0 ? undefined : process.argv[i + 1]; };
const client = option("--client") || "claude";
const api = option("--api") || "anthropic-messages";
const model = option("--model") || "deepseek-v4-flash-thinking";
const base = new URL(option("--api-base"));
assert.ok(["127.0.0.1", "localhost", "[::1]"].includes(base.hostname) && ["http:", "https:"].includes(base.protocol));
assert.ok(["claude", "pi"].includes(client));
const root = await fs.mkdtemp(path.join(await fs.realpath(os.tmpdir()), "fta-cli-loop-"));
const profile = path.join(root, "profile"); await fs.mkdir(profile);
const nonce = randomUUID();
await fs.writeFile(path.join(root, "source.txt"), "nonce=" + nonce + "\nstage=pending\n");
const prompt = "In this working directory (" + root.replaceAll("\\", "/") + "), read source.txt, "
  + "use the edit tool to change stage=pending to stage=complete without changing its nonce, "
  + "and write report.json with exactly the stage and nonce values from that file. "
  + "Then read both files back to verify the saved results. Use relative file paths for tool calls. Do not use the network or touch files outside this working directory.";
let executable, args;
const env = { ...process.env, NO_COLOR: "1" };
if (client === "claude") {
  executable = option("--claude-exe"); assert.ok(executable, "Supply --claude-exe");
  Object.assign(env, { CLAUDE_CONFIG_DIR: profile, ANTHROPIC_BASE_URL: base.origin, ANTHROPIC_API_KEY: "local",
    ANTHROPIC_AUTH_TOKEN: "", CLAUDE_CODE_OAUTH_TOKEN: "", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: "1",
    DISABLE_TELEMETRY: "1", DISABLE_AUTOUPDATER: "1", DISABLE_ERROR_REPORTING: "1" });
  args = ["-p", "--model", model, "--output-format", "stream-json", "--verbose", "--no-session-persistence",
    "--safe-mode", "--restricted", "--strict-mcp-config", "--tools", "Read,Write,Edit", "--allowedTools", "Read,Write,Edit",
    "--permission-mode", "acceptEdits", "--permission-prompts", "none", "--prompt-suggestions", "false", prompt];
} else {
  const piRoot = option("--pi-root"); assert.ok(piRoot, "Supply --pi-root");
  executable = process.execPath;
  await fs.writeFile(path.join(profile, "settings.json"), JSON.stringify({ extensions: [] }));
  await fs.writeFile(path.join(profile, "models.json"), JSON.stringify({ providers: { "free-token-api": {
    api, apiKey: "local", baseUrl: base.origin + (api === "anthropic-messages" ? "" : "/v1"),
    models: [{ id: model, name: "Agent loop test", reasoning: true, input: ["text"], contextWindow: 65536, maxTokens: 4096 }],
  } } }));
  Object.assign(env, { PI_CODING_AGENT_DIR: profile, FREETOKENAPI_PI_CACHE_DIR: path.join(root, "cache") });
  args = [path.join(piRoot, "dist", "bundle", "cli.js"), "-p", "--mode", "json", "--provider", "free-token-api", "--model", model,
    "--no-session", "--no-context-files", "--no-skills", "--no-prompt-templates", "--tools", "read,write,edit", "--thinking", "medium", prompt];
}
const before = await fetch(new URL("/__live_test_trace", base)).then(r => r.json());
const result = await new Promise((resolve, reject) => {
  const child = spawn(executable, args, { cwd: root, env, stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
  let stdout = "", stderr = "";
  child.stdout.setEncoding("utf8"); child.stderr.setEncoding("utf8");
  child.stdout.on("data", chunk => stdout += chunk); child.stderr.on("data", chunk => stderr += chunk);
  const timer = setTimeout(async () => {
    await fs.writeFile(path.join(root, "cli-output.jsonl"), stdout).catch(() => {});
    child.kill(); reject(new Error("Isolated CLI test timed out; diagnostics: " + root));
  }, 300000);
  child.on("error", error => { clearTimeout(timer); reject(error); });
  child.on("close", code => { clearTimeout(timer); resolve({ code, stdout, stderr }); });
});
const rows = result.stdout.split(/\r?\n/).filter(Boolean).flatMap(line => { try { return [JSON.parse(line)]; } catch { return []; } });
const tools = rows.flatMap(row => {
  const message = client === "claude" ? (row.type === "assistant" ? row.message : undefined) : (row.type === "message_end" ? row.message : undefined);
  return (message?.content || []).filter(part => ["tool_use", "toolCall"].includes(part.type)).map(part => part.name);
});
const stage = await fs.readFile(path.join(root, "source.txt"), "utf8");
let report; try { report = JSON.parse(await fs.readFile(path.join(root, "report.json"), "utf8")); } catch {}
const after = await fetch(new URL("/__live_test_trace", base)).then(r => r.json());
const verified = stage === "nonce=" + nonce + "\nstage=complete\n" && report?.stage === "complete" && report?.nonce === nonce;
if (!verified) await fs.writeFile(path.join(root, "cli-output.jsonl"), result.stdout);
console.log(JSON.stringify({ client, api, model, exitCode: result.code, tools, filesVerified: verified,
  upstreamRequests: after.upstream_requests - before.upstream_requests, webConversations: after.web_conversations - before.web_conversations,
  workspace: root, diagnostics: verified ? undefined : result.stdout.slice(-2500) + result.stderr.slice(-1000) }));
assert.equal(result.code, 0, result.stderr);
assert.ok(verified, "CLI stopped before applying and saving the requested changes");
assert.ok(["read", "write", "edit"].every(name => tools.some(tool => tool.toLowerCase() === name)), "Required local tools were not executed");
assert.ok(tools.filter(name => name.toLowerCase() === "read").length >= 3, "CLI skipped the requested read-back verification");
