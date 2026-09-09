// Explicit live smoke test. Uses only a generated PDF and a loopback FreeTokenAPI.
import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const option = name => { const index = process.argv.indexOf(name); return index < 0 ? undefined : process.argv[index + 1]; };
const piRoot = option("--pi-root") || process.env.PI_PACKAGE_DIR;
const modelId = option("--model") || "qwen3.8-max";
const thinking = option("--thinking") || "off";
if (!piRoot) throw new Error("Supply --pi-root with the installed pi-coding-agent package directory.");
const upstream = new URL(option("--api-base") || "http://127.0.0.1:8000");
assert.ok(["127.0.0.1", "localhost", "[::1]"].includes(upstream.hostname));
const extension = fileURLToPath(new URL("../pi-extension/index.ts", import.meta.url));
const root = await fs.mkdtemp(path.join(os.tmpdir(), "freetokenapi-pi-cli-"));
const agent = path.join(root, "agent"); await fs.mkdir(agent);
const pdfPath = path.join(root, "文档 sample.pdf");
const marker = "NATIVE-PDF-583";
const stream = Buffer.from(`BT /F1 18 Tf 20 80 Td (Check marker: ${marker}) Tj ET`);
const objects = [
  "<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
  "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 360 120] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
  "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", `<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`,
];
let pdf = Buffer.from("%PDF-1.4\n"), offsets = [];
for (let i = 0; i < objects.length; i++) { offsets.push(pdf.length); pdf = Buffer.concat([pdf, Buffer.from(`${i + 1} 0 obj\n${objects[i]}\nendobj\n`)]); }
const xref = pdf.length;
pdf = Buffer.concat([pdf, Buffer.from(`xref\n0 6\n0000000000 65535 f \n${offsets.map(value => `${String(value).padStart(10, "0")} 00000 n \n`).join("")}trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`)]);
await fs.writeFile(pdfPath, pdf);
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const captures = [];
const mockRead = process.argv.includes("--mock-read");
const allowRead = process.argv.includes("--allow-read") || mockRead;
const server = http.createServer(async (request, response) => {
  try {
    const chunks = []; for await (const chunk of request) chunks.push(chunk);
    const body = JSON.parse(Buffer.concat(chunks));
    const messages = Array.isArray(body.input) ? body.input : body.messages;
    const parts = (messages || []).flatMap(message => typeof message.content === "string" ? [{ type: "text", text: message.content }] : message.content || []);
    const files = parts.filter(part => ["file", "input_file", "document"].includes(part.type));
    assert.equal(files.length, 1, "Pi did not send exactly one native file part");
    const file = files[0];
    const encoded = file.type === "document" ? file.source.data : (file.type === "file" ? file.file.file_data : file.file_data).split(",")[1];
    assert.equal(sha(Buffer.from(encoded, "base64")), sha(pdf), "PDF bytes were changed");
    assert.ok(!parts.some(part => typeof part.text === "string" && (part.text.includes("%PDF-") || part.text.includes("FTA-FILE:"))), "Binary text/extension marker leaked into ordinary prompt text");
    captures.push({ path: request.url, fileType: file.type, bytes: pdf.length, hasTools: (body.tools?.length || 0) > 0 });
    if (process.argv.includes("--diagnostics")) console.error(JSON.stringify({ request: captures.length, toolChoice: body.tool_choice,
      roles: (messages || []).map(message => message.role || message.type),
      lastTool: (messages || []).filter(message => message.role === "tool" || message.type === "function_call_output").slice(-1).map(message => String(message.content || message.output).slice(0, 500)) }));
    if (captures.length > 3 && process.argv.includes("--diagnostics")) throw new Error("Diagnostic turn limit reached");
    if (mockRead) {
      const python = String.raw`import os,sys,json
os.environ.update({'PYTHON_DOTENV_DISABLED':'1','DEEPSEEK_TOKENS':'','QWEN_TOKENS':'','FREETOKENAPI_LOG_FILE':'','FREETOKENAPI_CACHE_DISABLED':'1','FREETOKENAPI_USAGE_ENABLED':'0'})
from fastapi.testclient import TestClient
from fastapi.responses import StreamingResponse
from freetokenapi.api import openai as api
incoming=json.load(sys.stdin)
async def fake(req):
    done=any(m.role=='tool' for m in req.messages)
    if done:
        assert any(m.role=='tool' and 'already provided as a native file attachment' in str(m.content) for m in req.messages)
    async def events():
        common={'id':'chatcmpl-native-doc-test','object':'chat.completion.chunk','created':0,'model':req.model}
        delta={'role':'assistant','content':incoming['marker']} if done else {'role':'assistant','tool_calls':[{'index':0,'id':'call_read_pdf','type':'function','function':{'name':'read','arguments':json.dumps({'path':incoming['pdfPath']})}}]}
        yield 'data: '+json.dumps({**common,'choices':[{'index':0,'delta':delta,'finish_reason':None}]})+'\n\n'
        yield 'data: '+json.dumps({**common,'choices':[{'index':0,'delta':{},'finish_reason':'stop' if done else 'tool_calls'}],'usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12}})+'\n\n'
        yield 'data: [DONE]\n\n'
    return StreamingResponse(events(),media_type='text/event-stream')
api._chat_completions_qwen=fake
api._chat_completions_deepseek=fake
r=TestClient(api.app).post(incoming['path'],json=incoming['body'])
print(json.dumps({'status':r.status_code,'headers':dict(r.headers),'body':r.text}))
`;
      const workspace = fileURLToPath(new URL("../", import.meta.url));
      const executable = process.platform === "win32" ? path.join(workspace, ".venv", "Scripts", "python.exe") : path.join(workspace, ".venv", "bin", "python");
      const processResult = spawnSync(executable, ["-B", "-X", "utf8", "-c", python], { cwd: workspace, input: JSON.stringify({ path: request.url, body, pdfPath, marker }), encoding: "utf8", windowsHide: true });
      assert.equal(processResult.status, 0, processResult.stderr);
      const result = JSON.parse(processResult.stdout.trim());
      response.writeHead(result.status, { "content-type": result.headers["content-type"] });
      response.end(result.body);
      return;
    }
    if (process.argv.includes("--current-code")) {
      const workspace = fileURLToPath(new URL("../", import.meta.url));
      const executable = process.platform === "win32" ? path.join(workspace, ".venv", "Scripts", "python.exe") : path.join(workspace, ".venv", "bin", "python");
      const bridge = path.join(workspace, "tests", "_live_deepseek_bridge.py");
      const output = spawnSync(executable, ["-B", "-X", "utf8", bridge], { cwd: workspace, input: JSON.stringify({ path: request.url, body }), encoding: "utf8", windowsHide: true, timeout: 180000 });
      assert.equal(output.status, 0, output.stderr);
      const result = JSON.parse(output.stdout.trim());
      console.error(JSON.stringify({ currentCodeStatus: result.status, trace: result.trace }));
      response.writeHead(result.status, { "content-type": result.headers["content-type"] });
      response.end(result.body);
      return;
    }
    const result = await fetch(new URL(request.url, upstream), { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal: AbortSignal.timeout(180000) });
    if (process.argv.includes("--diagnostics") && result.status >= 500) {
      const error = await result.text();
      console.error(JSON.stringify({ upstreamStatus: result.status, detail: error.slice(0, 800) }));
      response.writeHead(400, { "content-type": "application/json" });
      response.end(JSON.stringify({ error: { message: "Diagnostic stop; see the relay log." } }));
      return;
    }
    response.writeHead(result.status, { "content-type": result.headers.get("content-type") || "application/json" });
    let raw = "";
    for await (const chunk of result.body) { response.write(chunk); if (process.argv.includes("--diagnostics")) raw += Buffer.from(chunk).toString("utf8"); }
    if (process.argv.includes("--diagnostics")) {
      const summaries = [];
      if (result.status >= 400) {
        try { const data = JSON.parse(raw); summaries.push({ error: String(data.error?.message || data.detail || "Upstream failure").slice(0, 600) }); }
        catch { summaries.push({ error: "Non-JSON upstream failure" }); }
      }
      for (const line of raw.split("\n")) if (line.startsWith("data:")) {
        try {
          const event = JSON.parse(line.slice(5));
          if (event.error) summaries.push({ error: event.error });
          for (const choice of event.choices || []) {
            if (choice.delta?.tool_calls) summaries.push({ tools: choice.delta.tool_calls });
            if (choice.finish_reason) summaries.push({ finish: choice.finish_reason });
          }
        } catch {}
      }
      console.error(JSON.stringify({ responseStatus: result.status, summaries }));
    }
    response.end();
  } catch (error) {
    console.error(`Relay guard/error: ${error.message}`);
    response.writeHead(400, { "content-type": "application/json" });
    response.end(JSON.stringify({ error: { message: `Native-file test guard: ${error.message}` } }));
  }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const port = server.address().port;
await fs.writeFile(path.join(agent, "settings.json"), JSON.stringify({ extensions: [extension], defaultProvider: "free-token-api", defaultModel: modelId }));

function run(args) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [path.join(piRoot, "dist", "bundle", "cli.js"), ...args], {
      cwd: root, env: { ...process.env, PI_CODING_AGENT_DIR: agent, FREETOKENAPI_PI_CACHE_DIR: path.join(root, "cache"), NO_COLOR: "1" },
      stdio: ["ignore", "pipe", "pipe"], windowsHide: true,
    });
    let stdout = "", stderr = "";
    child.stdout.setEncoding("utf8"); child.stderr.setEncoding("utf8");
    child.stdout.on("data", data => { stdout += data; }); child.stderr.on("data", data => { stderr += data; });
    const timeout = setTimeout(() => { console.error(JSON.stringify({ timeout: true, captures, stdout: stdout.slice(-1500), stderr: stderr.slice(-1500) })); child.kill(); reject(new Error("Pi CLI smoke test timed out")); }, 210000);
    child.on("error", error => { clearTimeout(timeout); reject(error); });
    child.on("close", code => { clearTimeout(timeout); resolve({ code, stdout, stderr }); });
  });
}

try {
  const entryModes = process.argv.includes("--all-entries") ? ["cli-at", "inline-at", "flag"] : ["cli-at"];
  for (const api of (option("--api") ? [option("--api")] : ["openai-completions", "openai-responses", "anthropic-messages"])) {
    await fs.writeFile(path.join(agent, "models.json"), JSON.stringify({ providers: { "free-token-api": {
      api, apiKey: "local", baseUrl: `http://127.0.0.1:${port}${api === "anthropic-messages" ? "" : "/v1"}`,
      models: [{ id: modelId, name: "Native file test", reasoning: true, input: modelId.startsWith("deepseek-v4-flash") ? ["text"] : ["text", "image"], contextWindow: 65536, maxTokens: 4096 }],
    } } }));
    for (const entry of entryModes) {
      captures.length = 0;
      const question = "Do not search the web. Read the attached PDF and return only its exact check marker.";
      const args = ["-p", "--provider", "free-token-api", "--model", modelId, "--no-session", "--no-tools", "--no-context-files", "--no-skills", "--no-prompt-templates", "--thinking", thinking];
      if (allowRead) args.splice(args.indexOf("--no-tools"), 1, "--tools", "read");
      if (entry === "cli-at") args.push("@" + pdfPath, question);
      else if (entry === "inline-at") args.push(`${question}\n@"${pdfPath}"`);
      else args.push("--attach", pdfPath, question);
      const result = await run(args);
      console.log(JSON.stringify({ api, model: modelId, entry, exitCode: result.code, nativeRequests: captures, markerRead: result.stdout.includes(marker), diagnostics: result.code ? result.stderr.slice(-2000) : undefined }));
      assert.equal(result.code, 0, result.stderr);
      assert.ok(captures.length > 0 && (allowRead || captures.every(item => !item.hasTools)), "A read tool must not be required");
      assert.ok(result.stdout.includes(marker), result.stdout + result.stderr);
    }
  }
  if (process.argv.includes("--all-entries")) {
    const sessions = path.join(root, "sessions");
    const args = ["-p", "--provider", "free-token-api", "--model", modelId, "--session-dir", sessions,
      "--no-tools", "--no-context-files", "--no-skills", "--no-prompt-templates", "--thinking", thinking];
    captures.length = 0;
    const first = await run([...args, "@" + pdfPath, "Do not browse. Return only the exact check marker in the PDF."]);
    assert.equal(first.code, 0, first.stderr); assert.ok(first.stdout.includes(marker));
    const sessionFiles = (await fs.readdir(sessions)).filter(name => name.endsWith(".jsonl"));
    assert.equal(sessionFiles.length, 1);
    const transcript = await fs.readFile(path.join(sessions, sessionFiles[0]), "utf8");
    assert.ok(transcript.includes("freetokenapi-native-files-v1"), "Attachment metadata was not persisted");
    assert.ok(!transcript.includes("%PDF-") && !transcript.includes(pdf.toString("base64")), "PDF binary content leaked into Pi's session transcript");
    assert.ok(path.resolve(pdfPath).startsWith(path.resolve(root) + path.sep));
    await fs.rename(pdfPath, pdfPath + ".moved");
    captures.length = 0;
    try {
      const resumed = await run([...args, "-c", "Do not browse. What was the exact check marker in the previously attached PDF? Reply only the marker."]);
      console.log(JSON.stringify({ api: "anthropic-messages", entry: "resume-without-original-file", exitCode: resumed.code, nativeRequests: captures, markerRead: resumed.stdout.includes(marker) }));
      assert.equal(resumed.code, 0, resumed.stderr);
      assert.ok(captures.length > 0 && resumed.stdout.includes(marker), resumed.stdout + resumed.stderr);
    } finally { await fs.rename(pdfPath + ".moved", pdfPath); }
  }
} finally {
  server.closeAllConnections(); await new Promise(resolve => server.close(resolve));
  console.error(`Isolated Pi test profile (no real credentials): ${root}`);
}
