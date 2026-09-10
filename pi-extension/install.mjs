#!/usr/bin/env node
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const extension = fileURLToPath(new URL("./index.ts", import.meta.url));
const agentDir = process.env.PI_CODING_AGENT_DIR || path.join(os.homedir(), ".pi", "agent");
const settingsPath = path.join(agentDir, "settings.json");
const uninstall = process.argv.includes("--uninstall");
const check = process.argv.includes("--check");
try {
  let original = "";
  try { original = await fs.readFile(settingsPath, "utf8"); }
  catch (error) { if (error.code !== "ENOENT") throw error; }
  const settings = original ? JSON.parse(original.replace(/^\uFEFF/, "")) : {};
  if (!settings || typeof settings !== "object" || Array.isArray(settings)) throw new Error("Pi settings must be a JSON object.");
  const extensions = settings.extensions ?? [];
  if (!Array.isArray(extensions) || extensions.some(value => typeof value !== "string")) throw new Error("Existing Pi extensions configuration is not a string array; it was not changed.");
  const normalized = value => {
    if (value.startsWith("~/")) value = path.join(os.homedir(), value.slice(2));
    const result = path.resolve(agentDir, value);
    return process.platform === "win32" ? result.toLowerCase() : result;
  };
  const equal = value => normalized(value) === normalized(extension);
  const installed = extensions.some(equal);
  if (check) {
    console.log(JSON.stringify({ installed, extension, settingsPath }));
  } else {
    settings.extensions = uninstall ? extensions.filter(value => !equal(value)) : installed ? extensions : [...extensions, extension];
    await fs.mkdir(agentDir, { recursive: true });
    let current = "";
    try { current = await fs.readFile(settingsPath, "utf8"); } catch (error) { if (error.code !== "ENOENT") throw error; }
    if (current !== original) throw new Error("Pi settings changed concurrently; retry the installer.");
    const temporary = settingsPath + `.native-files-${process.pid}.tmp`;
    try {
      await fs.writeFile(temporary, JSON.stringify(settings, null, 2) + "\n", { flag: "wx", mode: 0o600 });
      await fs.rename(temporary, settingsPath);
    } finally { await fs.unlink(temporary).catch(() => {}); }
    console.log(uninstall ? "Native files extension unregistered; cached documents were not deleted." : "Native files extension registered. Restart Pi or use /reload; select free-token-api / Qwen or DeepSeek Web.");
    console.log(settingsPath);
  }
} catch (error) {
  console.error(`Pi extension configuration failed: ${error.message}`);
  process.exitCode = 1;
}
