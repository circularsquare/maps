// Memory readings for the memory checks (t2802-memory.mjs, t3160-phone-start.mjs): Chrome's private bytes by process
// type, and each JS context's heap, ArrayBuffer backing stores and WASM memories over a raw CDP connection.
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const run = promisify(execFile);
export const MB = 1048576;
export const mb = (x) => Math.round(x / MB);

/** private bytes, MB, of the Chrome whose command line holds `profile`: all processes, the renderer (the tab), the GPU */
export async function chromeMemory(profile) {
  const needle = profile.replace(/'/g, "''");
  const ps = `Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Where-Object { $_.CommandLine -like '*${needle}*' } | ForEach-Object { $t = if ($_.CommandLine -match '--type=(\\S+)') { $matches[1] } else { 'browser' }; '{0} {1}' -f $t, $_.PrivatePageCount }`;
  const { stdout } = await run('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', ps], { windowsHide: true });
  const by = {};
  let total = 0;
  for (const l of stdout.trim().split(/\r?\n/).filter(Boolean)) {
    const [type, priv] = l.trim().split(' ');
    by[type] = (by[type] ?? 0) + mb(+priv);
    total += mb(+priv);
  }
  return { total, renderer: by.renderer ?? 0, gpu: by['gpu-process'] ?? 0 };
}

/** a raw CDP connection to the browser, so a worker's session can be spoken to (Playwright's CDPSession can't) */
export async function rawCdp(port) {
  let v;
  for (let k = 0; k < 50 && !v; k++) v = await fetch(`http://127.0.0.1:${port}/json/version`).then((r) => r.json()).catch(() => new Promise((r) => setTimeout(r, 200)));
  const ws = new WebSocket(v.webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  let id = 0;
  const wait = new Map();
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    const w = m.id && wait.get(m.id);
    if (!w) return;
    wait.delete(m.id);
    if (m.error) w.j(new Error(m.error.message)); else w.r(m.result);
  };
  const send = (method, params = {}, sessionId) => new Promise((r, j) => {
    const i = ++id;
    wait.set(i, { r, j });
    ws.send(JSON.stringify({ id: i, method, params, ...(sessionId ? { sessionId } : {}) }));
    setTimeout(() => { if (wait.delete(i)) j(new Error('timeout ' + method)); }, 15000);
  });
  return { send, close: () => ws.close(), sessions: new Map() };
}

const shortName = (url) => url.replace(/^.*\/(.*?)(\?.*)?$/, '$1').replace(/-[A-Za-z0-9_-]{8}\.js$/, '.js') || url;

/** a context's WASM memories, MB (they are not in its backing stores): every WebAssembly.Memory it holds, by
 * queryObjects, which collects garbage first, so only at the stages */
async function wasmOf(cdp, s) {
  try {
    const { result: proto } = await cdp.send('Runtime.evaluate', { expression: 'WebAssembly.Memory.prototype' }, s);
    const { objects } = await cdp.send('Runtime.queryObjects', { prototypeObjectId: proto.objectId }, s);
    const { result } = await cdp.send('Runtime.callFunctionOn', { objectId: objects.objectId, functionDeclaration: 'function () { return this.reduce((a, m) => a + m.buffer.byteLength, 0); }', returnByValue: true }, s);
    return mb(result.value);
  } catch { return -1; }
}

/** each context's heap: [name, used MB, ArrayBuffer backing MB(, WASM MB with `withWasm`)], pages only under `base` */
export async function contexts(cdp, base, withWasm = false) {
  const { targetInfos } = await cdp.send('Target.getTargets', { filter: [{}] });
  const out = [];
  for (const t of targetInfos) {
    if (t.type !== 'page' && t.type !== 'worker') continue;
    if (t.type === 'page' && !t.url.startsWith(base)) continue;
    try {
      let s = cdp.sessions.get(t.targetId);
      if (!s) {
        s = (await cdp.send('Target.attachToTarget', { targetId: t.targetId, flatten: true })).sessionId;
        cdp.sessions.set(t.targetId, s);
      }
      const h = await cdp.send('Runtime.getHeapUsage', {}, s);
      const row = [t.type === 'page' ? 'page' : shortName(t.url), mb(h.usedSize), mb(h.backingStorageSize ?? 0)];
      if (withWasm) row.push(await wasmOf(cdp, s));
      out.push(row);
    } catch { /* a worker that just ended */ }
  }
  return out.sort((a, b) => b[1] + b[2] - a[1] - a[2]);
}
