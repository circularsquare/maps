// Opt-in, bounded diagnostics. A separate worker owns the log so a blocked page or clock can
// still be diagnosed after reload. Never send saves, account details or feedback automatically.
import type { EditOp } from "../workers/protocol";

declare const __TW_BUILD__: string | undefined;
let recorder: Worker | undefined;
let recording = false;
export const tracing = () => recording;
let phaseId = 0;
export function tracePhase<T>(name: string, run: () => T): T {
  if (!recording) return run();
  const id = ++phaseId, at = performance.now();
  trace("phase-start", { id, name });
  try { return run(); }
  finally { trace("phase-end", { id, name, ms: performance.now() - at }); }
}
export function trace(event: string, detail: unknown = null) {
  if (!recording) return;
  try { recorder?.postMessage({ event, detail }); } catch { /* diagnostics must not affect play */ }
}

export function editTrace(op: EditOp) {
  const out: Record<string, unknown> = { op: op.op };
  for (const [key, value] of Object.entries(op)) {
    if (typeof value === "number" || typeof value === "boolean") out[key] = value;
    else if (Array.isArray(value) && value.every((v) => typeof v === "number")) {
      out[key] = value.slice(0, 256);
      if (value.length > 256) out[key + "Count"] = value.length;
    }
  }
  return out;
}

export function attachClockTrace(worker: Worker) {
  if (!recorder) return;
  try {
    const channel = new MessageChannel();
    recorder.postMessage({ clock: channel.port1 }, [channel.port1]);
    worker.postMessage({ kind: "freezeTrace", port: channel.port2 }, [channel.port2]);
  } catch { /* diagnostics must not affect startup */ }
}

function start() {
  let enabled = false;
  try {
    const option = new URLSearchParams(location.search).get("diagnostics");
    if (option === "1" || option === "0") localStorage.setItem("tw-freeze-trace", option);
    enabled = localStorage.getItem("tw-freeze-trace") === "1";
  } catch { enabled = new URLSearchParams(location.search).get("diagnostics") === "1"; }
  if (!enabled) return;
  try { recorder = new Worker(new URL("../workers/freezeTrace.worker.ts", import.meta.url), { type: "module" }); }
  catch { return; }
  recording = true;
  trace("session", { build: typeof __TW_BUILD__ === "string" ? __TW_BUILD__ : "dev", browser: navigator.userAgent,
    cores: navigator.hardwareConcurrency, memoryGB: (navigator as Navigator & { deviceMemory?: number }).deviceMemory });
  const pulse = () => {
    const memory = (performance as Performance & { memory?: { usedJSHeapSize: number; jsHeapSizeLimit: number } }).memory;
    trace("pulse", { visible: document.visibilityState === "visible",
      heapUsedMB: memory ? Math.round(memory.usedJSHeapSize / 1048576) : undefined,
      heapLimitMB: memory ? Math.round(memory.jsHeapSizeLimit / 1048576) : undefined });
  };
  pulse();
  setInterval(pulse, 1000);
  document.addEventListener("visibilitychange", pulse);
  window.addEventListener("error", (e) => trace("page-error", { message: e.message.slice(0, 1000), stack: String(e.error?.stack ?? "").slice(0, 2000) }));
  window.addEventListener("unhandledrejection", (e) => trace("page-rejection", String(e.reason?.stack ?? e.reason).slice(0, 2000)));
  const panel = document.createElement("div");
  panel.style.cssText = "position:fixed;bottom:8px;right:8px;z-index:10000;background:#fff;color:#222;padding:8px;border:1px solid #888;border-radius:5px;font:12px sans-serif;max-width:300px";
  const status = document.createElement("div");
  status.textContent = "Diagnostics starting…"; // i18n-ignore: opt-in developer diagnostic
  const download = document.createElement("button");
  download.textContent = "Download diagnostics"; // i18n-ignore: opt-in developer diagnostic
  download.disabled = true;
  download.onclick = () => recorder?.postMessage({ download: true });
  const stop = document.createElement("button");
  stop.textContent = "Stop recording"; // i18n-ignore: opt-in developer diagnostic
  stop.onclick = () => {
    try { localStorage.setItem("tw-freeze-trace", "0"); } catch { /* unavailable storage */ }
    trace("stopped");
    recorder?.postMessage({ stop: true });
    recording = false;
    const url = new URL(location.href); url.searchParams.delete("diagnostics"); history.replaceState(null, "", url);
    status.textContent = "Recording stopped. Download before closing."; // i18n-ignore
    stop.disabled = true;
  };
  recorder.onmessage = (e) => {
    if (e.data.ready) { if (recording) status.textContent = "Diagnostics recording locally"; download.disabled = false; } // i18n-ignore
    if (e.data.error) { status.textContent = "Diagnostics storage unavailable; download before closing."; download.disabled = false; } // i18n-ignore
    if (e.data.report) {
      const url = URL.createObjectURL(new Blob([JSON.stringify(e.data.report, null, 2)], { type: "application/json" }));
      const a = document.createElement("a"); a.href = url; a.download = "trainsworld-diagnostics.json"; a.click();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    }
  };
  panel.append(status, download, stop);
  if (document.body) document.body.append(panel);
  else document.addEventListener("DOMContentLoaded", () => document.body.append(panel), { once: true });
}
if (typeof document !== "undefined") start();
