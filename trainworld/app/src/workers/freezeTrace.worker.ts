// Receives clock checkpoints directly through a port, independently of the page's event loop.
export {};
type Entry = { at: number; source: string; event: string; detail: unknown; wasmMB?: number };
type Session = { id: string; started: number; updated: number; metadata?: unknown; pageHealth?: unknown; clockHealth?: unknown; activePhases: Record<string, { at: number; name: string }>; events: Entry[] };
const session: Session = { id: crypto.randomUUID(), started: Date.now(), updated: Date.now(), activePhases: {}, events: [] };
let db: IDBDatabase | undefined;
let previous: Session[] = [];
let stopped = false;
let dirty = false;
const pulses = new Map<string, { at: number; stalled: boolean }>();
let visible = true;
let clockPort: MessagePort | undefined;

function record(source: string, event: string, detail: unknown = null, wasmMB?: number) {
  session.updated = Date.now();
  session.events.push({ at: session.updated, source, event, detail, wasmMB });
  if (session.events.length > 400) session.events.shift();
  dirty = true;
}
function receive(source: string, msg: { event: string; detail?: any; wasmMB?: number }) {
  if (stopped) return;
  if (source === "page" && (msg.event === "phase-start" || msg.event === "phase-end")) {
    const { id, name, ms } = msg.detail;
    if (msg.event === "phase-start") session.activePhases[id] = { at: Date.now(), name };
    else {
      delete session.activePhases[id];
      if (ms > 100) record(source, "slow-phase", { name, ms });
    }
    dirty = true;
    return;
  }
  if (source === "clock" && typeof msg.wasmMB === "number") {
    session.clockHealth = { at: Date.now(), wasmMB: msg.wasmMB };
    dirty = true;
  }
  if (msg.event === "session") session.metadata = msg.detail;
  if (msg.event === "pulse") {
    const old = pulses.get(source);
    if (old?.stalled) record(source, "responsive-again", { gapMs: Date.now() - old.at });
    pulses.set(source, { at: Date.now(), stalled: false });
    if (source === "page") {
      visible = msg.detail?.visible !== false;
      session.pageHealth = { at: Date.now(), ...msg.detail };
      dirty = true;
    }
  } else record(source, msg.event, msg.detail, msg.wasmMB);
}
function persist() {
  if (!db || !dirty) return;
  dirty = false;
  try {
    const tx = db.transaction("sessions", "readwrite");
    tx.objectStore("sessions").put(session);
    tx.onabort = () => { dirty = true; postMessage({ error: true }); };
  } catch { dirty = true; postMessage({ error: true }); }
}
const ready = new Promise<void>((resolve) => {
  const req = indexedDB.open("trainsworld-freeze-trace", 1);
  req.onupgradeneeded = () => req.result.createObjectStore("sessions", { keyPath: "id" });
  req.onerror = () => { postMessage({ error: true }); resolve(); };
  req.onsuccess = () => {
    db = req.result;
    const read = db.transaction("sessions").objectStore("sessions").getAll();
    read.onsuccess = () => {
      const all = (read.result as Session[]).sort((a, b) => b.started - a.started);
      previous = all.slice(0, 3);
      const tx = db!.transaction("sessions", "readwrite");
      for (const s of all.slice(3)) tx.objectStore("sessions").delete(s.id);
      persist(); postMessage({ ready: true }); resolve();
    };
    read.onerror = () => { postMessage({ error: true }); resolve(); };
  };
});
setInterval(() => {
  if (!stopped) for (const [source, p] of pulses) {
    // Background tabs and OS sleep can delay timers; preserve visibility and gaps as evidence,
    // not a verdict that the browser crashed.
    if (visible && !p.stalled && Date.now() - p.at > 5000) {
      p.stalled = true;
      record(source, "heartbeat-gap", { gapMs: Date.now() - p.at, visible });
    }
  }
  persist();
}, 1000);
onmessage = async (e) => {
  const msg = e.data;
  if (msg.clock) {
    clockPort?.close(); clockPort = msg.clock;
    pulses.delete("clock");
    clockPort!.onmessage = (event) => receive("clock", event.data);
    record("recorder", "clock-attached");
  } else if (msg.download) {
    await ready;
    persist();
    postMessage({ report: { format: 1, exported: Date.now(), previous, current: session } });
  } else if (msg.stop) { stopped = true; persist(); }
  else receive("page", msg);
};
