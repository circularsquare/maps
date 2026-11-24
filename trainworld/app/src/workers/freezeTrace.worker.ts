// Receives clock checkpoints directly through a port, independently of the page's event loop.
export {};
type Entry = { at: number; source: string; event: string; detail: unknown; wasmMB?: number };
type Session = { id: string; started: number; updated: number; metadata?: unknown; pageHealth?: unknown; clockHealth?: unknown; activePhases: Record<string, { at: number; name: string }>; events: Entry[];
  /** T-4206: calls too frequent to log one by one (the clock's once-a-second "today" query): how many, and the slowest */
  counts?: Record<string, { n: number; maxMs: number }> };
const session: Session = { id: crypto.randomUUID(), started: Date.now(), updated: Date.now(), activePhases: {}, events: [], counts: {} };
// T-4206: each kind of event has its own share of the log, so a busy one (queries, trips arriving) cannot push out the
// demand side or the edits, and the whole stays bounded: about 10 minutes of a busy session, 30 minutes of an editing one
const CAPS: Record<string, number> = { demand: 4000, edit: 2000, other: 600 };
const kept: Record<string, number> = { demand: 0, edit: 0, other: 0 };
const EDIT_EVENTS = /^(edit-click|clock-post|clock-answer|clock-error|handle-|apply-|build-state|edited|error|settle|slow-)/;
const category = (event: string) => (event.startsWith("dm-") ? "demand" : EDIT_EVENTS.test(event) ? "edit" : "other");
const PREVIOUS_EVENTS = 800;
const open = new Map<string, number>();
const count = (key: string, ms: number) => {
  const c = ((session.counts ??= {})[key] ??= { n: 0, maxMs: 0 });
  c.n++; c.maxMs = Math.max(c.maxMs, Math.round(ms));
};
let db: IDBDatabase | undefined;
let previous: Session[] = [];
let stopped = false;
let dirty = false;
const pulses = new Map<string, { at: number; stalled: boolean }>();
let visible = true;
let clockPort: MessagePort | undefined;

function record(source: string, event: string, detail: unknown = null, wasmMB?: number) {
  session.updated = Date.now();
  // an event stamped by its sender (a demand milestone that happened before it was seen) keeps its own time
  let at = session.updated;
  if (detail && typeof detail === "object" && typeof (detail as { t?: unknown }).t === "number") { at = (detail as { t: number }).t; delete (detail as { t?: number }).t; }
  const cat = category(event);
  session.events.push({ at, source, event, detail, wasmMB });
  if (++kept[cat] > CAPS[cat]) {
    const old = session.events.findIndex((x) => category(x.event) === cat);
    if (old >= 0) { session.events.splice(old, 1); kept[cat]--; }
  }
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
  if (source === "page" && (msg.event === "receive-start" || msg.event === "receive-end")) {
    const key = "receive:" + msg.detail?.kind;
    if (msg.event === "receive-start") session.activePhases[key] = { at: Date.now(), name: key };
    else {
      delete session.activePhases[key];
      count(key, msg.detail?.ms ?? 0);
      if (msg.detail?.ms > 30) record(source, "slow-receive", msg.detail);
    }
    dirty = true;
    return;
  }
  if (source === "clock" && msg.detail?.kind === "query" && (msg.event === "handle-start" || msg.event === "handle-end")) {
    const key = String(msg.detail.seq);
    if (msg.event === "handle-start") open.set(key, Date.now());
    else {
      const ms = Date.now() - (open.get(key) ?? Date.now());
      open.delete(key);
      count("query:" + msg.detail.q, ms);
      if (ms > 100) record(source, "slow-query", { ...msg.detail, ms });
    }
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
      for (const s of previous) s.events = s.events.slice(-PREVIOUS_EVENTS);
      const tx = db!.transaction("sessions", "readwrite");
      for (const s of all.slice(3)) tx.objectStore("sessions").delete(s.id);
      persist(); postMessage({ ready: true }); resolve();
    };
    read.onerror = () => { postMessage({ error: true }); resolve(); };
  };
});
let tick = 0;
setInterval(() => {
  if (!stopped) for (const [source, p] of pulses) {
    // Background tabs and OS sleep can delay timers; preserve visibility and gaps as evidence,
    // not a verdict that the browser crashed.
    if (visible && !p.stalled && Date.now() - p.at > 5000) {
      p.stalled = true;
      record(source, "heartbeat-gap", { gapMs: Date.now() - p.at, visible });
    }
  }
  // a big log is written every third second: the newest few seconds are what an abrupt death can lose
  if (session.events.length < 1500 || ++tick % 3 === 0) persist();
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
    postMessage({ report: { format: 2, exported: Date.now(), previous: previous.map((s) => ({ ...s, events: s.events.slice(-PREVIOUS_EVENTS) })), current: session } });
  } else if (msg.stop) { stopped = true; persist(); }
  else receive("page", msg);
};
