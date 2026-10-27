// T-348: what the page's main thread does in its long tasks, from a trace of checks/t348-lag.mjs.
//   node trainworld/checks/t348-analyze.mjs <trace.json> [minMs=50]
// Per long task (a top-level task of the renderer's main thread over minMs): its start (ms after
// navigation), wall and CPU time, and where it went: the CPU samples' functions (self time and
// time inside, by name and source line), garbage collection, message events, style and layout.
import { readFileSync } from 'node:fs';
const file = process.argv[2];
const MIN = Number(process.argv[3] ?? 50);
const trace = JSON.parse(readFileSync(file, 'utf8'));
const evs = trace.traceEvents ?? trace;

// the main thread: the CrRendererMain with the most events
const names = new Map();
for (const e of evs) if (e.ph === 'M' && e.name === 'thread_name') names.set(`${e.pid}:${e.tid}`, e.args.name);
const count = new Map();
for (const e of evs) { const k = `${e.pid}:${e.tid}`; if (names.get(k) === 'CrRendererMain') count.set(k, (count.get(k) ?? 0) + 1); }
const main = [...count].sort((a, b) => b[1] - a[1])[0][0];
const [pid, tid] = main.split(':').map(Number);
const on = evs.filter((e) => e.pid === pid && e.tid === tid);
const nav = on.find((e) => e.name === 'navigationStart' || e.name === 'NavigationTimingStart' || e.name === 'markAsMainFrame')?.ts
  ?? evs.find((e) => e.pid === pid && e.name === 'navigationStart')?.ts ?? on[0].ts;

// CPU samples of the main thread
const nodes = new Map(); // id -> {name, parent}
const samples = []; // [ts, nodeId]
const profileIds = new Set(evs.filter((e) => e.name === 'Profile' && e.pid === pid && e.tid === tid).map((e) => e.id));
const chunks = evs.filter((e) => e.name === 'ProfileChunk' && e.pid === pid && profileIds.has(e.id));
const startTs = evs.find((e) => e.name === 'Profile' && e.pid === pid && e.tid === tid)?.args?.data?.startTime ?? 0;
let t = startTs;
for (const c of chunks) {
  const p = c.args.data.cpuProfile ?? {};
  for (const n of p.nodes ?? []) {
    const f = n.callFrame;
    const url = (f.url || '').replace(/.*\/assets\//, '').replace(/.*\//, '');
    nodes.set(n.id, { name: `${f.functionName || '(anon)'} ${url}${f.url ? ':' + (f.lineNumber + 1) : ''}`.trim(), parent: n.parent });
  }
  const deltas = c.args.data.timeDeltas ?? [];
  (p.samples ?? []).forEach((id, i) => { t += deltas[i] ?? 0; samples.push([t, id]); });
}
samples.sort((a, b) => a[0] - b[0]);
const interval = samples.length > 1 ? (samples.at(-1)[0] - samples[0][0]) / samples.length : 100;

const top = on.filter((e) => e.ph === 'X' && (e.name === 'RunTask' || e.name === 'ThreadControllerImpl::RunTask') && e.dur >= MIN * 1000);
// keep outermost only
top.sort((a, b) => a.ts - b.ts);
const tasks = [];
for (const e of top) if (!tasks.length || e.ts >= tasks.at(-1).ts + tasks.at(-1).dur) tasks.push(e);

const KEEP = /GC|Garbage|FunctionCall|EventDispatch|UpdateLayoutTree|Layout|Paint|ParseHTML|EvaluateScript|v8.compile|V8.Deserialize|Deserialize|PostMessage|TimerFire|FireAnimationFrame|RunMicrotasks|v8.run|HitTest|Commit|PrePaint|ScheduleStyleRecalculation/;
const fmt = (us) => (us / 1000).toFixed(0);
let total = 0, totalCpu = 0;
const overallSelf = new Map(), overallIn = new Map();
for (const task of tasks) {
  const a = task.ts, b = task.ts + task.dur;
  total += task.dur; totalCpu += task.tdur ?? 0;
  const inner = on.filter((e) => e.ph === 'X' && e !== task && e.ts >= a && e.ts < b && KEEP.test(e.name));
  const byName = new Map();
  for (const e of inner) {
    let k = e.name;
    if (e.name === 'FunctionCall') { const d = e.args?.data ?? {}; k = `FunctionCall ${d.functionName || '(anon)'} ${(d.url || '').replace(/.*\//, '')}:${d.lineNumber ?? ''}`; }
    if (e.name === 'EventDispatch') k = `EventDispatch ${e.args?.data?.type}`;
    byName.set(k, (byName.get(k) ?? 0) + e.dur);
  }
  const self = new Map(), incl = new Map();
  for (const [ts, id] of samples) {
    if (ts < a || ts >= b) continue;
    const n = nodes.get(id);
    if (!n) continue;
    self.set(n.name, (self.get(n.name) ?? 0) + interval);
    overallSelf.set(n.name, (overallSelf.get(n.name) ?? 0) + interval);
    const seen = new Set();
    for (let p = n; p; p = nodes.get(p.parent)) {
      if (seen.has(p.name)) continue;
      seen.add(p.name);
      incl.set(p.name, (incl.get(p.name) ?? 0) + interval);
      overallIn.set(p.name, (overallIn.get(p.name) ?? 0) + interval);
    }
  }
  console.log(`\n== task at ${fmt(a - nav)} ms: ${fmt(task.dur)} ms wall, ${task.tdur ? fmt(task.tdur) : '?'} ms CPU`);
  console.log('  events:', [...byName].sort((x, y) => y[1] - x[1]).slice(0, 8).map(([k, v]) => `${k} ${fmt(v)}`).join(' | '));
  console.log('  self:  ', [...self].sort((x, y) => y[1] - x[1]).slice(0, 10).map(([k, v]) => `${k} ${fmt(v)}`).join(' | '));
  console.log('  inside:', [...incl].filter(([k]) => !/^\(root\)|^\(program\)/.test(k)).sort((x, y) => y[1] - x[1]).slice(0, 14).map(([k, v]) => `${k} ${fmt(v)}`).join(' | '));
}
console.log(`\n${tasks.length} tasks over ${MIN} ms, ${fmt(total)} ms wall, ${fmt(totalCpu)} ms CPU`);
console.log('overall self:', [...overallSelf].sort((x, y) => y[1] - x[1]).slice(0, 15).map(([k, v]) => `${k} ${fmt(v)}`).join(' | '));
console.log('overall inside:', [...overallIn].filter(([k]) => !/^\(root\)|^\(program\)/.test(k)).sort((x, y) => y[1] - x[1]).slice(0, 25).map(([k, v]) => `${k} ${fmt(v)}`).join(' | '));
