// T-835: how the frequency each stop sees across a change of period compares with what demand
// assumes, on the real New York save with demand's own riders (riders.f32 from capture.mjs).
//   node analyse.mjs <riders.f32> <out.json>     (cwd: the maps root)
import { readFile, writeFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';

const RIDERS = process.argv[2], OUT = process.argv[3];
const ROOT = 'C:/Users/anita/projects/maps/trainworld';
const mod = await import('file:///' + ROOT + '/app/src/wasm/track/sim.js');
await mod.default({ module_or_path: await readFile(ROOT + '/app/src/wasm/track/sim_bg.wasm') });
const raw = gunzipSync(await readFile(ROOT + '/T-007-real-nyc.save'));
const hlen = raw.readUInt32LE(4);
const api = new mod.TrackApi(-73.985, 40.758, true);
if (!api.load(new Uint8Array(raw.subarray(8 + hlen)))) throw new Error('load failed');
const fb = await readFile(RIDERS);
const packed = new Float32Array(fb.buffer.slice(fb.byteOffset, fb.byteOffset + fb.length));

const DAY = 86400, H = 3600;
const PERIOD_HOURS = [[6, 10], [10, 16], [16, 20], [20, 24], [0, 6]];
const PERIOD_LEVEL = [0, 1, 0, 1, 2];
const SHAPE = ['tent', 'tent', 'tent', 'falling', 'valley'];
const CRUSH = 160;
const peak = Array.from(packed.slice(0, 5));

// ---- riders per line
const riders = new Map();
for (let i = 5; i + 2 <= packed.length;) {
  const id = packed[i], n = packed[i + 1];
  i += 2;
  const len = 5 * 2 * n;
  const board = new Float64Array(len), alight = new Float64Array(len);
  for (let p = 0; p < 5; p++) for (let k = 0; k < 2 * n; k++) {
    board[p * 2 * n + k] = Math.max(0, packed[i + p * 4 * n + k]);
    alight[p * 2 * n + k] = Math.max(0, packed[i + p * 4 * n + 2 * n + k]);
  }
  i += 2 * len;
  riders.set(id, { n, board, alight });
}

// ---- the account's arrival shape (boarding.rs)
const shapeCum = (s, c, x) => {
  const tent = x <= 0.5 ? 2 * x * x - x : 3 * x - 2 * x * x - 1;
  const g = s === 'tent' ? tent : s === 'falling' ? x - x * x : -tent;
  return x + c * g;
};
const strength = (s, hours, f) => {
  const even = Math.round(hours) % 2 === 0;
  const busiest = s === 'valley' || (s === 'tent' && even) ? 1 - 2 / hours : 1 - 1 / hours;
  return busiest <= 0 ? 0 : Math.min(1, Math.max(0, (f - 1) / busiest));
};
const C = PERIOD_HOURS.map(([a, b], p) => strength(SHAPE[p], b - a, peak[p]));
function cum(per, t) {
  const d = Math.floor(t / DAY), s = t - d * DAY;
  let tot = d * per.reduce((a, b) => a + b, 0);
  for (let p = 0; p < 5; p++) {
    const a = PERIOD_HOURS[p][0] * H, b = PERIOD_HOURS[p][1] * H;
    if (s >= b) tot += per[p];
    else if (s > a) tot += per[p] * shapeCum(SHAPE[p], C[p], (s - a) / (b - a));
  }
  return tot;
}
const periodAt = (t) => { const h = Math.floor(t / H) % 24; return PERIOD_HOURS.findIndex(([a, b]) => h >= a && h < b); };

function alightShares(lr, r) {
  const n = lr.n, out = new Float64Array(5 * n), dayIn = new Float64Array(n), dayOff = new Float64Array(n);
  for (let p = 0; p < 5; p++) {
    const base = (p * 2 + r) * n;
    let on = 0;
    for (let k = 0; k < n; k++) {
      const a = lr.alight[base + k];
      dayIn[k] += on; dayOff[k] += a;
      out[p * n + k] = on > 1e-6 ? Math.min(1, Math.max(0, a / on)) : -1;
      on = Math.max(0, on + lr.board[base + k] - a);
    }
  }
  for (let p = 0; p < 5; p++) {
    for (let k = 0; k < n; k++) if (out[p * n + k] < 0) out[p * n + k] = dayIn[k] > 1e-6 ? Math.min(1, dayOff[k] / dayIn[k]) : 0;
    out[p * n + n - 1] = 1;
  }
  return out;
}

// ---- departures: the line's lattice with movable period starts (seconds into the day, per run)
// starts: [[sec, period]] in day order. Mirrors service.rs lattice_deps_local.
function lattice(tph, off, r, starts, d0, d1) {
  const out = [];
  let last = null;
  const pts = (day, i) => {
    const [s, p] = starts[i];
    const lev = PERIOD_LEVEL[p], t = tph[lev];
    if (!(t > 0)) return null;
    const hw = H / t;
    const start = day * DAY + s;
    const end = i + 1 < starts.length ? day * DAY + starts[i + 1][0] : (day + 1) * DAY + starts[0][0];
    const ps = start + (r === 1 ? off[lev] : 0);
    let n = Math.ceil((end - ps) / hw);
    while (n > 0 && ps + (n - 1) * hw >= end) n--;
    while (ps + n * hw < end) n++;
    return n > 0 ? { ps, hw, n, lev } : null;
  };
  for (let i = starts.length - 1; i >= 0; i--) { const q = pts(d0 - 1, i); if (q) { last = q.ps + (q.n - 1) * q.hw; break; } }
  for (let day = d0; day <= d1; day++) for (let i = 0; i < starts.length; i++) {
    const q = pts(day, i);
    if (!q) continue;
    let k = 0;
    if (last !== null) while (k < q.n && q.ps + k * q.hw - last < 0.5 * q.hw) k++;
    for (; k < q.n; k++) out.push({ dep: q.ps + k * q.hw, lev: q.lev, hw: q.hw });
    last = q.ps + (q.n - 1) * q.hw;
  }
  return out;
}
const BASE_STARTS = [[0, 4], [6 * H, 0], [10 * H, 1], [16 * H, 2], [20 * H, 3]];

// ---- the account (boarding.rs simulate, warm, no eras): per stop and hour of day D
function simulate(trips, stopDep, lr, r, cars, D) {
  const n = lr.n, cap = CRUSH * cars, m = trips.length;
  const alpha = alightShares(lr, r);
  const t0 = D * DAY + 3 * H;
  const time = (i, k) => trips[i].dep + stopDep[trips[i].lev][k];
  const carry = new Float64Array(m);
  const res = { arrA: new Float64Array(24), left: new Float64Array(24), arrived: new Float64Array(24), wait1: new Float64Array(24), waitX: new Float64Array(24),
    trains: Array.from({ length: n }, () => new Float64Array(24)), arrK: Array.from({ length: n }, () => new Float64Array(24)),
    w1K: Array.from({ length: n }, () => new Float64Array(24)), leftK: Array.from({ length: n }, () => new Float64Array(24)) };
  const order = [...Array(m).keys()];
  for (let k = 0; k < n; k++) {
    order.sort((a, b) => time(a, k) - time(b, k));
    const per = [0, 1, 2, 3, 4].map((p) => lr.board[(p * 2 + r) * n + k]);
    let q = 0, cPrev = NaN, tPrev = NaN;
    for (const i of order) {
      const t = time(i, k);
      const cNow = cum(per, t);
      const a = Number.isNaN(cPrev) ? 0 : Math.max(0, cNow - cPrev);
      const qb = q + a;
      const p = periodAt(t);
      let on = carry[i];
      on -= on * alpha[p * n + k];
      const b = Math.min(qb, Math.max(0, cap - on));
      const load = on + b, qa = qb - b;
      carry[i] = k + 1 < n ? load : 0;
      const kept = trips[i].dep >= t0 && trips[i].dep < t0 + DAY;
      if (kept && !Number.isNaN(tPrev)) {
        const h = Math.floor((t % DAY) / H);
        const gap = t - tPrev;
        res.left[h] += Math.min(a, qa);
        res.leftK[k][h] += Math.min(a, qa);
        res.arrived[h] += a; res.arrK[k][h] += a;
        // rider-s waiting for the first train: the integral of (A(s) - A(tPrev)) over the gap
        // by the hour each rider arrived (res.arrA, res.wait1)
        const steps = Math.max(1, Math.ceil(gap / 60));
        let c0 = cPrev;
        for (let j = 1; j <= steps; j++) {
          const s0 = tPrev + (j - 1) * gap / steps, s1 = tPrev + j * gap / steps;
          const c1 = cum(per, s1), da = c1 - c0, ha = Math.floor((((s0 + s1) / 2) % DAY) / H);
          res.wait1[ha] += da * (t - (s0 + s1) / 2); res.w1K[k][ha] += da * (t - (s0 + s1) / 2);
          res.arrA[ha] += da;
          c0 = c1;
        }
        res.waitX[h] += q * gap;          // rider-s of those already left behind, waiting through this gap
        res.trains[k][h] += 1;
      }
      cPrev = cNow; tPrev = t; q = qa;
    }
  }
  return res;
}

// ---- lines
const li = api.lines_info();
const lines = [];
for (let i = 0; i < li.length; i += 22) {
  if (!li[i + 3]) continue;
  const id = li[i];
  lines.push({ id, name: api.line_name(id), tph: [li[i + 5], li[i + 6], li[i + 7]], cars: li[i + 8], lenM: li[i + 9], rt: [li[i + 10], li[i + 11], li[i + 12]], trains: [li[i + 13], li[i + 14], li[i + 15]], turn: li[i + 20] });
}
const meta = api.profile_meta();
const D = 3;
// the drawn departures of day D-1..D+1 from the game itself (to check the JS lattice)
const drawn = new Map();
for (let h = (D - 1) * 24; h < (D + 2) * 24; h++) {
  const tr = api.trips(h * H);
  for (let j = 0; j < tr.length; j += 4) {
    const prof = tr[j], dep = Math.round((h * H + tr[j + 1]) * 1000) / 1000;
    const key = prof + ':' + dep;
    drawn.set(key, { prof, dep });
  }
}
const byProf = new Map();
for (const { prof, dep } of drawn.values()) { if (!byProf.has(prof)) byProf.set(prof, []); byProf.get(prof).push(dep); }
let mismatch = 0;
const mis = [];
const out = { peak, C, lines: [] };
const tot = {};
for (const L of lines) {
  const lr = riders.get(L.id);
  if (!lr) continue;
  const stopDep = [0, 1].map((r) => [0, 1, 2].map((lev) => api.stop_departures(L.id, r, lev)));
  const stopT = [0, 1].map((r) => [0, 1, 2].map((lev) => api.stop_times(L.id, r, lev)));
  if (stopDep[0][0].length !== lr.n) continue;
  const tripLen = (r, lev) => meta[((L.id * 2 + r) * 3 + lev) * 4 + 2];
  const off = [0, 1, 2].map((lev) => (L.tph[lev] > 0 ? tripLen(0, lev) % (H / L.tph[lev]) : 0));
  // check: whole-line fleet offset or none, against the drawn trips
  const runT = [0, 1].map((r) => stopT[r][1].at(-1)); // one-way run at medium, s
  const mk = (r, shiftUp, shiftDown, offv) => {
    // shiftUp: seconds earlier for a period whose level runs more trains than the one before; shiftDown likewise
    const starts = BASE_STARTS.map(([s, p], i) => {
      const prev = BASE_STARTS[(i + 4) % 5][1];
      const up = L.tph[PERIOD_LEVEL[p]] > L.tph[PERIOD_LEVEL[prev]];
      return [s - (up ? shiftUp : shiftDown), p];
    });
    // keep the first start at or after 0 by rotating (day boundary): shift the whole thing into [0, DAY)
    const norm = starts.map(([s, p]) => [((s % DAY) + DAY) % DAY, p]).sort((a, b) => a[0] - b[0]);
    return lattice(L.tph, offv, r, norm, D - 1, D + 1);
  };
  // which offset does the game use?
  const base0 = [0, 1].map((r) => mk(r, 0, 0, off));
  const baseZ = [0, 1].map((r) => mk(r, 0, 0, [0, 0, 0]));
  const hit = (deps, r) => deps.filter((d) => d.dep >= D * DAY && d.dep < (D + 1) * DAY && (byProf.get((L.id * 2 + r) * 3 + d.lev) ?? []).some((x) => Math.abs(x - d.dep) < 0.05)).length;
  const nDay = (deps) => deps.filter((d) => d.dep >= D * DAY && d.dep < (D + 1) * DAY).length;
  const useOff = hit(base0[1], 1) >= hit(baseZ[1], 1) ? off : [0, 0, 0];
  const base = [0, 1].map((r) => mk(r, 0, 0, useOff));
  for (let r = 0; r < 2; r++) {
    const drawnN = [0, 1, 2].reduce((s, lev) => s + (byProf.get((L.id * 2 + r) * 3 + lev) ?? []).filter((x) => x >= D * DAY && x < (D + 1) * DAY).map((x) => Math.round(x)).filter((x, i, a) => a.indexOf(x) === i && a.indexOf(x - 1) < 0).length, 0);
    if (hit(base[r], r) !== nDay(base[r]) || drawnN !== nDay(base[r])) { mismatch++; mis.push([L.name, r, hit(base[r], r), nDay(base[r]), drawnN]); }
  }
  const opts = {
    now: [0, 1].map((r) => base[r]),
    half: [0, 1].map((r) => mk(r, runT[r] / 2, runT[r] / 2, useOff)),
    upEarly: [0, 1].map((r) => mk(r, runT[r], 0, useOff)),
    upHalf: [0, 1].map((r) => mk(r, runT[r] / 2, 0, useOff)),
  };
  const rec = { id: L.id, name: L.name, n: lr.n, km: L.lenM / 1000, runMin: runT.map((x) => x / 60), tph: L.tph, cars: L.cars, boardDay: lr.board.reduce((a, b) => a + b, 0), opt: {} };
  for (const [name, deps] of Object.entries(opts)) {
    const o = { arrA: new Float64Array(24), left: new Float64Array(24), arrived: new Float64Array(24), wait1: new Float64Array(24), waitX: new Float64Array(24), carKm: 0, carKmH: new Float64Array(24) };
    for (let r = 0; r < 2; r++) {
      const s = simulate(deps[r], stopDep[r], lr, r, L.cars, D);
      for (let h = 0; h < 24; h++) for (const f of ['left', 'arrived', 'wait1', 'waitX', 'arrA']) o[f][h] += s[f][h];
      for (const d of deps[r]) if (d.dep >= D * DAY + 3 * H && d.dep < (D + 1) * DAY + 3 * H) { o.carKm += L.cars * L.lenM / 1000; o.carKmH[Math.floor((d.dep % DAY) / H)] += L.cars * L.lenM / 1000; }
      if (name === 'now') {
        // a mid-line stop's trains per hour, both directions (stop nearest half the run)
        const k = stopT[r][1].reduce((b, t, j) => (Math.abs(t - runT[r] / 2) < Math.abs(stopT[r][1][b] - runT[r] / 2) ? j : b), 0);
        (rec.mid ??= []).push({ r, k, tMin: stopT[r][1][k] / 60, trains: Array.from(s.trains[k]), arr: Array.from(s.arrK[k]).map(Math.round), left: Array.from(s.leftK[k]).map(Math.round) });
      }
    }
    rec.opt[name] = { arrA: Array.from(o.arrA), left: Array.from(o.left), arrived: Array.from(o.arrived), wait1: Array.from(o.wait1), waitX: Array.from(o.waitX), carKm: o.carKm, carKmH: Array.from(o.carKmH) };
    const T = (tot[name] ??= { arrA: new Float64Array(24), left: new Float64Array(24), arrived: new Float64Array(24), wait1: new Float64Array(24), waitX: new Float64Array(24), carKm: 0 });
    for (let h = 0; h < 24; h++) for (const f of ['left', 'arrived', 'wait1', 'waitX', 'arrA']) T[f][h] += o[f][h];
    T.carKm += o.carKm;
  }
  // demand's view: riders of each period wait half the period's headway
  const nominal = new Float64Array(24);
  for (let h = 0; h < 24; h++) {
    const p = periodAt(h * H), t = L.tph[PERIOD_LEVEL[p]];
    nominal[h] = t > 0 ? rec.opt.now.arrA[h] * (H / t) / 2 : NaN;
  }
  rec.nominalWait1 = Array.from(nominal);
  const tn = (tot.nominal ??= { wait1: new Float64Array(24) });
  for (let h = 0; h < 24; h++) if (Number.isFinite(nominal[h])) tn.wait1[h] += nominal[h];
  out.lines.push(rec);
}
out.mismatchRuns = mismatch;
out.total = Object.fromEntries(Object.entries(tot).map(([k, v]) => [k, Object.fromEntries(Object.entries(v).map(([f, a]) => [f, typeof a === 'number' ? a : Array.from(a)]))]));
await writeFile(OUT, JSON.stringify(out));
console.log('lines', out.lines.length, 'runs whose JS departures differ from the drawn', mismatch, JSON.stringify(mis.slice(0, 12)));
