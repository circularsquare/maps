// T-842: on the real New York save with demand's riders (a riders.f32 from T-835-capture.mjs),
// boardings by class of line and the account's wait for the first train against demand's half
// headway, with riders spread over the whole period (`plain`) or only while a train is due within
// their tolerance (`masked`, the account as built, track/boarding.rs `Arrivals::with_mask`).
//   node T-842-analyse.mjs <riders.f32> <out.json> [root]     (root: the trainworld folder)
import { readFile, writeFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';

const RIDERS = process.argv[2], OUT = process.argv[3];
const ROOT = process.argv[4] ?? 'C:/Users/anita/projects/maps/trainworld';
const mod = await import('file:///' + ROOT + '/app/src/wasm/track/sim.js');
await mod.default({ module_or_path: await readFile(ROOT + '/app/src/wasm/track/sim_bg.wasm') });
const raw = gunzipSync(await readFile(ROOT + '/T-007-real-nyc.save'));
const api = new mod.TrackApi(-73.985, 40.758, true);
if (!api.load(new Uint8Array(raw.subarray(8 + raw.readUInt32LE(4))))) throw new Error('load failed');
const fb = await readFile(RIDERS);
const packed = new Float32Array(fb.buffer.slice(fb.byteOffset, fb.byteOffset + fb.length));

const DAY = 86400, H = 3600;
const PERIOD_HOURS = [[6, 10], [10, 16], [16, 20], [20, 24], [0, 6]];
const PERIOD_LEVEL = [0, 1, 0, 1, 2];
const SHAPE = ['tent', 'tent', 'tent', 'falling', 'valley'];
const CRUSH = 160;
const TOL_MIN = 1800, TOL_MAX = 6 * H;
const within = (w) => (w <= TOL_MIN ? 1 : w >= TOL_MAX ? 0 : Math.log(TOL_MAX / w) / Math.log(TOL_MAX / TOL_MIN));
const peak = Array.from(packed.slice(0, 5));

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

const shapeCum = (s, c, x) => {
  const tent = x <= 0.5 ? 2 * x * x - x : 3 * x - 2 * x * x - 1;
  return x + c * (s === 'tent' ? tent : s === 'falling' ? x - x * x : -tent);
};
const strength = (s, hours, f) => {
  const even = Math.round(hours) % 2 === 0;
  const busiest = s === 'valley' || (s === 'tent' && even) ? 1 - 2 / hours : 1 - 1 / hours;
  return busiest <= 0 ? 0 : Math.min(1, Math.max(0, (f - 1) / busiest));
};
const C = PERIOD_HOURS.map(([a, b], p) => strength(SHAPE[p], b - a, peak[p]));
const periodAt = (t) => { const h = Math.floor(((t % DAY) + DAY) % DAY / H); return PERIOD_HOURS.findIndex(([a, b]) => h >= a && h < b); };
// the share of period p's riders arrived by s seconds into the day
const W = (p, s) => { const [a, b] = PERIOD_HOURS[p]; return shapeCum(SHAPE[p], C[p], Math.min(1, Math.max(0, (s - a * H) / ((b - a) * H)))); };

/** A stop's cumulative arrivals: by the shape alone, or masked by the tolerance (60 s knots). */
function arrivals(per, deps, masked) {
  const N = DAY / 60, M = new Float64Array(N + 1);
  const dd = deps.map((d) => ((d % DAY) + DAY) % DAY).sort((a, b) => a - b);
  const next = (t) => { for (const d of dd) if (d > t) return d - t; return dd.length ? dd[0] + DAY - t : Infinity; };
  const inc = new Float64Array(N), sh = new Float64Array(N), norm = new Float64Array(5);
  for (let j = 0; j < N; j++) {
    const s0 = j * 60, s1 = s0 + 60, p = periodAt(s0);
    inc[j] = W(p, s1) - W(p, s0);
    sh[j] = masked ? within(next(s0 + 30)) : 1;
    norm[p] += inc[j] * sh[j];
  }
  for (let j = 0; j < N; j++) {
    const p = periodAt(j * 60);
    const f = norm[p] > 1e-9 ? sh[j] / norm[p] : 1;
    M[j + 1] = M[j] + per[p] * inc[j] * f;
  }
  const day = M[N];
  return (t) => { const d = Math.floor(t / DAY), s = t - d * DAY, j = Math.min(N - 1, Math.floor(s / 60)); return d * day + M[j] + (M[j + 1] - M[j]) * (s / 60 - j); };
}

function alightShares(lr, r) {
  const n = lr.n, out = new Float64Array(5 * n), dayIn = new Float64Array(n), dayOff = new Float64Array(n);
  for (let p = 0; p < 5; p++) {
    let on = 0;
    for (let k = 0; k < n; k++) {
      const a = lr.alight[(p * 2 + r) * n + k];
      dayIn[k] += on; dayOff[k] += a;
      out[p * n + k] = on > 1e-6 ? Math.min(1, Math.max(0, a / on)) : -1;
      on = Math.max(0, on + lr.board[(p * 2 + r) * n + k] - a);
    }
  }
  for (let p = 0; p < 5; p++) {
    for (let k = 0; k < n; k++) if (out[p * n + k] < 0) out[p * n + k] = dayIn[k] > 1e-6 ? Math.min(1, dayOff[k] / dayIn[k]) : 0;
    out[p * n + n - 1] = 1;
  }
  return out;
}

// the account (boarding.rs simulate, warm) over the drawn trips: rider-seconds waiting for the
// first train and riders, by the period each arrived in, on day D
function simulate(trips, stopDep, lr, r, cars, D, masked) {
  const n = lr.n, cap = CRUSH * cars, m = trips.length;
  const alpha = alightShares(lr, r);
  const t0 = D * DAY + 3 * H;
  const time = (i, k) => trips[i].dep + stopDep[trips[i].lev][k];
  const carry = new Float64Array(m);
  const res = { arr: new Float64Array(5), wait1: new Float64Array(5) };
  const order = [...Array(m).keys()];
  for (let k = 0; k < n; k++) {
    order.sort((a, b) => time(a, k) - time(b, k));
    const per = [0, 1, 2, 3, 4].map((p) => lr.board[(p * 2 + r) * n + k]);
    const cum = arrivals(per, order.map((i) => time(i, k)).filter((t) => t >= D * DAY && t < (D + 1) * DAY), masked);
    let q = 0, cPrev = NaN, tPrev = NaN;
    for (const i of order) {
      const t = time(i, k), cNow = cum(t);
      const a = Number.isNaN(cPrev) ? 0 : Math.max(0, cNow - cPrev);
      const qb = q + a;
      let on = carry[i];
      on -= on * alpha[periodAt(t) * n + k];
      const b = Math.min(qb, Math.max(0, cap - on));
      carry[i] = k + 1 < n ? on + b : 0;
      if (trips[i].dep >= t0 && trips[i].dep < t0 + DAY && !Number.isNaN(tPrev)) {
        const gap = t - tPrev, steps = Math.max(1, Math.ceil(gap / 60));
        let c0 = cPrev;
        for (let j = 1; j <= steps; j++) {
          const s1 = tPrev + j * gap / steps, mid = s1 - gap / steps / 2, c1 = cum(s1), p = periodAt(mid);
          res.wait1[p] += (c1 - c0) * (t - mid);
          res.arr[p] += c1 - c0;
          c0 = c1;
        }
      }
      cPrev = cNow; tPrev = t; q = qb - b;
    }
  }
  return res;
}

const li = api.lines_info();
const lines = [];
for (let i = 0; i < li.length; i += 22) if (li[i + 3]) lines.push({ id: li[i], name: api.line_name(li[i]), tph: [li[i + 5], li[i + 6], li[i + 7]], cars: li[i + 8] });
const D = 3;
const byProf = new Map();
for (let h = (D - 1) * 24; h < (D + 2) * 24; h++) {
  const tr = api.trips(h * H);
  for (let j = 0; j < tr.length; j += 4) {
    const prof = tr[j], dep = Math.round((h * H + tr[j + 1]) * 1000) / 1000;
    if (!byProf.has(prof)) byProf.set(prof, new Set());
    byProf.get(prof).add(dep);
  }
}
const classOf = (t) => (Math.min(...t) >= 2.5 ? 'frequent' : Math.min(t[0], t[1]) >= 2.5 ? 'frequent, no night' : t[0] <= 2 ? 'sparse' : 'middling');
const out = { lines: [], classes: {}, total: {} };
for (const L of lines) {
  const lr = riders.get(L.id);
  if (!lr) continue;
  const stopDep = [0, 1].map((r) => [0, 1, 2].map((lev) => api.stop_departures(L.id, r, lev)));
  if (stopDep[0][0].length !== lr.n) continue;
  const deps = [0, 1].map((r) => {
    const v = [];
    for (let lev = 0; lev < 3; lev++) for (const dep of byProf.get((L.id * 2 + r) * 3 + lev) ?? []) v.push({ dep, lev });
    v.sort((a, b) => a.dep - b.dep);
    return v.filter((d, i) => i === 0 || d.dep - v[i - 1].dep > 0.5);
  });
  const rec = { name: L.name, cls: classOf(L.tph), tph: L.tph, board: lr.board.reduce((a, b) => a + b, 0) };
  for (const masked of [false, true]) {
    const key = masked ? 'masked' : 'plain';
    const o = { arr: new Float64Array(5), wait1: new Float64Array(5), nominal: new Float64Array(5) };
    for (let r = 0; r < 2; r++) {
      const s = simulate(deps[r], stopDep[r], lr, r, L.cars, D, masked);
      for (let p = 0; p < 5; p++) {
        o.arr[p] += s.arr[p]; o.wait1[p] += s.wait1[p];
        const t = L.tph[PERIOD_LEVEL[p]];
        o.nominal[p] += t > 0 ? s.arr[p] * (H / t) / 2 : 0;
      }
    }
    rec[key] = { arr: Array.from(o.arr), wait1H: Array.from(o.wait1, (x) => x / H), nominalH: Array.from(o.nominal, (x) => x / H) };
    for (const g of [rec.cls, 'all']) {
      const c = ((out.classes[g] ??= { lines: 0, board: 0 })[key] ??= { arr: 0, wait1H: 0, nominalH: 0 });
      c.arr += o.arr.reduce((a, b) => a + b, 0);
      c.wait1H += o.wait1.reduce((a, b) => a + b, 0) / H;
      c.nominalH += o.nominal.reduce((a, b) => a + b, 0) / H;
    }
  }
  for (const g of [rec.cls, 'all']) { out.classes[g].lines++; out.classes[g].board += rec.board; }
  out.lines.push(rec);
}
await writeFile(OUT, JSON.stringify(out));
for (const [g, c] of Object.entries(out.classes)) {
  const f = (x) => `${Math.round(x.wait1H)} h against ${Math.round(x.nominalH)} (${((x.wait1H / x.nominalH - 1) * 100).toFixed(1)}%)`;
  console.log(`${g}: ${c.lines} lines, ${Math.round(c.board)} boardings; plain ${f(c.plain)}; masked ${f(c.masked)}`);
}
