// T-2552 check: the load bar's model (app/src/game/loadBar.ts) on the timings of Anita's North America save (81 zones,
// 2 solving workers; notes/T-2524.md): the solve stage moves steadily and its time left stays honest through every
// round, the last one's quick converged period included, and a zone's own bar keeps moving through its opening and
// the solve that covers it.
//   node --experimental-strip-types trainworld/checks/t2552-load-bar.mjs
import { keepPlace, solveStage, zoneShare, FEW_MS, nextLeft } from '../app/src/game/loadBarCore.ts';

const results = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

// the import's whole solve (notes/T-2524.md), seconds: sent, each round's end, each period's landing per round
const SENT = 54.1;
const ROUND_AT = [109.4, 145.5, 172.4, 197.0];
const LANDS = [
  [83.8, 84.7, 95.3, 101.8, 109.4],
  [121.4, 122.3, 132.6, 136.8, 145.5],
  [154.7, 156.2, 166.8, 169.9, 172.3],
  [172.9, 182.9, 188.2, 191.6, 197.0],
];
const PERIODS = 5, WAVES = 3, ROUNDS = 3, SETUP = 15000;

/** the solve stage every 0.25 s from the solve's sending to its final, as the card would see it */
function run(prior, later) {
  const out = [];
  let shown = null, at = null;
  for (let t = SENT; t < ROUND_AT[ROUNDS]; t += 0.25) {
    const published = ROUND_AT.findLastIndex((x) => x <= t);
    const roundAt = ROUND_AT.filter((x) => x <= t).map((x) => x * 1000);
    const r = published + 1;
    const landed = LANDS[r].filter((x) => x <= t).length;
    const firstLand = LANDS.map((l, i) => (i <= r && l[0] <= t ? l[0] * 1000 : undefined));
    const st = solveStage(SENT * 1000, roundAt, published, landed, firstLand, ROUNDS, prior * 1000, later, t * 1000, PERIODS, WAVES, t >= LANDS[0][0] ? SETUP : 0);
    shown = nextLeft(shown, st.left, at === null ? 0 : (t - at) * 1000);
    at = t;
    out.push({ t, share: st.share, left: shown, real: (ROUND_AT[ROUNDS] - t) * 1000 });
  }
  return out;
}

for (const [name, prior, later] of [['remembered (a reload)', 143, 0.6], ['modelled, half the real solve', 70, 0.5], ['modelled, twice the real solve', 290, 0.5]]) {
  const r = run(prior, later);
  // the card's bar with the solve as 70% of it after a 30% data stage, kept in place as the client keeps it
  const place = { best: 0 };
  let last = -1, since = SENT, longest = 0, at = 0;
  for (const x of r) {
    const bar = Math.floor(100 * keepPlace(place, Math.min(0.99, 0.3 + 0.7 * x.share)));
    if (bar !== last) { if (x.t - since > longest) { longest = x.t - since; at = since; } last = bar; since = x.t; }
  }
  report(`${name}: the bar never sits 20 s on one percent`, longest <= 20, `longest ${longest.toFixed(1)} s from ${at.toFixed(1)} s`);
  // "A few seconds left" only in the last 15 s, and the time left never under a third of the real one past the first minute
  const few = r.filter((x) => x.left < FEW_MS && x.real > 15000);
  report(`${name}: "A few seconds left" only near the end`, few.length === 0, few.length ? `from ${few[0].t.toFixed(1)} s, ${(few[0].real / 1000).toFixed(0)} s still to go` : '');
  const low = r.filter((x) => x.t > SENT + 60 && x.left < x.real / 3);
  report(`${name}: time left at least a third of the real one after the first minute`, low.length === 0, low.length ? `${low.length} ticks, first at ${low[0].t.toFixed(1)} s: ${(low[0].left / 1000).toFixed(0)} s shown, ${(low[0].real / 1000).toFixed(0)} s real` : '');
}

// a zone loading outside the load in hand: its download, then its opening, then the solve covering it
{
  const dlMs = 3000, openMs = 2000, solveMs = 140000;
  const a = zoneShare({ dlMs, dl: 1, openMs, open: 1, solveMs, solve: 0 });
  const b = zoneShare({ dlMs, dl: 1, openMs, open: 1, solveMs, solve: 0.5 });
  report('a zone in a solve: open is not nearly full', a < 0.1, a.toFixed(3));
  report('a zone in a solve: moves with the solve', b > a + 0.4 && b < 0.99, b.toFixed(3));
  const c = zoneShare({ dlMs, dl: 1, openMs, open: 1, solveMs: 0, solve: 0 });
  report('a bare zone: full once open', c === 1, String(c));
}

console.log(results.every(Boolean) ? 'T-2552 PASSED' : 'T-2552 FAILED');
process.exit(results.every(Boolean) ? 0 : 1);
