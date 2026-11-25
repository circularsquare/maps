// T-2552 check: the load bar's model (app/src/game/loadBar.ts) on the timings of Anita's North America save (81 zones,
// 2 solving workers; notes/T-2524.md): the solve stage's time left stays honest through every round, the last one's
// quick converged period included, and a zone's own bar moves through its opening and the solve that covers it.
// T-4264 (Anita, 11-25): a bar on a guess never shows past half; on a bounded share it runs from its place to the top,
// never back.
//   node --experimental-strip-types trainworld/checks/t2552-load-bar.mjs
import { barPlace, solveStage, zoneShare, FEW_MS, nextLeft, guessShown, GUESS_TOP, BAR_TOP } from '../app/src/game/loadBarCore.ts';
import { editLeftOf } from '../app/src/game/editLeft.ts';

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
  // the card's bar with the solve as 70% of it after a 30% data stage: the round model is a guess, so half at most
  const place = { best: 0 };
  let top = 0;
  for (const x of r) top = Math.max(top, barPlace(place, Math.min(0.99, 0.3 + 0.7 * x.share), null));
  report(`${name}: the round model's bar stays at half or under`, top <= GUESS_TOP + 1e-9, `top ${(100 * top).toFixed(1)}%`);
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

// T-4264: a bounded share (the workers' units' time left, honest here) runs the bar from where the guess left it to the
// top in step with the clock; a bound that lapses (a unit past its expected time) holds the bar, never back
{
  const place = { best: 0 };
  for (let t = 0; t < 30; t++) barPlace(place, t / 40, null);
  const from = place.best, T = 300;
  let back = false, prev = from, at90 = null;
  for (let t = 30; t < T; t++) {
    const lapsed = t >= 150 && t < 170;
    const b = barPlace(place, lapsed ? 0.95 : 0.4, lapsed ? null : t / T);
    if (b < prev - 1e-9) back = true;
    if (at90 === null && b >= 0.9) at90 = t;
    prev = b;
  }
  report('a bounded bar: from the guess\'s place to near the top', from <= GUESS_TOP && prev >= 0.95 && prev <= BAR_TOP, `from ${(100 * from).toFixed(0)}% to ${(100 * prev).toFixed(1)}%`);
  report('a bounded bar: never back, through a lapse', !back);
  report('a bounded bar: past 90% only in the last tenth of the wait', at90 !== null && at90 >= 0.85 * T, `90% at ${at90} s of ${T}`);
}

// T-4264: the workers' units bound a solve's time left only at this solve's own pace: not on the passes' prior alone,
// not past a unit's expected time, not once only the stretch after the last unit is left
{
  const prior = { pace: [0.2, 1], tailMs: 5000, gapMs: 9000 };
  const part = (o) => ({ stage: 1, busy: true, done: [0, 0], of: [0, 10], unitMs: [0, 0], at: 0, left: [0, 90_000], blind: [0, 0], cur: 10_000, curSize: 0, sizeMs: [0, 0], ran: [0, 0], ranExp: [0, 0], ranN: [0, 0], viewLeft: [0, 0], viewBlind: [0, 0], curView: false, leftQ: [], ranQ: [], expQ: [], nQ: [], curQ: -1, ...o });
  const ran = { done: [0, 2], ran: [0, 24_000], ranExp: [0, 20_000], ranN: [0, 2] };
  const cases = [
    ['the prior alone', part({}), 2000, false],
    ['two units ended here', part(ran), 2000, true],
    ['the unit in hand past its time', part(ran), 30_000, false],
    ['only the stretch after the last unit', part({ ...ran, busy: false, cur: null, left: [0, 0], done: [0, 10] }), 2000, false],
  ];
  for (const [name, p, now, want] of cases) {
    const got = editLeftOf([p], now, prior);
    report(`units bound the time left: ${name} -> ${want}`, got !== null && got.bounded === want, JSON.stringify(got));
  }
}

// T-2638: the bar shown against the clock on two measured loads (the model's fraction at each tenth of the load's
// time; notes/T-2638.md): within 12 points of the time from 30% to 80%, the time skewed so its 90% is the bar's
// half (Anita, 11-22; T-4264: a guess goes no further), and a model's 30% shown at about 7%
{
  const MEASURED = {
    'North America import': [0.29, 0.29, 0.45, 0.60, 0.68, 0.76, 0.83, 0.90, 0.97],
    'New York real save import': [0.34, 0.42, 0.52, 0.60, 0.66, 0.75, 0.84, 0.89, 0.95],
  };
  const skew = (x) => Math.min(GUESS_TOP, x * GUESS_TOP / 0.9);
  for (const [name, model] of Object.entries(MEASURED)) {
    // (to 80%: past it the skew's steep end turns a few points of the model into many of the bar)
    const off = model.map((m, i) => guessShown(m) - skew((i + 1) / 10)).slice(2, 8);
    const worst = Math.max(...off.map(Math.abs));
    report(`${name}: the bar within 12 points of the skewed time from 30% to 80%`, worst <= 0.12, `worst ${(100 * worst).toFixed(0)} points`);
  }
  report('a model\'s 30% is shown at 5-10%', guessShown(0.3) >= 0.05 && guessShown(0.3) <= 0.1, guessShown(0.3).toFixed(3));
  report('a guess never past half', guessShown(0.999) <= GUESS_TOP && guessShown(5) <= GUESS_TOP, guessShown(0.999).toFixed(3));
}

console.log(results.every(Boolean) ? 'T-2552 PASSED' : 'T-2552 FAILED');
process.exit(results.every(Boolean) ? 0 : 1);
