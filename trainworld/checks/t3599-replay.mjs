// T-3599: replays the edits t3599-time-left.mjs recorded (artifacts/trainworld/t3599/<name>.json) against the
// time-left estimate (app/src/game/editLeft.ts, smoothed by loadBarCore.ts `nextLeft`), with no browser: at each
// tenth of each wait, the estimate against the time really left (to the end of the "being recomputed" tag while
// it shows, else to the final). For trying a change to the estimate without solving again. Each file starts from
// the first-edit guesses, and each run from what the runs before it left, as in play (`noteEditSolve`).
//   node trainworld/checks/t3599-replay.mjs <name.json> [...]
import { readFileSync } from 'node:fs';
import { join, isAbsolute } from 'node:path';
import { ARTIFACTS, loadApp } from './lib.mjs';

const { editLeft, editPrior, noteEditSolve, nextLeft, PACE_GUESS, TAIL_GUESS_MS, GAP_GUESS_MS } = await loadApp("export { editLeft, editPrior, noteEditSolve, PACE_GUESS, TAIL_GUESS_MS, GAP_GUESS_MS } from '@app/src/game/editLeft.ts'; export { nextLeft } from '@app/src/game/loadBarCore.ts';");
const files = process.argv.slice(2);
if (!files.length) { console.log('usage: t3599-replay.mjs <name.json> ...'); process.exit(2); }
const show = (ms) => (ms === null ? 'none' : ms < 60000 ? '<1 min' : `${Math.round(ms / 60000)} min`);
let off = 0, total = 0, errSum = 0, errN = 0;
for (const f of files) {
  const d = JSON.parse(readFileSync(isAbsolute(f) ? f : join(ARTIFACTS, 't3599', f), 'utf8'));
  Object.assign(editPrior, { pace: [...PACE_GUESS], tailMs: TAIL_GUESS_MS, gapMs: GAP_GUESS_MS });
  for (const r of d.runs) {
    const s = r.solves.find((x) => x.version === r.version) ?? r.solves[0];
    const work = s?.work ?? [];
    const n = Math.max(0, ...work.map((w) => w.i + 1));
    const real = Math.max(r.finalMs, r.tagEndMs);
    console.log(`\n${f} ${r.spec}: ${r.what}; ${(real / 1000).toFixed(0)} s, ${n} workers, ${work.length} work messages; prior pace ${editPrior.pace.map((x) => x.toFixed(2)).join('/')}, tail ${(editPrior.tailMs / 1000).toFixed(1)} s, gap ${(editPrior.gapMs / 1000).toFixed(1)} s`);
    const REC = /being recomputed/;
    const recEnd = (r.samples.filter((x) => REC.test(x.text ?? '')).at(-1)?.t ?? -250) + 250;
    const shown = { all: null, view: null };
    let k = 0;
    const last = Array.from({ length: n }, () => null);
    const line = [];
    for (let t = 0; t <= real; t += 250) {
      while (k < work.length && work[k].at <= t) { last[work[k].i] = work[k]; k++; }
      const part = t < recEnd ? 'view' : 'all';
      const all = n > 0 && last.every((x) => x !== null);
      const est = all ? editLeft(last, t, editPrior, part) : null;
      const was = shown[part];
      if (est !== null) shown[part] = { ms: nextLeft(was ? was.ms : null, est, was ? t - was.at : 0), at: t };
      const s = shown[part];
      line.push({ t, est, shown: s ? Math.max(0, s.ms - (t - s.at)) : null, part });
    }
    for (let q = 0.1; q < 0.95; q += 0.1) {
      const at = q * real, x = line.findLast((y) => y.t <= at), left = (x?.part === 'view' ? recEnd : real) - at;
      const v = x?.shown ?? null;
      const ok = v === null || (left < 60000 ? v < 80000 : v < 60000 ? left < 80000 : Math.abs(v - left) <= Math.max(left / 3, 20000));
      if (q > 0.2) { total++; if (!ok) off++; if (v !== null && left > 20000) { errSum += Math.abs(v - left) / left; errN++; } }
      console.log(`  ${Math.round(q * 100)}% ${x?.part ?? ''}: ${(left / 1000).toFixed(0)} s left, estimate ${x?.est === null ? 'none' : (x.est / 1000).toFixed(0) + ' s'}, shown ${v === null ? 'none' : (v / 1000).toFixed(0) + ' s'} (${show(v)})${ok ? '' : '  <- off'}`);
    }
    // what the next solve here starts from, as demandClient.ts notes it at the final
    const ends = work.filter((w) => !w.busy && w.done[0] + w.done[1] > 0);
    const lastEnd = ends.length ? ends.at(-1).at : null;
    const finals = Array.from({ length: n }, () => null);
    for (const w of work) finals[w.i] = w;
    const localEnd = work.filter((w) => !w.busy && w.stage === 0 && w.done[0] > 0).at(-1)?.at, wholeAt = work.find((w) => w.busy && w.stage === 1)?.at;
    if (n && finals.every((x) => x !== null) && lastEnd !== null) noteEditSolve(finals, r.finalMs - lastEnd, localEnd !== undefined && wholeAt !== undefined && wholeAt > localEnd ? wholeAt - localEnd : null);
  }
}
console.log(`\n${off} of ${total} samples past 20% off by more than a third; mean error ${errN ? Math.round((100 * errSum) / errN) : 0}% where over 20 s were left`);
