// T-2682: the edit note settles in a new New York game whose zone in view is still shown rough (coarse)
// when the player edits. Each step waits for the settled note (not the T-2626 estimate), LIMIT_S at most:
// a line built the moment the game opens (still loading), with more trains on it the moment New York shows
// its first, coarse, figures (one note for both); more trains again once settled; a second line after.
// Before T-2682's fix the first step gave no note (the edits came before any figures to compare against). A trace of the edit note's inputs every
// 250 ms goes to artifacts/trainworld/t2682/trace.jsonl.
//   TW_CORES=2 node trainworld/checks/t2682-editnote.mjs [url]
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, NOISE, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6174/';
const OUT = join(ARTIFACTS, 't2682');
const LIMIT_S = Number(process.env.LIMIT_S ?? 150);
await mkdir(OUT, { recursive: true });
const t0 = Date.now();
const log = (...a) => console.log(`${((Date.now() - t0) / 1000).toFixed(1).padStart(6)}s`, ...a);
let fails = 0;
const check = (name, ok, detail = '') => { if (!ok) fails++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); };

const LINE1 = [[-73.995, 40.728], [-73.988, 40.742], [-73.981, 40.756], [-73.974, 40.770]];
const LINE2 = [[-73.990, 40.692], [-73.975, 40.686], [-73.960, 40.680]];

/** samples what the note reads, every 250 ms, into window.__t2682 (a row only when something changed) */
const startTrace = (page) => page.evaluate(async () => {
  const d = await import('/src/game/demand.ts'), n = await import('/src/game/editNote.ts'), lp = await import('/src/game/loading.ts');
  const rows = window.__t2682 = [];
  let last = '';
  const t0 = performance.now();
  setInterval(() => {
    const w = window.tw.world.value, v = d.demandView.peek(), note = n.editNote.peek();
    const r = {
      world: w?.version ?? -1, net: d.networkWorld.peek(), fin: d.finalWorld.peek(), upd: d.demandUpdating.peek(), settling: d.editsSettling.peek(),
      loading: !!lp.loadProgress.peek(), rough: [...d.roughZones.peek()].sort().join(), refining: d.refiningZone.peek(), view: v?.version ?? -1, final: !!v?.final,
      lines: w?.save.lines.map((l) => `${l.num}:${w.lineStats[l.id]?.status}`).join(),
      note: note ? (note.estimating ? 'estimating' : `trips ${note.trips.from}->${note.trips.to}`) : null,
    };
    const k = JSON.stringify(r);
    if (k !== last) { last = k; rows.push({ s: +((performance.now() - t0) / 1000).toFixed(2), ...r }); }
  }, 250);
});

/** the settled note (not the estimate), or null after LIMIT_S */
const dump = async (page) => writeFile(join(OUT, 'trace.jsonl'), (await page.evaluate(() => window.__t2682)).map((r) => JSON.stringify(r)).join('\n') + '\n');
const waitNote = async (page) => {
  const r = await waitNote1(page);
  await dump(page);
  return r;
};
const waitNote1 = async (page) => {
  const t = Date.now();
  while (Date.now() - t < LIMIT_S * 1000) {
    const n = await page.evaluate(async () => { const v = (await import('/src/game/editNote.ts')).editNote.value; return v && !v.estimating ? { trips: v.trips, money: v.money, line: v.line?.id ?? null } : null; });
    if (n) return { n, s: (Date.now() - t) / 1000 };
    await page.waitForTimeout(250);
  }
  return { n: null, s: LIMIT_S };
};

const state = (page) => page.evaluate(async () => {
  const d = await import('/src/game/demand.ts');
  return { rough: [...d.roughZones.peek()], refining: d.refiningZone.peek(), fin: d.finalWorld.peek(), world: window.tw.world.value.version };
});

/** a line through `pts` ([lng, lat]) through the game's own edit(), so the note counts it (lib's buildLine goes round it) */
const build = (page, pts, name, colour) => page.evaluate(async ([PTS, name, colour]) => {
  const tw = window.tw, { edit } = await import('/src/game/world.ts');
  const xy = PTS.map(([lng, lat]) => { const p = tw.lngLatToPack(lng, lat); return [p.x, p.y]; });
  const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
  const pause = (ms) => new Promise((d) => setTimeout(d, ms));
  await edit({ op: 'route', from: { kind: 'free', x: xy[0][0], y: xy[0][1], level: -1 }, to: { kind: 'free', x: xy[1][0], y: xy[1][1], level: -1 }, pis: [], single: false });
  for (let k = 2; k < xy.length; k++) { await pause(300); await edit({ op: 'route', from: { kind: 'node', node: near(xy[k - 1]) }, to: { kind: 'free', x: xy[k][0], y: xy[k][1], level: -1 }, pis: [], single: false }); }
  await pause(500);
  for (const p of xy) { await edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await pause(200); }
  await edit({ op: 'constructAll' });
  await pause(500);
  const r = await edit({ op: 'addLine', stops: xy.map(near), name, colour });
  if (r?.line !== undefined) await edit({ op: 'setTrainCount', line: r.line, trains: [6, 4, 3] });
  return r?.line ?? null;
}, [pts, name, colour]);

const { ctx, close } = await launchChrome({ prefix: 'tw-t2682-' });
const errors = [];
try {
  const page = await ctx.newPage();
  collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => void window.tw.map.jumpTo({ center: [-73.982, 40.752], zoom: 12.5 }));
  await startTrace(page);
  // the line at once, while the game still loads; then, the moment New York shows its first (coarse)
  // figures and its fine ones are being worked out, more trains on it: one note for both
  const s0 = await state(page);
  log('line 1 at', JSON.stringify(s0));
  const one = await build(page, LINE1, 'Check', '#0a7d3b');
  let s1 = await state(page);
  for (let t = Date.now(); !s1.rough.length && Date.now() - t < 60000; s1 = await state(page)) await page.waitForTimeout(100);
  const early = await page.evaluate(async () => { const v = (await import('/src/game/editNote.ts')).editNote.value; return !!v && !v.estimating; });
  log('trains while rough at', JSON.stringify(s1), early ? '(the line note came first)' : '');
  await page.evaluate(async (num) => (await import('/src/game/world.ts')).edit({ op: 'setTrainCount', line: num, trains: [8, 6, 4] }), one);
  const a = await waitNote(page);
  log('line 1 note', JSON.stringify(a.n));
  check('the zone was shown rough at the trains edit', s1.rough.length > 0 && !early, `rough: ${s1.rough.join(',') || 'none'}, refining: ${s1.refining}`);
  check('a line built while the game loads, and trains added while its zone is rough, give a note', !!a.n && a.n.trips.from === 0 && a.n.trips.to > 0, `${a.s.toFixed(1)} s after the trains edit`);
  await page.waitForTimeout(9000); // the note goes after NOTE_MS
  const s2 = await state(page);
  log('trains edit at', JSON.stringify(s2));
  await page.evaluate(async (num) => (await import('/src/game/world.ts')).edit({ op: 'setTrainCount', line: num, trains: [12, 8, 6] }), one);
  const b = await waitNote(page);
  log('trains note', JSON.stringify(b.n));
  check('more trains on it gives a note', !!b.n, `${b.s.toFixed(1)} s; rough at the edit: ${s2.rough.join(',') || 'none'}`);
  await page.waitForTimeout(9000);
  const s3 = await state(page);
  log('line 2 at', JSON.stringify(s3));
  await build(page, LINE2, 'Second', '#b0372a');
  const c = await waitNote(page);
  log('line 2 note', JSON.stringify(c.n));
  check('a second line built afterwards gives a note', !!c.n, `${c.s.toFixed(1)} s; rough at the edit: ${s3.rough.join(',') || 'none'}`);
  await page.close();
  check('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
console.log(`${fails ? 'FAIL' : 'PASS'}  t2682 edit note, ${((Date.now() - t0) / 1000).toFixed(0)} s; trace ${join(OUT, 'trace.jsonl')}`);
process.exit(fails ? 1 : 0);
