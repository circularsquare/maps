// T-2626: the edit note's profit estimate (game/profitEstimate.ts) against the solve, on the real New York
// save, from T-2604's experiment (artifacts/trainworld/t2604/edits.mjs): seven schedule edits, each made,
// settled, put back and settled (14 changes). For each: the estimate the note shows once the edit lands
// (running cost and fares, read from the note itself), then the weekday's network fares and running cost
// at the settled final. Reports the profit change's error (mean |estimated - solved|, worst, sign right).
// Screenshots of the note estimating and settled, light and dark, for the first edit.
//   node trainworld/checks/t2626-profit-estimate.mjs [url] [outdir]      (DEMAND_WORKERS=1 by default; ~1 hour)
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { REAL_SAVE, collectErrors, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6541/';
const OUT = process.argv[3] ?? 'C:/Users/anita/projects/maps/artifacts/trainworld/profitest';
const WORKERS = process.env.DEMAND_WORKERS ?? 1;
const H = [4, 6, 4, 4, 6], LEVEL = [0, 0, 0, 1, 2];
const DAY = 0;
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
await mkdir(OUT, { recursive: true });

const snap = (page) => page.evaluate(([H, LEVEL, di]) => {
  const tw = window.tw, w = tw.world.value, m = tw.moneyState.value, v = window.twDemand.days()[di].view;
  const inc = tw.incomeByDay.value[di];
  let fares = 0, run = 0;
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id];
    if (!st || st.status !== 'running') continue;
    const ld = v.lines.get(l.id);
    const tph = ld?.tph ?? [l.tph.high, l.tph.medium, l.tph.low];
    const cars = ld?.cars ?? st.cars;
    const perKm = st.bus?.ferry ? m.runCostBoatKm : st.bus ? m.runCostBusKm : st.fast ? m.hsRunCostTrainKm + cars * m.hsRunCostCarKm : m.runCostTrainKm + cars * m.runCostCarKm;
    H.forEach((h, p) => { run += tph[LEVEL[p]] * 2 * st.km * perKm * m.economy * h; });
    const lm = inc?.lines.get(l.id);
    if (lm) fares += lm.perHour.reduce((s, x, p) => s + x * H[p], 0) * 1e6;
  }
  return { version: v.version, final: v.final, fares, run };
}, [H, LEVEL, DAY]);

const probe = (page, id) => page.evaluate(([di, id]) => {
  const d = window.twDemand, s = d?.log?.filter((x) => !x.rough).at(-1), v = d?.days?.()[di]?.view;
  const ld = v?.lines.get(id);
  return { version: v?.version ?? 0, final: !!v?.final, logv: s?.version ?? -1, waiting: !!d?.client?.waiting, tph: ld?.tph ?? null, cars: ld?.cars ?? null, nstops: ld?.stops?.length ?? null, est: !!ld?.estimate };
}, [DAY, id]);

/** the note as the page has it: its estimate and the rows' text */
const note = (page) => page.evaluate(async () => {
  const { editNote } = await import('/src/game/editNote.ts');
  const n = editNote.value;
  const el = document.querySelector('.edit-note');
  return n ? { estimating: !!n.estimating, est: n.est ?? null, money: n.money, text: el?.innerText ?? '', tip: [...document.querySelectorAll('.edit-note .edit-row')].map((r) => r.getAttribute('data-tip')) } : null;
});

async function shoot(page, name) {
  const box = await page.evaluate(() => { const r = document.querySelector('.edit-note')?.getBoundingClientRect(); return r && { x: r.x, y: r.y, width: r.width, height: r.height }; });
  if (!box) return;
  const clip = { x: Math.max(0, box.x - 16), y: Math.max(0, box.y - 16), width: Math.max(box.width, 220) + 32, height: box.height + 32 };
  for (const shade of ['light', 'dark']) {
    await page.evaluate((s) => { document.documentElement.dataset.shade = s; }, shade);
    await page.waitForTimeout(150);
    await page.screenshot({ path: `${OUT}/${name}_${shade}.png`, clip });
  }
  await page.evaluate(() => { document.documentElement.dataset.shade = 'light'; });
}

async function settle(page, id, prev, after, shots) {
  const t0 = Date.now();
  let local = null, settledNote = null, shot = !shots;
  for (;;) {
    const p = await probe(page, id);
    const changed = p.tph && !p.est && (JSON.stringify(p.tph) !== JSON.stringify(prev.tph) || p.cars !== prev.cars || p.nstops !== prev.nstops);
    if (changed && local === null) local = (Date.now() - t0) / 1000;
    if (changed && p.final && !p.waiting && p.version === p.logv && p.version > after) {
      const n = await note(page);
      if (n && !n.estimating) {
        await page.waitForTimeout(1200);
        settledNote = await note(page);
        if (!shot) { await shoot(page, `${shots}_settled`); shot = true; }
      }
      const v = p.version, tf = (Date.now() - t0) / 1000;
      await page.waitForTimeout(settledNote ? 3000 : 12000);
      const q = await probe(page, id);
      if (q.version === v && q.final) return { local, final: tf, settledNote: settledNote ?? (await note(page)) };
    }
    if (Date.now() - t0 > 1200000) throw new Error('settle timeout');
    await page.waitForTimeout(300);
  }
}

const b = await launchChrome({ prefix: 'tw-t2626-' });
const res = { when: new Date().toISOString(), base: BASE, workers: WORKERS, edits: [] };
const errors = [];
const save = () => writeFile(`${OUT}/estimate.json`, JSON.stringify(res, null, 1));
try {
  const page = await b.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&lod=fine&demandWorkers=${WORKERS}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  log('loaded');
  await page.waitForFunction((di) => { const d = window.twDemand, v = d?.days?.()[di]?.view, s = d?.log?.filter((x) => !x.rough).at(-1); return v && s && v.final && v.version === s.version && v.version > 0 && !d.client.waiting && v.lines.size > 100; }, DAY, { timeout: 900000, polling: 1000 });
  await page.waitForTimeout(15000);
  log('base final');
  const picks = await page.evaluate((di) => {
    const w = window.tw.world.value, v = window.twDemand.days()[di].view;
    const ls = w.save.lines.filter((l) => { const s = w.lineStats[l.id]; return s && s.status === 'running' && !s.bus && v.lines.get(l.id); });
    ls.sort((a, b) => v.lines.get(b.id).ridersPerDay - v.lines.get(a.id).ridersPerDay);
    return { count: ls.length, ranked: ls.map((l) => ({ id: l.id, num: l.num, name: l.name, cars: l.cars ?? 0, trains: w.lineStats[l.id].trainsNeeded, stops: [...l.stops] })) };
  }, DAY);
  const RANKS = (process.env.RANKS ?? '0,4,15,40,80').split(',').map(Number).filter((r) => r < picks.count);
  const plan = [];
  RANKS.forEach((r, i) => {
    const l = picks.ranked[r], t = l.trains, f = i % 2 === 0 ? 1.5 : 0.67;
    let hi = Math.max(1, Math.round(t[0] * f));
    if (hi === t[0]) hi = f > 1 ? t[0] + 1 : Math.max(1, t[0] - 1);
    plan.push({ rank: r, line: l, kind: 'trains', go: { op: 'setTrainCount', line: l.num, trains: [hi, t[1], t[2]] }, back: { op: 'setTrainCount', line: l.num, trains: t } });
  });
  { const l = picks.ranked[1]; plan.push({ rank: 1, line: l, kind: 'cars', go: null, back: { op: 'lineCars', line: l.num, cars: l.cars } }); }
  { const l = picks.ranked[3]; plan.push({ rank: 3, line: l, kind: 'dropStop', go: { op: 'dropStop', line: l.num, index: Math.floor(l.stops.length / 2) }, back: { op: 'setStops', line: l.num, stops: l.stops } }); }
  let cur = await snap(page);
  const detail = (id) => page.evaluate(([di, id]) => { const ld = window.twDemand.days()[di].view.lines.get(id); return { tph: ld.tph, cars: ld.cars, nstops: ld.stops?.length }; }, [DAY, id]);
  let n = 0;
  for (const p of plan) {
    const id = p.line.id;
    for (const dir of ['go', 'back']) {
      const before = cur, d0 = await detail(id);
      let op = p[dir];
      if (p.kind === 'cars' && dir === 'go') op = { op: 'lineCars', line: p.line.num, cars: Math.max(2, d0.cars - 2) };
      const first = n++ === 0;
      const t0 = Date.now();
      // through the game's own `edit()`, which notes the edit for the edit note
      const ok = await page.evaluate(async (op) => (await (await import('/src/game/world.ts')).edit(op)).ok, op);
      log(p.kind, dir, p.line.name, JSON.stringify(op), ok);
      // the estimate: the note estimating (up to 8 s; a stop edit gets none)
      let est = null;
      for (let k = 0; k < (p.kind === 'dropStop' || dir === 'back' && p.kind === 'dropStop' ? 20 : 60); k++) {
        const x = await note(page);
        if (x?.estimating) { est = x; break; }
        await page.waitForTimeout(150);
      }
      const estSecs = (Date.now() - t0) / 1000;
      if (est && first) { await page.waitForTimeout(300); await shoot(page, 'estimating'); }
      const s = await settle(page, id, d0, before.version, first ? 'note' : null);
      cur = await snap(page);
      const e = { kind: p.kind, dir, name: p.line.name, op, ok, estSecs, estimate: est && { running: est.est.running, fares: est.est.fares, text: est.text, tip: est.tip },
        local: s.local, final: s.final, settled: s.settledNote && { text: s.settledNote.text, tip: s.settledNote.tip },
        solved: { fares: cur.fares - before.fares, running: cur.run - before.run } };
      res.edits.push(e);
      log('  estimate', est ? `${Math.round(est.est.fares - est.est.running)}` : 'none', 'solved', Math.round(e.solved.fares - e.solved.running), 'in', estSecs, 's');
      await save();
    }
  }
  // the report
  const rows = res.edits.filter((e) => e.kind !== 'dropStop');
  const err = rows.map((e) => Math.abs((e.estimate ? e.estimate.fares - e.estimate.running : 0) - (e.solved.fares - e.solved.running)));
  const sign = rows.filter((e) => Math.sign(e.estimate ? e.estimate.fares - e.estimate.running : 0) === Math.sign(e.solved.fares - e.solved.running)).length;
  res.summary = { changes: rows.length, meanErrorM: err.reduce((a, c) => a + c, 0) / err.length / 1e6, worstM: Math.max(...err) / 1e6, signRight: `${sign}/${rows.length}`,
    stopEditsWithEstimate: res.edits.filter((e) => e.kind === 'dropStop' && e.estimate).length };
  log('SUMMARY', JSON.stringify(res.summary));
} catch (e) {
  log('FAILED', e.message);
  res.error = e.message;
} finally {
  res.errors = errors.slice(0, 8);
  await save();
  await b.close();
}
