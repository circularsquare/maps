// T-2773: after a small headway change, does any line's shown riders (or money) dip while pending?
//   node trainworld/checks/t2773-pending-dip.mjs [url]      (from the maps root)
// Samples every 0.5 s: each line's shown riders (pending figure or the count), its profit shown, the pending
// flag, and the income the clock is paid (sum of incomeByDay); prints each line's worst dip against its settled value.
import { readFile } from 'node:fs/promises';
import { launchChrome, importSave, ANGLE, REAL_SAVE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6171/';
const { ctx, close } = await launchChrome({ prefix: 'tw-t2773-', size: [1200, 800], args: ANGLE });
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.goto(`${BASE}?debug=1&new=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) >= 20, undefined, { timeout: 300000, polling: 250 });
  await page.waitForFunction(() => { const d = window.twDemand, v = d?.view?.(); return !!v?.final && !d.client.waiting; }, undefined, { timeout: 300000, polling: 500 }).catch(() => console.log('no final'));
  await page.waitForTimeout(4000);
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.selection.value = null; ui.tab.value = 'money'; await new Promise((r) => setTimeout(r, 300)); ui.tab.value = 'lines'; });
  await page.waitForFunction(async () => { const td = await import('/src/game/today.ts'); const d = window.twDemand, v = d?.view?.(); return td.today.value.size > 5 && !!v?.final && !d.client.waiting && !document.querySelector('.load-card, .loading-card'); }, undefined, { timeout: 500000, polling: 1000 }).catch(() => console.log('no today'));
  await page.waitForTimeout(3000);
  await page.screenshot({ path: 'C:/Users/anita/projects/maps/artifacts/trainworld/riderdip/probe.png' });
  const r = await page.evaluate(async () => {
    const tw = window.tw;
    const td = await import('/src/game/today.ts');
    const lf = await import('/src/game/lineFigures.ts');
    const mm = await import('/src/game/money.ts');
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const ui = await import('/src/game/ui.ts');
    ui.selection.value = null; ui.tab.value = 'money'; await sleep(300); ui.tab.value = 'lines';
    await sleep(12000);
    const w = () => tw.world.value;
    const snap = () => {
      const o = {}, tod = td.today.value;
      for (const l of w().save.lines) {
        const st = w().lineStats[l.id];
        if (st?.status !== 'running') continue;
        const k = td.pendingOf(l);
        const d = tod.get(l.id);
        const riders = k ? td.pendingRiders(l) : d?.riders ?? null;
        const net = d && td.profitKnown(l) ? d.dayFares - td.runningToday(l, st) - st.upkeep : null;
        o[l.id] = { riders, net, pend: !!k };
      }
      const inc = mm.incomeByDay.value[0];
      return { lines: o, income: inc ? inc.perHour.reduce((a, b) => a + b, 0) : 0, t: performance.now() };
    };
    const base = snap();
    // the busiest line, its headway 15% longer
    const tod = td.today.value;
    const l = w().save.lines.filter((x) => w().lineStats[x.id]?.status === 'running').sort((a, b) => (tod.get(b.id)?.riders ?? 0) - (tod.get(a.id)?.riders ?? 0))[0];
    const tph = [l.tph.high, l.tph.medium, l.tph.low];
    const secs = tph.map((n) => (n > 0 ? Math.round((3600 / n) * 1.4) : 0));
    const out = { line: l.name, tph, secs, samples: [] };
    const t0 = performance.now();
    const editP = tw.client.edit({ op: 'setHeadway', line: Number(l.num), seconds: secs }).then((a) => { out.ok = a?.ok; out.after = JSON.stringify(w().save.lines.find((x) => x.id === l.id).tph); });
    const solving = () => { const d = window.twDemand, v = d?.view?.(); return !(v?.final && !d.client.waiting); };
    let settledFor = 0;
    while (performance.now() - t0 < 150000) {
      const s = snap(); s.t = Math.round(s.t - t0); s.solving = solving(); s.npend = lf.linesPending.value.size;
      out.samples.push(s);
      settledFor = !s.solving && !s.npend && s.t > 12000 ? settledFor + 1 : 0;
      if (settledFor > 6) break;
      await sleep(500);
    }
    out.base = base;
    return out;
  });
  const fin = r.samples[r.samples.length - 1];
  console.log('edited', r.line, JSON.stringify(r.tph), '->', JSON.stringify(r.secs), 'ok', r.ok, 'after', r.after, 'samples', r.samples.length, 'first-settled-after-ms', fin.t);
  const worst = [];
  for (const id of Object.keys(r.base.lines)) {
    const b = r.base.lines[id], f = fin.lines[id];
    let lo = Infinity, loNet = Infinity, seenPend = false;
    for (const s of r.samples) { const x = s.lines[id]; if (!x) continue; if (x.pend) seenPend = true; if (x.riders !== null && x.riders < lo) lo = x.riders; if (x.net !== null) loNet = Math.min(loNet, x.net); }
    worst.push({ id, before: Math.round(b.riders ?? 0), final: Math.round(f?.riders ?? 0), min: Math.round(lo), dip: b.riders ? +(1 - lo / Math.min(b.riders, f?.riders ?? b.riders)).toFixed(3) : 0, netBefore: Math.round(b.net ?? 0), netFinal: Math.round(f?.net ?? 0), netMin: Math.round(loNet), seenPend });
  }
  worst.sort((a, b) => b.dip - a.dip);
  console.log('worst lines by dip:'); for (const x of worst.slice(0, 8)) console.log(JSON.stringify(x));
  console.log('pending lines max', Math.max(...r.samples.map((s) => s.npend)));
  const inc = r.samples.map((s) => s.income), i0 = r.base.income;
  console.log('income/day sum: base', i0.toFixed(3), 'min', Math.min(...inc).toFixed(3), 'final', fin.income.toFixed(3));
  console.log('timeline (t, solving, npend, income):'); for (const s of r.samples.filter((_, i) => i % 4 === 0)) console.log(s.t, s.solving, s.npend, s.income.toFixed(3));
} finally {
  await close();
}
