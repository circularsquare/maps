// T-3298 scene: a tram line A-B' and a train line B-C, B' `gap` metres west of B (vertical lines).
//   node scene.mjs <gap m> [street] [url]
import { openGame } from '../checks/lib/play.mjs';

const args = process.argv.slice(2);
const GAP = Number(args[0] ?? 250) / 1000;
const STREET = args.includes('street');
const BASE = args.find((a) => /^https?:/.test(a)) ?? 'http://localhost:6197/';
const C = [-73.93, 40.66];
const P = (dx, dy) => [C[0] + dx / (111.32 * Math.cos((C[1] * Math.PI) / 180)), C[1] + dy / 111.32];
const TRAM_LEN = Number(process.env.TRAM_KM ?? 2.5);

const g = await openGame({ url: BASE, city: 'nyc', mode: 'sandbox', ui: { shade: 'light', display: { dayShown: 'weekday' }, v: 2 }, prefix: 'tw-t3641-', query: { demandWorkers: 1 } });
try {
  await g.act.camera(P(0, 2), 12.5);
  // train: B at (0,0), C at (0,6)
  await g.act.tool('track', { type: 'regular', tracks: 2, level: 0, snap: { real: false, roads: false }, keepPlacing: false });
  const rail = await g.act.drawTrack([P(0, -0.4), P(0, 2), P(0, 4.3)]);
  console.log('rail', rail.ok);
  await g.act.camera(P(0, 0), 15);
  console.log('B', JSON.stringify(await g.act.placeStation(P(0, 0))));
  await g.act.camera(P(0, 4), 15);
  console.log('C', JSON.stringify(await g.act.placeStation(P(0, 4))));
  // tram: A at (-gap, -TRAM_LEN), B' at (-gap, 0)
  await g.act.camera(P(-GAP, -1.2), 12.8);
  await g.act.tool('track', { type: 'tram', tracks: 2, level: 0, snap: { real: false, streets: STREET }, keepPlacing: false });
  const tram = await g.act.drawTrack([P(-GAP, -TRAM_LEN - 0.2), P(-GAP, -TRAM_LEN / 2), P(-GAP, 0.15)]);
  console.log('tram', tram.ok, JSON.stringify(tram.segments.map((s) => s.notes)).slice(0, 300));
  await g.act.camera(P(-GAP, -TRAM_LEN), 15);
  console.log('A', JSON.stringify(await g.act.placeStation(P(-GAP, -TRAM_LEN))).slice(0, 200));
  await g.act.camera(P(-GAP, 0), 15);
  console.log("B'", JSON.stringify(await g.act.placeStation(P(-GAP, 0))).slice(0, 200));
  await g.act.construct();
  const train = await g.act.newLine([P(0, 0), P(0, 4)], { name: 'Train', zoom: 15 });
  await g.act.setTrains([6, 4, 3]);
  const tl = await g.act.newLine([P(-GAP, -TRAM_LEN), P(-GAP, 0)], { name: 'Yellow', zoom: 15 });
  for (const [i, n] of [6, 4, 3].entries()) await g.act.type('Type trams on the line', String(n), { within: '#inspector', nth: i });
  await g.page.waitForTimeout(800);
  console.log('lines', JSON.stringify([train, tl]).slice(0, 400));
  const st = await g.page.evaluate(() => window.tw.world.value.save.stations.map((s) => ({ num: s.num, name: s.name, tram: s.tram, complex: s.complex ?? s.group ?? null, x: s.x, y: s.y })));
  console.log('stations', JSON.stringify(st));
  if (args.includes('link')) {
    const r = await g.page.evaluate(async () => { const st = window.tw.world.value.save.stations; const b = st.find((s) => !s.tram && s.name === st.find((x) => !x.tram).name), b2 = st.filter((s) => s.tram).at(-1); return window.tw.client.edit({ op: 'link', a: b.num, b: b2.num, on: true }); });
    console.log('link', JSON.stringify(r).slice(0, 300));
  }
  await g.waitDemand({ timeout: 600000 });
  const sched = await g.page.evaluate(() => { const w = window.tw.world.value; return w.save.lines.map((l) => ({ name: l.name, stats: w.lineStats[l.id] ? Object.fromEntries(Object.entries(w.lineStats[l.id]).filter(([k, v]) => typeof v !== 'object')) : null })); });
  console.log('lineStats', JSON.stringify(sched).slice(0, 1500));
  for (const [label, from, to] of [['home near A to C', P(-GAP - 0.1, -TRAM_LEN), P(0.1, 4)], ['C to home near A', P(0.1, 4), P(-GAP - 0.1, -TRAM_LEN)]]) {
    await g.act.camera(P(0, 1), 12);
    await g.page.evaluate(() => { window.tw.selection.value = null; });
    await g.page.evaluate(([a, b]) => {
      const p = window.tw.lngLatToPack(a[0], a[1]), q = window.tw.lngLatToPack(b[0], b[1]);
      window.tw.selection.value = { kind: 'trip', from: [p.x, p.y], to: [q.x, q.y] };
    }, [from, to]);
    await g.page.waitForFunction(() => /By car/.test(document.querySelector('#inspector')?.innerText ?? ''), undefined, { timeout: 120000, polling: 500 });
    await g.page.waitForTimeout(1500);
    const text = await g.page.evaluate(() => document.querySelector('#inspector')?.innerText ?? '');
    console.log('MIX', await g.page.evaluate(() => window.__mix));
    console.log(`==== ${label}\n${text}\n====`);
    await g.shot(`t3641/gap${Math.round(GAP * 1000)}${STREET ? 's' : ''}${args.includes('link') ? 'link' : ''}-${label.replace(/ /g, '_')}.png`);
  }
  console.log('errors', g.pageErrors?.().slice(0, 3));
} catch (e) {
  console.log('FAILED', e.stack ?? e.message);
  await g.shot('t3641/fail.png').catch(() => {});
} finally {
  await g.close();
}







