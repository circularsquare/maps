// T-2236: the map's crowding marks show trains' load past the rated load. On the real save (New York),
// in the Traffic track display, it checks that every crowding mark sits on a stretch whose trains run
// past 100% of their rated load and none on a stretch under it, against an estimate of its own: the
// drawn trips' account loads over a whole day (every hour's slice), each trip binned by when it leaves
// the stretch's first stop (its line's stop times), the busiest hour's mean per direction over the
// rated load. Then light and dark screenshots of the most crowded stretch with its hover.
//   node trainworld/checks/t2236-crowding.mjs <url> [outdir]      (from the maps root)
// Prints one JSON summary and PASS or FAIL; writes <outdir>/t2236-crowding.json.
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, REAL_SAVE, ARTIFACTS } from './lib.mjs';

const [BASE, OUT = `${ARTIFACTS}t2236`] = process.argv.slice(2);
if (!BASE) { console.error('usage: node checks/t2236-crowding.mjs <url> [outdir]'); process.exit(2); }
/** the estimate bins trips by their timetable, the account by its own: within this of 100% the two may differ */
const MARGIN = 0.05;
await mkdir(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t2236-', size: [1100, 760] });
let pass = false;
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log(`page error: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error' && !/favicon|openfreemap|AJAXError/.test(m.text())) console.log(`console: ${m.text().slice(0, 300)}`); });
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday', trackColour: 'traffic', fullMarks: true }, v: 2 })); } catch { /* */ } });
  await page.goto(`${BASE}${BASE.includes('?') ? '&' : '?'}debug=1&new=1&opened=all&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 100, undefined, { timeout: 120000 });
  await demandFinal(page, { polling: 250 });
  const day = await page.evaluate(() => {
    const c = window.tw.clock, d = Math.floor(c.now() / 86400);
    c.setSpeed?.(0);
    window.tw.display.value = { ...window.tw.display.peek(), trackColour: 'traffic', fullMarks: true };
    return d;
  });
  // every hour's slice of the day: each trip's loads once
  const trips = {};
  for (let h = 0; h < 24; h++) {
    const epoch = day * 86400 + h * 3600;
    await page.evaluate((t) => window.tw.clock.set(t), epoch + 1800);
    await page.waitForFunction((e) => window.tw.renderer.board?.epoch === e, epoch, { timeout: 60000, polling: 100 });
    Object.assign(trips, await page.evaluate((e) => {
      const r = window.tw.renderer, b = r.board, d = r.trips, out = {};
      for (let i = 0; i < b.trips; i++) {
        const o = b.offs[i * 2];
        if (!(o >= 0)) continue;
        const prof = d[i * 2], dep = e + d[i * 2 + 1];
        const line = Math.floor(prof / 6), run = Math.floor(prof / 3) % 2, level = prof % 3;
        const n = window.tw.world.value.save.lines.find((l) => l.num === line)?.stops.length ?? 0;
        if (n < 2) continue;
        out[`${prof}|${Math.round(dep)}`] = { line, run, level, dep, loads: Array.from(b.loads.subarray(o, o + n - 1)) };
      }
      return out;
    }, epoch));
  }
  // back to 08:00, and the marks drawn then
  await page.evaluate((t) => window.tw.clock.set(t), day * 86400 + 8 * 3600 + 600);
  await page.waitForFunction((e) => window.tw.renderer.board?.epoch === e, day * 86400 + 8 * 3600, { timeout: 60000, polling: 100 });
  await page.waitForTimeout(1500);
  const drawn = await page.evaluate(() => {
    const tw = window.tw, r = tw.renderer, L = r.net.lines, f = r.linesFull;
    const keys = Object.fromEntries([...r.fullKeys].map(([k, v]) => [k, v]));
    const segs = {}, wrong = [];
    for (let i = 0; i < L.count; i++) {
      const v = f[i * 2];
      if (!(v > 0)) continue;
      const k = L.colour[i] * 4096 + L.stretch[i], c = keys[k];
      if (!c || Math.abs(c.load - v) > 1e-4) wrong.push({ seg: i, key: k, drawn: v, key_load: c?.load ?? null });
      (segs[k] ??= []).push(i);
    }
    const lines = {};
    for (const l of tw.world.value.save.lines) {
      const st = tw.world.value.lineStats[l.id];
      if (st?.status !== 'running') continue;
      lines[l.num] = { name: l.name, stops: l.stops.length, cars: st.cars, bus: !!st.bus, fromStart: st.fromStartByLevel ?? [], fromStartS: st.fromStartS };
    }
    return { keys, segs, wrong, lines };
  });
  // the estimate: per line, stretch, direction and hour, the mean load of the trips leaving its first stop
  const sums = new Map();
  for (const t of Object.values(trips)) {
    const L = drawn.lines[t.line];
    if (!L || L.bus) continue;
    const fs = L.fromStart[t.level]?.length === L.stops ? L.fromStart[t.level] : L.fromStartS;
    const end = fs[fs.length - 1] ?? 0;
    for (let s = 0; s < t.loads.length; s++) {
      const at = t.run === 0 ? fs[s] : end - fs[s + 1];
      const h = Math.floor((((t.dep + (at ?? 0)) % 86400) + 86400) % 86400 / 3600);
      const k = `${t.line * 4096 + s}|${t.run}|${h}`;
      const a = sums.get(k) ?? [0, 0];
      a[0] += t.loads[s]; a[1] += 1;
      sums.set(k, a);
    }
  }
  const est = new Map();
  for (const [k, [sum, n]] of sums) {
    const key = Number(k.split('|')[0]), L = drawn.lines[Math.floor(key / 4096)];
    const v = sum / n / (160 * Math.max(1, L.cars));
    if (v > (est.get(key) ?? 0)) est.set(key, v);
  }
  const marked = Object.keys(drawn.keys).map(Number);
  const markedUnder = marked.filter((k) => (est.get(k) ?? 0) < 1).map((k) => ({ key: k, line: drawn.lines[Math.floor(k / 4096)]?.name, stretch: k % 4096, mark: drawn.keys[k].load, estimate: est.get(k) ?? 0 }));
  const unmarkedOver = [...est].filter(([k, v]) => v >= 1.005 && !drawn.keys[k]).map(([k, v]) => ({ key: k, line: drawn.lines[Math.floor(k / 4096)]?.name, stretch: k % 4096, estimate: v }));
  const undrawn = marked.filter((k) => !drawn.segs[k]?.length);
  const far = [...markedUnder.filter((x) => x.estimate < 1 - MARGIN), ...unmarkedOver.filter((x) => x.estimate > 1 + MARGIN)];
  const diffs = marked.filter((k) => est.has(k)).map((k) => Math.abs(drawn.keys[k].load - est.get(k)));
  const summary = {
    trips: Object.keys(trips).length, marks: marked.length, stretches: est.size,
    stretchesOver100: [...est.values()].filter((v) => v >= 1.005).length,
    markLoad: { min: Math.min(...marked.map((k) => drawn.keys[k].load)), max: Math.max(...marked.map((k) => drawn.keys[k].load)) },
    markVsEstimate: { mean: diffs.reduce((a, b) => a + b, 0) / Math.max(1, diffs.length), max: Math.max(0, ...diffs) },
    segmentsWrong: drawn.wrong.length, marksNotDrawn: undrawn.length,
    markedUnder100: markedUnder.length, unmarkedOver100: unmarkedOver.length, outsideMargin: far.length,
  };
  pass = marked.length > 0 && drawn.wrong.length === 0 && far.length === 0 && Math.min(...marked.map((k) => drawn.keys[k].load)) >= 1.005;
  await writeFile(`${OUT}/t2236-crowding.json`, JSON.stringify({ summary, markedUnder, unmarkedOver, far, wrong: drawn.wrong, undrawn }, null, 1));
  console.log(JSON.stringify(summary));

  // the most crowded drawn stretch, hovered, in both shades
  const best = marked.filter((k) => drawn.segs[k]?.length).sort((a, b) => drawn.keys[b].load - drawn.keys[a].load)[0];
  if (best !== undefined) {
    const segs = drawn.segs[best];
    const p = await page.evaluate((i) => {
      const tw = window.tw, s = tw.renderer.net.lines.seg;
      const [lng, lat] = tw.toLngLat((s[4 * i] + s[4 * i + 2]) / 2, (s[4 * i + 1] + s[4 * i + 3]) / 2);
      return { lng, lat };
    }, segs[segs.length >> 1]);
    for (const shade of ['light', 'dark']) {
      await page.evaluate(([c, sh]) => { window.tw.shade.value = sh; window.tw.map.jumpTo({ center: c, zoom: 14.5 }); }, [[p.lng, p.lat], shade]);
      await page.waitForTimeout(2500);
      const xy = await page.evaluate((c) => {
        const q = window.tw.map.project(c), r = window.tw.map.getContainer().getBoundingClientRect();
        return { x: q.x + r.left, y: q.y + r.top };
      }, [p.lng, p.lat]);
      let shown = '';
      for (const [dx, dy] of [[0, 0], [2, 0], [-2, 0], [0, 2], [0, -2], [4, 0], [-4, 0], [0, 4], [0, -4], [6, 6], [-6, -6]]) {
        await page.mouse.move(xy.x + dx, xy.y + dy);
        await page.waitForTimeout(700);
        shown = await page.evaluate(() => [...document.querySelectorAll('body *')].filter((e) => e.children.length === 0 || e.childNodes.length > 1).map((e) => e.textContent).find((t) => /^Peak load/.test(t ?? '')) ?? '');
        if (shown) break;
      }
      console.log(`${shade}: ${drawn.lines[Math.floor(best / 4096)]?.name} stretch ${best % 4096}, hover: ${shown || 'none'}`);
      await page.screenshot({ path: `${OUT}/${shade}-crowded.png` });
    }
  }
} finally {
  await close();
}
console.log(pass ? 'PASS t2236 crowding marks by load' : 'FAIL t2236 crowding marks by load');
process.exit(pass ? 0 : 1);
