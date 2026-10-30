// T-1866 probe: a destination bubble's far end (homes) on the real New York save, by purpose and size.
//   node probe.mjs [url]   (cwd anywhere; writes artifacts/trainworld/t1866/)
import { readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
const lib = await import(pathToFileURL('C:/Users/anita/projects/maps/trainworld/checks/lib.mjs').href);
const { launchChrome, importSave, demandFinal, REAL_SAVE, GL } = lib;

const BASE = (process.argv[2] ?? 'http://localhost:6166/').replace(/\/?$/, '/');
const OUT = 'C:/Users/anita/projects/maps/artifacts/trainworld/t1866/';
const PICKS = (process.env.PICKS ?? 'midtown:-73.985:40.755,nyu:-73.9965:40.7295,columbia:-73.9626:40.8075,dtbk:-73.99:40.693,jamaica:-73.80:40.70').split(',').map((s) => { const [name, lng, lat] = s.split(':'); return { name, lng: +lng, lat: +lat }; });
const ZOOM = Number(process.env.ZOOM ?? 13);

const { ctx, close } = await launchChrome({ prefix: 'tw-t1866-', args: GL });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
console.log('chrome up');
try {
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&demandWorkers=2&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twViews && window.twDemand?.client, undefined, { timeout: 240000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { strict: false, lines: 0, timeout: 900000 });
  console.log('demand final');
  await page.evaluate(async () => {
    const t = window.tw;
    t.display.value = { ...t.display.value, commuters: true, commuterEnd: 'work', dayShown: 'weekday' };
    const ui = await import('/src/game/ui.ts');
    ui.purposeFilter.value = 'all';
    window.__t1866 = { fc: await import('/src/game/farChunks.ts'), dm: await import('/src/game/demand.ts') };
  });
  const results = [];
  for (const pick of PICKS) {
    await page.evaluate(() => { window.tw.selection.value = null; });
    await page.evaluate(([lng, lat, z]) => void window.tw.map.jumpTo({ center: [lng, lat], zoom: z }), [pick.lng, pick.lat, ZOOM]);
    await page.waitForTimeout(3000);
    const sel = await page.evaluate(() => {
      const t = window.tw, W = t.map.getContainer().clientWidth, H = t.map.getContainer().clientHeight;
      // the bubble under the view's centre, else the nearest within 60 px
      for (const r of [0, 10, 20, 30, 45, 60]) for (let a = 0; a < (r ? 12 : 1); a++) {
        const x = W / 2 + r * Math.cos(a * Math.PI / 6), y = H / 2 + r * Math.sin(a * Math.PI / 6);
        const s = window.twViews.click(x, y, false);
        if (s?.kind === 'commuters') return s;
      }
      return null;
    });
    if (!sel) { console.log(pick.name, 'no bubble'); continue; }
    await page.evaluate((s) => { window.tw.selection.value = s; }, sel);
    await page.waitForFunction(() => window.twViews.stats.commuters.farProbe?.(), undefined, { timeout: 180000, polling: 250 });
    await page.waitForTimeout(800);
    const r = await page.evaluate(async (sel) => {
      const { fc, dm } = window.__t1866;
      const raw = await window.twDemand.flows(sel.end, Uint32Array.from(sel.cells));
      const cc = await dm.cityCells();
      const xy = cc.xy;
      const dealt = fc.dealFlows(raw, 'all', xy);
      const tot = (fe, i) => fe.farRail[i] + fe.farWalk[i] + fe.farDrive[i];
      const by = {};
      for (const [k, fe] of raw.byPurpose ?? []) {
        const m = new Map();
        fe.cells.forEach((c, i) => m.set(c, tot(fe, i)));
        const vals = [...m.values()].sort((a, b) => b - a);
        const sum = vals.reduce((a, b) => a + b, 0);
        by[k] = { m, cells: vals.length, sum, top: vals.slice(0, 25).map((v) => +v.toFixed(1)), over100: vals.filter((v) => v >= 100).length, over500: vals.filter((v) => v >= 500).length, under10: vals.filter((v) => v < 10).length,
          shareOver100: vals.filter((v) => v >= 100).reduce((a, b) => a + b, 0) / (sum || 1), shareUnder10: vals.filter((v) => v < 10).reduce((a, b) => a + b, 0) / (sum || 1) };
      }
      // the far end before and after the display's grouping, by size class; purposes of each class from the raw cells
      const classes = [[0, 10], [10, 20], [20, 100], [100, 500], [500, 1e9]];
      const classify = (fe, raw) => {
        const out = classes.map(([lo, hi]) => ({ lo, hi, n: 0, sum: 0, purposes: {} }));
        let all = 0;
        fe.cells.forEach((c, i) => {
          const v = tot(fe, i);
          all += v;
          const k = out.find((o) => v >= o.lo && v < o.hi);
          k.n++; k.sum += v;
          if (raw) for (const [p, b] of Object.entries(by)) { const x = b.m.get(c) ?? 0; if (x > 0) k.purposes[p] = (k.purposes[p] ?? 0) + x; }
        });
        return { cells: fe.cells.length, all, classes: out.map((o) => ({ ...o, share: o.sum / (all || 1) })) };
      };
      const rawAll = classify(raw, true), dealtAll = classify(dealt, false);
      // the drawn level's bubbles
      const fp = window.twViews.stats.commuters.farProbe();
      const drawn = [];
      for (let i = 0; i < fp.group.length; i++) drawn.push(fp.train[i] + fp.walk[i] + fp.drive[i]);
      drawn.sort((a, b) => b - a);
      const dsum = drawn.reduce((a, b) => a + b, 0);
      const dcls = classes.map(([lo, hi]) => { const v = drawn.filter((x) => x >= lo && x < hi); const s = v.reduce((a, b) => a + b, 0); return { lo, hi, n: v.length, sum: s, share: s / (dsum || 1) }; });
      // the selected cells' own trips to them by purpose (`purposes` timing, legs)
      for (const b of Object.values(by)) delete b.m;
      return { selCells: sel.cells.length, totals: { rail: raw.rail, walk: raw.walk, drive: raw.drive, cut: raw.cutRail + raw.cutWalk + raw.cutDrive }, by, rawAll, dealtAll, drawn: { n: drawn.length, sum: dsum, top: drawn.slice(0, 30).map((v) => +v.toFixed(0)), classes: dcls, level: window.tw.map.getZoom() } };
    }, sel);
    r.name = pick.name;
    results.push(r);
    console.log(pick.name, JSON.stringify({ sel: r.selCells, drawn: r.drawn.n, top: r.drawn.top.slice(0, 10), classes: r.drawn.classes.map((c) => `${c.lo}+:${c.n}/${(100 * c.share).toFixed(0)}%`) }));
    if (process.env.SHOTS !== '0') {
      for (const z of (process.env.SHOT_ZOOMS ?? '12,11').split(',').map(Number)) {
        await page.evaluate(([lng, lat, z]) => void window.tw.map.jumpTo({ center: [lng, lat], zoom: z }), [pick.lng, pick.lat, z]);
        await page.waitForTimeout(2500);
        await page.screenshot({ path: `${OUT}${pick.name}-z${z}.png` });
        // the drawn bubbles at that zoom
        const lv = await page.evaluate(() => { const fp = window.twViews.stats.commuters.farProbe(); const v = []; for (let i = 0; i < fp.group.length; i++) v.push(fp.train[i] + fp.walk[i] + fp.drive[i]); v.sort((a, b) => b - a); return v; });
        r[`drawnZ${z}`] = { n: lv.length, top: lv.slice(0, 30).map((v) => +v.toFixed(0)), under20: lv.filter((v) => v < 20).length, over500: lv.filter((v) => v >= 500).length };
      }
    }
  }
  await writeFile(OUT + 'probe.json', JSON.stringify(results, null, 1));
  console.log('errors', errors.slice(0, 3));
} finally {
  await close();
}
process.exit(0);
