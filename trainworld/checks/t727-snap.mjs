// Snapping the track tool to the real network (T-727d): in a new game with the snap switch on, two
// clicks from Tokyo Station to Shin-Osaka and two from Penn Station to Emeryville each place the
// whole route as one edit, and one undo takes each away; Tokyo to London (two networks that never
// meet) is refused with "No real track joins these points"; a second click away from real track
// draws an ordinary segment. Brussels to Rotterdam snaps across the Belgian-Dutch border; in
// Mumbai, with coverage switched off as if no graph were built, there is no ring and the clicks draw an ordinary segment (T-913).
// Reports the router's and the fit's times and the page's longest task.
//   node trainworld/checks/t727-snap.mjs [url]     (from the maps root; about 2 minutes)
// Writes artifacts/trainworld/t727-snap.json and t727d-*.png screenshots.
import { mkdtemp, rm, readdir, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const BASE = process.argv[2] ?? 'http://localhost:6170/';
const OUT = fileURLToPath(new URL('../../artifacts/trainworld/', import.meta.url));

async function playwright() {
  try { return await import('playwright'); } catch { /* */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright');
}
const { chromium } = await playwright();
await mkdir(OUT, { recursive: true });
const profile = await mkdtemp(join(tmpdir(), 'tw-t727-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1200, height: 800 }, args: ['--use-angle=d3d11'] });
let failed = false;
const report = {};
const result = (ok, what, detail = '') => {
  failed ||= !ok;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}${detail ? '  (' + detail + ')' : ''}`);
};
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  page.on('console', (m) => { if (m.type() === 'error') console.log('console', m.text()); });
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools, undefined, { timeout: 180000 });
  await page.evaluate(async () => {
    const ui = await import('/src/game/ui.ts');
    ui.tool.value = 'track';
    ui.snapReal.value = true;
    window.__long = [];
    new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__long.push({ at: e.startTime, ms: e.duration }); }).observe({ type: 'longtask', buffered: false });
  });
  const shot = (name) => page.screenshot({ path: join(OUT, `t727d-${name}.png`) });

  /** jump there, wait for the real network's tiles, move the pointer onto the point; returns the ring */
  const go = (lng, lat, zoom) => page.evaluate(async ([lng, lat, zoom]) => {
    const { map, tools } = window.tw;
    map.jumpTo({ center: [lng, lat], zoom });
    for (let i = 0; i < 40; i++) {
      await new Promise((r) => setTimeout(r, 250));
      if (map.areTilesLoaded() && map.getStyle().layers.some((l) => l.id.startsWith('tw-real-'))) break;
    }
    await new Promise((r) => setTimeout(r, 600));
    const p = map.project([lng, lat]);
    tools.move({ point: { x: p.x, y: p.y }, lngLat: map.unproject([p.x, p.y]), originalEvent: {} });
    return tools.real.ringAt;
  }, [lng, lat, zoom]);
  const click = (lng, lat, shift = false) => page.evaluate(([lng, lat, shift]) => {
    const { map, tools } = window.tw;
    const p = map.project([lng, lat]);
    tools.click({ point: { x: p.x, y: p.y }, lngLat: map.unproject([p.x, p.y]), originalEvent: { shiftKey: shift, detail: 1 } });
  }, [lng, lat, shift]);
  /** wait until the route at the pointer is fitted and previewed, or refused */
  const settle = () => page.evaluate(async () => {
    const t0 = performance.now(), r = window.tw.tools.real;
    for (;;) {
      const ph = r.spot?.phase;
      if (ph === 'done' || ph === 'none' || ph === 'off' || !r.spot) break;
      if (performance.now() - t0 > 60000) break;
      await new Promise((res) => setTimeout(res, 50));
    }
    await new Promise((res) => setTimeout(res, 200));
    const pv = window.tw.tools.lastPreview;
    return { phase: r.spot?.phase ?? null, ms: performance.now() - t0, last: r.last, preview: pv ? { ok: pv.ok, km: pv.lengthM / 1000, cost: pv.cost, issues: pv.issues } : null, tag: window.tw.tools.tag.textContent };
  });
  const count = () => page.evaluate(() => ({ edges: window.tw.world.value.edges.length, notice: (window.__notice ?? null) }));
  const notice = () => page.evaluate(async () => (await import('/src/game/world.ts')).notice.value?.text ?? '');
  const waitEdges = (n) => page.waitForFunction((n) => window.tw.world.value.edges.length !== n, n, { timeout: 30000 }).catch(() => null);
  const undo = () => page.evaluate(async () => { const W = await import('/src/game/world.ts'); return (await W.edit({ op: 'undo' }, true)).ok; });
  const longest = (since) => page.evaluate((since) => Math.max(0, ...window.__long.filter((e) => e.at >= since).map((e) => e.ms)), since);
  const now = () => page.evaluate(() => performance.now());

  // ---- Tokyo Station to Shin-Osaka
  for (const [name, a, b, zA, zB, km] of [
    ['tokyo-osaka', [139.7671, 35.6812], [135.5001, 34.7334], 13, 13, [500, 530]],
    ['nyc-emeryville', [-73.9935, 40.7506], [-122.2916, 37.8405], 13, 13, [5200, 5600]],
    // T-913: Belgium and the Netherlands' graphs, joined at their border (on to Amsterdam fails at
    // ground level on water by Schiphol, T-914)
    ['brussels-rotterdam', [4.3571, 50.8455], [4.4686, 51.9244], 13, 13, [125, 165]],
    // T-727e, T-919: China station to station, each click on the high-speed platform track (at
    // Tianjin the metro below is nearer a few metres off, and a route from rail to it fails)
    ['beijing-tianjin', [116.37853, 39.86549], [117.20939, 39.13414], 13, 13, [110, 125]],
    ['hongqiao-hangzhou', [121.31702, 31.19403], [120.2122, 30.29045], 13, 13, [150, 170]],
    // T-923: a click on Tianjin station that rings on the metro below still finds the high-speed line,
    // as the end and as the start
    ['beijing-tianjin-metro', [116.37853, 39.86549], [117.2105, 39.1361], 13, 13, [110, 125]],
    ['tianjin-metro-beijing', [117.2105, 39.1361], [116.37853, 39.86549], 13, 13, [110, 125]],
  ]) {
    await page.evaluate(() => window.tw.tools.reset());
    const e0 = (await count()).edges;
    const ringA = await go(...a, zA);
    result(!!ringA, `${name}: a ring at the first click`);
    await click(...a);
    const ringB = await go(...b, zB);
    result(!!ringB, `${name}: a ring at the second click`);
    const t0 = await now();
    const s = await settle();
    report[name] = s;
    const okKm = s.preview?.ok && s.preview.km >= km[0] && s.preview.km <= km[1];
    result(s.phase === 'done' && okKm, `${name}: route previewed`, `${s.preview?.km?.toFixed(1)} km, ${s.last?.corners} corners, search ${s.last?.searchMs?.toFixed(0)} ms over ${s.last?.states} states, fit ${s.last?.fitMs?.toFixed(0)} ms, preview ${s.last?.previewMs?.toFixed(0)} ms, worst ${s.last?.worst?.toFixed(1)} m; tag "${s.tag}"`);
    // the same route again once the zones' loading has quietened: the longest main-thread task
    // from the pointer's rest to the fitted preview is the snap's own
    await page.waitForTimeout(5000);
    await page.evaluate(([lng, lat]) => { const { map, tools } = window.tw; const p = map.project([lng, lat]); const ev = (x, y) => ({ point: { x, y }, lngLat: map.unproject([x, y]), originalEvent: {} }); tools.move(ev(p.x + 60, p.y + 60)); tools.move(ev(p.x, p.y)); }, b);
    const tq = await now();
    const again = await settle();
    report[name].again = { ...again, longestTaskMs: await longest(tq) };
    result(again.phase === 'done' && report[name].again.longestTaskMs < 200, `${name}: the page stays responsive while the route is found and fitted`, `longest main-thread task ${report[name].again.longestTaskMs.toFixed(0)} ms, rest to preview ${again.ms.toFixed(0)} ms`);
    const view = await page.evaluate(([a, b]) => { const m = window.tw.map, v = { center: m.getCenter(), zoom: m.getZoom() }; m.fitBounds([[Math.min(a[0], b[0]), Math.min(a[1], b[1])], [Math.max(a[0], b[0]), Math.max(a[1], b[1])]], { padding: 80, animate: false }); return v; }, [a, b]);
    await page.waitForTimeout(1500);
    await shot(`${name}-preview`);
    // back to the view the pointer rested in
    await page.evaluate((v) => void window.tw.map.jumpTo(v), view);
    await page.waitForTimeout(500);
    const t1 = await now();
    await click(...b);
    await waitEdges(e0);
    await page.waitForTimeout(500);
    const e1 = (await count()).edges, n1 = await notice();
    report[name].placed = { edges: e1 - e0, notice: n1, longestTaskMs: await longest(t0), clickToPlacedLongest: await longest(t1) };
    result(e1 - e0 > 20 && /Blueprint added/.test(n1), `${name}: placed by the second click`, `${e1 - e0} pieces; "${n1}"; longest main-thread task ${report[name].placed.longestTaskMs.toFixed(0)} ms`);
    await page.evaluate(([a, b]) => void window.tw.map.fitBounds([[Math.min(a[0], b[0]), Math.min(a[1], b[1])], [Math.max(a[0], b[0]), Math.max(a[1], b[1])]], { padding: 80, animate: false }), [a, b]);
    await page.waitForTimeout(1500);
    await shot(`${name}-placed`);
    const u = await undo();
    await page.waitForTimeout(500);
    const e2 = (await count()).edges;
    result(u && e2 === e0, `${name}: one undo takes it all away`, `${e2} pieces after, ${e0} before`);
  }

  // ---- no route: Tokyo to London
  {
    await page.evaluate(() => window.tw.tools.reset());
    await go(139.7671, 35.6812, 13);
    await click(139.7671, 35.6812);
    await go(-0.1236, 51.5308, 13); // St Pancras
    const s = await settle();
    report.noRoute = s;
    await shot('no-route');
    const e0 = (await count()).edges;
    await click(-0.1236, 51.5308);
    await page.waitForTimeout(1000);
    const n = await notice(), e1 = (await count()).edges;
    result(s.phase === 'none' && /No real track joins these points/.test(s.tag) && e1 === e0 && /No real track joins/.test(n), 'Tokyo to London: refused, nothing placed', `tag "${s.tag}", notice "${n}", search ${s.last?.searchMs?.toFixed(0)} ms over ${s.last?.states} states`);
    await page.keyboard.press('Escape');
    await page.evaluate(() => window.tw.tools.reset());
  }

  // ---- a second click away from real track: an ordinary segment
  {
    const e0 = (await count()).edges;
    // Winnemucca, Nevada: the Zephyr's station, then the town 1.5 km south, no track near
    await go(-117.7363, 40.9706, 15);
    await click(-117.7363, 40.9706);
    const ring = await go(-117.7360, 40.9570, 15);
    await page.waitForTimeout(800);
    await click(-117.7360, 40.9570);
    await waitEdges(e0);
    await page.waitForTimeout(400);
    const e1 = (await count()).edges, n = await notice();
    result(!ring && e1 > e0 && e1 - e0 <= 3 && /Blueprint added/.test(n), 'away from real track: an ordinary segment', `${e1 - e0} pieces, "${n}"`);
    await shot('ordinary');
  }

  // ---- T-913: where no region has a graph, no ring and no snap, though the line is drawn. Every
  // region has one since T-727e, so Mumbai stands in with the coverage switched off.
  {
    await page.evaluate(() => { window.tw.tools.reset(); window.tw.tools.real.covers = () => false; });
    const e0 = (await count()).edges;
    // Mumbai CSMT, then 1.2 km north along the line
    const ring = await go(72.8355, 18.9398, 13);
    const drawn = await page.evaluate(() => { const { map, tools } = window.tw, p = map.project([72.8355, 18.9398]); return tools.real.nearestDrawn(p.x, p.y); });
    await click(72.8355, 18.9398);
    const ring2 = await go(72.8330, 18.9505, 13);
    const spot = await page.evaluate(() => window.tw.tools.real.spot?.phase ?? null);
    await click(72.8330, 18.9505);
    await waitEdges(e0);
    await page.waitForTimeout(400);
    const e1 = (await count()).edges, n = await notice();
    result(drawn && !ring && !ring2 && !spot && e1 > e0 && e1 - e0 <= 3 && /Blueprint added/.test(n), 'no graph (Mumbai): no ring, an ordinary segment', `drawn line at the click ${drawn ? 'yes' : 'no'}; ${e1 - e0} pieces, "${n}"`);
    await shot('no-graph');
    await page.evaluate(() => { delete window.tw.tools.real.covers; window.tw.tools.reset(); });
  }
  await writeFile(join(OUT, 't727-snap.json'), JSON.stringify(report, null, 1));
} catch (e) {
  failed = true;
  console.log('FAIL  snap check  ' + e.message);
} finally {
  await ctx.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
process.exitCode = failed ? 1 : 0;
