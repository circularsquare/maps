// T-147 / T-163 browser check: the complex mini-map in the station inspector and every member's
// platform box drawn selected, on New York's real network (T-007 save) in the real app.
// Needs a Vite dev server (BASE, default http://localhost:5861; the probe imports
// /src/game/complexView.ts) and Playwright (PLAYWRIGHT_MODULE, a file URL to its index.mjs).
// PROFILE_DIR: where the throwaway Chrome profile goes. Run from anywhere:
//   node trainworld/notes/t147-browser.mjs
// Screenshots: artifacts/trainworld/T-147-*.png. Results: artifacts/trainworld/notes-outputs/t147-browser.json.
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, '..');
const ART = join(ROOT, '..', 'artifacts', 'trainworld');
const BASE = process.env.BASE ?? 'http://localhost:5861';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't147-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--enable-unsafe-swiftshader'] });
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(BASE + '/?debug=1&new=1&paused=1&v=' + Date.now());
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client, undefined, { timeout: 240000 });
  const bytes = await readFile(join(ROOT, 'T-007-real-nyc.save'));
  await page.evaluate(data => { const bytes = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes }, [bytes]); }, Array.from(bytes));
  await page.waitForFunction(() => window.twDemand.view()?.round === 3 && window.twDemand.view().lines.size > 100, undefined, { timeout: 600000 });
  // the complexes to look at: the station in each with the most lines
  const picks = await page.evaluate(async () => {
    const cv = await import('/src/game/complexView.ts');
    window.__cv = cv;
    const c = cv.complexes.value, w = window.tw.world.value, st = w.save.stations;
    const nLines = id => w.save.lines.filter(l => l.stops.includes(id)).length;
    const best = members => members.reduce((a, b) => nLines(b) > nLines(a) ? b : a);
    const byName = re => {
      const ms = c.members.filter(m => m.length > 1 && m.some(id => re.test(st.find(s => s.id === id).name)));
      return ms.sort((a, b) => b.length - a.length)[0];
    };
    const out = {};
    const add = (key, members) => {
      if (!members) return;
      const id = best(members), s = st.find(s => s.id === id);
      out[key] = { id, name: s.name, lng: s.lng, lat: s.lat, members: members.length };
    };
    add('penn', byName(/Penn/));
    add('grandcentral', byName(/Grand Central/));
    add('timessq', byName(/Times Sq/));
    // a two-station complex with riders, busiest first
    const dv = window.twDemand.view();
    const pairs = c.members.filter(m => m.length === 2 && m.every(id => dv.stations.get(id)));
    pairs.sort((a, b) => b.reduce((x, id) => x + dv.stations.get(id).boardings, 0) - a.reduce((x, id) => x + dv.stations.get(id).boardings, 0));
    add('pair', pairs[0]);
    return out;
  });
  await page.evaluate(() => { const d = document.querySelector('#debug'); if (d) d.style.display = 'none'; });
  // a selected station's commuter catchment draws only under All purposes or Work: another
  // purpose keeps it off the map, so the platform boxes can be seen
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).purposeFilter.value = 'shopping'; });
  const results = {};
  for (const [key, p] of Object.entries(picks)) {
    await page.evaluate(({ id, lng, lat, members }) => {
      window.tw.display.value = { ...window.tw.display.value, complexView: true, stationRiders: false, commuters: false };
      window.tw.select({ kind: 'station', station: id });
      window.tw.map.jumpTo({ center: [lng, lat], zoom: members > 10 ? 15.6 : 16.4 });
    }, p);
    await page.waitForTimeout(3000);
    await page.screenshot({ path: join(ART, `T-147-${key}.png`) });
    const panel = await page.$('#inspector');
    if (panel) await panel.screenshot({ path: join(ART, `T-147-${key}-panel.png`) });
    results[key] = await page.evaluate(() => {
      const svg = document.querySelector('.complex-map svg');
      const r = window.tw.renderer;
      return {
        svg: svg ? { w: svg.getAttribute('width'), h: svg.getAttribute('height'), strokes: svg.querySelectorAll('.complex-map-platform').length, bundles: svg.querySelectorAll(':scope > g').length - 1, labels: [...svg.querySelectorAll('.complex-map-label')].map(t => t.textContent), scale: svg.querySelector('.complex-map-scale')?.textContent, firstTip: svg.querySelector('.complex-map-platform title')?.textContent } : null,
        highlighted: r.selectedAlso?.size ?? null,
        caption: document.querySelector('.complex-map p')?.textContent ?? null,
      };
    });
    results[key].pick = p;
  }
  // a click on a platform in the mini-map selects that station and stays in complex view
  const click = await page.evaluate(async (id) => {
    window.tw.select({ kind: 'station', station: id });
    await new Promise(r => setTimeout(r, 800));
    const gs = [...document.querySelectorAll('.complex-map-platform')];
    const before = window.tw.selection.value?.station;
    const target = gs[gs.length - 1];
    target?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await new Promise(r => setTimeout(r, 800));
    return { before, after: window.tw.selection.value?.station, complexView: window.tw.display.value.complexView };
  }, picks.timessq?.id ?? picks.pair.id);
  // station view: only the selected box enlarged
  const stationView = await page.evaluate(async (p) => {
    window.tw.display.value = { ...window.tw.display.value, complexView: false };
    window.tw.select({ kind: 'station', station: p.id });
    window.tw.map.jumpTo({ center: [p.lng, p.lat], zoom: 15.6 });
    await new Promise(r => setTimeout(r, 1500));
    return { highlighted: window.tw.renderer.selectedAlso.size, miniMap: !!document.querySelector('.complex-map') };
  }, picks.penn);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(ART, 'T-147-penn-station-view.png') });
  const gl = await page.evaluate(() => window.tw.renderer.gl.getError());
  const result = { picks, results, click, stationView, errors, gl };
  console.log(JSON.stringify(result, null, 1));
  await writeFile(join(here, '../../artifacts/trainworld/notes-outputs', 't147-browser.json'), JSON.stringify(result, null, 2));
} finally { await ctx.close(); }
