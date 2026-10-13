// T-176 browser check: transfer curves in the complex mini-map, on New York's real network (T-007
// save) in the real app. Needs a Vite dev server (BASE; the probe imports /src/game/complexView.ts)
// and Playwright (PLAYWRIGHT_MODULE, a file URL to its index.mjs). PROFILE_DIR: where the
// throwaway Chrome profile goes. Run from anywhere:
//   node trainworld/notes/t176-browser.mjs
// Screenshots: artifacts/trainworld/T-176-*.png. Results: artifacts/trainworld/notes-outputs/t176-browser.json.
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, '..');
const ART = join(ROOT, '..', 'artifacts', 'trainworld');
const BASE = process.env.BASE ?? 'http://localhost:6417';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't176-'));
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
  // the complexes to look at: the station in each with the most lines (as t147-browser.mjs)
  const picks = await page.evaluate(async () => {
    const cv = await import('/src/game/complexView.ts');
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
      out[key] = { id, name: s.name, lng: s.lng, lat: s.lat, members };
    };
    add('penn', byName(/Penn/));
    add('grandcentral', byName(/Grand Central/));
    add('timessq', byName(/Times Sq/));
    const dv = window.twDemand.view();
    const pairs = c.members.filter(m => m.length === 2 && m.every(id => dv.stations.get(id)));
    pairs.sort((a, b) => b.reduce((x, id) => x + dv.stations.get(id).boardings, 0) - a.reduce((x, id) => x + dv.stations.get(id).boardings, 0));
    add('pair', pairs[0]);
    return out;
  });
  await page.evaluate(() => { const d = document.querySelector('#debug'); if (d) d.style.display = 'none'; });
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).purposeFilter.value = 'shopping'; });
  const results = {};
  for (const [key, p] of Object.entries(picks)) {
    // the query itself: time, pairs, and the biggest by line (names from the save)
    const q = await page.evaluate(async (members) => {
      const w = window.tw.world.value, ln = new Map(w.save.lines.map(l => [l.id, l.name])), sn = new Map(w.save.stations.map(s => [s.id, s.name]));
      const t = performance.now();
      const r = await window.twDemand.transfers(members);
      const ms = performance.now() - t;
      if (!r) return { ms, answer: null };
      const inside = new Set(members);
      const both = r.pairs.filter(x => inside.has(x.fromStation) && inside.has(x.toStation));
      const total = both.reduce((a, x) => a + x.riders, 0);
      return {
        ms: Math.round(ms), pairs: r.pairs.length, pairsInside: both.length, ridersInside: Math.round(total),
        ridersWalkingOut: Math.round(r.pairs.filter(x => !(inside.has(x.fromStation) && inside.has(x.toStation))).reduce((a, x) => a + x.riders, 0)),
        top: both.slice(0, 10).map(x => `${ln.get(x.fromLine)} (${sn.get(x.fromStation)}) -> ${ln.get(x.toLine)} (${sn.get(x.toStation)}): ${Math.round(x.riders)}`),
      };
    }, p.members);
    await page.evaluate(({ id, lng, lat, members }) => {
      window.tw.display.value = { ...window.tw.display.value, complexView: true, stationRiders: false, commuters: false };
      window.tw.select({ kind: 'station', station: id });
      window.tw.map.jumpTo({ center: [lng, lat], zoom: members.length > 10 ? 15.6 : 16.4 });
    }, p);
    await page.waitForFunction(() => document.querySelectorAll('.complex-map-change').length > 0, undefined, { timeout: 30000 }).catch(() => {});
    await page.waitForTimeout(1500);
    await page.screenshot({ path: join(ART, `T-176-${key}.png`) });
    const map = await page.$('.complex-map');
    if (map) await map.screenshot({ path: join(ART, `T-176-${key}-map.png`) });
    results[key] = {
      query: q,
      map: await page.evaluate(() => {
        const svg = document.querySelector('.complex-map svg');
        if (!svg) return null;
        return {
          w: svg.getAttribute('width'), h: svg.getAttribute('height'),
          curves: [...svg.querySelectorAll('.complex-map-change')].map(c => ({ width: Number(c.previousElementSibling.getAttribute('stroke-width')).toFixed(1), title: c.querySelector('title')?.textContent })),
          bandTips: [...svg.querySelectorAll('.complex-map-platform title')].map(t => t.textContent).filter(t => t.includes('change')).slice(0, 4),
          captions: [...document.querySelectorAll('.complex-map p')].map(p => p.textContent),
        };
      }),
      members: p.members.length, name: p.name,
    };
  }
  const gl = await page.evaluate(() => window.tw.renderer.gl.getError());
  const result = { results, errors, gl };
  console.log(JSON.stringify(result, null, 1));
  await writeFile(join(here, '../../artifacts/trainworld/notes-outputs', 't176-browser.json'), JSON.stringify(result, null, 2));
} finally { await ctx.close(); }
