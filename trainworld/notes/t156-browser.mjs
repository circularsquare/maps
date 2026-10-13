// T-156 browser check: the Homes bubbles catch up over a game day after a new line opens. A new
// game, paused, Homes bubbles on Manhattan; a five-station elevated line is built and constructed,
// then the clock runs at the fastest speed and stops at set game times after the new result landed
// for a screenshot and the hover tags of the bubbles at the line's stations.
// Needs a Vite dev server (BASE) and Playwright (PLAYWRIGHT_MODULE). PROFILE_DIR: where the
// throwaway Chrome profile goes. Run from the maps root: node trainworld/notes/t156-browser.mjs.
// Screenshots: artifacts/trainworld/T-156-*.png. Results: artifacts/trainworld/notes-outputs/t156-browser.json.
import { writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const here = dirname(fileURLToPath(import.meta.url));
const ART = join(here, '..', '..', 'artifacts', 'trainworld');
const BASE = process.env.BASE ?? 'http://localhost:6571';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't156-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--enable-unsafe-swiftshader'] });
// Harlem to Fulton St, roughly under Lexington and Broadway
const STOPS = [[-73.9376, 40.8044], [-73.9554, 40.7795], [-73.9772, 40.7527], [-73.9903, 40.7359], [-74.0076, 40.7101]];
const result = { checks: [], frames: [] };
const check = (name, ok, detail) => { result.checks.push({ name, ok: !!ok, detail }); console.log(ok ? 'ok  ' : 'FAIL', name, JSON.stringify(detail ?? '')); };
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(`${BASE}/?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client && window.twDemand.view()?.round === 3, undefined, { timeout: 300000 });
  await page.evaluate(() => {
    const d = document.querySelector('#debug'); if (d) d.style.display = 'none';
    window.tw.display.value = { ...window.tw.display.value, commuters: true, commuterEnd: 'home', intercity: true, bubbleSize: 1 };
    window.tw.map.jumpTo({ center: [-73.972, 40.757], zoom: 11.9 });
  });
  await page.waitForFunction(() => window.twViews.stats.commuters.builds > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(1500);
  // the hover tags of the bubbles at the line's stops (pixels a little off the stop itself)
  const sample = () => page.evaluate((stops) => stops.map(ll => {
    const p = window.tw.map.project(ll);
    return window.twViews.hover(p.x + 4, p.y + 4)?.join(' / ') ?? null;
  }), STOPS);
  const frame = async (key) => {
    await page.waitForTimeout(1200);
    const at = await page.evaluate(() => ({ t: window.tw.clock.now(), builds: window.twViews.stats.commuters.builds, split: window.twDemand.cities().get('nyc')?.split }));
    const tags = await sample();
    result.frames.push({ key, ...at, tags });
    console.log(key, JSON.stringify({ ...at, tags }));
    await page.screenshot({ path: join(ART, `T-156-${key}.png`) });
  };
  await frame('0-before');

  // the line, built and constructed with the clock paused
  const before = await page.evaluate(() => window.twDemand.view().version);
  await page.evaluate(async (stops) => {
    const t = window.tw, run = async op => { const r = await t.client.edit(op); if (!r.ok) throw Error(JSON.stringify({ op, r })); return r; };
    const nodes = [];
    for (let i = 1; i < stops.length; i++) {
      const b = t.lngLatToPack(...stops[i]);
      const from = i === 1 ? (() => { const a = t.lngLatToPack(...stops[0]); return { kind: 'free', x: a.x, y: a.y, level: 1 }; })() : { kind: 'node', node: nodes.at(-1) };
      await run({ op: 'route', from, to: { kind: 'free', x: b.x, y: b.y, level: 1 }, pis: [], single: false });
      const e = t.world.value.edges.at(-1);
      if (i === 1) nodes.push(e.a);
      nodes.push(e.b);
    }
    for (const [i, n] of nodes.entries()) await run({ op: 'addStation', at: { kind: 'node', node: n }, platform: 200, name: 'Test ' + (i + 1) });
    const line = (await run({ op: 'addLine', stops: nodes, name: 'Test line', colour: '#1f6fd1' })).line;
    await run({ op: 'setTrainCount', line, trains: [10, 8, 4] });
    await run({ op: 'constructAll' });
  }, STOPS);
  await page.waitForFunction(v => window.twDemand.view()?.version > v && window.twDemand.view().round === 3 && window.twDemand.view().lines.size === 1, before, { timeout: 300000 });
  await frame('1-new-result-paused');
  const paused = result.frames.at(-1);
  check('paused: the bubbles still show the old result', JSON.stringify(paused.tags) === JSON.stringify(result.frames[0].tags), { before: result.frames[0].tags, after: paused.tags });

  // run at the fastest speed, stopping at these game hours after the result landed
  const t0 = paused.t;
  for (const h of [2, 4, 8, 12, 18, 26, 30]) {
    await page.evaluate(() => window.tw.clock.setSpeed(2));
    await page.waitForFunction(([t0, h]) => window.tw.clock.now() >= t0 + h * 3600, [t0, h], { timeout: 300000, polling: 50 });
    await page.evaluate(() => window.tw.clock.setRunning(false));
    await frame(`2-after-${String(h).padStart(2, '0')}h`);
  }
  const trainPct = (tag) => Number(/Train (\d+)%/.exec(tag ?? '')?.[1] ?? NaN);
  const series = STOPS.map((_, i) => result.frames.map(f => trainPct(f.tags[i])));
  result.series = series;
  const moved = series.filter(s => s.at(-1) > s[0]);
  check('the train share at the stops rises through the day', moved.length >= 3 && moved.every(s => s.every((v, k) => k === 0 || v >= s[k - 1] - 1)), series);
  check('caught up after a day: the bubbles stop changing', JSON.stringify(result.frames.at(-1).tags) === JSON.stringify(result.frames.at(-2).tags) || series.every(s => Math.abs(s.at(-1) - s.at(-2)) <= 1), result.frames.slice(-2).map(f => f.tags));
  const builds = result.frames.map(f => f.builds);
  result.builds = builds;
  result.errors = errors;
  check('no page errors', errors.length === 0, errors);
  await writeFile(join(here, '../../artifacts/trainworld/notes-outputs', 't156-browser.json'), JSON.stringify(result, null, 2));
} finally { await ctx.close(); }
