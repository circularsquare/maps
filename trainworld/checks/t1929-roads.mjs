// T-1929 in a new game: road crossings priced at ground level. A ghost across Midtown at ground and
// at +1 (the tag's crossings, ground dearer than the viaduct), one in rural Iowa (0-2 a km), then the
// Midtown line as blueprint: its "Road crossings" quote row through an export and reload, a station
// splitting it, and Construct paying it. Pictures of the tags in artifacts/trainworld/t1929/. About
// two minutes.
// Usage: node trainworld/checks/t1929-roads.mjs [url] (default http://localhost:6180/)

import { mkdirSync, readFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6180/';
const sep = BASE.includes('?') ? '&' : '?';
const OUT = join(ARTIFACTS, 't1929');
mkdirSync(OUT, { recursive: true });
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) failed++; };

const { ctx, close } = await launchChrome({ prefix: 'tw-t1929-' });
try {
  const page = await ctx.newPage();
  const errors = [];
  collectErrors(page, errors);
  await page.goto(`${BASE}${sep}debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });

  const px = (x, y) => page.evaluate(([x, y]) => { const r = window.tw.map.getCanvas().getBoundingClientRect(), p = window.tw.map.project([x, y]); return [r.left + p.x, r.top + p.y]; }, [x, y]);
  const tagText = () => page.evaluate(() => [...document.querySelectorAll('.tool-tag')].map((t) => t.textContent).join(' | '));
  const money = (s) => Number(s.replace(/,/g, '').slice(0, -1)) * ({ k: 0.001, M: 1, B: 1000 })[s.at(-1)];
  /** a ghost from a to b at the level picked with Shift+digit `key`: its tag, read once the roads are in */
  const ghost = async (a, b, key, shot, place = false, zoom = 14) => {
    await page.evaluate(([c, zoom]) => void window.tw.map.jumpTo({ center: c, zoom }), [[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], zoom]);
    await page.waitForTimeout(2000);
    await page.keyboard.press('Escape');
    await page.keyboard.press('1');
    await page.keyboard.press(`Shift+Digit${key}`);
    await page.waitForTimeout(300);
    const [x0, y0] = await px(...a), [x1, y1] = await px(...b);
    await page.mouse.move(x0, y0);
    await page.mouse.click(x0, y0);
    await page.waitForTimeout(400);
    await page.mouse.move(x1, y1, { steps: 6 });
    // the tiles land and the ghost is asked again
    await page.waitForTimeout(5000);
    await page.mouse.move(x1 + 1, y1, { steps: 1 });
    await page.waitForTimeout(1500);
    const tag = await tagText();
    await page.screenshot({ path: join(OUT, `${shot}.png`), clip: { x: Math.max(0, x1 - 300), y: Math.max(0, y1 - 160), width: 600, height: 320 } });
    if (place) await page.mouse.click(x1 + 1, y1);
    await page.waitForTimeout(500);
    await page.keyboard.press('Escape');
    const cost = /cost [\d.]+x, \$([\d.,]+[kMB])/.exec(tag);
    const roads = /([\d,]+) road crossings? \+\$([\d.,]+[kMB])/.exec(tag);
    const km = Number(/([\d.]+) km/.exec(tag)?.[1] ?? NaN);
    return { tag, km, cost: cost ? money(cost[1]) : NaN, n: roads ? Number(roads[1].replace(/,/g, '')) : 0, roadCost: roads ? money(roads[2]) : 0 };
  };

  // ---- Midtown, due north from 34th Street to Central Park South (across the grid at an angle), at
  // ground (Shift+4) and at +1 (Shift+3)
  const mA = [-73.9845, 40.7485], mB = [-73.9845, 40.7645];
  const g = await ghost(mA, mB, 4, 'midtown-ground');
  check(g.n / g.km >= 8 && Math.abs(g.roadCost - 6 * g.n) < 0.6, `Midtown at ground: ${g.n} crossings in ${g.km} km, ${(g.n / g.km).toFixed(1)} a km (tag "${g.tag}")`);
  await page.evaluate(() => { window.tw.shade.value = 'dark'; });
  await ghost(mA, mB, 4, 'midtown-ground-dark');
  await page.evaluate(() => { window.tw.shade.value = 'light'; });
  const v = await ghost(mA, mB, 3, 'midtown-viaduct');
  check(v.n === 0 && g.cost > v.cost, `the same at +1 crosses none and costs less: $${v.cost}M against $${g.cost}M at ground (tag "${v.tag}")`);

  // ---- rural Iowa, 4 km east at ground
  const rA = [-94.325, 41.75], rB = [-94.277, 41.75];
  const r = await ghost(rA, rB, 4, 'iowa-ground', false, 13);
  check(r.km > 3.5 && r.n / r.km <= 2, `rural Iowa at ground: ${r.n} crossings in ${r.km} km (tag "${r.tag}")`);

  // ---- the Midtown line as blueprint: its quote row, through a reload and a split, then paid
  await ghost(mA, mB, 4, 'midtown-placed', true);
  const roadRow = () => page.evaluate(async () => {
    const d = await window.tw.client.query({ what: 'blueprintQuote' });
    let n = 0, cost = 0, total = 0;
    for (let i = 0; i < d.length; i += 7) { total += d[i + 6]; if (d[i] === 7) { n += d[i + 5]; cost += d[i + 6]; } }
    return { n, cost, total };
  });
  const placed = await roadRow();
  await page.keyboard.press('1');
  await page.waitForTimeout(1500);
  const panel = page.locator('.blueprint-breakdown').first();
  if (await panel.isVisible().catch(() => false)) await panel.screenshot({ path: join(OUT, 'breakdown.png') });
  await page.keyboard.press('Escape');
  check(placed.n > 0 && Math.abs(placed.n - g.n) <= 2, `the placed line's quote row: ${placed.n} crossings, $${placed.cost.toFixed(0)}M (the ghost had ${g.n})`);
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const bytes = await readFile(await (await dl).path());
  await page.evaluate(() => { window.tw.client.lastLoad = null; });
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.edges.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(6000);
  const reloaded = await roadRow();
  check(reloaded.n === placed.n, `after an export and reload: ${reloaded.n} crossings`);
  const split = await page.evaluate(async ([lng, lat]) => {
    const tw = window.tw, e = tw.world.value.edges.find((e) => !e.built && !e.crossover), p = tw.lngLatToPack(lng, lat);
    return tw.client.edit({ op: 'addStation', at: { kind: 'edge', edge: e.id, x: p.x, y: p.y }, platform: 200, name: 'Midtown' });
  }, [(mA[0] + mB[0]) / 2, (mA[1] + mB[1]) / 2]);
  const after = await roadRow();
  check(split.ok && after.n === placed.n, `a station splitting it keeps them: ${after.n} (${JSON.stringify(split.issues ?? [])})`);
  const paid = await page.evaluate(() => window.tw.client.edit({ op: 'constructAll' }));
  check(paid.ok && paid.charge >= after.total - 1e-6, `Construct pays the quote: $${paid.charge?.toFixed(1)}M against $${after.total.toFixed(1)}M`);

  // a contact sheet of the three tags
  const files = ['midtown-ground', 'midtown-viaduct', 'iowa-ground'].map((f) => join(OUT, `${f}.png`));
  const sheet = await ctx.newPage();
  await sheet.setViewportSize({ width: 1812, height: 320 });
  await sheet.setContent(`<body style="margin:0;display:flex;gap:6px">${files.map((f) => `<img src="data:image/png;base64,${readFileSync(f).toString('base64')}">`).join('')}</body>`);
  await sheet.screenshot({ path: join(OUT, 'tags.png') });
  await sheet.close();
  check(errors.length === 0, `no page errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
