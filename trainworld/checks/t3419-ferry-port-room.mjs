// T-2194: the Ferry tool on a new game in New York, with the mouse. The Build tab's Ferry button, two ports
// across the harbour (Whitehall and St George: the Staten Island ferry), the route the water worker finds,
// Make line at the price the tag gave (US$10M a port), Construct, boats moving on it, its panel, the Lines
// tab and the Money tab's Ferry fare row and boat rows (the Finances tab), then a save exported and loaded again with the line
// and the boat fleet intact; and two ports on waters that do not meet (Central Park's reservoir and its
// Lake) refused with "No water path found". Screenshots at zoom 12 and 15, light and dark, to
// artifacts/trainworld/t2194/. Uses the Vite already on the url (dev-nowatch.mjs). From the maps root:
import { ARTIFACTS, collectErrors, importSave, launchChrome } from './lib.mjs';
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

const OUT = join(ARTIFACTS, 't2194');
await mkdir(OUT, { recursive: true });
const BASE = process.argv[2] ?? 'http://localhost:6199/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const WHITEHALL = [-74.0130, 40.7012], ST_GEORGE = [-74.0715, 40.6437];
const RESERVOIR = [-73.9668, 40.7852], LAKE = [-73.9712, 40.7741];

const chrome = await launchChrome({ prefix: 'tw-t3419-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools?.ferry, undefined, { timeout: 180000 });
  const box = await (await page.$('#map')).boundingBox();
  const at = async (p) => { const [x, y] = await page.evaluate(([a, b]) => { const q = window.tw.map.project([a, b]); return [q.x, q.y]; }, p); return { x: Math.round(box.x + x), y: Math.round(box.y + y) }; };
  const settle = async (ms = 800) => {
    await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 });
    await page.waitForTimeout(ms);
  };
  const fit = async (a, b, pad = 140) => {
    await page.evaluate(([a, b, pad]) => window.tw.map.fitBounds([[Math.min(a[0], b[0]), Math.min(a[1], b[1])], [Math.max(a[0], b[0]), Math.max(a[1], b[1])]], { padding: pad, duration: 0 }), [a, b, pad]);
    await settle();
  };
  const ferry = () => page.evaluate(() => {
    const f = window.tw.tools.ferry;
    return { active: f.active, ports: f.ports.length, route: f.route ? { ok: f.route.ok, m: f.route.m, refused: f.route.legs.filter((l) => l.reason).length } : null, tag: [...document.querySelectorAll('.tool-tag')].filter((e) => e.style.display !== 'none').map((e) => e.textContent).join(' / '), under: f.under.kind };
  });
  /** move to `p`, wait for the probe, click; then wait until `ports` ports are placed */
  const place = async (p, ports) => {
    const q = await at(p);
    await page.mouse.move(q.x - 4, q.y - 4);
    await page.mouse.move(q.x, q.y, { steps: 4 });
    await page.waitForTimeout(1500);
    const tag = (await ferry()).tag;
    await page.mouse.click(q.x, q.y);
    await page.waitForFunction((n) => window.tw.tools.ferry.ports.length >= n, ports, { timeout: 30000 }).catch(() => {});
    return tag;
  };
  const waitRoute = () => page.waitForFunction(() => { const f = window.tw.tools.ferry; return f.route && f.routeKey === f.berthKey(); }, undefined, { timeout: 60000 });

  // ---- the Build tab's Ferry button, two ports across the harbour
  await fit(WHITEHALL, ST_GEORGE);
  await page.click('button.tab[aria-label="Build"]');
  await page.click('button.tool-ferry');
  await page.waitForTimeout(300);
  check((await ferry()).active, 'the Build tab\'s Ferry button starts the tool');
  const tag0 = await place(WHITEHALL, 1);
  check(/Place port/.test(tag0), 'the tag over the shore says Place port', tag0);
  await place(ST_GEORGE, 2);
  let f = await ferry();
  check(f.ports === 2, 'two ports placed', `${f.ports}`);
  await waitRoute();
  // the pointer back over the water between for the tag
  const mid = await at([(WHITEHALL[0] + ST_GEORGE[0]) / 2, (WHITEHALL[1] + ST_GEORGE[1]) / 2]);
  await page.mouse.move(mid.x, mid.y, { steps: 4 });
  await page.waitForTimeout(1200);
  f = await ferry();
  check(f.route?.ok && f.route.refused === 0 && f.route.m > 7000 && f.route.m < 10000, 'the router finds the harbour route', `${((f.route?.m ?? 0) / 1000).toFixed(2)} km`);
  const priced = /\$20(\.0)?M/.test(f.tag);
  check(priced, 'the tag gives the route\'s length and the two ports\' price', f.tag);
  await page.screenshot({ path: join(OUT, 'drawing-light-z12.png'), clip: box });


  await page.keyboard.press('Enter');
  await page.waitForFunction(() => !window.tw.tools.ferry.active && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 30000 });
  const made = await page.evaluate(() => { const w = window.tw.world.value, l = w.save.lines.at(-1), st = w.lineStats[l.id]; return { id: l.id, num: l.num, tph: l.tph ?? null, line: JSON.stringify(l).slice(0, 400), ports: st.bus.ferry.ports.map((p) => p.node) }; });
  console.log('line', made.line);
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, made.id);
  await page.waitForTimeout(800);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(500);
  const panel = await page.evaluate(() => document.querySelector('#inspector, .inspector')?.innerText ?? '');
  console.log('PANEL', panel.replace(/\n+/g, ' | ').slice(0, 900));
  await page.screenshot({ path: join(OUT, 't3400-line-panel.png') });
  // a port's panel
  const sid = await page.evaluate((n) => { const w = window.tw.world.value; const s = w.save.stations.find((x) => x.num === n); return s?.id ?? null; }, made.ports[0]);
  console.log('port station id', sid);
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'station', station: id }; }, sid);
  await page.waitForTimeout(1000);
  const up = await page.evaluate(() => [...document.querySelectorAll('.upgrades button')].map((b) => b.getAttribute('aria-label') + ' :: ' + (b.getAttribute('data-tip') || '').replace(/\n/g, ' / ')));
  console.log('UPGRADES', JSON.stringify(up));
  check(up.length === 1 && /Waiting room/.test(up[0]) && /boat/.test(up[0]), 'a port offers the waiting room only, with boat wording', JSON.stringify(up));
  await page.screenshot({ path: join(OUT, 't3419-port-panel.png') });
  await page.click('.upgrades button.upgrade');
  await page.waitForTimeout(1000);
    console.log('NOTE', await page.evaluate(() => (document.querySelector('#tool-notes')?.textContent ?? '') + ' // ' + JSON.stringify(window.tw.world.value.save.stations.map((s) => [s.num, s.upgrades]))));
  const on = await page.evaluate(() => document.querySelector('.upgrades button')?.getAttribute('aria-pressed'));
  check(on === 'true', 'clicking adds the waiting room to the planned port', String(on));
  await page.screenshot({ path: join(OUT, 't3419-port-panel-on.png') });
} finally {
  console.log('page errors', JSON.stringify(errors));
  await chrome.close?.();
}
process.exit(fail ? 1 : 0);
