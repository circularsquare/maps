// T-1475: bus tweaks. A new game in New York with a rail line, three bus lines (one constructed, one
// planned, one a copy) and a copy of the rail line; screenshots light and dark of a hover on the bus
// route, the Lines tab's bullets and the bus ends on the map, into artifacts/trainworld/t1475/ named by
// the label; checks (labelled "after") that the hover names a bus, shows no corridor, new buses are
// "Bus n" and a copy keeps its line's symbol. Uses the Vite already on `port` (dev-nowatch.mjs). From
// the maps root:
//   node trainworld/checks/t1475-shots.mjs <port> <before|after>
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, launchChrome, collectErrors, demandFinal, acrossLine } from './lib.mjs';

const OUT = join(ARTIFACTS, 't1475');
await mkdir(OUT, { recursive: true });
const port = Number(process.argv[2] ?? 6175);
const label = process.argv[3] ?? 'after';
const url = `http://localhost:${port}/`;
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

const chrome = await launchChrome({ prefix: 'tw-t1475-', size: [1400, 900], deviceScaleFactor: 2 });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const [lng, lat] = [-73.87, 40.75];
  await page.evaluate(([a, b]) => window.tw.map.jumpTo({ center: [a, b], zoom: 12 }), [lng, lat]);
  await page.waitForTimeout(2500);
  const rail = await acrossLine(page, [lng, lat], 2);
  check(rail !== null, 'a rail line built');
  await demandFinal(page, { polling: 250, lines: 0 });
  const bus = (from, to, build) => page.evaluate(async ([from, to, build]) => {
    const tw = window.tw, c = tw.client;
    const route = [from, to].flatMap(([a, b]) => { const p = tw.lngLatToPack(a, b); return [p.x, p.y]; });
    const add = await c.edit({ op: 'addBusLine', route, name: '', colour: '#d7263d' });
    if (build) await c.edit({ op: 'construct', edges: [], stations: [], buses: [add.line] });
    return add.line;
  }, [from, to, build]);
  const a = await bus([lng, lat + 0.04], [lng, lat + 0.0005], true);
  const b = await bus([lng - 0.04, lat - 0.02], [lng + 0.03, lat - 0.015], false);
  await page.waitForTimeout(1500);
  const names = () => page.evaluate(() => window.tw.world.value.save.lines.map((l) => ({ num: l.num, name: l.name, letter: l.letter, symbol: l.symbol })));
  const copyOf = async (line) => (await page.evaluate((n) => window.tw.client.edit({ op: 'copyLine', line: n }), line)).line;
  await page.evaluate(async (n) => {
    const l = window.tw.world.value.save.lines.find((x) => x.num === n);
    await window.tw.client.edit({ op: 'lineLook', line: n, name: 'Line 7', colour: l.colour });
  }, rail);
  const railCopy = await copyOf(rail);
  const busCopy = await copyOf(a);
  // the copy lies on its line's route: move it east so the hover finds the line itself
  await page.evaluate(async ([line, lng, lat]) => {
    const tw = window.tw;
    const route = [[lng + 0.035, lat + 0.04], [lng + 0.035, lat + 0.005]].flatMap(([x, y]) => { const p = tw.lngLatToPack(x, y); return [p.x, p.y]; });
    await tw.client.edit({ op: 'busRoute', line, route });
  }, [busCopy, lng, lat]);
  await page.waitForTimeout(1500);
  const list = await names();
  console.log(JSON.stringify(list));
  const get = (n) => list.find((l) => l.num === n);
  check(/^Bus \d+$/.test(get(a)?.name) && /^Bus \d+$/.test(get(b)?.name), `new bus lines are named Bus n (${get(a)?.name}, ${get(b)?.name})`);
  check(get(railCopy)?.letter === get(rail)?.letter, `a rail copy keeps the symbol (${get(rail)?.letter} -> ${get(railCopy)?.letter})`);
  check(get(busCopy)?.letter === get(a)?.letter, `a bus copy keeps the symbol (${get(a)?.letter} -> ${get(busCopy)?.letter})`);
  await page.evaluate(() => { window.tw.selection.value = null; });
  await demandFinal(page, { polling: 250, lines: 0 });

  const map = await page.$('#map');
  const box = await map.boundingBox();
  const shades = ['light', 'dark'];
  const shade = async (s) => { await page.evaluate((s) => { window.tw.shade.value = s; }, s); await page.waitForTimeout(700); };

  // the hover: on the constructed bus route, and on the planned one
  await page.evaluate(([a, b]) => window.tw.map.jumpTo({ center: [a, b], zoom: 12.4 }), [lng, lat + 0.015]);
  await page.waitForTimeout(2000);
  const at = (p) => page.evaluate(([a, b]) => { const q = window.tw.map.project([a, b]); return [q.x, q.y]; }, p);
  const tagText = () => page.evaluate(() => { const t = document.querySelector('.track-tag'); return t && t.style.display !== 'none' ? t.innerText : ''; });
  for (const s of shades) {
    await shade(s);
    const [x, y] = await at([lng, lat + 0.022]);
    await page.mouse.move(box.x + x + 40, box.y + y);
    await page.mouse.move(box.x + x, box.y + y, { steps: 4 });
    await page.waitForTimeout(800);
    const tag = await tagText();
    console.log(s, 'hover tag:', tag.replace(/\n/g, ' | '));
    if (s === 'light' && label === 'after') {
      check(/\bbus\b/i.test(tag), 'the hover says it is a bus');
    }
    await page.screenshot({ path: join(OUT, `${label}-hover-${s}.png`), clip: { x: Math.max(0, box.x + x - 260), y: Math.max(0, box.y + y - 170), width: 520, height: 340 } });
    await page.mouse.move(box.x + 5, box.y + 5);
  }
  await shade('light');
  // the Traffic track display says the riders on the stretch
  await page.evaluate(() => { const t = window.tw; t.display.value = { ...t.display.value, trackColour: 'traffic' }; });
  await page.waitForTimeout(1500);
  for (const [name, p] of [['middle', [lng, lat + 0.022]], ['near-end', [lng, lat + 0.004]]]) {
    const [x, y] = await at(p);
    await page.mouse.move(box.x + x + 40, box.y + y);
    await page.mouse.move(box.x + x, box.y + y, { steps: 4 });
    await page.waitForTimeout(800);
    const tag = await tagText();
    console.log('traffic hover', name, ':', tag.replace(/\n/g, ' | '));
    if (label === 'after') check(/riders/.test(tag), `the hover on the ${name} of the route has riders`);
    await page.screenshot({ path: join(OUT, `${label}-hover-traffic-${name}.png`), clip: { x: Math.max(0, box.x + x - 260), y: Math.max(0, box.y + y - 170), width: 520, height: 340 } });
    await page.mouse.move(box.x + 5, box.y + 5);
  }
  await page.evaluate(() => { const t = window.tw; t.display.value = { ...t.display.value, trackColour: 'line' }; });

  // the Lines tab and the map ends
  await page.click('[aria-label="Lines"]');
  await page.waitForTimeout(800);
  for (const s of shades) {
    await shade(s);
    const d = await (await page.$('#dock')).boundingBox();
    await page.screenshot({ path: join(OUT, `${label}-lines-tab-${s}.png`), clip: { x: d.x, y: d.y, width: d.width, height: 280 } });
  }
  await shade('light');
  await page.evaluate((n) => { window.tw.selection.value = { kind: 'line', line: String(n) }; }, a);
  await page.waitForTimeout(1000);
  for (const s of shades) {
    await shade(s);
    const d = await (await page.$('#dock')).boundingBox();
    await page.screenshot({ path: join(OUT, `${label}-panel-${s}.png`), clip: { x: d.x, y: d.y + 280, width: d.width, height: 260 } });
  }
  await shade('light');
  await page.evaluate(() => { window.tw.selection.value = null; });
  await page.evaluate(([a, b]) => window.tw.map.jumpTo({ center: [a, b], zoom: 16 }), [lng, lat + 0.04]);
  await page.waitForTimeout(2000);
  for (const s of shades) {
    await shade(s);
    const [x, y] = await at([lng, lat + 0.04]);
    await page.mouse.move(box.x + 5, box.y + 5);
    await page.screenshot({ path: join(OUT, `${label}-ends-${s}.png`), clip: { x: Math.max(0, box.x + x - 140), y: Math.max(0, box.y + y - 100), width: 280, height: 200 } });
  }
  await shade('light');
  check(errors.length === 0, `no console errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
