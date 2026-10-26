// T-1414, T-1415, T-1417, T-1420 on the real save: a click on a station with a train on it picks the
// station; the station panel's "Station upgrades" heading; the stop list without "Add all" and the A
// key adding every station the selected line passes; a double click on the blueprint's disabled
// total enabling every disabled piece. Screenshots into artifacts/trainworld/t1414/.
//   node trainworld/checks/t1414-ui.mjs [url]     (from the maps root; a running dev server)
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, NOISE, REAL_SAVE, collectErrors, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6171/';
const OUT = join(ARTIFACTS, 't1414');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

const { ctx, close } = await launchChrome({ prefix: 'tw-t1414-', size: [1500, 950] });
const errors = [];
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.waitForTimeout(2000);

  // ---- T-1414: a station with a train on it
  await page.evaluate(() => {
    const tw = window.tw;
    tw.clock.setRunning(false);
    tw.clock.set(Math.floor(tw.clock.now() / 86400) * 86400 + 8.5 * 3600);
    tw.selection.value = null;
    tw.overlay.invalidate();
  });
  await page.waitForFunction(() => window.tw.renderer.tripCount > 0, undefined, { timeout: 60000 });
  const both = await page.evaluate(() => {
    const tw = window.tw, r = tw.renderer, gpu = r.probe(tw.clock.now()), a = r.net?.anchor ?? tw.anchor();
    const st = tw.world.value.save.stations.filter((s) => s.built);
    const m = (lat) => [111320 * Math.cos((lat * Math.PI) / 180), 110574];
    let best = null;
    for (let i = 0; i < r.tripCount; i++) {
      if (!(gpu[i * 2] < 1e8)) continue;
      const [lng, lat] = tw.toLngLat(gpu[i * 2], gpu[i * 2 + 1], a);
      const [mx, my] = m(lat);
      for (const s of st) {
        const d = Math.hypot((s.lng - lng) * mx, (s.lat - lat) * my);
        if (!best || d < best.d) best = { d, lng, lat, id: s.id, name: s.name };
      }
    }
    return best;
  });
  console.log(`closest train to a station: ${both?.d.toFixed(1)} m from ${both?.name}`);
  await page.evaluate((t) => { window.tw.map.jumpTo({ center: [t.lng, t.lat], zoom: 16.5 }); window.tw.overlay.invalidate(); }, both);
  await page.waitForTimeout(900);
  const xy = await page.evaluate((t) => { const r = window.tw.map.getContainer().getBoundingClientRect(), p = window.tw.map.project([t.lng, t.lat]); return [r.left + p.x, r.top + p.y]; }, both);
  // what the pick finds at the train's own point, with trains and stations both there
  const hit = await page.evaluate(([x, y]) => { const r = window.tw.map.getContainer().getBoundingClientRect(); return window.tw.pick(window.tw.overlay, x - r.left, y - r.top, true); }, xy);
  await page.mouse.click(xy[0], xy[1]);
  await page.waitForTimeout(400);
  const sel = await page.evaluate(() => window.tw.selection.value);
  check(sel?.kind === 'station' && sel.station === both.id, `a click on a station with a train on it picks the station (pick ${JSON.stringify(hit)}, selected ${JSON.stringify(sel)})`);

  // ---- T-1415: the heading
  await page.evaluate((id) => { window.tw.select({ kind: 'station', station: id }); }, both.id);
  await page.waitForSelector('.upgrades', { timeout: 30000 });
  await page.waitForTimeout(800);
  const head = await page.evaluate(() => { const u = document.querySelector('.upgrades'); const h = u?.previousElementSibling; return h ? { text: h.textContent, cls: h.className } : null; });
  check(head?.text === 'Station upgrades' && head.cls === 'h', `the upgrades row has the heading "Station upgrades" (${JSON.stringify(head)})`);
  const box = await (await page.$('.upgrades')).boundingBox();
  await page.screenshot({ path: join(OUT, 'station-upgrades-heading.png'), clip: { x: box.x - 20, y: Math.max(0, box.y - 200), width: 420, height: 320 } });

  // ---- T-1417: the stop list
  // the real save's lines stop everywhere, so make two expresses: copies stopping at three stops
  await page.evaluate(async () => {
    const c = window.tw.client, w = window.tw.world.value;
    const src = w.save.lines.filter((l) => l.stops.length >= 9).slice(0, 2);
    for (const l of src) {
      const copy = await c.edit({ op: 'copyLine', line: l.num });
      await c.edit({ op: 'setStops', line: copy.line, stops: [l.stops[0], l.stops[3], l.stops[l.stops.length - 1]].map(Number) });
    }
    await new Promise((d) => setTimeout(d, 1500));
  });
  const lines = await page.evaluate(() => {
    const w = window.tw.world.value;
    return w.save.lines.map((l) => ({ id: l.id, name: l.name, n: l.stops.length, passed: w.lineStats[l.id]?.passed?.length ?? 0 })).filter((l) => l.passed > 1);
  });
  console.log(`lines passing stations: ${lines.length} (${lines.slice(0, 5).map((l) => `${l.name}:${l.passed}`).join(', ')})`);
  if (lines.length < 2) throw new Error('need two lines that pass stations');
  const [one, two] = lines;
  await page.evaluate(() => { document.activeElement?.blur?.(); window.tw.select(null); });
  await page.evaluate((id) => window.tw.select({ kind: 'line', line: id }), one.id);
  await page.waitForSelector('button.related-toggle', { timeout: 30000 });
  await page.waitForTimeout(500);
  await page.keyboard.press('l');
  await page.waitForSelector('li.passed', { timeout: 30000 });
  await page.waitForTimeout(500);
  const text = await page.evaluate(() => document.querySelector('.inspector, #inspector, .panel')?.innerText ?? document.body.innerText);
  check(!/Add all/.test(await page.evaluate(() => document.body.innerText)), 'no "Add all" row in the stop list');
  const stopsShot = async (file) => {
    const h = await page.$('button.related-toggle');
    await h.evaluate((el) => el.closest('.h').scrollIntoView({ block: 'start' }));
    await page.waitForTimeout(200);
    const hb = await (await h.evaluateHandle((el) => el.closest('.h'))).boundingBox();
    const rb = await (await page.$('.related') ?? await page.$('ul.stops')).boundingBox();
    const vp = page.viewportSize();
    const top = Math.max(0, hb.y - 8), bottom = Math.min(vp.height, rb.y + rb.height + 10);
    await page.screenshot({ path: join(OUT, file), clip: { x: Math.max(0, hb.x - 12), y: top, width: Math.min(vp.width, hb.width + 24), height: bottom - top } });
  };
  await stopsShot('stops-editing.png');
  // a gap's row, once: the stations show with Add and Cancel
  await page.click('li.passed button');
  await page.waitForSelector('li.adding', { timeout: 10000 });
  await page.waitForTimeout(300);
  check(!!(await page.$('li.confirm')), 'a click on a gap row shows Add and Cancel');
  await stopsShot('stops-pending.png');
  await page.click('li.confirm >> text=Cancel');
  await page.waitForTimeout(300);
  check(!(await page.$('li.adding')), 'Cancel puts the rows back');
  // a gap's row, twice: adds at once
  const n0 = await page.evaluate((id) => window.tw.world.value.save.lines.find((l) => l.id === id).stops.length, one.id);
  // as a hand does it: the panel redraws between the two clicks
  const gb = await (await page.$('li.passed button')).boundingBox();
  const gx = gb.x + gb.width / 2, gy = gb.y + gb.height / 2;
  await page.mouse.click(gx, gy);
  await page.waitForTimeout(120);
  await page.mouse.click(gx, gy, { clickCount: 2 });
  await page.waitForTimeout(1200);
  console.log('after dblclick:', JSON.stringify(await page.evaluate(() => ({ adding: document.querySelectorAll('li.adding').length, passedRows: document.querySelectorAll('li.passed').length, row: !!document.querySelector('.passed-row'), stops: document.querySelectorAll('ul.stops > li').length, sel: window.tw.selection.value, lines: window.tw.world.value.save.lines.map((l) => l.name + ':' + l.stops.length).join(' ').slice(-200) }))));
  const n1 = await page.evaluate((id) => window.tw.world.value.save.lines.find((l) => l.id === id).stops.length, one.id);
  check(n1 > n0, `a double click on a gap row adds its stations (${n0} -> ${n1} stops)`);
  // the A key, on another line, with its panel open and not editing
  await page.evaluate(() => { document.activeElement?.blur?.(); window.tw.select(null); });
  await page.evaluate(() => window.tw.tools?.exit?.());
  await page.evaluate((id) => window.tw.select({ kind: 'line', line: id }), two.id);
  await page.waitForSelector('button.related-toggle', { timeout: 30000 });
  await page.waitForTimeout(500);
  await page.mouse.move(2, 2);
  const before = await page.evaluate((id) => window.tw.world.value.save.lines.find((l) => l.id === id).stops.length, two.id);
  await page.keyboard.press('a');
  await page.waitForFunction(([id, n]) => window.tw.world.value.save.lines.find((l) => l.id === id).stops.length > n, [two.id, before], { timeout: 15000 }).catch(() => {});
  const after = await page.evaluate((id) => ({ n: window.tw.world.value.save.lines.find((l) => l.id === id).stops.length, left: window.tw.world.value.lineStats[id].passed.length }), two.id);
  check(after.n === before + two.passed, `the A key adds all ${two.passed} passed stations (${before} -> ${after.n} stops, ${after.left} still passed)`);
  await page.waitForTimeout(600);
  await stopsShot('stops-after-a-key.png');
  // A with no line open does nothing
  await page.evaluate(() => { window.tw.select(null); });
  const total = await page.evaluate(() => window.tw.world.value.save.lines.reduce((s, l) => s + l.stops.length, 0));
  await page.keyboard.press('a');
  await page.waitForTimeout(500);
  check(total === await page.evaluate(() => window.tw.world.value.save.lines.reduce((s, l) => s + l.stops.length, 0)), 'A with no line selected changes nothing');

  // ---- T-1420: the disabled total
  await page.keyboard.press('5');
  await page.waitForTimeout(500);
  const made = await page.evaluate(async () => {
    const tw = window.tw, c = tw.client;
    const pack = ([lng, lat]) => { const p = tw.lngLatToPack(lng, lat); return [p.x, p.y]; };
    const a = pack([-73.45, 40.78]), b = pack([-73.40, 40.78]);
    const r = await c.edit({ op: 'route', from: { kind: 'free', x: a[0], y: a[1], level: -1 }, to: { kind: 'free', x: b[0], y: b[1], level: -1 }, pis: [], single: false });
    await new Promise((d) => setTimeout(d, 800));
    const edges = tw.world.value.edges.filter((e) => !e.built).map((e) => e.id);
    if (edges.length) await c.edit({ op: 'blueprintOff', edges, stations: [], off: true }, true);
    await new Promise((d) => setTimeout(d, 800));
    return { ok: r?.ok, edges: edges.length, off: tw.world.value.edges.filter((e) => !e.built && e.off).length };
  });
  console.log(`blueprint: ${JSON.stringify(made)}`);
  await page.waitForSelector('.off-cost', { timeout: 15000 });
  const tab = await page.$('.off-cost');
  await tab.scrollIntoViewIfNeeded();
  await tab.hover();
  await page.waitForTimeout(700);
  const tag = await page.evaluate(() => { const t = document.querySelector('.tool-tag.ui-tag'); return t && t.style.display !== 'none' ? t.innerText : ''; });
  check(/Double click to enable them all/.test(tag), `the disabled total's hover (${tag.replace(/\n/g, ' | ')})`);
  const hb = await page.$('.blueprint-head');
  const bb = await hb.boundingBox();
  await page.screenshot({ path: join(OUT, 'blueprint-disabled-hover.png'), clip: { x: Math.max(0, bb.x - 20), y: Math.max(0, bb.y - 20), width: 460, height: 180 } });
  await tab.dblclick();
  await page.waitForTimeout(1500);
  const offAfter = await page.evaluate(() => window.tw.world.value.edges.filter((e) => !e.built && e.off).length);
  check(offAfter === 0 && !(await page.$('.off-cost')), `a double click on the disabled total enables every piece (${made.off} off -> ${offAfter})`);

  check(errors.length === 0, `no console errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally {
  await close();
}
process.exit(fail ? 1 : 0);
