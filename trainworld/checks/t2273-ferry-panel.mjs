// T-2273, T-2274, T-2275: the ferry tool's tag, its pointer leg, new ferry lines' headways, ferry lines'
// widths by riders in the Traffic view, and the ferry panel's port list (a click on a port's dot takes it
// out, a drag of its row moves it, undo puts it back). From the maps root, on a Vite of your own:
//   node trainworld/checks/t2273-ferry-panel.mjs [dev url]
// Loads T-2195's real-ferries.save (artifacts/trainworld/t2195/, five NYC ferry lines; run t2195-ferry-demand.mjs
// to make it) and waits for its demand: about 2-3 minutes on 3 workers, claim 5 cores. Pictures in
// artifacts/trainworld/ferries/.
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';

const OUT = join(ARTIFACTS, 'ferries');
await mkdir(OUT, { recursive: true });
const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6177/';
const WORKERS = Number(process.env.T2273_WORKERS ?? 3);
/** `T2273_ONLY=tool`: the ferry tool's part alone, on a new game */
const TOOL_ONLY = process.env.T2273_ONLY === 'tool';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = await launchChrome({ prefix: 'tw-t2273-', size: [1300, 850], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
const errors = [];
collectErrors(page, errors);
try {
  await page.goto(`${BASE}?debug=1&new=1&opened=all&demandWorkers=${WORKERS}&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && window.tw.tools?.ferry, undefined, { timeout: 180000 });
  if (!TOOL_ONLY) {
  await importSave(page, await readFile(join(ARTIFACTS, 't2195', 'real-ferries.save')));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  await demandFinal(page, { timeout: 1200000, polling: 1000, lines: 100 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));

  const ferries = await page.evaluate(() => {
    const w = window.tw.world.value;
    return w.save.lines.filter((l) => w.lineStats[l.id]?.bus?.ferry).map((l) => ({ id: l.id, num: l.num, name: l.name, ports: w.lineStats[l.id].bus.ferry.ports.length }));
  });
  const er = ferries.find((f) => f.name === 'East River') ?? ferries.sort((a, b) => b.ports - a.ports)[0];
  check(!!er && er.ports >= 4, 'a ferry line with four ports or more', JSON.stringify(ferries.map((f) => `${f.name} ${f.ports}`)));

  // ---- T-2274: the Traffic view's widths on the map and in the panel
  await page.evaluate(() => {
    const tw = window.tw;
    tw.display.value = { ...tw.display.value, trackColour: 'traffic' };
    tw.map.jumpTo({ center: [-73.985, 40.72], zoom: 12.6 });
  });
  await wait(2500);
  const widths = await page.evaluate((num) => {
    const b = window.tw.renderer.buses;
    return Array.from(b.traffic?.get(num) ?? []);
  }, er.num);
  check(widths.length === er.ports - 1 && new Set(widths.map((v) => v.toFixed(3))).size > 1, 'the East River line\'s stretches have widths of their own in the Traffic view', widths.map((v) => v.toFixed(2)).join(' '));
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, er.id);
  await wait(1500);
  const stretches = await page.locator('.ferry-ports .stretch').count();
  check(stretches === 2 * (er.ports - 1), 'the ferry panel draws each stretch by its riders', `${stretches} halves`);
  const tip = await page.locator('.ferry-ports .stretch').nth(1).getAttribute('data-tip');
  check(/riders/.test(tip ?? ''), 'a stretch\'s hover gives its riders', JSON.stringify(tip));
  await page.screenshot({ path: join(OUT, 'traffic-light.png') });
  // the map's hover on the route tells the stretch under it (each stretch its own riders)
  const hover = await page.evaluate((num) => {
    const tw = window.tw, w = tw.world.value, l = w.save.lines.find((x) => x.num === num), b = w.lineStats[l.id].bus;
    const out = [];
    for (let k = 0; k + 1 < b.ferry.ports.length; k++) {
      const [a, c] = [b.ferry.ports[k].berth, b.ferry.ports[k + 1].berth];
      // the route point nearest the middle of the two berths
      let best = Infinity, at = null;
      for (let i = 0; i < b.routeLL.length; i += 2) {
        const d = (b.routeLL[i] - (a[0] + c[0]) / 2) ** 2 + (b.routeLL[i + 1] - (a[1] + c[1]) / 2) ** 2;
        if (d < best) { best = d; at = [b.routeLL[i], b.routeLL[i + 1]]; }
      }
      const p = tw.map.project(at);
      out.push(tw.renderer.buses.stretchAt(tw.overlay, p.x, p.y, num));
    }
    return out;
  }, er.num);
  check(hover.every((k, i) => k === i), 'the route\'s hover finds each stretch between ports', hover.join(' '));

  // ---- T-2275: a click on a port's dot takes it out, undo puts it back, a drag moves it
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, trackColour: 'line' }; });
  await wait(500);
  const portsOf = () => page.evaluate((id) => window.tw.world.value.lineStats[id].bus.ferry.ports.map((p) => p.node), er.id);
  const before = await portsOf();
  await page.locator('.ferry-ports li').nth(2).locator('.stop-hit').click();
  await page.waitForFunction(([id, n]) => window.tw.world.value.lineStats[id]?.bus?.ferry?.ports.length === n, [er.id, before.length - 1], { timeout: 60000 });
  const less = await portsOf();
  check(JSON.stringify(less) === JSON.stringify(before.filter((_, i) => i !== 2)), 'a click on the third port\'s dot takes it out of the line', `${before.join(',')} -> ${less.join(',')}`);
  await page.evaluate(() => window.tw.client.edit({ op: 'undo' }));
  await page.waitForFunction(([id, n]) => window.tw.world.value.lineStats[id]?.bus?.ferry?.ports.length === n, [er.id, before.length], { timeout: 60000 });
  check(JSON.stringify(await portsOf()) === JSON.stringify(before), 'undo puts it back');
  await wait(500);
  const rows = page.locator('.ferry-ports li');
  const a = await rows.nth(0).locator('.stop-name').boundingBox(), c = await rows.nth(2).boundingBox();
  await page.mouse.move(a.x + 20, a.y + a.height / 2);
  await page.mouse.down();
  for (let k = 1; k <= 8; k++) { await page.mouse.move(a.x + 20, a.y + a.height / 2 + ((c.y + c.height * 0.8 - a.y - a.height / 2) * k) / 8); await wait(30); }
  await page.screenshot({ path: join(OUT, 'drag-light.png') });
  await page.mouse.up();
  const moved = [before[1], before[2], before[0], ...before.slice(3)];
  await page.waitForFunction(([id, m]) => JSON.stringify(window.tw.world.value.lineStats[id]?.bus?.ferry?.ports.map((p) => p.node)) === m, [er.id, JSON.stringify(moved)], { timeout: 60000 }).catch(() => {});
  check(JSON.stringify(await portsOf()) === JSON.stringify(moved), 'dragging the first port below the third moves it there', `${before.join(',')} -> ${(await portsOf()).join(',')}`);
  await page.evaluate(() => window.tw.client.edit({ op: 'undo' }));
  await wait(1000);
  }

  // ---- T-2273: the ferry tool's tag stays up while the pointer moves; the pointer's leg; new headways
  await page.evaluate(() => { window.tw.selection.value = null; window.tw.map.jumpTo({ center: [-74.018, 40.748], zoom: 13.2 }); });
  await wait(1500);
  await page.keyboard.press('o');
  await wait(300);
  const xy = (ll) => page.evaluate((ll) => { const p = window.tw.map.project(ll); const r = window.tw.map.getCanvas().getBoundingClientRect(); return [r.left + p.x, r.top + p.y]; }, ll);
  // two ports on the Hudson, where the save has none: Hoboken, then Manhattan's W 39th St, clicked on
  // landings the router gives; the pointer rests on the Manhattan shore for the pictures in between
  const landings = [];
  for (const near of [[-74.0275, 40.7360], [-74.0040, 40.7600]]) {
    const ll = await page.evaluate(async ([lng, lat]) => {
      const water = window.tw.tools.ferry.water, k = 1 / (111320 * Math.cos((lat * Math.PI) / 180)), j = 1 / 110540;
      for (const r of [0, 50, 100, 150]) for (let a = 0; a < (r ? 8 : 1); a++) {
        const s = await water.port([lng + r * Math.cos((a * Math.PI) / 4) * k, lat + r * Math.sin((a * Math.PI) / 4) * j]);
        if (s?.ok) return s.landing;
      }
      return [lng, lat];
    }, near);
    landings.push(ll);
  }
  const place = async (ll) => {
    const [x, y] = await xy(ll);
    await page.mouse.move(x, y);
    await wait(400);
    await page.mouse.click(x, y);
    await wait(1500);
  };
  await place(landings[0]);
  const placed = await page.evaluate(() => window.tw.tools.ferry.ports?.length ?? null);
  // sweep the pointer from the river onto the Manhattan shore and back, a step a frame, noting the tag
  const [x0, y0] = await xy([-74.016, 40.752]), [x1, y1] = await xy([-73.99, 40.757]);
  let hidden = 0, shown = 0, texts = new Set();
  for (let k = 0; k <= 120; k++) {
    const f = k <= 60 ? k / 60 : (120 - k) / 60;
    await page.mouse.move(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f);
    await wait(16);
    const t = await page.evaluate(() => { const el = document.querySelector('.tool-tag.build-tag'); return el && el.style.display !== 'none' ? el.innerText : null; });
    if (t === null) hidden++;
    else { shown++; texts.add(t.split('\n')[0]); }
  }
  check(hidden === 0, 'the ferry tool\'s tag stays up as the pointer moves', `${shown} shown, ${hidden} hidden over 121 moves; first rows: ${[...texts].join(' | ')} (ports placed ${placed})`);
  const [sx, sy] = await xy(landings[1]);
  await page.mouse.move(sx + 14, sy);
  await wait(800);
  await page.screenshot({ path: join(OUT, 'tool-light.png') });
  await page.evaluate(() => { window.tw.shade.value = 'dark'; });
  await wait(1500);
  await page.mouse.move(sx + 15, sy);
  await wait(800);
  await page.screenshot({ path: join(OUT, 'tool-dark.png') });
  await page.evaluate(() => { window.tw.shade.value = 'light'; });
  await wait(1000);
  await place(landings[1]);
  check((await page.evaluate(() => window.tw.tools.ferry.ports?.length)) === 2, 'two ports placed by clicks');
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => window.tw.tools.ferry.active === false, undefined, { timeout: 30000 }).catch(() => {});
  await wait(1000);
  const auto = await page.evaluate(() => {
    const w = window.tw.world.value, l = w.save.lines.at(-1);
    return w.lineStats[l.id]?.bus?.ferry ? w.lineStats[l.id].autoS : null;
  });
  check(JSON.stringify(auto) === '[1200,1800,0]', 'a new ferry line runs a boat every 20 and 30 minutes and none at low demand', JSON.stringify(auto));
  await wait(800);
  await page.screenshot({ path: join(OUT, 'new-line-panel.png') });
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-2273 FAILED: ${fail}` : 'T-2273 PASSED');
process.exit(fail ? 1 : 0);
