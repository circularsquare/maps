// T-3169, T-3168, T-3170, T-3188: the track bar's Advanced row and flyovers drawn with the track tool.
//  A  the Advanced expander at the bottom right of the track bar opens a third row holding Target max speed and
//     Flyover target max speed, right-aligned under it with the bar's own row gap; its open state survives a reload.
//     Pictures closed and open, light and dark; and the phone bar's speed pop with both steppers.
//  B  a double branch drawn off a double main with the ramp switch on: the preview has a flyover leg, the map shows
//     a curve label for the leg's slowest curve beside the route's own (T-3168), the slowest label is the tag's
//     speed, and a higher flyover target gives a faster leg (T-3169).
//  C  the same with High speed picked lays a flyover too (T-3170), and a branch drawn nearly parallel (T-3188).
// Pictures in artifacts/trainworld/t3169-*.png.
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6111):
//   node trainworld/checks/t3169-flyover-speed.mjs [dev url]
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6111/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const RC = [-73.93, 40.675];

const chrome = await launchChrome({ prefix: 'tw-t3169-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const open = async (size, ui) => {
  const page = await chrome.ctx.newPage();
  if (size) await page.setViewportSize({ width: size[0], height: size[1] });
  await page.addInitScript((ui) => { try { if (!sessionStorage.getItem('t3169')) { localStorage.setItem('trainworld-ui', JSON.stringify(ui)); sessionStorage.setItem('t3169', '1'); } } catch {} }, ui);
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  return page;
};
const UI = { shade: 'light', v: 7, build: { level: 0, keep: false, ramp: true } };

try {
  // ---- A: the track bar
  const page = await open(null, UI);
  // `s`: { tool, shade, type, design, fly, ramp }, each set where given
  const setUi = (s) => page.evaluate(async (s) => {
    const ui = await import('/src/game/ui.ts');
    if (s.type !== undefined) ui.setTrackType(s.type);
    if (s.design !== undefined) ui.designSpeed.value = s.design;
    if (s.fly !== undefined) ui.flyoverSpeed.value = s.fly;
    if (s.ramp !== undefined) ui.autoRamp.value = s.ramp;
    if (s.shade !== undefined) ui.shade.value = s.shade;
    if (s.tool !== undefined) ui.tool.value = s.tool;
  }, s);
  await setUi({ tool: 'track' });
  await wait(500);
  const bar = page.locator('.track-bar');
  const shotBar = async (name) => { const b = await bar.boundingBox(); await page.screenshot({ path: join(ARTIFACTS, `t3169-${name}.png`), clip: { x: b.x - 30, y: b.y - 20, width: b.width + 60, height: b.height + 40 } }); };
  check(await page.locator('.tb-adv-row').count() === 0, 'A: the Advanced row is closed at first');
  await shotBar('bar-closed-light');
  await page.locator('.tb-advanced').click();
  await wait(300);
  const geo = await page.evaluate(() => {
    const rows = [...document.querySelectorAll('.track-bar > .tb-row')].map((r) => r.getBoundingClientRect());
    const adv = document.querySelector('.tb-advanced').getBoundingClientRect();
    const speeds = [...document.querySelectorAll('.tb-adv-row .tb-speed')].map((s) => s.getBoundingClientRect());
    const steppers = [...document.querySelectorAll('.tb-adv-row .stepper')].map((s) => s.getBoundingClientRect());
    const labels = [...document.querySelectorAll('.tb-adv-row .tb-speed > span:first-child')].map((s) => s.textContent);
    return { rows: rows.map((r) => [r.top, r.bottom, r.left, r.right]), adv: [adv.top, adv.bottom, adv.left, adv.right], speeds: speeds.map((r) => [r.left, r.right, r.top, r.bottom]), steppers: steppers.map((r) => [r.top, r.bottom]), labels };
  });
  check(geo.rows.length === 3, 'A: Advanced adds a third row', JSON.stringify(geo.rows));
  check(geo.labels.join('|') === 'Target max speed|Flyover target max speed', 'A: the row holds the two target speeds', geo.labels.join('|'));
  if (geo.rows.length === 3) {
    const gap1 = geo.rows[1][0] - geo.rows[0][1], gap2 = geo.rows[2][0] - geo.rows[1][1];
    check(Math.abs(gap1 - gap2) < 1.5, 'A: the third row sits the bar\'s own row gap below the second', `${gap1.toFixed(1)} vs ${gap2.toFixed(1)} px`);
    const right = geo.speeds[geo.speeds.length - 1][1];
    check(Math.abs(right - geo.adv[3]) < 1.5, 'A: the row is right-aligned under the Advanced button', `${right.toFixed(1)} vs ${geo.adv[3].toFixed(1)}`);
    check(geo.steppers.length === 2 && Math.abs(geo.steppers[0][0] - geo.steppers[1][0]) < 0.5, 'A: the two steppers sit level');
  }
  await shotBar('bar-open-light');
  await setUi({ shade: 'dark' });
  await wait(500);
  await shotBar('bar-open-dark');
  await setUi({ shade: 'light' });
  // the flyover stepper steps and is remembered
  const before = await page.evaluate(async () => (await import('/src/game/ui.ts')).flyoverSpeed.value);
  await page.locator('.tb-adv-row .tb-speed').nth(1).locator('button').last().click();
  await wait(300);
  const after = await page.evaluate(async () => (await import('/src/game/ui.ts')).flyoverSpeed.value);
  check(before === 80 && after === 100, 'A: flyover target starts at 80 km/h and steps to 100', `${before} -> ${after}`);
  await page.reload();
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  await setUi({ tool: 'track' });
  await wait(500);
  const kept = await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); return { adv: ui.buildAdvanced.value, fly: ui.flyoverSpeed.value, row: document.querySelectorAll('.tb-adv-row').length }; });
  check(kept.adv && kept.fly === 100 && kept.row === 1, 'A: after a reload the row is open and the flyover target kept', JSON.stringify(kept));
  await setUi({ fly: 80 });

  // ---- B and C: flyovers drawn with the track tool
  await page.evaluate(([c]) => {
    const tw = window.tw, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + dy });
    const ll = (p) => { const q = tw.packToLngLat(p.x, p.y); return Array.isArray(q) ? q : [q.lng, q.lat]; };
    window.__t = { P, ll };
  }, [RC]);
  const box = await (await page.$('#map')).boundingBox();
  const jump = async (p, zoom) => { await page.evaluate(([p, z]) => window.tw.map.jumpTo({ center: window.__t.ll(window.__t.P(p[0], p[1])), zoom: z }), [p, zoom]); await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {}); await wait(600); };
  const at = (p) => page.evaluate((p) => { const q = window.tw.map.project(window.__t.ll(window.__t.P(p[0], p[1]))); return [q.x, q.y]; }, p).then(([x, y]) => [box.x + x, box.y + y]);
  const ghost = async () => {
    const t0 = Date.now();
    let last = null;
    for (;;) {
      const g = await page.evaluate(() => {
        const a = window.tw.tools.lastPreview;
        if (!a) return null;
        let top = Infinity;
        for (const v of a.speeds ?? []) if (v >= 0) top = Math.min(top, v);
        const labels = [...document.querySelectorAll('.pi-label')].filter((e) => e.style.display !== 'none' && e.offsetParent).map((e) => e.textContent);
        return { ok: a.ok, top: Math.round(top), verts: Array.from(a.verts).map((v) => Math.round(v)), legs: a.legs ? Array.from(a.legs) : [], noFlyover: !!a.noFlyover, labels, key: window.tw.tools.previewKey };
      });
      if (g && last && JSON.stringify(g) === JSON.stringify(last) && Date.now() - t0 > 900) return g;
      if (Date.now() - t0 > 20000) return g;
      last = g;
      await wait(250);
    }
  };
  // a main along y = dy, and a branch drawn from (0, dy) toward `to` with the tool; Esc after
  const branch = async (dy, to, name, setup) => {
    await page.evaluate(async ([dy]) => { const { P } = window.__t; await window.tw.client.edit({ op: 'route', from: { kind: 'free', ...P(-3000, dy), level: 0 }, to: { kind: 'free', ...P(4000, dy), level: 0 }, pis: [], single: false }); }, [dy]);
    await wait(800);
    await setUi({ ...setup, tool: 'track' });
    await jump([900, dy + to[1] / 3], 14.6);
    const [sx, sy] = await at([0, dy]);
    const [ex, ey] = await at([to[0], dy + to[1]]);
    await page.mouse.move(sx + 15, sy + 15, { steps: 2 });
    await page.mouse.move(sx, sy, { steps: 3 });
    await wait(300);
    await page.mouse.click(sx, sy);
    await wait(400);
    await page.mouse.move(ex - 30, ey, { steps: 3 });
    await page.mouse.move(ex, ey, { steps: 3 });
    const g = await ghost();
    if (name) await page.screenshot({ path: join(ARTIFACTS, `t3169-${name}.png`) });
    lastEnd = [ex, ey];
    return g;
  };
  let lastEnd = null;
  const reset = async () => { await page.keyboard.press('Escape'); await wait(300); await page.keyboard.press('Escape'); await wait(300); };
  const kmhOf = (l) => Number(String(l).replace(/[^\d.]/g, ''));

  const b80 = await branch(0, [2600, 1500], 'drawn-80', { type: 2, design: 160, fly: 80, ramp: true, tool: 'track' });
  check(b80?.ok && b80.legs.length === 4 && !b80.noFlyover, 'B: a double branch off double track lays a flyover with one leg', JSON.stringify(b80?.legs));
  check(b80 && b80.legs[3] >= 79, 'B: at the 80 km/h flyover target its leg\'s slowest curve holds 80 km/h', `${b80?.legs[3]?.toFixed(1)} km/h`);
  // closer in, where the leg's label and the route's own do not overlap
  await jump([650, 200], 16.4);
  const near = await ghost();
  await page.screenshot({ path: join(ARTIFACTS, 't3169-labels-80.png') });
  const ls = (near?.labels ?? []).map(kmhOf);
  check(ls.some((v) => Math.abs(v - Math.round(b80.legs[3])) <= 1) && ls.some((v) => v > b80.legs[3] + 5), 'B: the map labels both the leg\'s curve and the route\'s own (T-3168)', near?.labels.join(', '));
  check(ls.length && Math.min(...ls) === b80.top, 'B: the slowest label on the map is the tag\'s speed', `${Math.min(...ls)} vs ${b80?.top}`);
  check(b80.labels.map(kmhOf).includes(b80.top), 'B: zoomed out, where the two overlap, the slower shows', b80.labels.join(', '));
  await reset();

  const b20 = await branch(4000, [2600, 1500], null, { fly: 20 });
  check(b20?.ok && b20.legs.length === 4 && b20.legs[3] < b80.legs[3] - 10, 'B: a 20 km/h flyover target lays a tighter leg', `${b20?.legs[3]?.toFixed(1)} vs ${b80?.legs[3]?.toFixed(1)} km/h`);
  await reset();

  const hsr = await branch(8000, [2600, 1500], 'drawn-hsr', { fly: 80, type: 'hsr' });
  check(hsr?.ok && hsr.legs.length === 4 && !hsr.noFlyover && /"hsr":true/.test(hsr.key ?? ''), 'C: a high-speed branch lays a flyover (T-3170)', JSON.stringify({ legs: hsr?.legs, no: hsr?.noFlyover }));
  // built: its legs are one-way single high-speed track
  const edgesBefore = await page.evaluate(() => window.tw.world.value.edges.map((e) => e.id));
  await page.mouse.click(lastEnd[0], lastEnd[1]);
  await wait(2000);
  const made = await page.evaluate((b) => window.tw.world.value.edges.filter((e) => !b.includes(e.id)).map((e) => ({ tracks: e.tracks, hsr: !!e.hsr, oneWay: !!e.oneWay })), edgesBefore);
  const singles = made.filter((e) => e.tracks === 1);
  check(singles.length >= 2 && singles.every((e) => e.hsr && e.oneWay), 'C: the built high-speed flyover\'s legs are one-way single high-speed track', JSON.stringify(made));
  await reset();
  await page.evaluate(async () => { const c = window.tw.client; await c.edit({ op: 'constructAll' }); });
  await wait(1500);
  await jump([700, 8150], 16.6);
  await page.screenshot({ path: join(ARTIFACTS, 't3169-hsr-built.png') });

  const shallow = await branch(12000, [3500, 250], 'drawn-shallow', { type: 2, design: 160 });
  check(shallow?.ok && shallow.legs.length === 4 && !shallow.noFlyover, 'C: a branch leaving at about 4 degrees lays a flyover (T-3188)', JSON.stringify({ legs: shallow?.legs, no: shallow?.noFlyover }));
  await reset();

  // phone: the speed button's pop holds both steppers
  const phone = await open([390, 844], { ...UI, build: { ...UI.build, advanced: false } });
  await phone.evaluate(async () => { (await import('/src/game/ui.ts')).tool.value = 'track'; });
  await wait(600);
  await phone.locator('.tb-speed-btn').dispatchEvent('click');
  await wait(400);
  const pop = await phone.evaluate(() => [...document.querySelectorAll('.tb-pop-speeds .tb-speed > span:first-child')].map((s) => s.textContent));
  check(pop.length === 2, 'A: on a phone the speed button opens both steppers', pop.join('|'));
  await phone.screenshot({ path: join(ARTIFACTS, 't3169-phone-pop.png') });

  const bad = errors.filter((e) => !/favicon|404|openfreemap|AJAXError/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3169 FAILED: ${fail}` : 'T-3169 PASSED');
process.exit(fail ? 1 : 0);
