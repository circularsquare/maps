// T-2194: the Ferry tool on a new game in New York, with the mouse. The Build tab's Ferry button, two ports
// across the harbour (Whitehall and St George: the Staten Island ferry), the route the water worker finds,
// Make line at the price the tag gave (US$10M a port), Construct, boats moving on it, its panel, the Lines
// tab and the Money tab's Ferry fare row and boat rows (the Finances tab), then a save exported and loaded again with the line
// and the boat fleet intact; and two ports on waters that do not meet (Central Park's reservoir and its
// Lake) refused with "No water path found". Screenshots at zoom 12 and 15, light and dark, to
// artifacts/trainworld/t2194/. Uses the Vite already on the url (dev-nowatch.mjs). From the maps root:
//   node trainworld/checks/t2194-ferry-tool.mjs [url]
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

const chrome = await launchChrome({ prefix: 'tw-t2194-', size: [1400, 900], deviceScaleFactor: 1 });
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

  // ---- Make line, then Construct
  const cash0 = await page.evaluate(() => window.tw.world.value.save.cash);
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => !window.tw.tools.ferry.active && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 30000 });
  const made = await page.evaluate(() => {
    const w = window.tw.world.value, l = w.save.lines.at(-1), st = w.lineStats[l.id];
    return { id: l.id, num: l.num, name: l.name, ferry: !!st?.bus?.ferry, ports: st?.bus?.ferry?.ports.length, km: st?.km, notice: document.querySelector('#tool-notes')?.textContent ?? '' };
  });
  check(made.ferry && made.ports === 2 && /^Ferry \d+$/.test(made.name), 'Make line adds a planned ferry line over the two ports', `${made.name}, ${made.ports} ports`);
  check(/Ferry line added: .*\$20(\.0)?M to construct/.test(made.notice), 'the notice says the line and its price', made.notice.slice(0, 120));
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, made.id);
  await page.waitForTimeout(800);
  const panel = await page.evaluate(() => document.querySelector('#inspector, .inspector')?.textContent ?? document.body.textContent);
  check(/Edit route/.test(panel) && /Ports/.test(panel) && /Boats/.test(panel), 'the ferry panel opens with its ports and boats', '');
  // the new line's name box is open (it asks for a name): keep the name, as a player presses Enter, so the
  // colour row it opened folds away before the click (a press outside it folds it under the pointer)
  await page.keyboard.press('Enter');
  await page.waitForTimeout(500);
  const btn = page.locator('.callout button:has-text("Construct for")').first();
  await btn.waitFor({ timeout: 20000 });
  const quote = await btn.textContent();
  await btn.click();
  await page.waitForFunction((id) => window.tw.world.value.lineStats[id]?.bus?.built, made.id, { timeout: 30000 }).catch(async () => { console.log('DEBUG', await page.evaluate(() => JSON.stringify({ notes: document.querySelector('#tools')?.textContent, dialog: document.querySelector('dialog[open]')?.textContent, sel: window.tw.selection.value }))); throw new Error('not constructed'); });
  await page.waitForTimeout(1500);
  const after = await page.evaluate(() => ({ cash: window.tw.world.value.save.cash }));
  const spent = (cash0 - after.cash) / 1e6;
  const boats = await page.evaluate((id) => Math.max(...window.tw.world.value.lineStats[id].trainsNeeded), made.id);
  check(Math.abs(spent - (20 + 8 * boats)) < 0.05, 'Construct charges US$10M a port and US$8M a boat', `spent ${spent.toFixed(2)} for ${boats} boats; button "${quote}"`);
  const ports = await page.evaluate(() => window.tw.world.value.save.stations.filter((s) => s.port).map((s) => ({ name: s.name, built: s.built })));
  check(ports.length === 2 && ports.every((p) => p.built), 'both ports constructed', ports.map((p) => p.name).join(', '));
  const labels = await page.evaluate(() => (window.tw.labels?.labels ?? []).map((l) => `${l.el.textContent}${l.hidden ? ' (hidden)' : ''}`));
  check(ports.every((p) => labels.some((t) => t.startsWith(p.name))), 'the ports\' names are labels on the map', labels.join(', '));

  // ---- boats move on it: the renderer's own positions two game minutes apart (the clock paused at 8:30)
  await page.evaluate(() => {
    const tw = window.tw;
    tw.clock.setRunning(false);
    tw.clock.set(Math.floor(tw.clock.now() / 86400) * 86400 + 8.5 * 3600);
    tw.overlay.invalidate();
  });
  await page.waitForFunction(() => window.tw.renderer.tripCount > 0, undefined, { timeout: 60000 });
  const moved = await page.evaluate(() => {
    const tw = window.tw, r = tw.renderer, t = tw.clock.now(), a = r.net?.anchor ?? tw.anchor();
    const p0 = r.probe(t), p1 = r.probe(t + 120);
    let live = 0, moving = 0, first = null;
    for (let i = 0; i < r.tripCount; i++) {
      if (p0[i * 2] >= 1e8 || p1[i * 2] >= 1e8) continue;
      live++;
      const [x0, y0] = tw.toLngLat(p0[i * 2], p0[i * 2 + 1], a), [x1, y1] = tw.toLngLat(p1[i * 2], p1[i * 2 + 1], a);
      const m = Math.hypot((x1 - x0) * 84000, (y1 - y0) * 111000);
      if (m > 100) moving++;
      first ??= { lng: x0, lat: y0, m };
    }
    return { live, moving, first };
  });
  check(moved.live > 0 && moved.moving > 0, 'boats move along the route', `${moved.moving}/${moved.live} boats moved over 100 m in two game minutes`);
  if (moved.first) {
    await page.evaluate((t) => { window.tw.map.jumpTo({ center: [t.lng, t.lat], zoom: 16 }); window.tw.overlay.invalidate(); }, moved.first);
    await settle(800);
    await page.screenshot({ path: join(OUT, 'boat-light-z16.png'), clip: box });
    const xy = await at([moved.first.lng, moved.first.lat]);
    await page.mouse.click(xy.x, xy.y);
    await page.waitForTimeout(500);
    const boat = await page.evaluate(() => ({ sel: window.tw.selection.value?.kind, text: document.querySelector('.title .name')?.textContent ?? '' }));
    check(boat.sel === 'train' && / boat$/.test(boat.text), 'a click on a boat opens its panel', boat.text);
  }

  // ---- the Lines tab and the Money tab
  await page.click('button.tab[aria-label="Lines"]');
  await page.waitForTimeout(600);
  const lines = await page.evaluate(() => document.body.innerText);
  check(lines.includes(made.name), 'the Lines tab lists the ferry line', made.name);
  check(!!(await page.$('button.new-ferry')), 'the Lines tab has the ferry button');
  await page.click('button.tab[aria-label="Finances"]');
  await page.waitForTimeout(600);
  const money = await page.evaluate(() => document.body.innerText);
  check(/Ferry\s*\n?\s*1\.00x/.test(money) || (money.includes('Ferry') && money.includes('1.00x')), 'the Money tab has the Ferry fare row at 1.00x');
  check(money.includes('Boats owned') && money.includes('Spare boats'), 'the Money tab has the boat rows');
  // step the ferry multiple up once; it is saved with the game
  await page.evaluate(() => window.tw.client.edit({ op: 'fareFerry', mul: 1.25 }));
  await page.waitForTimeout(500);

  // ---- screenshots: zoom 12 and 15, light and dark
  const shot = async (name, centre, zoom) => {
    await page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z }), [centre, zoom]);
    await settle(1500);
    await page.screenshot({ path: join(OUT, name), clip: box });
  };
  await page.evaluate(() => { window.tw.selection.value = null; });
  const centre = [(WHITEHALL[0] + ST_GEORGE[0]) / 2, (WHITEHALL[1] + ST_GEORGE[1]) / 2];
  await shot('line-light-z12.png', centre, 12);
  await shot('whitehall-light-z15.png', [WHITEHALL[0] - 0.003, WHITEHALL[1] - 0.002], 15);
  await page.evaluate(() => window.tw.shade.value = 'dark');
  await page.waitForTimeout(800);
  await shot('line-dark-z12.png', centre, 12);
  await shot('whitehall-dark-z15.png', [WHITEHALL[0] - 0.003, WHITEHALL[1] - 0.002], 15);
  await page.evaluate(() => window.tw.shade.value = 'light');

  // ---- save and load again
  const before = await page.evaluate((id) => {
    const w = window.tw.world.value, st = w.lineStats[id];
    return { route: st.bus.routeLL.length, built: st.bus.built, ports: st.bus.ferry.ports.map((p) => p.node), fleet: window.tw.moneyState?.value?.fleetFerry ?? null };
  }, made.id);
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const saved = await readFile(await (await dl).path());
  await page.evaluate(() => { window.tw.client.lastLoad = null; });
  await importSave(page, saved);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length === 1, undefined, { timeout: 120000 });
  await page.waitForTimeout(2000);
  const back = await page.evaluate((id) => {
    const w = window.tw.world.value, st = w.lineStats[id];
    return st?.bus?.ferry ? { route: st.bus.routeLL.length, built: st.bus.built, ports: st.bus.ferry.ports.map((p) => p.node), fleet: window.tw.moneyState?.value?.fleetFerry ?? null, ferryFare: window.tw.moneyState?.value?.fares?.ferry } : null;
  }, made.id);
  check(!!back && back.route === before.route && back.built && back.ports.join() === before.ports.join(), 'the saved ferry line loads with its route and ports', JSON.stringify(back));
  check(!!back && back.fleet !== null && back.fleet === before.fleet && back.fleet >= boats, 'the boat fleet is saved and restored', `${before.fleet} -> ${back?.fleet}`);
  check(back?.ferryFare === 1.25, 'the ferry fare multiple is saved', `${back?.ferryFare}`);

  // ---- two inland ports on waters that do not meet
  await fit(RESERVOIR, LAKE, 250);
  await page.keyboard.press('o');
  await page.waitForTimeout(300);
  check((await ferry()).active, 'the O key starts the ferry tool');
  await place(RESERVOIR, 1);
  await place(LAKE, 2);
  await waitRoute();
  const lm = await at([(RESERVOIR[0] + LAKE[0]) / 2, (RESERVOIR[1] + LAKE[1]) / 2]);
  await page.mouse.move(lm.x + 30, lm.y, { steps: 3 });
  await page.waitForTimeout(1200);
  f = await ferry();
  check(f.ports === 2 && f.route?.refused === 1 && /No water path found/.test(f.tag), 'two inland ports: the tag says No water path found', f.tag);
  await page.screenshot({ path: join(OUT, 'no-water-path-light.png'), clip: box });
  const n0 = await page.evaluate(() => window.tw.world.value.save.lines.length);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(1500);
  const refused = await page.evaluate(() => ({ lines: window.tw.world.value.save.lines.length, notice: document.querySelector('#tool-notes')?.textContent ?? '' }));
  check(refused.lines === n0 && /No water path found/.test(refused.notice), 'two inland ports: Make line is refused', refused.notice.slice(0, 120));
  await page.keyboard.press('Escape');
  await page.keyboard.press('Escape');

  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  check(false, 'ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-2194 FAILED: ${fail}` : 'T-2194 PASSED');
process.exit(fail ? 1 : 0);
