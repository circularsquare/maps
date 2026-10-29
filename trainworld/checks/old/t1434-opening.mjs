// T-1434: opening day. A planned line is constructed; its Lines tab riders count up after
// "constructing..." and nothing plays but the construct's snap; its stations' names carry no
// bunting (T-1759). From the maps root, with a running dev server:
//   node trainworld/checks/t1434-opening.mjs <port>
import { readFile } from 'node:fs/promises';
import { GL, NOISE, REAL_SAVE, collectErrors, demandFinal, importSave, launchChrome } from '../lib.mjs';

const BASE = `http://localhost:${process.argv[2] ?? 6171}/`;
const { ctx, close } = await launchChrome({ prefix: 'tw-t1434-', size: [1400, 900], args: [...GL, '--autoplay-policy=no-user-gesture-required'] });
const errors = [];
let fails = 0;
const check = (name, ok, detail = '') => { if (!ok) fails++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); };
try {
  const page = await ctx.newPage();
  collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
  await page.addInitScript(() => {
    try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {}
    window.__snd = { buf: 0, osc: 0 };
    const P = BaseAudioContext.prototype;
    const cb = P.createBufferSource, co = P.createOscillator;
    P.createBufferSource = function () { window.__snd.buf++; return cb.call(this); };
    P.createOscillator = function () { window.__snd.osc++; return co.call(this); };
  });
  await page.goto(`${BASE}?debug=1&new=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.mouse.click(700, 450);
  await page.waitForTimeout(2500);
  await page.evaluate(async () => { const u = await import('/src/game/ui.ts'); u.tab.value = 'lines'; });

  // a planned line: track and stations as blueprint, the line over them
  const made = await page.evaluate(async () => {
    const tw = window.tw, c = tw.client;
    const km = 2, lat = 40.745, lng = -73.99, dLng = km / (111.32 * Math.cos((lat * Math.PI) / 180));
    const xy = [-1, 0, 1].map((k) => { const p = tw.lngLatToPack(lng + k * dLng, lat); return [p.x, p.y]; });
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const NAMES = ['Irving Place', 'Madison Square', 'Hudson Yards'];
    await c.edit({ op: 'route', from: { kind: 'free', x: xy[0][0], y: xy[0][1], level: -1 }, to: { kind: 'free', x: xy[1][0], y: xy[1][1], level: -1 }, pis: [], single: false });
    await pause(300);
    await c.edit({ op: 'route', from: { kind: 'node', node: near(xy[1]) }, to: { kind: 'free', x: xy[2][0], y: xy[2][1], level: -1 }, pis: [], single: false });
    await pause(500);
    for (const p of xy) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: NAMES[xy.indexOf(p)] }); await pause(200); }
    const r = await c.edit({ op: 'addLine', stops: xy.map(near), name: 'Opening', colour: '#d9480f' });
    await c.edit({ op: 'setTrainCount', line: r.line, trains: [6, 4, 3] });
    await pause(800);
    const w = tw.world.value;
    const l = w.save.lines.at(-1);
    return { ok: r?.ok, id: l?.id, status: w.lineStats[l?.id]?.status, lng, lat };
  });
  check('a line planned over blueprint', made.ok && made.status === 'planned', JSON.stringify(made));
  const opens = () => page.evaluate(async () => { const m = await import('/src/game/openingDay.ts'); return [...m.openings.value.keys()]; });
  check('nothing on its opening day yet', (await opens()).length === 0);

  // the camera on the line, zoomed to 13; sample the riders cell as it opens
  await page.evaluate(({ lng, lat }) => { window.tw.map.jumpTo({ center: [lng, lat], zoom: 13 }); }, made);
  await page.waitForTimeout(1200);
  const snd0 = await page.evaluate(() => ({ ...window.__snd }));
  await page.evaluate(async () => { const w = await import('/src/game/world.ts'); await w.edit({ op: 'constructAll' }); });
  const series = [];
  const t0 = Date.now();
  while (Date.now() - t0 < 14000) {
    const s = await page.evaluate((id) => {
      const rows = [...document.querySelectorAll('#dock .rows li')];
      const li = rows.find((e) => e.querySelector('.line-name')?.textContent === 'Opening');
      return { t: performance.now(), riders: li?.querySelector('.w-riders')?.textContent ?? null };
    }, made.id);
    series.push([Math.round(Date.now() - t0), s.riders]);
    await page.waitForTimeout(100);
  }
  const compact = [];
  for (const [t, r] of series) if (!compact.length || compact.at(-1)[1] !== r) compact.push([t, r]);
  console.log('riders cell over time (ms, text):', JSON.stringify(compact));
  const ids = await opens();
  check('the line is on its opening day', ids.includes(String(made.id)), JSON.stringify(ids));
  const nums = compact.map(([, r]) => Number((r ?? '').replace(/,/g, ''))).filter((n) => Number.isFinite(n));
  check('the riders cell starts at 0 and rises to a figure', nums.length >= 3 && nums[0] < nums.at(-1) / 4 && nums.every((n, i) => i === 0 || n >= nums[i - 1]), JSON.stringify(nums));
  const snd1 = await page.evaluate(() => ({ ...window.__snd }));
  check('the opening itself makes no sound: a snap only', snd1.buf - snd0.buf === 1 && snd1.osc - snd0.osc === 0, JSON.stringify({ buf: snd1.buf - snd0.buf, osc: snd1.osc - snd0.osc }));

  // T-1759: no pennants under the names
  const names = await page.locator('.stn-label').count();
  check('three station names and no bunting on any', names >= 3 && (await page.locator('.stn-label.opening').count()) === 0 && await page.evaluate(() => [...document.querySelectorAll('.stn-label')].every((e) => getComputedStyle(e, '::before').content === 'none' && !e.style.getPropertyValue('--rib'))), `${names} labels`);

  // one game day later it is gone
  await page.evaluate(() => { const c = window.tw.clock; c.set(c.now() + 86400 + 300); });
  await page.waitForTimeout(1500);
  check('after a game day the opening is over', (await opens()).length === 0);
  await page.waitForTimeout(500);
  const plain = await page.evaluate(() => [...document.querySelectorAll('#dock .rows li')].find((e) => e.querySelector('.line-name')?.textContent === 'Opening')?.querySelector('.w-riders')?.textContent);
  console.log('riders cell after the day:', plain);
  // a loaded game shows no opening days: the real save's lines were running before the page was
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 20, undefined, { timeout: 120000 });
  await demandFinal(page, { polling: 250, lines: 100 });
  await page.waitForTimeout(1500);
  check('the real save loads with no opening days', (await opens()).length === 0);
  check('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
process.exit(fails ? 1 : 0);
