// T-728, T-731: the new game sheet, sandbox money and real-network starts, through the game's own
// buttons in a fresh profile. The start menu's sheet: the search ("hague", "sao", nonsense), then a
// sandbox in Tokyo from its real network (slot "Tokyo sandbox", lines running, cash hidden, a $10B+
// train order allowed). The saves panel's sheet: a normal game in London (slot "London", no real
// network box, cash shown) and a map start (slot "World", zoomed out). The real New York save loaded
// from a file refuses the same order, Settings makes it a sandbox after the confirmation, and the
// order then goes through; the switch has no way back. Every slot left keeps its bytes, and a reload's
// start menu marks the sandboxes. Screenshots (light and dark) go to artifacts/trainworld/t728-*.png.
//   node trainworld/checks/t728-newgame.mjs [url]     (from the maps root; default localhost:6150)
// Prints PASS or FAIL lines and exits 1 on a failure. A few minutes.
import { mkdtemp, rm, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const BASE = process.argv[2] ?? 'http://localhost:6150/';
const REAL = fileURLToPath(new URL('../T-007-real-nyc.save', import.meta.url));
const ART = fileURLToPath(new URL('../../artifacts/trainworld/', import.meta.url));
const shot = (name) => join(ART, `t728-${name}.png`);

async function playwright() {
  if (process.env.PLAYWRIGHT_MODULE) return import(process.env.PLAYWRIGHT_MODULE);
  try { return await import('playwright'); } catch { /* not installed here */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright');
}

let failed = false;
const report = (name, ok, detail) => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

const { chromium } = await playwright();
const profile = await mkdtemp(join(tmpdir(), 'tw-t728-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1300, height: 900 }, args: ['--use-angle=d3d11'] });
try {
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  const wait = (ms) => page.waitForTimeout(ms);
  const state = () => page.evaluate(async () => {
    const s = (await import('/src/workers/clockClient.ts')).saveSlotsState.value;
    return { active: s.active, slots: s.slots.map((x) => ({ key: x.key, name: x.name, mode: x.mode })) };
  });
  const stored = (key) => page.evaluate(async (key) => {
    const S = await import('/src/workers/saveFile.ts');
    const r = await S.readStored(key);
    if (!r) return null;
    const h = [...new Uint8Array(await crypto.subtle.digest('SHA-256', r.bytes))].map((b) => b.toString(16).padStart(2, '0')).join('');
    const { header } = await S.decodeSave(r.bytes);
    return { hash: h.slice(0, 16), name: r.name, mode: r.mode, hmode: header.mode, start: header.start, opened: header.opened, lines: r.lines };
  }, key);
  const game = () => page.evaluate(() => {
    const c = window.tw.map.getCenter();
    return { lng: c.lng, lat: c.lat, zoom: window.tw.map.getZoom(), lines: window.tw.world.value?.save.lines.length ?? -1, mode: window.tw.moneyState.value?.mode, cash: window.tw.moneyState.value?.cash, bar: document.querySelector('#topbar .cash')?.textContent ?? '', time: document.querySelector('#topbar .clock')?.textContent ?? '' };
  });
  const switched = async (from) => {
    await page.evaluate(async () => { window.__cc = await import('/src/workers/clockClient.ts'); });
    await page.waitForFunction((from) => window.__cc.saveSlotsState.value.active !== from, from, { timeout: 120000, polling: 200 });
    return (await state()).active;
  };
  const flush = async () => { await page.evaluate(() => window.tw.client.post({ kind: 'flush' })); await wait(1500); };
  const panel = async () => {
    await page.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = true; });
    await page.waitForSelector('.saves-panel .save-files');
    await wait(200);
  };
  /** a train order far beyond $3B on the first line: [ok, issues, cash after] */
  const bigOrder = () => page.evaluate(async () => {
    const l = window.tw.world.value.save.lines[0];
    const a = await window.tw.client.edit({ op: 'setTrainCount', line: l.num, trains: [600, 600, 600] });
    await new Promise((r) => setTimeout(r, 300));
    return { ok: a.ok, issues: a.issues.join(' '), charge: a.charge, cash: window.tw.moneyState.value.cash };
  });
  const names = () => page.$$eval('.new-game-hits li', (li) => li.map((x) => x.textContent));
  const find = async (text) => { await page.fill('.new-game-find', text); await wait(150); return names(); };
  const near = (g, c) => Math.abs(g.lng - c[0]) < 0.5 && Math.abs(g.lat - c[1]) < 0.5;

  // 1. the start menu's sheet, in a profile with nothing saved
  await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
  await page.waitForSelector('.start-menu .start-new', { timeout: 120000 });
  await page.click('.start-menu .start-new');
  await page.waitForSelector('.start-menu .new-game');
  const cities = await page.$$eval('.new-game-cities button', (b) => b.map((x) => x.textContent));
  const hague = await find('hague'), sao = await find('sao'), none = await find('qqqzz'), aom = await find('AOMO');
  report('the five cities', cities.join(',') === 'New York,London,Paris,Tokyo,Toronto', `(${cities.join(', ')})`);
  report('search', hague[0]?.startsWith('Rotterdam [The Hague]') && hague[0].endsWith('Netherlands') && sao[0]?.startsWith('São Paulo') && none.length === 0 && aom[0]?.startsWith('Aomori'),
    `(hague: ${hague.join(' | ')}; sao: ${sao.join(' | ')}; nonsense: ${none.length} rows; AOMO: ${aom.join(' | ')})`);
  await find('osa');
  await page.screenshot({ path: shot('sheet-light') });
  await page.fill('.new-game-find', '');
  await page.click('.new-game-mode [data-mode="sandbox"]');
  const realNormal = await page.locator('.new-game-real').count();
  await page.click('.new-game-cities [data-city="tyo"]');
  const realBox = await page.locator('.new-game-real input').isChecked().catch(() => false);
  const label = await page.textContent('.new-game .start-new');
  report('real network box', realNormal === 0 && realBox && label === 'Start in Tokyo', `(before a city ${realNormal}, Tokyo sandbox ticked ${realBox}, button "${label}")`);
  await page.click('.new-game .start-new');
  await page.waitForFunction(() => window.tw?.world?.value?.save.lines.length > 0 && window.twDemand?.view?.()?.final && !window.tw.clock.held.peek(), undefined, { timeout: 300000, polling: 500 });
  await wait(1500);
  let s = await state();
  const T = s.active;
  let g = await game();
  report('sandbox in Tokyo from its real network', near(g, [139.77, 35.68]) && g.lines > 20 && g.mode === 'sandbox' && g.bar === 'Sandbox' && /0[78]:\d\d$/.test(g.time),
    `(centre ${g.lng.toFixed(2)}, ${g.lat.toFixed(2)} z${g.zoom.toFixed(1)}, ${g.lines} lines, mode ${g.mode}, top bar "${g.bar}", clock "${g.time}")`);
  await page.locator('#topbar').screenshot({ path: shot('topbar-light') });
  const order = await bigOrder();
  report('a sandbox refuses nothing for money', order.ok && order.charge > 3000 && order.cash < 0, `(ok ${order.ok}, paid $${Math.round(order.charge)}M, cash $${Math.round(order.cash)}M ${order.issues})`);
  await flush();
  const t = await stored(T);
  report('slot named after the start, header recorded', t?.name === 'Tokyo sandbox' && t.mode === 'sandbox' && t.hmode === 'sandbox' && t.start === 'tyo' && t.opened?.join() === 'tyo',
    `(${T} "${t?.name}", summary ${t?.mode}, header ${t?.hmode}, start ${t?.start}, opened ${JSON.stringify(t?.opened)})`);

  // 2. the saves panel's sheet: a normal game in London, found by search
  await panel();
  await page.click('.saves-panel .save-files button:has-text("New game")');
  await page.waitForSelector('.saves-panel .new-game');
  await page.fill('.saves-panel .new-game-find', 'lond');
  await wait(150);
  await page.keyboard.press('Enter');
  const lLabel = await page.textContent('.saves-panel .new-game .start-new');
  const lReal = await page.locator('.saves-panel .new-game-real').count();
  await page.click('.saves-panel .new-game .start-new');
  const L = await switched(T);
  await page.waitForFunction(() => window.tw.world.value.save.lines.length === 0 && window.tw.moneyState.value?.mode === 'normal', undefined, { timeout: 60000 });
  await wait(1500);
  g = await game();
  const tLeft = await stored(T); // written as it was left; untouched from here on
  report('normal game in London', lLabel === 'Start in London' && lReal === 0 && near(g, [-0.12, 51.5]) && g.bar.startsWith('$') && tLeft.mode === "sandbox",
    `(button "${lLabel}", real box ${lReal}, centre ${g.lng.toFixed(2)}, ${g.lat.toFixed(2)}, top bar "${g.bar}", Tokyo slot left at ${tLeft.hash})`);

  // 3. the real New York save from a file: normal, refuses the order; Settings makes it a sandbox once
  const [fc] = await Promise.all([page.waitForEvent('filechooser'), (async () => { await panel(); await page.click('.saves-panel .save-files button:has-text("Load a file")'); })()]);
  await fc.setFiles(REAL);
  const N = await switched(L);
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await wait(1500);
  const refused = await bigOrder();
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).tab.value = 'settings'; });
  await wait(300);
  const before = await page.textContent('.game-mode');
  await page.click('.game-mode button');
  await page.waitForSelector('dialog.confirmation');
  await page.screenshot({ path: shot('confirm-light') });
  await page.click('dialog.confirmation .btn.danger');
  await page.waitForFunction(() => window.tw.moneyState.value?.mode === 'sandbox', undefined, { timeout: 30000 });
  await wait(1500);
  const after = await page.textContent('.game-mode');
  const allowed = await bigOrder();
  await flush();
  const n = await stored(N);
  report('one-way switch to sandbox', !refused.ok && refused.issues.includes('Money:') && before.includes('Normal') && after === 'Sandbox' && allowed.ok && n?.mode === 'sandbox' && n.hmode === 'sandbox',
    `(normal: ${refused.issues}; settings "${before}" -> "${after}"; then ok ${allowed.ok}; slot ${n?.mode}/${n?.hmode})`);

  // 4. a map start: the sheet's defaults
  await panel();
  await page.click('.saves-panel .save-files button:has-text("New game")');
  const mLabel = await page.textContent('.saves-panel .new-game .start-new');
  await page.click('.saves-panel .new-game .start-new');
  const W = await switched(N);
  await page.waitForFunction(() => window.tw.world.value.save.lines.length === 0, undefined, { timeout: 60000 });
  await wait(2000);
  const nLeft = await stored(N);
  g = await game();
  // an empty game's slot is written once something is built: none yet
  s = await state();
  report('map start', mLabel === 'Start on the map' && g.zoom < 3 && g.mode === 'normal' && !s.slots.some((x) => x.key === W), `(button "${mLabel}", zoom ${g.zoom.toFixed(1)}, mode ${g.mode}, slot written ${s.slots.some((x) => x.key === W)})`);

  // 5. the slots left behind, and the start menu after a reload, dark
  const kept = (await stored(T)).hash === tLeft.hash && (await stored(N)).hash === nLeft.hash;
  await page.evaluate(() => { localStorage.setItem('trainworld-ui', JSON.stringify({ ...JSON.parse(localStorage.getItem('trainworld-ui') ?? '{}'), shade: 'dark' })); });
  await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
  await page.waitForSelector('.start-menu .start-save', { timeout: 120000 });
  await wait(800);
  const marks = await page.$$eval('.start-save', (b) => b.map((x) => `${x.querySelector('strong')?.textContent}`));
  await page.screenshot({ path: shot('menu-dark') });
  await page.click('.start-menu .start-new');
  await page.click('.new-game-mode [data-mode="sandbox"]');
  await page.click('.new-game-cities [data-city="par"]');
  await page.screenshot({ path: shot('sheet-dark') });
  report('slots kept, sandboxes marked', kept && marks.some((m) => m === 'Tokyo sandboxSandbox') && marks.some((m) => m === 'T-007-real-nycSandbox'),
    `(${marks.join(' | ')})`);

  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  console.log(`screenshots: ${['sheet-light', 'topbar-light', 'confirm-light', 'menu-dark', 'sheet-dark'].map(shot).join(', ')}`);
} finally {
  await ctx.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
process.exit(failed ? 1 : 0);
