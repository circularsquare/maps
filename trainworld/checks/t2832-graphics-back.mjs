// T-2832/T-2833: a lost WebGL context that comes back clears the graphics notice and restarts the clock (only if the
// notice paused it); the 6 s count runs only while the page is visible. T-2833: a 4 GB desktop keeps the market and
// shows no note; a phone gets neither. Your own Vite:
//   TW_CORES=2 node trainworld/checks/t2832-graphics-back.mjs http://localhost:6177/   (about 3 minutes)
import { GL, launchChrome, playwright } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6177/';
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const { devices } = await playwright();
const phone = { ...devices['Pixel 7'], deviceScaleFactor: 3 };
delete phone.defaultBrowserType;
const phoneSize = [phone.viewport.width, phone.viewport.height];
delete phone.viewport;

const open = async (chrome, init) => {
  const page = await chrome.ctx.newPage();
  if (init) await page.addInitScript(init);
  await page.goto(`${BASE}?debug=1&demandWorkers=1&new=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.map, undefined, { timeout: 180000 });
  await page.waitForFunction(() => window.twDemand?.view?.()?.final, undefined, { timeout: 180000 }).catch(() => {});
  await page.waitForTimeout(1500);
  return page;
};
const lose = (page) => page.evaluate(() => { const e = window.tw.map.getCanvas().getContext('webgl2').getExtension('WEBGL_lose_context'); e.loseContext(); window.__lc = e; });
const restore = (page) => page.evaluate(() => window.__lc.restoreContext());
const noticeUp = async (page) => !!(await page.$('.demand-stopped-note'));
const running = (page) => page.evaluate(() => window.tw.clock.running.value);
const setRunning = (page, on) => page.evaluate((v) => window.tw.clock.setRunning(v), on);
const visibility = (page, state) => page.evaluate((s) => {
  Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => s });
  document.dispatchEvent(new Event('visibilitychange'));
}, state);

// T-2832, run 1: desktop
{
  const chrome = await launchChrome({ prefix: 'tw-t2832a-', args: GL, size: [1400, 900] });
  try {
    const page = await open(chrome);
    await setRunning(page, true);
    await lose(page);
    await page.waitForSelector('.demand-stopped-note', { timeout: 12000 }).catch(() => {});
    check(await noticeUp(page), 'a context lost over 6 s shows the notice');
    check((await running(page)) === false, 'and the clock is paused');
    await restore(page);
    await page.waitForTimeout(1500);
    check(!(await noticeUp(page)), 'restored: the notice clears');
    check((await running(page)) === true, 'restored: the clock runs again');

    // a clock the player paused stays paused
    await setRunning(page, false);
    await lose(page);
    await page.waitForSelector('.demand-stopped-note', { timeout: 12000 }).catch(() => {});
    check(await noticeUp(page), 'player-paused: the notice shows');
    await restore(page);
    await page.waitForTimeout(1500);
    check(!(await noticeUp(page)), 'player-paused: the notice clears on restore');
    check((await running(page)) === false, 'player-paused: the clock stays paused');
    await setRunning(page, true);

    // hidden: no count; visible again: a fresh 6 s
    await visibility(page, 'hidden');
    await lose(page);
    await page.waitForTimeout(8000);
    check(!(await noticeUp(page)), 'hidden for 8 s: no notice');
    check((await running(page)) === true, 'hidden for 8 s: the clock is not paused by it');
    await visibility(page, 'visible');
    await page.waitForTimeout(3000);
    check(!(await noticeUp(page)), 'visible 3 s after: still no notice (a fresh 6 s)');
    await page.waitForSelector('.demand-stopped-note', { timeout: 8000 }).catch(() => {});
    check(await noticeUp(page), 'visible and still lost: the notice shows after 6 s');
    await restore(page);
    await page.waitForTimeout(1500);
    check(!(await noticeUp(page)), 'restored again: the notice clears');
    // hide mid-count cancels it
    await lose(page);
    await page.waitForTimeout(2000);
    await visibility(page, 'hidden');
    await page.waitForTimeout(6000);
    check(!(await noticeUp(page)), 'hidden mid-count: the count is cancelled');
    await restore(page);
    await visibility(page, 'visible');
  } finally { await chrome.close(); }
}

// T-2833: market and note by device
// the market's rule: with a city selected, the zones whose market packs the client wants (empty = no market)
const marketOpen = (page) => page.evaluate(() => {
  const c = window.twDemand?.client;
  const ac = window.twDemand.activeCity;
  const city = typeof ac === 'function' ? ac() : ac?.value ?? ac;
  window.tw.select({ kind: 'city', city });
  const n = c.marketZones(null).size;
  window.tw.select(null);
  return n > 0;
});
const note = async (page) => !!(await page.$('.no-market-note'));
const noteText = (page) => page.$eval('.no-market-note span', (n) => n.textContent).catch(() => null);
const dmN = (n) => `Object.defineProperty(Navigator.prototype, 'deviceMemory', { configurable: true, get: () => ${n} });`;
const dm4 = () => Object.defineProperty(Navigator.prototype, 'deviceMemory', { configurable: true, get: () => 4 });
{
  const chrome = await launchChrome({ prefix: 'tw-t2833a-', args: GL, size: [1400, 900] });
  try {
    const page = await open(chrome, dm4);
    check((await page.evaluate(() => navigator.deviceMemory)) === 4, 'desktop: deviceMemory reads 4');
    check(!(await page.evaluate(() => /Android|iPhone|Mobi/.test(navigator.userAgent))), 'desktop: a desktop user agent');
    await page.waitForTimeout(3000);
    const text = await noteText(page);
    check(/4GB of RAM or less/.test(text ?? '') && !/Intercity/.test(text ?? ''), `4 GB desktop: only the memory sentence (${text})`);
    check(await marketOpen(page), '4 GB desktop: the market is open');
  } finally { await chrome.close(); }
}
{
  const chrome = await launchChrome({ prefix: 'tw-t2839a-', args: GL, size: [1400, 900] });
  try {
    const page = await open(chrome, dmN(8));
    await page.waitForTimeout(3000);
    check(!(await note(page)), '8 GB desktop: no note');
  } finally { await chrome.close(); }
}
{
  // T-2838: phone reporting 4: combined note, no market; toggle on (the Settings tab), reload: market, memory sentence only
  const chrome = await launchChrome({ prefix: 'tw-t2833b-', args: GL, size: phoneSize, ...phone });
  try {
    const init = () => {
      Object.defineProperty(Navigator.prototype, 'deviceMemory', { configurable: true, get: () => 4 });
      if (!localStorage.getItem('trainworld-ui')) localStorage.setItem('trainworld-ui', JSON.stringify({ tab: 'settings', v: 12 }));
    };
    const page = await open(chrome, init);
    await page.waitForTimeout(3000);
    const text = await noteText(page);
    check(/4GB of RAM or less\. .*Intercity trips are off on mobile/.test(text ?? ''), `phone reporting 4: one note, memory first (${text})`);
    check(!(await marketOpen(page)), 'phone: no market');
    const label = page.getByText('Calculate intercity trips');
    check(await label.count() === 1, 'phone: the toggle is in Settings');
    await label.click();
    await page.waitForTimeout(500);
    check(!/Intercity/.test((await noteText(page)) ?? 'x'), 'toggle on: the note drops the intercity sentence at once');
    await page.close();
    const page2 = await open(chrome, init);
    check(await page2.evaluate(() => JSON.parse(localStorage.getItem('trainworld-ui')).display?.mobileIntercity === true), 'the setting survives a reload');
    await page2.waitForTimeout(3000);
    const t2 = await noteText(page2);
    check(/4GB of RAM or less/.test(t2 ?? '') && !/Intercity/.test(t2 ?? ''), `toggle on after reload: memory sentence only (${t2})`);
    check(await marketOpen(page2), 'toggle on: the market opens');
    await page2.getByText('Calculate intercity trips').click();
    await page2.waitForTimeout(500);
    check(/Intercity trips are off/.test((await noteText(page2)) ?? ''), 'toggle off: the sentence is back');
  } finally { await chrome.close(); }
}
{
  // a phone that does not report its memory: the intercity sentence alone
  const chrome = await launchChrome({ prefix: 'tw-t2833c-', args: GL, size: phoneSize, ...phone });
  try {
    const page = await open(chrome, () => Object.defineProperty(Navigator.prototype, 'deviceMemory', { configurable: true, get: () => undefined }));
    await page.waitForTimeout(3000);
    const text = await noteText(page);
    check(/^Intercity trips are off on mobile/.test(text ?? '') && !/RAM/.test(text ?? ''), `phone not reporting: intercity sentence only (${text})`);
    check(!(await marketOpen(page)), 'phone not reporting: no market');
  } finally { await chrome.close(); }
}
process.exitCode = fail ? 1 : 0;
