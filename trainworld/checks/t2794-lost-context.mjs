// T-2794: the notices a phone can hit. Run 1: a lost WebGL context that never comes back shows the graphics notice
// with its Reload button and pauses the game; a context lost and restored within the wait shows nothing. Run 2:
// a load that fails out of memory says so (`tw.failDemand`, ?debug=1). Emulated phone, against your own Vite.
//   TW_CORES=3 node trainworld/checks/t2794-lost-context.mjs http://localhost:6133/   (about 2 minutes)
import { join } from 'node:path';
import { ARTIFACTS, GL, launchChrome, collectErrors, playwright } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6133/';
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const { devices } = await playwright();
const phone = { ...devices['Pixel 7'], deviceScaleFactor: 3 };
delete phone.defaultBrowserType;
const size = [phone.viewport.width, phone.viewport.height];
delete phone.viewport;
const open = async (chrome) => {
  const page = await chrome.ctx.newPage();
  await page.goto(`${BASE}?debug=1&demandWorkers=1&new=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.map, undefined, { timeout: 180000 });
  await page.waitForFunction(() => window.twDemand?.view?.()?.final, undefined, { timeout: 180000 }).catch(() => {});
  return page;
};
const lose = (page) => page.evaluate(() => { const e = window.tw.map.getCanvas().getContext('webgl2').getExtension('WEBGL_lose_context'); e.loseContext(); window.__lc = e; });
const note = (page) => page.$eval('.demand-stopped-note', (n) => n.textContent).catch(() => null);

{
  const chrome = await launchChrome({ prefix: 'tw-t2794c-', args: GL, size, ...phone });
  try {
    const page = await open(chrome);
    await lose(page);
    await page.waitForTimeout(1000);
    await page.evaluate(() => window.__lc.restoreContext());
    await page.waitForTimeout(7500);
    check(!(await note(page)), 'a context restored within the wait shows no notice');
    await lose(page);
    await page.waitForSelector('.demand-stopped-note', { timeout: 12000 }).catch(() => {});
    const text = await note(page);
    console.log('notice:', text);
    check(/stopped drawing/.test(text ?? ''), 'a context that stays lost shows the graphics notice');
    check(!!(await page.$('.demand-stopped-note .btn')), 'with a Reload button');
    check((await page.evaluate(() => window.tw.clock.running.value)) === false, 'the game is paused');
    await page.screenshot({ path: join(ARTIFACTS, 't2794', 'lost-context.png') });
  } finally { await chrome.close(); }
}
{
  const chrome = await launchChrome({ prefix: 'tw-t2794d-', args: GL, size, ...phone });
  try {
    const errors = [];
    const page = await open(chrome);
    collectErrors(page, errors, null);
    await page.evaluate(() => window.tw.failDemand('RangeError: WebAssembly.Memory.grow(): Maximum memory size exceeded'));
    await page.waitForFunction(() => /ran out of memory/i.test(document.body.innerText), undefined, { timeout: 15000 }).catch(() => {});
    const text = await page.evaluate(() => document.body.innerText.split('\n').filter((l) => /memory/i.test(l)).join(' | '));
    console.log('memory text on the page:', text);
    check(/ran out of memory/i.test(text), `a memory failure says it ran out of memory (${text})`);
  } finally { await chrome.close(); }
}
process.exitCode = fail ? 1 : 0;
