// T-3582: a browser that has WebGL2 but refuses the page a context (Chrome blocks a page after its context was lost)
// gets the restart screen, at startup and mid-game (where the autosave is written); one with no WebGL2 keeps the
// T-3307 screen. The refusal is faked: getContext("webgl2") returns null on a new canvas once `__refuse` is set.
// Your own Vite:
//   TW_CORES=3 node trainworld/checks/t3582-graphics-gone.mjs http://localhost:6165/   (about 2 minutes)
import { join } from 'node:path';
import { mkdir } from 'node:fs/promises';
import { ARTIFACTS, GL, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6165/';
const OUT = join(ARTIFACTS, 't3582');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

/** `refuse`: refuse from the start, with the browser's `reason`; `missing`: no WebGL2 at all */
const init = ({ refuse = false, missing = false, reason = 'Web page caused context loss and was blocked' }) => {
  if (missing) { delete window.WebGL2RenderingContext; return; }
  window.__refuse = refuse;
  const orig = HTMLCanvasElement.prototype.getContext;
  const had = new WeakSet();
  HTMLCanvasElement.prototype.getContext = function (type, ...rest) {
    if (type === 'webgl2' && window.__refuse && !had.has(this)) {
      this.dispatchEvent(new WebGLContextEvent('webglcontextcreationerror', { statusMessage: reason }));
      return null;
    }
    const c = orig.call(this, type, ...rest);
    if (c) had.add(this);
    return c;
  };
};
const screenText = (page) => page.$eval('.loading-screen .big', (n) => n.textContent).catch(() => null);

async function startup(name, opts, want) {
  const chrome = await launchChrome({ prefix: `tw-t3582${name}-`, args: GL });
  try {
    const page = await chrome.ctx.newPage();
    await page.addInitScript(init, opts);
    await page.goto(`${BASE}?new=1&v=${Date.now()}`);
    await page.waitForSelector('.loading-screen .big', { timeout: 60000 }).catch(() => {});
    const text = await screenText(page);
    check(want.test(text ?? ''), `startup ${name}: ${text}`);
    await page.screenshot({ path: join(OUT, `startup-${name}.png`) });
  } finally { await chrome.close(); }
}

await startup('refused', { refuse: true }, /stopped working.*close every browser window/i);
await startup('refused-no-reason', { refuse: true, reason: '' }, /stopped working.*close every browser window/i);
await startup('missing', { missing: true }, /no WebGL2/);

// mid-game: the map's context is lost, and a new one is refused: after 6 s the restart screen covers the game,
// which is paused; the context coming back takes the screen away and runs the clock again
{
  const chrome = await launchChrome({ prefix: 'tw-t3582m-', args: GL });
  try {
    const page = await chrome.ctx.newPage();
    await page.addInitScript(init, {});
    await page.goto(`${BASE}?debug=1&demandWorkers=1&new=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.tw?.map, undefined, { timeout: 180000 });
    await page.waitForFunction(() => window.twDemand?.view?.()?.final, undefined, { timeout: 180000 }).catch(() => {});
    await page.waitForTimeout(1500);
    await page.evaluate(() => { const c = window.tw.client, f = c.saveBeforeReload.bind(c); window.__saves = 0; c.saveBeforeReload = () => { window.__saves++; return f(); }; });
    await page.evaluate(() => window.tw.clock.setRunning(true));
    await page.evaluate(() => { const e = window.tw.map.getCanvas().getContext('webgl2').getExtension('WEBGL_lose_context'); e.loseContext(); window.__lc = e; window.__refuse = true; });
    await page.waitForTimeout(3000);
    check(!(await screenText(page)), 'nothing yet 3 s after the loss');
    await page.waitForSelector('.loading-screen .big', { timeout: 8000 }).catch(() => {});
    const text = await screenText(page);
    check(/stopped working.*close every browser window/i.test(text ?? ''), `lost and refused: the restart screen (${text})`);
    check(!(await page.$('.demand-stopped-note')), 'and not the reload notice');
    check((await page.evaluate(() => window.tw.clock.running.value)) === false, 'the game is paused');
    check((await page.evaluate(() => window.__saves)) === 1, 'and its autosave asked for');
    await page.screenshot({ path: join(OUT, 'midgame-refused.png') });
    await page.evaluate(() => { window.__refuse = false; window.__lc.restoreContext(); });
    await page.waitForTimeout(1500);
    check(!(await screenText(page)), 'restored: the screen goes');
    check((await page.evaluate(() => window.tw.clock.running.value)) === true, 'restored: the clock runs again');
  } finally { await chrome.close(); }
}
process.exitCode = fail ? 1 : 0;
