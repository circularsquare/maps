// T-2657: demand stopped for good. On a running game, `window.tw.failDemand()` (?debug=1) stops the pool as
// a dead worker would; then: the clock is paused, the notice sits at the top centre with a Reload page
// button and the error line, the old tag under the demand buttons is gone, the player can run the clock
// again, and Reload page reloads. Screenshots (light and dark) to artifacts/trainworld/stopnote/.
//   node trainworld/checks/t2657-demand-stopped.mjs [app url]   (from the maps root; default localhost:6167)
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { launchChrome, ANGLE, ARTIFACTS, NOISE } from './lib.mjs';

const APP = process.argv[2] ?? 'http://localhost:6167/';
const OUT = join(ARTIFACTS, 'stopnote');
mkdirSync(OUT, { recursive: true });
let failed = false;
const report = (name, ok, detail = '') => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

const d = await launchChrome({ prefix: 'tw-t2657-', size: [1300, 800], args: ANGLE });
const errors = [];
const p = await d.ctx.newPage();
p.on('pageerror', (e) => errors.push(e.message));
p.on('console', (m) => { if (m.type() === 'error' && !NOISE.test(m.text()) && !/demand:/.test(m.text())) errors.push(m.text()); });
try {
  await p.goto(`${APP}?debug=1&new=1&demandWorkers=1&v=${Date.now()}`, { waitUntil: 'commit', timeout: 240000 });
  await p.waitForFunction(() => window.tw?.world?.value && window.tw?.clock, undefined, { timeout: 180000 });
  await p.addStyleTag({ content: '#debug-root { display: none !important; }' });
  // after startup only: a failure while the loading card is up has its own screen
  await p.evaluate(async () => { window.__startup = (await import('/src/game/loading.ts')).startup; });
  await p.waitForFunction(() => window.__startup.value === 'ready', undefined, { timeout: 420000, polling: 2000 });
  await p.evaluate(() => window.tw.clock.setRunning(true));
  report('clock runs before', await p.evaluate(() => window.tw.clock.running.value));
  report('no notice before', await p.locator('.demand-stopped-note').count() === 0);

  await p.evaluate(() => window.tw.failDemand('worker 1: its replacement could not reopen its packs: DataCloneError: forced by the check'));
  await p.locator('.demand-stopped-note').waitFor({ timeout: 10000 });
  report('the clock is paused', await p.evaluate(() => !window.tw.clock.running.value));
  const box = await p.locator('.demand-stopped-note').boundingBox();
  const mid = box.x + box.width / 2;
  report('top centre', Math.abs(mid - 650) < 330 && box.y < 120, JSON.stringify(box));
  report('old tag gone', await p.locator('.demand-stopped').count() === 0);
  console.log('   text:', JSON.stringify(await p.locator('.demand-stopped-note').innerText()));
  await p.screenshot({ path: join(OUT, 'notice-light.png') });
  await p.evaluate(() => { document.documentElement.dataset.shade = 'dark'; });
  await p.waitForTimeout(300);
  await p.screenshot({ path: join(OUT, 'notice-dark.png') });
  await p.evaluate(() => { document.documentElement.dataset.shade = 'light'; });

  await p.evaluate(() => window.tw.clock.setRunning(true));
  await p.waitForTimeout(500);
  report('the player can run it again', await p.evaluate(() => window.tw.clock.running.value));

  p.on('dialog', (dlg) => dlg.accept().catch(() => {})); // a test page holds the unload (autosave is off)
  const nav = p.waitForNavigation({ timeout: 20000 }).then(() => true, () => false);
  await p.locator('.demand-stopped-note .btn').click();
  report('Reload page reloads', await nav);
  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} catch (e) {
  report('check ran', false, String(e?.stack ?? e));
  await p.screenshot({ path: join(OUT, 'failure.png') }).catch(() => {});
} finally {
  await d.close?.().catch(() => {});
}
process.exit(failed ? 1 : 0);
