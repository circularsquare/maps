// Holds one headless Chrome (real GPU) with the all-Japan save loaded; measure.mjs connects over CDP 9338.
import { readFile, writeFile } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, GL } from 'file:///C:/Users/anita/projects/maps/trainworld/checks/lib.mjs';

const OUT = 'C:/Users/anita/AppData/Local/Temp/claude/c--Users-anita-projects-maps/f28006bc-4241-4116-9970-a65fc772bb95/scratchpad/t1618/';
const log = (s) => console.log(`${new Date().toISOString().slice(11, 19)} ${s}`);
const { ctx } = await launchChrome({ prefix: 'tw-t1618-', size: [1920, 1080], args: [...GL, '--remote-debugging-port=9338'] });
await writeFile(OUT + 'browser.pid', String(ctx.browser()?.process?.()?.pid ?? ''));
const page = ctx.pages()[0] ?? await ctx.newPage();
page.on('pageerror', (e) => log('pageerror ' + e.message));
await page.goto('http://localhost:6158/?debug=1&new=1&paused=1&demandWorkers=2');
await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
log('new game up');
await importSave(page, await readFile('C:/Users/anita/projects/maps/artifacts/trainworld/t737-japan-trains.save'));
await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 600000 });
log('save in: ' + JSON.stringify(await page.evaluate(() => ({ lines: window.tw.world.value.save.lines.length, stations: window.tw.world.value.save.stations.length }))));
const t = Date.now();
await demandFinal(page, { timeout: 3600000, polling: 2000, lines: 100 });
log(`demand final after ${((Date.now() - t) / 1000).toFixed(0)} s`);
await writeFile(OUT + 'ready', 'yes');
await new Promise(() => {});
