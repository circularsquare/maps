// T-1507: load a save on BASE in headless Chrome (no demand wait unless WAIT=1), then run each
// <name>.js dropped in CMD (an async function body run in the page; its answer to <name>.out.json).
// A file named `export` in CMD saves the game to OUTSAVE; a file `stop` ends it. Made the high-speed
// save: start it on t1156-all-china.save, drop edit.js in CMD, then `export`.
//   node trainworld/notes/T-1507/probe.mjs <base> <save> <cmd dir> <out save>   (from the maps root)
import { readFile, readdir, writeFile, rm, mkdir, copyFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { NOISE, launchChrome, collectErrors, importSave, demandFinal } from '../../checks/lib.mjs';

const [BASE, SAVE, CMD, OUTSAVE] = process.argv.slice(2);
await mkdir(CMD, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t1507-' });
const errors = [];
try {
  const page = await ctx.newPage();
  collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const t0 = Date.now();
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  console.log(`loaded ${Math.round((Date.now() - t0) / 1000)} s`);
  if (process.env.WAIT) {
    await demandFinal(page, { polling: 1000, timeout: 1800000 });
    console.log(`demand final ${Math.round((Date.now() - t0) / 1000)} s`);
  }
  console.log('ready');
  for (;;) {
    if (existsSync(join(CMD, 'stop'))) { await rm(join(CMD, 'stop')); break; }
    if (existsSync(join(CMD, 'export'))) {
      await rm(join(CMD, 'export'));
      const dl = page.waitForEvent('download', { timeout: 120000 });
      await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
      await copyFile(await (await dl).path(), OUTSAVE);
      console.log(`exported to ${OUTSAVE}`);
    }
    for (const f of (await readdir(CMD)).filter((f) => f.endsWith('.js'))) {
      const body = await readFile(join(CMD, f), 'utf8');
      await rm(join(CMD, f));
      const a = Date.now();
      let res;
      try { res = await page.evaluate(`(async () => { ${body} })()`); } catch (e) { res = { error: e.message.split('\n').slice(0, 3).join(' ') }; }
      await writeFile(join(CMD, f.replace(/\.js$/, '.out.json')), JSON.stringify(res, null, 1));
      console.log(`${f} done in ${Math.round((Date.now() - a) / 1000)} s; ${errors.length} page errors`);
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
} catch (e) {
  console.log('ERROR', e.message.split('\n')[0]);
} finally {
  console.log('errors', JSON.stringify(errors.slice(0, 20)));
  await close();
}
