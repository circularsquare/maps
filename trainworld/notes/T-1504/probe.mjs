// T-1504: the China network's longer trips, read from the game itself. Loads t1156-all-china.save
// on BASE in headless Chrome, waits for demand final, writes the lines and the weekday view's riders
// by purpose to artifacts/trainworld/T-1504/start.json, then stays open: every `<name>.js` dropped
// in CMD (an async function body run in the page, `return` its answer) is run once and its answer
// written to `<name>.out.json` beside it. A file `stop` ends it. TAG names start.json
// (start-<tag>.json), QUERY adds to the page's query string.
//   node trainworld/notes/T-1504/probe.mjs [base] [cmd dir]      (from the maps root; 6 min to load)
import { mkdir, readFile, readdir, writeFile, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, MAPS, NOISE, launchChrome, collectErrors, importSave, demandFinal } from '../../checks/lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6504/';
const CMD = process.argv[3] ?? join(ARTIFACTS, 'T-1504', 'cmd');
const OUT = join(ARTIFACTS, 'T-1504');
await mkdir(CMD, { recursive: true });
const SAVE = join(MAPS, 't1156-all-china.save');

const { ctx, close } = await launchChrome({ prefix: 'tw-t1504-' });
const errors = [];
try {
  const page = await ctx.newPage();
  collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  // QUERY: more of the page's query string (perfOff=chunks solves every zone pair, T-780)
  await page.goto(`${BASE}?debug=1&new=1&paused=1${process.env.QUERY ? `&${process.env.QUERY}` : ''}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const t0 = Date.now();
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.waitForTimeout(5000);
  await demandFinal(page, { polling: 1000, timeout: 1800000 });
  await page.waitForTimeout(2000);
  await demandFinal(page, { polling: 1000, timeout: 600000 });
  console.log(`demand final ${Math.round((Date.now() - t0) / 1000)} s`);

  const start = await page.evaluate(() => {
    const w = window.tw.world.value, toLL = window.tw.packToLngLat;
    const lines = w.save.lines.map((l) => {
      const small = {};
      for (const [k, v] of Object.entries(l)) if (typeof v !== 'object' || (Array.isArray(v) && v.length < 12)) small[k] = v;
      small.stops = (l.stops ?? []).map((s) => { const n = w.nodes.get(s) ?? w.nodes.get(Number(s)); return n ? toLL(n.x, n.y).map((x) => +x.toFixed(4)) : null; });
      return small;
    });
    const v = window.twDemand.days()[0].view;
    const plain = {};
    for (const [k, x] of Object.entries(v)) if (typeof x === 'number' || typeof x === 'string' || typeof x === 'boolean') plain[k] = x;
    const purposes = {};
    for (const [p, d] of v.purposes ?? []) {
      const per = {};
      let sum = 0;
      for (const [id, l] of d.lines) { per[id] = Math.round(l.ridersPerDay); sum += l.ridersPerDay; }
      purposes[p] = { sum: Math.round(sum), per };
    }
    return { lineKeys: Object.keys(w.save.lines[0]), lines, view: { plain, keys: Object.keys(v), purposes } };
  });
  const tag = process.env.TAG ? `-${process.env.TAG}` : '';
  await writeFile(join(OUT, `start${tag}.json`), JSON.stringify(start));
  console.log(`wrote start${tag}.json; ready for commands in ${CMD}`);

  for (;;) {
    if (existsSync(join(CMD, 'stop'))) { await rm(join(CMD, 'stop')); break; }
    for (const f of (await readdir(CMD)).filter((f) => f.endsWith('.js'))) {
      const body = await readFile(join(CMD, f), 'utf8');
      await rm(join(CMD, f));
      const a = Date.now();
      let res;
      try { res = await page.evaluate(`(async () => { ${body} })()`); } catch (e) { res = { error: e.message.split('\n').slice(0, 3).join(' ') }; }
      await writeFile(join(CMD, f.replace(/\.js$/, '.out.json')), JSON.stringify(res));
      console.log(`${f} done in ${Math.round((Date.now() - a) / 1000)} s; ${errors.length} page errors`);
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
} catch (e) {
  console.log('ERROR', e.message.split('\n')[0]);
} finally {
  await writeFile(join(OUT, 'errors.json'), JSON.stringify(errors.slice(0, 20)));
  await close();
}
