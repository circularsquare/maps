// settle2 (from bubfix/repro.mjs): Mettmann line, then setTrainCount [2,2,2]; the zone's All-purposes Homes sum each second
//   node flicker.mjs <baseUrl> <label>
import { writeFile, readFile } from 'node:fs/promises';
const L = 'file:///C:/Users/anita/projects/maps/.tw-trees/settle/trainworld/checks/lib.mjs';
const { launchChrome, collectErrors, demandFinal, buildLine, importSave, REAL_SAVE } = await import(L);
const [base, label, MODE = 'met', QS = ''] = process.argv.slice(2);
const SP = 'C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/f3eca8a5-6550-4b65-9245-def721281ba7/scratchpad/settle2/';
const { ctx, close } = await launchChrome({ profile: SP + 'prof-' + label, size: [1400, 900] });
const out = { label, samples: [] };
const errors = [];
try {
  const page = await ctx.newPage();
  collectErrors(page, errors);
  const nyc = MODE === 'nyc';
  await page.goto(`${base}?debug=1&new=1&paused=1&opened=all${nyc ? '' : '&city=met'}${QS}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 300000 });
  const C = nyc ? { c: [-73.985, 40.758], z: 9 } : { c: [6.97, 51.25], z: 11 };
  const pts = [[6.85, 51.23], [6.97, 51.25], [7.08, 51.26]];
  if (nyc) {
    await importSave(page, await readFile(REAL_SAVE));
    await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  }
  for (let k = 0; k < 4 && !nyc; k++) {
    await page.evaluate(([c]) => window.tw.map.jumpTo({ center: c, zoom: 11 }), [C.c]);
    await page.waitForTimeout(15000);
    const r = await buildLine(page, pts, { trains: [6, 4, 3] });
    console.log('build', JSON.stringify(r));
    if (r?.ok) break;
  }
  await page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z }), [C.c, C.z]);
  await demandFinal(page, { strict: true, timeout: 1500000 });
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, commuters: true, commuterEnd: 'home', trains: false }; });
  await page.waitForTimeout(6000);
  const sample = () => page.evaluate(async () => {
    const d = await import('/src/game/demand.ts');
    const m = await d.cellModes('all');
    if (!m) return null;
    const s = (b) => m[b].reduce((a, v) => a + v, 0);
    return Math.round(s('homeRail') + s('homeWalk') + s('homeDrive') + s('homeNone'));
  });
  out.before = await sample();
  console.log('before', out.before);
  const t0 = Date.now();
  if (nyc) { let wv = -1, since = Date.now(); while (Date.now() - since < 5000) { await page.waitForTimeout(500); const v = await page.evaluate(() => window.tw.world.value.version); if (v !== wv) { wv = v; since = Date.now(); } } await demandFinal(page, { strict: true, timeout: 600000 }); out.before = await sample(); }
  out.edit = await page.evaluate((nyc) => { const w = window.tw.world.value, l = w.save.lines[0], t = nyc ? w.lineStats[l.id].trainsNeeded.map((x) => x + 1) : [2, 2, 2]; window.tw.client.edit({ op: 'setTrainCount', line: l.num, trains: t }); return JSON.stringify(t); }, nyc);
  while (Date.now() - t0 < (nyc ? 150000 : 40000)) {
    await page.waitForTimeout(Date.now() - t0 < 15000 ? 100 : 1000);
    out.samples.push([Math.round((Date.now() - t0) / 100) / 10, await sample(), await page.evaluate(() => window.twDemand.client.version > (window.twDemand.view()?.version ?? 0) || !window.twDemand.view()?.final), await page.evaluate(() => window.twDemand.view()?.version)]);
  }
  await demandFinal(page, { strict: true, timeout: 600000 }).catch(() => {});
  await page.waitForTimeout(3000);
  out.after = await sample();
  out.solving = out.samples.filter((s) => s[2]).length;
  console.log('after', out.after, 'samples while solving', out.solving);
  console.log(out.samples.map((s) => s[0] + ':' + s[1] + (s[2] ? '*' : '') + 'v' + s[3]).join(' '));
} catch (e) { console.log('ERR', e.stack ?? e.message); }
finally {
  await writeFile(SP + `out-${label}.json`, JSON.stringify(out, null, 1));
  console.log('errors', errors.length, errors.slice(0, 5));
  await close();
}
