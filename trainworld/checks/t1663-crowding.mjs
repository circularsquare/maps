// T-1663: screenshots of the crowding red outline on the real save (New York) in the Traffic track
// display: a stretch with a typical wait, a long one, and track at capacity, at zoom 13 and 15.
//   node trainworld/checks/t1663-crowding.mjs <url> <outdir> [light|dark]      (from the maps root)
import { readFile, mkdir } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, REAL_SAVE } from './lib.mjs';

const [BASE, OUT, SHADE = 'light'] = process.argv.slice(2);
if (!BASE || !OUT) { console.error('usage: node checks/t1663-crowding.mjs <url> <outdir> [light|dark]'); process.exit(2); }
await mkdir(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-crowd-', size: [1000, 700] });
try {
  const page = await ctx.newPage();
  await page.addInitScript((shade) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade, display: { dayShown: 'weekday', trackColour: 'traffic', fullMarks: true }, v: 2 })); } catch { /* */ } }, SHADE);
  await page.goto(`${BASE}${BASE.includes('?') ? '&' : '?'}debug=1&new=1&opened=all&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 100, undefined, { timeout: 120000 });
  await demandFinal(page, { polling: 250 });
  await page.evaluate(() => { const c = window.tw.clock; c.set(Math.floor(c.now() / 86400) * 86400 + 8 * 3600); c.setSpeed?.(0); window.tw.display.value = { ...window.tw.display.peek(), trackColour: 'traffic', fullMarks: true }; });
  await page.waitForTimeout(1500);
  await demandFinal(page, { polling: 250 }).catch(() => console.log('demand never final'));
  await page.waitForTimeout(3000);
  console.log(await page.evaluate(() => JSON.stringify({ colour: window.tw.display.value.trackColour, marks: window.tw.display.value.fullMarks, keys: window.tw.renderer.fullKeys.size, edges: window.tw.renderer.fullEdges?.size })));
  const spots = await page.evaluate(() => {
    const tw = window.tw, r = tw.renderer, f = r.linesFull, s = r.net.lines.seg;
    const at = (i) => { const [lng, lat] = tw.toLngLat((s[4 * i] + s[4 * i + 2]) / 2, (s[4 * i + 1] + s[4 * i + 3]) / 2); return { lng, lat, v: f[i] }; };
    const pos = [], neg = [];
    for (let i = 0; i < f.length; i++) (f[i] > 0 ? pos : f[i] < 0 ? neg : []).push(i);
    pos.sort((a, b) => f[a] - f[b]);
    return { nPos: pos.length, nNeg: neg.length, typical: pos.length ? at(pos[pos.length >> 1]) : null, long: pos.length ? at(pos[pos.length - 1]) : null, full: neg.length ? at(neg[neg.length >> 1]) : null };
  });
  console.log(JSON.stringify(spots));
  for (const name of ['typical', 'long', 'full']) {
    const p = spots[name];
    if (!p) continue;
    for (const zoom of [13, 15]) {
      await page.evaluate(([c, z]) => { window.tw.map.jumpTo({ center: c, zoom: z }); }, [[p.lng, p.lat], zoom]);
      await page.waitForTimeout(2500);
      await page.screenshot({ path: `${OUT}/${SHADE}-${name}-z${zoom}.png` });
    }
  }
} finally {
  await close();
}
