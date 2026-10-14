// T-308 browser check: import the real (TWT2, legacy frame) NYC save, see it converted to Mercator
// and written back to its slot, reload and see it load from the slot without converting again.
import { mkdtemp, writeFile, readFile } from 'node:fs/promises';
import { join } from 'node:path';
const { chromium } = await import('file:///C:/Users/anita/AppData/Local/npm-cache/_npx/e41f203b7505f1fb/node_modules/playwright/index.mjs');
const HERE = 'C:/Users/anita/AppData/Local/Temp/claude/c--Users-anita-projects-maps/87037fc9-7e8f-4e0f-abce-7678aadf6b68/scratchpad/agent-saves308';
const ART = 'C:/Users/anita/projects/maps/artifacts/trainworld';
const BASE = process.env.BASE ?? 'http://localhost:5308';
const SAVE = process.env.SAVE ?? 'C:/Users/anita/projects/maps/trainworld/T-007-real-nyc.save';
const profile = await mkdtemp(join(HERE, 'chrome-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
const out = { profile, steps: {} };
const errors = [];
const measure = (page) => page.evaluate(() => {
  const t = window.twDemand, w = window.tw.world.value, v = t?.view?.();
  const km = w.edges.reduce((s, e) => s + e.lengthM, 0) / 1000;
  const rail = t ? [...t.cities()].reduce((s, [, c]) => s + (c.railTripsPerDay || 0), 0) : null;
  return { load: window.tw.client.lastLoad, frame: w.frame ?? null, lines: w.save.lines.length, stations: w.save.stations.length, edges: w.edges.length, trackKm: km, railTripsPerDay: rail, demandLines: v?.lines?.size ?? null, round: v?.round ?? null, cash: w.save.cash };
});
const slotInfo = (page) => page.evaluate(async () => {
  const db = await new Promise((res, rej) => { const r = indexedDB.open('anitabuilder', 1); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
  const s = db.transaction('saves', 'readonly').objectStore('saves');
  const keys = await new Promise(res => { const r = s.getAllKeys(); r.onsuccess = () => res(r.result); });
  const info = {};
  for (const k of keys) {
    const v = await new Promise(res => { const r = s.get(k); r.onsuccess = () => res(r.result); });
    const raw = new Uint8Array(await new Response(new Blob([v.bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
    const n = new DataView(raw.buffer).getUint32(4, true);
    info[k] = { trackMagic: new TextDecoder().decode(raw.subarray(8 + n, 8 + n + 4)), bytes: v.bytes.length, at: v.at, lines: v.lines, stations: v.stations, cash: v.cash };
  }
  return info;
});
try {
  const page = await ctx.newPage();
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(BASE + '/?debug=1&new=1&paused=1&v=' + Date.now());
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(2000);
  out.steps.fresh = await measure(page);
  out.steps.freshSlots = await slotInfo(page);
  // import the real save
  const bytes = await readFile(SAVE);
  await page.evaluate(data => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file', undefined, { timeout: 60000 });
  out.steps.afterImportLoad = await measure(page);
  await page.waitForFunction(() => window.twDemand.view()?.round === 3 && window.twDemand.view().lines.size > 100, undefined, { timeout: 300000 });
  await page.waitForTimeout(1500);
  out.steps.afterImport = await measure(page);
  await page.waitForTimeout(3000);
  out.steps.slotsAfterImport = await slotInfo(page);
  await page.screenshot({ path: `${ART}/T-308-imported.png` });
  // reload: it must come from the converted slot, without converting again
  await page.goto(BASE + '/?debug=1&paused=1&v=' + Date.now());
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client && window.tw.client.lastLoad, undefined, { timeout: 180000 });
  out.steps.reloadLoad = await measure(page);
  await page.waitForFunction(() => window.twDemand.view()?.round === 3 && window.twDemand.view().lines.size > 100, undefined, { timeout: 300000 });
  await page.waitForTimeout(1500);
  out.steps.afterReload = await measure(page);
  out.steps.slotsAfterReload = await slotInfo(page);
  await page.screenshot({ path: `${ART}/T-308-reloaded.png` });
  out.gl = await page.evaluate(() => window.tw.renderer.gl.getError());
  out.errors = errors.filter(e => !/favicon|DevTools|openfreemap|AJAXError|city pack/.test(e)).slice(0, 20);
} finally {
  await ctx.close();
}
await writeFile(join(HERE, 't308-out.json'), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));
