// T-308 browser check 2: old-format saves already sitting in browser slots (as hers would be).
// The real TWT2 save is put in IndexedDB as the autosave slot and as a second slot; the page loads
// the autosave (converted, slot rewritten, old bytes kept), then opens the second slot (converted),
// then reloads (no conversion).
import { mkdtemp, writeFile, readFile } from 'node:fs/promises';
import { join } from 'node:path';
const { chromium } = await import('file:///C:/Users/anita/AppData/Local/npm-cache/_npx/e41f203b7505f1fb/node_modules/playwright/index.mjs');
const HERE = 'C:/Users/anita/AppData/Local/Temp/claude/c--Users-anita-projects-maps/87037fc9-7e8f-4e0f-abce-7678aadf6b68/scratchpad/agent-saves308';
const BASE = process.env.BASE ?? 'http://localhost:5308';
const SAVE = 'C:/Users/anita/projects/maps/trainworld/T-007-real-nyc.save';
const profile = await mkdtemp(join(HERE, 'chrome-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
const out = { profile, steps: {} };
const errors = [];
const measure = (page) => page.evaluate(() => {
  const w = window.tw.world.value;
  return { load: window.tw.client.lastLoad, lines: w.save.lines.length, stations: w.save.stations.length, trackKm: w.edges.reduce((s, e) => s + e.lengthM, 0) / 1000 };
});
const slotInfo = (page) => page.evaluate(async () => {
  const db = await new Promise((res, rej) => { const r = indexedDB.open('anitabuilder', 1); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
  const s = db.transaction('saves', 'readonly').objectStore('saves');
  const keys = await new Promise(res => { const r = s.getAllKeys(); r.onsuccess = () => res(r.result); });
  const vals = await new Promise(res => { const r = s.getAll(); r.onsuccess = () => res(r.result); });
  const info = {};
  for (const [i, k] of keys.entries()) {
    const v = vals[i];
    const raw = new Uint8Array(await new Response(new Blob([v.bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
    const n = new DataView(raw.buffer).getUint32(4, true);
    info[k] = { trackMagic: new TextDecoder().decode(raw.subarray(8 + n, 8 + n + 4)), bytes: v.bytes.length, name: v.name, lines: v.lines };
  }
  return info;
});
try {
  let page = await ctx.newPage();
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errors.push(m.type() + ': ' + m.text()); });
  page.on('worker', w => w.on('console', m => errors.push('worker ' + m.type() + ': ' + m.text())));
  await page.goto(BASE + '/?debug=1&new=1&paused=1&v=' + Date.now());
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(2000);
  // the first run's page may write its (empty) game on the way out: let it finish, then seed from a
  // page that is not the app
  await page.close();
  await new Promise(r => setTimeout(r, 4000));
  const page2 = await ctx.newPage();
  await page2.goto(BASE + '/package.json');
  const bytes = await readFile(SAVE);
  await page2.evaluate(async (data) => {
    const db = await new Promise((res, rej) => { const r = indexedDB.open('anitabuilder', 1); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
    const put = (key, v) => new Promise((res, rej) => { const tx = db.transaction('saves', 'readwrite'); tx.objectStore('saves').put(v, key); tx.oncomplete = res; tx.onerror = () => rej(tx.error); });
    const b = Uint8Array.from(data);
    await put('autosave', { bytes: b, at: Date.now() - 5000, day: 5 });
    await put('slot-seeded-two', { bytes: b, at: Date.now() - 9000, day: 5, name: 'Old slot two' });
  }, Array.from(bytes));
  out.steps.seeded = await slotInfo(page2);
  // load it as the last game
  page2.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page2.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page2.on('worker', w => w.on('console', m => errors.push('worker ' + m.type() + ': ' + m.text())));
  page = page2;
  await page.goto(BASE + '/?debug=1&paused=1&v=' + Date.now());
  try {
    await page.waitForFunction(() => window.tw?.world.value && window.tw.client.lastLoad && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 90000 });
  } catch (e) {
    out.stuck = await page.evaluate(() => ({ lastLoad: window.tw?.client?.lastLoad, lines: window.tw?.world?.value?.save?.lines?.length, text: document.body.innerText.slice(0, 600) }));
    out.errors = errors;
    out.slotsStuck = await slotInfo(page);
    out.say = await page.evaluate(() => window.tw.say?.value ?? null);
    throw e;
  } finally { await writeFile(join(HERE, 't308-slots-out.json'), JSON.stringify(out, null, 1)); }
  await page.waitForTimeout(4000);
  out.steps.autosaveLoad = await measure(page);
  out.steps.afterAutosaveLoad = await slotInfo(page);
  // open the second slot (it is still TWT2)
  await page.evaluate(() => window.tw.client.post({ kind: 'saveSlot', action: 'open', key: 'slot-seeded-two' }));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file', undefined, { timeout: 60000 });
  await page.waitForTimeout(4000);
  out.steps.slotOpen = await measure(page);
  out.steps.afterSlotOpen = await slotInfo(page);
  // back to the first, converted: must not convert again
  await page.evaluate(() => window.tw.client.post({ kind: 'saveSlot', action: 'open', key: 'autosave' }));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.client.lastLoad.converted === false, undefined, { timeout: 60000 });
  out.steps.reopenFirst = await measure(page);
  out.steps.slotsAtEnd = await slotInfo(page);
  out.errors = errors.filter(e => !/favicon|DevTools|openfreemap|AJAXError|city pack/.test(e)).slice(0, 20);
} finally {
  await ctx.close();
}
await writeFile(join(HERE, 't308-slots-out.json'), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));
