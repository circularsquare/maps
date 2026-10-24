// T-1196, T-1197, T-1214: switching games and keeping saves. In headless Chrome with a temporary
// profile: the real New York save loaded into a slot; another slot opened from the saves panel's Open
// reloads the page into it (a navigation, the new slot active, its network drawn); the loading card
// names its stage and Cancel goes back to the start menu; a game too big for a demand worker is refused
// with a reason (the memory estimate for every Chinese zone); a copy folder (the page's private file
// system standing in for a folder the player picks) gets every slot as a .save file and the slot again
// after a save; the browser was asked to keep its storage.
//   node trainworld/checks/t1196-switch.mjs [url]     (from the maps root; default localhost:6146)
// Prints PASS or FAIL lines and exits 1 on a failure. About two minutes.
import { readFile } from 'node:fs/promises';
import { launchChrome, importSave, REAL_SAVE, ANGLE, ARTIFACTS } from './lib.mjs';

const ART = ARTIFACTS.replace(/[\\/]$/, '');

const BASE = process.argv[2] ?? 'http://localhost:6146/';
let failed = false;
const report = (name, ok, detail) => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

const { ctx, close } = await launchChrome({ prefix: 'tw-t1196-', size: [1200, 800], args: ANGLE });
try {
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  const wait = (ms) => page.waitForTimeout(ms);
  const active = () => page.evaluate(async () => (await import('/src/workers/clockClient.ts')).saveSlotsState.value.active).catch(() => null);
  const ready = () => page.waitForFunction(() => window.tw?.world?.value && window.tw.lastSaved, undefined, { timeout: 180000 });

  // two slots, each the real save loaded into its own
  await page.goto(`${BASE}?debug=1&continue=1&paused=1`);
  await ready();
  const load = async (from) => {
    const before = await page.evaluate(() => window.tw.lastSaved.peek() ?? 0);
    await importSave(page, await readFile(REAL_SAVE));
    await page.waitForFunction((from) => window.tw.world.value.save.lines.length === 109 && window.__cc?.saveSlotsState.value.active !== from, from, { timeout: 180000 });
    await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
    await page.waitForFunction((b) => (window.tw.lastSaved.peek() ?? 0) > b, before, { timeout: 60000 });
    return active();
  };
  await page.evaluate(async () => { window.__cc = await import('/src/workers/clockClient.ts'); });
  const first = await load(await active());
  const real = await load(first);
  const kept = await page.evaluate(async () => (await import('/src/game/saveSafety.ts')).storageKept.value);
  report('storage asked to be kept', kept !== null, `(persisted: ${kept})`);

  // T-1214: a folder (the private file system here) gets every slot, and the slot again at a save
  const copies = await page.evaluate(async () => {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle('copies', { create: true });
    window.tw.client.post({ kind: 'copyFolder', dir });
    await new Promise((r) => setTimeout(r, 3000));
    const names = [];
    for await (const [name, h] of dir.entries()) names.push(`${name}:${(await h.getFile()).size}`);
    return names;
  });
  report('copies in the folder', copies.filter((n) => / \(\w{4}\)\.save:\d{4,}$/.test(n)).length === 2, `(${copies.join(', ')})`);
  // the panel's line for it (the folder set as a pick would set it)
  await page.evaluate(async () => {
    const S = await import('/src/game/saveSafety.ts');
    S.copyFolder.value = { name: 'anitabuilder saves', state: 'on', at: Date.now() };
    (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = true;
  });
  await page.waitForSelector('.saves-panel .save-safety');
  await wait(500);
  await page.locator('.saves-panel').screenshot({ path: `${ART}/t1214-saves-panel.png` });
  await page.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = false; });

  // T-1197: an estimate for every Chinese zone is refused, New York's is not
  const refusal = await page.evaluate(async () => {
    const M = await import('/src/game/memoryCheck.ts'), C = await import('/src/game/cities.ts');
    const china = C.CITIES.filter((c) => c.country === 'CN').map((c) => c.id);
    return { n: china.length, china: M.memoryRefusal(china), need: M.memoryNeed(china), nyc: M.memoryRefusal(['nyc', 'nhv', 'hfd', 'aby', 'law']) };
  });
  report('a game too big is refused', !!refusal.china && refusal.nyc === null, `(${refusal.n} zones: "${refusal.china}"; ${refusal.need.workerGB.toFixed(1)} GB a worker; New York: ${refusal.nyc})`);

  // T-1196: Open in the saves panel reloads the page into the other slot
  await page.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = true; });
  await page.waitForSelector(`.save-slots li[data-slot="${first}"] .save-open`);
  const [nav] = await Promise.all([page.waitForNavigation({ timeout: 60000 }).then(() => true, () => false), page.click(`.save-slots li[data-slot="${first}"] .save-open`)]);
  await ready();
  await wait(1500);
  const after = { active: await active(), url: page.url(), lines: await page.evaluate(() => window.tw.world.value.save.lines.length) };
  report('open reloads into the slot', nav && after.active === first && after.lines === 109 && !/slot=/.test(after.url), `(navigated ${nav}, active ${after.active}, ${after.lines} lines, address ${after.url})`);

  // the card through its stages, then Cancel back to the start menu
  await page.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = true; });
  await page.waitForSelector(`.save-slots li[data-slot="${real}"] .save-open`);
  await Promise.all([page.waitForNavigation({ timeout: 60000 }), page.click(`.save-slots li[data-slot="${real}"] .save-open`)]);
  const steps = [];
  for (const end = Date.now() + 120000; Date.now() < end && !steps.includes('Calculating demand');) {
    const t = await page.evaluate(() => document.querySelector('.loading-card .big')?.textContent ?? null).catch(() => null);
    if (t && steps.at(-1) !== t) steps.push(t);
    await wait(100);
  }
  report('card stages in order', steps.join(' > ') === 'Loading > Loading city data > Calculating demand', `(${steps.join(' > ')})`);
  await Promise.all([page.waitForNavigation({ timeout: 30000 }).catch(() => null), page.click('.loading-card .load-cancel')]);
  await page.waitForSelector('.start-menu .start-save', { timeout: 60000 }).catch(() => null);
  const menu = await page.locator('.start-menu').count();
  report('cancel goes to the start menu', menu === 1, `(${page.url()})`);

  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
