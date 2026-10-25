// T-1251, T-1247: saving by hand and renaming in place, in two tabs of one browser. The real save
// is loaded into a slot of its own in tab A, then tab B opens on the same slot with autosave off.
// The Save button in A writes the slot at once and shows "Saved <time>". The Save button in B takes
// the slot's lease: B writes it and autosaves from then on, and A says autosave is off and writes
// nothing more. Ctrl+S in A takes it back. Then in A a double click on a save's name renames it on
// Enter and leaves it on Escape. Screenshots of the panel renaming and after the save go to artifacts/trainworld/.
//   node trainworld/checks/t1251-save-by-hand.mjs [url]     (from the maps root; default localhost:6161)
// Prints PASS or FAIL lines and exits 1 on a failure. A few minutes.
import { join } from 'node:path';
import { launchChrome, demandFinal, REAL_SAVE, ANGLE, ARTIFACTS } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6161/';

let failed = false;
const report = (name, ok, detail) => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

const { ctx, close } = await launchChrome({ prefix: 'tw-t1251-', size: [1300, 900], args: ANGLE });
try {
  const errors = [];
  const open = async () => {
    const page = await ctx.newPage();
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(`${BASE}?debug=1&paused=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
    await page.waitForTimeout(1500);
    return page;
  };
  const state = (page) => page.evaluate(async () => {
    const s = (await import('/src/workers/clockClient.ts')).saveSlotsState.value;
    return { active: s.active, writable: s.writable, reason: s.reason ?? '', slots: s.slots.map((x) => ({ key: x.key, name: x.name, at: x.at })) };
  });
  const stored = (page, key) => page.evaluate(async (key) => {
    const r = await (await import('/src/workers/saveFile.ts')).readStored(key);
    return r ? { at: r.at, name: r.name } : null;
  }, key);
  const panel = async (page) => {
    await page.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = true; });
    await page.waitForSelector('.saves-panel .save-files');
    await page.waitForTimeout(500);
  };
  const closePanel = (page) => page.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = false; });
  const fares = async (page, base) => {
    await page.evaluate((base) => window.tw.client.post({ kind: 'edit', seq: 990000 + Math.round(base * 100), op: { op: 'fares', base, perKm: 0.1 } }), base);
    await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
    await page.waitForTimeout(2000);
  };

  // tab A: the real save from a file, in a slot of its own (the page reloads into it, T-1196)
  const a = await open();
  const first = (await state(a)).active;
  await panel(a);
  const [fc] = await Promise.all([a.waitForEvent('filechooser'), a.click('.saves-panel .save-files button:has-text("Load a file")')]);
  await fc.setFiles(REAL_SAVE);
  let slot = null;
  for (const end = Date.now() + 180000; !slot && Date.now() < end; await a.waitForTimeout(300)) {
    try {
      const s = await a.evaluate(async () => window.tw?.world?.value ? (await import('/src/workers/clockClient.ts')).saveSlotsState.value.active : null);
      if (s && s !== first && s.startsWith('slot-')) slot = s;
    } catch { /* reloading */ }
  }
  await a.waitForFunction(() => window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(a, { strict: false, held: true, polling: 300 });
  await a.waitForTimeout(1500);

  // 1. with autosave on, Save writes now and says when
  await panel(a);
  const before = await stored(a, slot);
  await a.waitForTimeout(1100);
  await a.click('.save-slots li.on .save-now');
  await a.waitForSelector('.save-slots li.on .save-by-hand', { timeout: 30000 });
  const label = await a.locator('.save-slots li.on .save-by-hand').textContent();
  const after = await stored(a, slot);
  report('save with autosave on', (after?.at ?? 0) > (before?.at ?? 0) && /^Saved \d/.test(label ?? ''), `(slot written ${before?.at} -> ${after?.at}; "${label}")`);
  await a.screenshot({ path: join(ARTIFACTS, 't1251-saved.png') });
  await closePanel(a);

  // 2. tab B opens the same slot: autosave off there
  const b = await open();
  let sb = await state(b);
  report('second tab has autosave off', sb.active === slot && !sb.writable, `(B on ${sb.active}, writable ${sb.writable}: "${sb.reason}")`);

  // 3. Save in B takes the slot: B writes it, A stops
  await panel(b);
  const atBefore = (await stored(b, slot)).at;
  await b.click('.save-slots li.on .save-now');
  await b.waitForSelector('.save-slots li.on .save-by-hand', { timeout: 30000 });
  await b.waitForTimeout(1000);
  sb = await state(b);
  let sa = await state(a);
  const atB = (await stored(b, slot)).at;
  report('save in the second tab takes the slot', sb.writable && !sa.writable && atB > atBefore && /another tab/.test(sa.reason),
    `(B writable ${sb.writable}, A writable ${sa.writable}: "${sa.reason}"; slot ${atBefore} -> ${atB})`);
  await b.screenshot({ path: join(ARTIFACTS, 't1251-second-tab-saved.png') });

  // 4. A's changes no longer reach the slot
  await fares(a, 3.25);
  const atA = (await stored(a, slot)).at;
  report('the first tab stops autosaving', atA === atB, `(slot ${atB} -> ${atA} after an edit in A)`);

  // 5. Ctrl+S in A takes it back
  await a.bringToFront();
  await a.keyboard.press('Control+s');
  await a.waitForTimeout(2500);
  sa = await state(a);
  sb = await state(b);
  const atA2 = (await stored(a, slot)).at;
  report('Ctrl+S takes the slot back', sa.writable && !sb.writable && atA2 > atA, `(A writable ${sa.writable}, B writable ${sb.writable}; slot ${atA} -> ${atA2})`);

  // 6. rename in place in A, which holds the slot again: double click, type, Enter; then Escape
  await panel(a);
  await a.dblclick('.save-slots li.on .save-name');
  await a.waitForSelector('.save-slots li.on .save-name-field');
  await a.fill('.save-slots li.on .save-name-field', 'Renamed by hand');
  await a.screenshot({ path: join(ARTIFACTS, 't1251-renaming.png') });
  await a.press('.save-slots li.on .save-name-field', 'Enter');
  await a.waitForTimeout(1500);
  const r1 = (await stored(a, slot))?.name;
  const panelOpen = await a.locator('.saves-panel').count();
  await a.dblclick('.save-slots li.on .save-name');
  await a.fill('.save-slots li.on .save-name-field', 'Not kept');
  await a.press('.save-slots li.on .save-name-field', 'Escape');
  await a.waitForTimeout(1000);
  const r2 = (await stored(a, slot))?.name;
  const stillOpen = await a.locator('.saves-panel').count();
  const shown = await a.locator('.save-slots li.on .save-name').textContent();
  report('double click renames, Escape cancels', r1 === 'Renamed by hand' && r2 === 'Renamed by hand' && panelOpen === 1 && stillOpen === 1 && /Renamed by hand/.test(shown ?? ''),
    `(after Enter "${r1}", after Escape "${r2}", card "${shown}", panel open ${stillOpen})`);

  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
