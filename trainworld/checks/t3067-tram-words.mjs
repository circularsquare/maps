// T-3067: the rest of a tram's words (capacity kinds, markers, trouble hovers, issues, tips) and the Lines
// tab's Fleet heading. Prints each changed message for train and tram, fails on a train word in a tram
// text, and screenshots the Lines tab and a tram capacity hover. From the maps root:
//   node trainworld/checks/t3067-tram-words.mjs [url]
import { readFile } from 'node:fs/promises';
import { launchChrome, collectErrors, importSave, REAL_SAVE } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6194/';
const chrome = await launchChrome({ prefix: 'tw-t3067-', size: [1400, 1000] });
const errors = [];
let bad = 0;
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(4000);
  await page.click('[data-guide-tab="lines"]');
  await page.waitForTimeout(1500);
  await page.screenshot({ path: 'artifacts/trainworld/t3067-lines-tab.png' });
  const out = await page.evaluate(async () => {
    const { t } = await import('/src/i18n/index.ts');
    const { capacityKind, fullTip } = await import('/src/game/capacityKinds.ts');
    const { capacityTip } = await import('/src/map/capacityMarkers.ts');
    const r = { train: [], tram: [] };
    for (const v of ['train', 'tram']) {
      const tram = v === 'tram';
      for (const kind of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14]) {
        const k = capacityKind(kind, v);
        r[v].push(k.name, k.short, k.title, k.cause, k.help, k.fix, fullTip(kind, 12, v));
      }
      r[v].push(...capacityTip(2, 0.95, 11, 20, 0, v));
      r[v].push(t('troubleList.held', { vehicle: v }), t('troubleList.waits', { time: '30s', level: 'high', vehicle: v }),
        t('upgrades.waitingDoes', { vehicle: v }), t('upgrades.waitingDoesAll', { vehicle: v }), t('today.shortStretch', { vehicle: v }),
        t('issues.moneyTrains', { cost: '$5M', vehicle: v }));
      for (const k of ['platformLength', 'platformTooLong', 'platformJoint', 'platformNotLevel', 'platformCrossing', 'branchNoRoom']) r[v].push(t('issues.' + k, { tram, min: '20 m', max: '80 m' }));
      r[v].push(t('networkTips.nobody', { tram }), t('networkTips.fullestFull', { tram }), t('networkTips.fullestPct', { tram, share: 60 }),
        t('stopMarkers.reverseEnd', { tram }), t('stopMarkers.reverseAt', { tram, name: 'Canal St' }), t('stopMarkers.reverseSome', { tram }));
    }
    return r;
  });
  console.log(JSON.stringify(out.tram, null, 1));
  for (const s of out.tram) if (/\b(trains?|platforms?|terminals?|sidings?)\b/i.test(s) && !/Tram stop/.test(s)) { bad++; console.log('FAIL train word in tram text: ' + s); }
  for (const s of out.train) if (/\btrams?\b/i.test(s)) { bad++; console.log('FAIL tram word in train text: ' + s); }
  // a tram capacity hover on the page
  await page.evaluate(async () => {
    const { capacityTip } = await import('/src/map/capacityMarkers.ts');
    const d = document.createElement('div');
    d.id = 'cap-demo';
    d.style.cssText = 'position:fixed;left:420px;top:120px;background:#fff;color:#222;padding:8px 10px;border:1px solid #888;font:13px sans-serif;white-space:pre;z-index:99999';
    d.textContent = capacityTip(2, 0.95, 11, 20, 0, 'tram').join('\n');
    document.body.append(d);
  });
  await page.screenshot({ path: 'artifacts/trainworld/t3067-tram-capacity-hover.png', clip: { x: 400, y: 100, width: 420, height: 100 } });
  if (errors.length) { bad++; console.log('errors: ' + errors.slice(0, 3).join(' | ')); }
} finally {
  await chrome.close();
}
console.log(bad ? `FAIL ${bad}` : 'PASS t3067');
process.exit(bad ? 1 : 0);
