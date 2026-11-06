// T-2636: screenshots of panels in each language, for the translation review's last pass (docs/localization.md
// "Translating"), and each panel's row heights against English, so a fallback font that changes line height shows.
//   node trainworld/checks/l10n-shots.mjs <url> <out dir> [--langs en,ja,zh-Hans,zh-TW,pseudo] [--tabs settings,lines,...] [--ask <key>]
// A ?new=1 game in New York, paused, one demand worker. --tabs: dock tabs (build, lines, stations, money, cities,
// settings), each shot whole however long (settings by default). --ask <key>: also the confirm dialog showing that
// message (values from --values, JSON or a JSON file). Prints each shot's path and rows taller than English's.
import { readFileSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { launchChrome } from './lib.mjs';

const [url, out] = process.argv.slice(2);
const opt = (n, d) => (process.argv.includes(n) ? process.argv[process.argv.indexOf(n) + 1] : d);
if (!url || !out) { console.error('usage: node checks/l10n-shots.mjs <url> <out dir> [--langs ...] [--tabs ...] [--ask <key>]'); process.exit(2); }
const langs = opt('--langs', 'en,ja,zh-Hans,zh-TW').split(',');
const tabs = opt('--tabs', 'settings').split(',');
// --values: JSON, or a JSON file's path (easier to pass through cmd's quoting)
const rawValues = opt('--values', '{}');
const ask = opt('--ask', null), values = JSON.parse(rawValues.trim().startsWith('{') ? rawValues : readFileSync(rawValues, 'utf8'));
await mkdir(out, { recursive: true });

const { ctx, close } = await launchChrome({ prefix: 'tw-l10n-', size: [1400, 1000] });
const metrics = {};
try {
  for (const lang of langs) {
    const page = await ctx.newPage();
    const sep = url.includes('?') ? '&' : '?';
    await page.goto(`${url}${sep}debug=1&new=1&paused=1&demandWorkers=1&lang=${lang}&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value, undefined, { timeout: 180000 });
    await page.waitForFunction((l) => l === 'pseudo' || document.documentElement.lang === l || l === 'en', lang, { timeout: 30000 });
    await page.evaluate(() => document.fonts.ready);
    for (const tab of tabs) {
      await page.evaluate(async (t) => { const ui = await import('/src/game/ui.ts'); ui.selection.value = null; ui.tab.value = t; }, tab);
      await page.waitForTimeout(600);
      await page.evaluate(() => document.fonts.ready);
      // the pane unrolled to its whole height, so one shot holds every row
      const m = await page.evaluate(() => {
        const pane = document.querySelector('#dock .pane') ?? document.querySelector('#dock');
        const h = pane.scrollHeight;
        pane.style.maxHeight = 'none'; pane.style.height = h + 'px'; pane.style.overflow = 'visible';
        const dock = document.querySelector('#dock'); dock.style.height = 'auto'; dock.style.maxHeight = 'none'; dock.style.overflow = 'visible';
        const rows = [...pane.querySelectorAll('.slider, .check, .game-mode, .h, .stack .group')].map((el) => Math.round(el.getBoundingClientRect().height));
        const wide = [...pane.querySelectorAll('button, .btn')].filter((el) => el.scrollWidth > el.clientWidth + 1).map((el) => el.textContent.trim().slice(0, 40));
        return { paneHeight: h, rows, overflowing: wide, font: getComputedStyle(document.body).fontFamily };
      });
      await page.setViewportSize({ width: 1400, height: Math.max(1000, m.paneHeight + 120) });
      await page.waitForTimeout(300);
      const file = join(out, `${tab}-${lang}.png`);
      await page.locator('#dock').screenshot({ path: file });
      await page.setViewportSize({ width: 1400, height: 1000 });
      metrics[`${tab} ${lang}`] = m;
      console.log(`${file}  pane ${m.paneHeight}px, ${m.rows.length} rows${m.overflowing.length ? `, clipped: ${m.overflowing.join(' | ')}` : ''}`);
      await page.evaluate(() => { const p = document.querySelector('#dock .pane') ?? document.querySelector('#dock'); p.style.cssText = ''; document.querySelector('#dock').style.cssText = ''; });
    }
    if (ask) {
      await page.evaluate(async ([key, values]) => {
        const ui = await import('/src/game/ui.ts'), i18n = await import('/src/i18n/index.ts');
        ui.ask.value = { text: i18n.t(key, values), yes: i18n.t('remove.yes'), no: i18n.t('remove.no'), run() {} };
      }, [ask, values]);
      await page.waitForTimeout(400);
      const file = join(out, `ask-${lang}.png`);
      await page.screenshot({ path: file });
      console.log(file);
    }
    await page.close();
  }
} finally {
  await close();
}
// rows that came out taller than English's: a sign of a fallback font's line height, or of wrapping
for (const tab of tabs) {
  const base = metrics[`${tab} en`];
  if (!base) continue;
  for (const lang of langs.filter((l) => l !== 'en')) {
    const m = metrics[`${tab} ${lang}`];
    if (!m) continue;
    const taller = m.rows.map((h, i) => [i, h - (base.rows[i] ?? h)]).filter(([, d]) => d > 1);
    console.log(`${tab} ${lang}: pane ${m.paneHeight - base.paneHeight >= 0 ? '+' : ''}${m.paneHeight - base.paneHeight}px against English; ${taller.length ? `${taller.length} rows taller (${taller.map(([i, d]) => `#${i} +${d}`).join(', ')})` : 'no row taller'}; font ${m.font}`);
  }
}
