// T-2636: screenshots of panels in each language, for the translation review's last pass (docs/localization.md
// "Translating"), and each panel's row heights against English, so a fallback font that changes line height shows.
//   node trainworld/checks/l10n-shots.mjs <url> <out dir> [--langs en,ja,zh-Hans,zh-TW,pseudo] [--tabs settings,lines,...] [--ask <key>]
//     [--save] [--pick station,line,train,track,city,world] [--page]
// A ?new=1 game in New York, paused, one demand worker. --tabs: dock tabs (build, lines, stations, money, cities,
// settings), each shot whole however long (settings by default). --ask <key>: also the confirm dialog showing that
// message (values from --values, JSON or a JSON file). --save: the real save imported first (T-007-real-nyc.save),
// so tabs and inspectors have lines. --pick: an inspector each (the line with most stops, its first station, one of its
// trains, a track, New York, the all cities view), the dock shot with it open. --page: the whole window as well (the top bar, the map's
// tags). Prints each shot's path, buttons whose text is clipped, and rows taller than English's.
import { readFileSync } from 'node:fs';
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { launchChrome, importSave, REAL_SAVE } from './lib.mjs';

const [url, out] = process.argv.slice(2);
const opt = (n, d) => (process.argv.includes(n) ? process.argv[process.argv.indexOf(n) + 1] : d);
if (!url || !out) { console.error('usage: node checks/l10n-shots.mjs <url> <out dir> [--langs ...] [--tabs ...] [--ask <key>]'); process.exit(2); }
const langs = opt('--langs', 'en,ja,zh-Hans,zh-TW').split(',');
const tabs = opt('--tabs', 'settings').split(',').filter(Boolean);
const picks = opt('--pick', '').split(',').filter(Boolean);
const withSave = process.argv.includes('--save'), wholePage = process.argv.includes('--page');
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
    if (withSave) {
      await importSave(page, await readFile(REAL_SAVE));
      await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
      await page.waitForTimeout(3000);
    }
    // the dock unrolled to its whole height, so one shot holds every row
    // with an inspector open (`insp`), the tab's pane is hidden and the inspector unrolled instead
    const shootDock = async (name, insp = false) => {
      await page.waitForTimeout(insp ? 1500 : 600);
      await page.evaluate(() => document.fonts.ready);
      const m = await page.evaluate((insp) => {
        const dock = document.querySelector('#dock'), pane = dock.querySelector('.pane') ?? dock, ins = dock.querySelector('#inspector');
        window.__l10nStyles = [dock, pane, ins].filter(Boolean).map((el) => [el, el.style.cssText]);
        if (insp && ins) {
          pane.style.display = 'none';
          ins.style.flex = 'none'; ins.style.overflow = 'visible'; ins.style.maxHeight = 'none';
        } else {
          pane.style.maxHeight = 'none'; pane.style.height = pane.scrollHeight + 'px'; pane.style.overflow = 'visible';
        }
        dock.style.height = 'auto'; dock.style.maxHeight = 'none'; dock.style.overflow = 'visible';
        const root = insp && ins ? ins : pane;
        const rows = [...root.querySelectorAll('.slider, .check, .game-mode, .h, .stack .group, tr')].map((el) => Math.round(el.getBoundingClientRect().height));
        const wide = [...dock.querySelectorAll('button, .btn, .tab, .seg button')].filter((el) => el.offsetParent && el.scrollWidth > el.clientWidth + 1).map((el) => el.textContent.trim().slice(0, 40));
        return { paneHeight: root.scrollHeight, dockHeight: dock.scrollHeight, rows, overflowing: wide, font: getComputedStyle(document.body).fontFamily };
      }, insp);
      await page.setViewportSize({ width: 1400, height: Math.max(1000, m.dockHeight + 120) });
      await page.waitForTimeout(300);
      const file = join(out, `${name}-${lang}.png`);
      await page.locator('#dock').screenshot({ path: file });
      await page.setViewportSize({ width: 1400, height: 1000 });
      metrics[`${name} ${lang}`] = m;
      console.log(`${file}  pane ${m.paneHeight}px, ${m.rows.length} rows${m.overflowing.length ? `, clipped: ${m.overflowing.join(' | ')}` : ''}`);
      await page.evaluate(() => { for (const [el, css] of window.__l10nStyles ?? []) el.style.cssText = css; });
    };
    for (const tab of tabs) {
      await page.evaluate(async (t) => { const ui = await import('/src/game/ui.ts'); ui.selection.value = null; ui.tab.value = t; }, tab);
      await shootDock(tab);
    }
    for (const pick of picks) {
      const ok = await page.evaluate(async (kind) => {
        const ui = await import('/src/game/ui.ts'), w = window.tw.world.value;
        const lines = w.save.lines, stats = w.lineStats ?? {};
        const line = [...lines].sort((a, b) => (stats[b.id]?.ridersPerDay ?? 0) - (stats[a.id]?.ridersPerDay ?? 0) || b.stops.length - a.stops.length)[0];
        if (kind === 'world') ui.selection.value = { kind: 'world' };
        else if (!line) return false;
        else if (kind === 'line') ui.selection.value = { kind: 'line', line: line.id };
        else if (kind === 'track') ui.selection.value = { kind: 'track', edge: w.edges[0].id };
        else if (kind === 'city') ui.selection.value = { kind: 'city', city: (await import('/src/game/cities.ts')).viewedZone.value };
        else if (kind === 'station') ui.selection.value = { kind: 'station', station: line.stops[0] };
        else if (kind === 'train') ui.selection.value = { kind: 'train', line: line.id, profile: 0, dep: Math.floor(window.tw.clock.now() / 86400) * 86400 + 8 * 3600 };
        else return false;
        return true;
      }, pick);
      if (ok) await shootDock(`pick-${pick}`, true);
      else console.log(`pick ${pick}: nothing to pick (no lines; add --save)`);
    }
    if (wholePage) {
      await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.selection.value = null; ui.tab.value = 'build'; });
      await page.waitForTimeout(800);
      const wide = await page.evaluate(() => [...document.querySelectorAll('#game button, #game .btn, #game .tab, #game .clock-text, #game .cash, #game .city')]
        .filter((el) => el.offsetParent && el.scrollWidth > el.clientWidth + 1).map((el) => el.textContent.trim().slice(0, 40)));
      const file = join(out, `page-${lang}.png`);
      await page.screenshot({ path: file });
      console.log(`${file}${wide.length ? `  clipped: ${wide.join(' | ')}` : ''}`);
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
for (const tab of [...tabs, ...picks.map((p) => `pick-${p}`)]) {
  const base = metrics[`${tab} en`];
  if (!base) continue;
  for (const lang of langs.filter((l) => l !== 'en')) {
    const m = metrics[`${tab} ${lang}`];
    if (!m) continue;
    const taller = m.rows.map((h, i) => [i, h - (base.rows[i] ?? h)]).filter(([, d]) => d > 1);
    console.log(`${tab} ${lang}: pane ${m.paneHeight - base.paneHeight >= 0 ? '+' : ''}${m.paneHeight - base.paneHeight}px against English; ${taller.length ? `${taller.length} rows taller (${taller.map(([i, d]) => `#${i} +${d}`).join(', ')})` : 'no row taller'}; font ${m.font}`);
  }
}
