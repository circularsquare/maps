// T-1928, T-1930, T-1931 in a new game: the level price table (a ghost at each level on flat ground
// in Brooklyn, its tag read and photographed in both themes), the 30 km water rule (a viaduct and a
// cut and cover tunnel over the Dover Strait refused, 25 km of it and a bored tunnel not), and the
// upkeep rates by level. About two minutes.
// Usage: node trainworld/checks/old/t1928-prices.mjs [url] (default http://localhost:6170/)

import { mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from '../lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6170/';
const sep = BASE.includes('?') ? '&' : '?';
const OUT = join(ARTIFACTS, 't1928');
mkdirSync(OUT, { recursive: true });
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) failed++; };

const LEVELS = [3, 2, 1, 0, -1, -2, -3];
// US$M a km of double track at each level on flat ground (params.rs LEVEL_MULT x 75)
const PER_KM = { 3: 150, 2: 112.5, 1: 75, 0: 37.5, [-1]: 75, [-2]: 120, [-3]: 150 };
const MULT = { 3: 2, 2: 1.5, 1: 1, 0: 0.5, [-1]: 1, [-2]: 1.6, [-3]: 2 };

const { ctx, close } = await launchChrome({ prefix: 'tw-t1928-' });
try {
  const page = await ctx.newPage();
  const errors = [];
  collectErrors(page, errors);
  await page.goto(`${BASE}${sep}debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });

  const px = (x, y) => page.evaluate(([x, y]) => { const r = window.tw.map.getCanvas().getBoundingClientRect(), p = window.tw.map.project([x, y]); return [r.left + p.x, r.top + p.y]; }, [x, y]);
  const tagText = () => page.evaluate(() => [...document.querySelectorAll('.tool-tag')].map((t) => t.textContent).join(' | '));

  // ---- a ghost at each level, 1.5 km east from Flatbush
  const lng0 = -73.955, lat0 = 40.64, dLng = 1.5 / (111.32 * Math.cos((lat0 * Math.PI) / 180));
  await page.evaluate(([lng, lat]) => void window.tw.map.jumpTo({ center: [lng, lat], zoom: 13.5 }), [lng0 + dLng / 2, lat0]);
  await page.waitForTimeout(2500);
  const shots = { light: [], dark: [] };
  for (const theme of ['light', 'dark']) {
    await page.evaluate((t) => { window.tw.shade.value = t; }, theme);
    for (let k = 0; k < LEVELS.length; k++) {
      const level = LEVELS[k];
      await page.keyboard.press('Escape');
      await page.keyboard.press('1');
      await page.keyboard.press(`Shift+Digit${k + 1}`);
      await page.waitForTimeout(300);
      const [x0, y0] = await px(lng0, lat0), [x1, y1] = await px(lng0 + dLng, lat0);
      await page.mouse.move(x0, y0);
      await page.mouse.click(x0, y0);
      await page.waitForTimeout(400);
      await page.mouse.move(x1, y1, { steps: 6 });
      await page.waitForTimeout(2000);
      await page.mouse.move(x1 + 1, y1, { steps: 1 });
      await page.waitForTimeout(1000);
      const tag = await tagText();
      const m = /cost (\d+\.\d+)x, \$([\d.,]+)([kMB])/.exec(tag);
      if (theme === 'light') {
        const mult = m && Number(m[1]);
        const usd = (n, u) => Number(n.replace(/,/g, '')) * (u === 'B' ? 1000 : u === 'k' ? 0.001 : 1);
        // T-1929: the tag's price includes the roads crossed at ground level (T-2017: "22 crossings +$132M"), which this check leaves out
        const roads = /crossings? \+\$([\d.,]+)([kMB])/.exec(tag);
        const dollars = m && usd(m[2], m[3]) - (roads ? usd(roads[1], roads[2]) : 0);
        const km = Number(/([\d.]+) km/.exec(tag)?.[1] ?? NaN);
        check(!!m && Math.abs(mult - MULT[level]) < 0.011 && Math.abs(dollars / km - PER_KM[level]) < PER_KM[level] * 0.03 + 1, `level ${level}: tag "${tag}" (${MULT[level]}x, about ${PER_KM[level]} a km)`);
      }
      const clip = { x: Math.max(0, x1 - 260), y: Math.max(0, y1 - 150), width: 520, height: 300 };
      const file = join(OUT, `ghost-${theme}-${level}.png`);
      await page.screenshot({ path: file, clip });
      shots[theme].push([level, file]);
      await page.keyboard.press('Escape');
    }
  }
  // a contact sheet per theme
  for (const theme of ['light', 'dark']) {
    const html = `<body style="margin:0;background:${theme === 'light' ? '#ddd' : '#222'};display:grid;grid-template-columns:repeat(4,520px);gap:6px">` +
      shots[theme].map(([l, f]) => `<img src="data:image/png;base64,${readFileSync(f).toString('base64')}" title="${l}">`).join('') + '</body>';
    const sheet = await ctx.newPage();
    await sheet.setViewportSize({ width: 2130, height: 620 });
    await sheet.setContent(html);
    await sheet.screenshot({ path: join(OUT, `ghosts-${theme}.png`) });
    await sheet.close();
  }

  // ---- the 30 km water rule across the Dover Strait
  const dover = [1.30, 51.12], gris = [1.60, 50.87];
  const along = (t) => [dover[0] + (gris[0] - dover[0]) * t, dover[1] + (gris[1] - dover[1]) * t];
  const draw = (t0, t1, level) => page.evaluate(async ([a, b, level]) => {
    const tw = window.tw, c = tw.client;
    tw.map.jumpTo({ center: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], zoom: 9 });
    await new Promise((d) => setTimeout(d, 2500));
    const p = tw.lngLatToPack(a[0], a[1]), q = tw.lngLatToPack(b[0], b[1]);
    const r = await c.edit({ op: 'route', from: { kind: 'free', x: p.x, y: p.y, level }, to: { kind: 'free', x: q.x, y: q.y, level }, pis: [], single: false });
    return { ok: r.ok, issues: r.issues };
  }, [along(t0), along(t1), level]);
  // the ghost of a viaduct over it, with its reason, in both themes (before anything is drawn there)
  await page.evaluate(([c]) => void window.tw.map.jumpTo({ center: c, zoom: 9.5 }), [along(0.5)]);
  await page.waitForTimeout(2500);
  for (const theme of ['light', 'dark']) {
    await page.evaluate((t) => { window.tw.shade.value = t; }, theme);
    await page.keyboard.press('Escape');
    await page.keyboard.press('1');
    await page.keyboard.press('Shift+Digit3');
    const [x0, y0] = await px(...along(-0.1)), [x1, y1] = await px(...along(1.1));
    await page.mouse.move(x0, y0);
    await page.mouse.click(x0, y0);
    await page.waitForTimeout(500);
    await page.mouse.move(x1, y1, { steps: 8 });
    await page.waitForTimeout(3000);
    await page.mouse.move(x1 + 1, y1, { steps: 1 });
    await page.waitForTimeout(1500);
    const tag = await tagText();
    if (theme === 'light') check(/open water/.test(tag), `the ghost's tag gives the reason (${tag})`);
    await page.screenshot({ path: join(OUT, `strait-${theme}.png`) });
    await page.keyboard.press('Escape');
  }
  const long = await draw(-0.15, 1.15, 1);
  check(!long.ok && long.issues.includes('WaterTooLong'), `a 35 km viaduct over the strait is refused (${JSON.stringify(long)})`);
  const cut = await draw(-0.15, 1.15, -1);
  check(!cut.ok && cut.issues.includes('WaterTooLong'), `cut and cover over it is refused too (${JSON.stringify(cut)})`);
  const bored = await draw(-0.15, 1.15, -2);
  check(bored.ok, `a bored tunnel over it is allowed (${JSON.stringify(bored)})`);
  const short = await draw(0.3, 1.15, 1);
  check(short.ok, `a viaduct over the last 25 km of it is not (${JSON.stringify(short)})`);
  check(errors.length === 0, `no page errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
