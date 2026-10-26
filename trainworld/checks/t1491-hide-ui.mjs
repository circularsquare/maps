// T-1491: F2 hides the whole interface (dock, top bar, map control rows, trouble list, tool notes,
// MapLibre's controls) and brings it back; Esc brings it back too; the map takes the room and the
// network stays. A new game with a line on it. Screenshots into artifacts/trainworld/t1491/.
//   node trainworld/checks/t1491-hide-ui.mjs [url]     (from the maps root; a running dev server)
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, NOISE, acrossLine, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6171/';
const OUT = join(ARTIFACTS, 't1491');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

for (const shade of ['light', 'dark']) {
  const { ctx, close } = await launchChrome({ prefix: 'tw-t1491-', size: [1300, 800] });
  const errors = [];
  try {
    const page = await ctx.newPage();
    await page.addInitScript((s) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: s, v: 2 })); } catch {} }, shade);
    collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
    await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools, undefined, { timeout: 180000 });
    await page.waitForTimeout(2000);
    await acrossLine(page, [-73.95, 40.70], 3);
    await page.evaluate(() => { window.tw.map.jumpTo({ center: [-73.95, 40.70], zoom: 12.5 }); window.tw.overlay.invalidate(); });
    await page.waitForTimeout(2500);
    const state = () => page.evaluate(() => {
      const vis = (sel) => [...document.querySelectorAll(sel)].some((e) => getComputedStyle(e).display !== 'none' && e.getBoundingClientRect().width > 0);
      const m = document.getElementById('map').getBoundingClientRect();
      return { hidden: document.documentElement.classList.contains('ui-hidden'), dock: vis('#dock'), bar: vis('#topbar'), controls: vis('.map-top-left'), ctrl: vis('#map .maplibregl-control-container'), search: vis('.map-search'), mapW: Math.round(m.width), mapH: Math.round(m.height), canvasW: window.tw.map.getCanvas().clientWidth, canvasH: window.tw.map.getCanvas().clientHeight };
    });
    const shot = (name) => page.screenshot({ path: join(OUT, `t1491-${name}-${shade}.png`) });

    const a = await state();
    check(!a.hidden && a.dock && a.bar && a.controls, `${shade}: interface shown at the start (${JSON.stringify(a)})`);
    await shot('shown');

    await page.keyboard.press('F2');
    await page.waitForTimeout(800);
    const b = await state();
    check(b.hidden && !b.dock && !b.bar && !b.controls && !b.ctrl && !b.search, `${shade}: F2 hides dock, bar, map controls (${JSON.stringify(b)})`);
    check(b.mapW > a.mapW && b.mapH > a.mapH && b.canvasW === b.mapW && b.canvasH === b.mapH, `${shade}: the map takes the room and its canvas follows`);
    await shot('hidden');

    await page.keyboard.press('F2');
    await page.waitForTimeout(500);
    const c = await state();
    check(!c.hidden && c.dock && c.bar && c.controls && c.mapW === a.mapW && c.canvasW === a.canvasW, `${shade}: F2 again brings it back (${JSON.stringify(c)})`);

    await page.keyboard.press('F2');
    await page.waitForTimeout(400);
    await page.keyboard.press('Escape');
    await page.waitForTimeout(500);
    const d = await state();
    check(!d.hidden && d.dock && d.bar, `${shade}: Esc brings it back`);

    // the Controls panel lists F2
    await page.evaluate(async () => { (await import('/src/ui/HelpPanel.tsx')).helpOpen.value = true; });
    await page.waitForTimeout(400);
    const listed = await page.evaluate(() => [...document.querySelectorAll('.help-panel dt')].some((e) => e.textContent === 'F2'));
    check(listed, `${shade}: the Controls panel lists F2`);
    if (shade === 'light') await page.screenshot({ path: join(OUT, 't1491-controls-light.png'), clip: { x: 280, y: 30, width: 460, height: 620 } });
    check(errors.length === 0, `${shade}: no console errors ${errors.slice(0, 3).join(' | ')}`);
  } finally {
    await close();
  }
}
console.log(fail ? `${fail} FAILED` : 'ALL PASS');
process.exit(fail ? 1 : 0);
