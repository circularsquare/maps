// T-1495: the game's hover tags with coloured rows (a station with its line bullets and ridership dots,
// the track tool's cost, level and speed chips) on the real save, light and dark. Screenshots at 3x into
// artifacts/trainworld/t1491/ as t1495-<label>-<what>-<shade>.png; BEFORE=1 puts the old tag style back.
//   node trainworld/checks/t1495-tags.mjs <label> [url]     (from the maps root; a running dev server)
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, NOISE, REAL_SAVE, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';

const LABEL = process.argv[2] ?? 'after';
const BASE = process.argv[3] ?? 'http://localhost:6171/';
const OUT = join(ARTIFACTS, 't1491');
await mkdir(OUT, { recursive: true });

for (const shade of ['light', 'dark']) {
  const { ctx, close } = await launchChrome({ prefix: 'tw-t1495-', size: [1300, 800], deviceScaleFactor: 3 });
  const errors = [];
  try {
    const page = await ctx.newPage();
    await page.addInitScript((s) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: s, display: { dayShown: 'weekday' }, v: 2 })); } catch {} }, shade);
    collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
    await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
    await page.addStyleTag({ content: '.loading-card-wrap { display: none !important; }' });
    // BEFORE=1: the tags as they were before T-1495
    if (process.env.BEFORE) await page.addStyleTag({ content: '.tool-tag { padding: 0 4px !important; } .tag-chip { background: var(--chip-bg) !important; }' });
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
    await importSave(page, await readFile(REAL_SAVE));
    await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
    await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
    await demandFinal(page, { strict: false, timeout: 240000 });
    await page.waitForTimeout(2000);

    // the station with the most lines, and a point just along one of its tracks
    const t = await page.evaluate(() => {
      const w = window.tw.world.value, count = new Map(), by = new Map(w.save.stations.map((x) => [x.id, x]));
      for (const l of w.save.lines) for (let i = 0; i + 1 < l.stops.length; i++) {
        const a = by.get(l.stops[i]), b = by.get(l.stops[i + 1]);
        if (!a || !b || !a.built || !b.built) continue;
        const k = [a.id, b.id].sort().join('-');
        const c = count.get(k) ?? { n: new Set(), a, b };
        c.n.add(l.id);
        count.set(k, c);
      }
      const dist = (c) => Math.hypot((c.a.lng - c.b.lng) * 84000, (c.a.lat - c.b.lat) * 111000);
      const best = [...count.values()].filter((c) => dist(c) > 900 && dist(c) < 2500).sort((x, y) => y.n.size - x.n.size || dist(y) - dist(x))[0];
      const f = 0.45;
      return { n: best.n.size, station: [best.a.lng, best.a.lat], on: [best.a.lng + (best.b.lng - best.a.lng) * f, best.a.lat + (best.b.lat - best.a.lat) * f] };
    });
    console.log(shade, JSON.stringify(t));
    await page.evaluate((c) => {
      window.tw.map.jumpTo({ center: c, zoom: 16 });
      window.tw.display.value = { ...window.tw.display.value, stationUse: 'total' };
      window.tw.overlay.invalidate();
    }, t.on);
    await page.waitForTimeout(2500);
    const xy = (p) => page.evaluate((q) => { const r = window.tw.map.getContainer().getBoundingClientRect(), pt = window.tw.map.project(q); return [r.left + pt.x, r.top + pt.y]; }, p);
    const shot = async (what) => {
      await page.waitForTimeout(600);
      const box = await page.evaluate(() => {
        const tags = [...document.querySelectorAll('.tool-tag')].filter((e) => getComputedStyle(e).display !== 'none' && getComputedStyle(e).visibility !== 'hidden').map((e) => e.getBoundingClientRect()).filter((r) => r.width > 0 && r.height > 0 && Math.abs(r.left - window.__px) < 300 && Math.abs(r.top - window.__py) < 300);
        if (!tags.length) return null;
        const x0 = Math.min(...tags.map((r) => r.left)), y0 = Math.min(...tags.map((r) => r.top)), x1 = Math.max(...tags.map((r) => r.right)), y1 = Math.max(...tags.map((r) => r.bottom));
        return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
      });
      if (!box) { console.log(`${shade} ${what}: no tag`, JSON.stringify(await page.evaluate(() => ({ p: [window.__px, window.__py], tags: [...document.querySelectorAll('.tool-tag')].map((e) => [e.className, getComputedStyle(e).display, JSON.stringify(e.getBoundingClientRect())]) })))); return; }
      const clip = { x: Math.max(0, box.x - 16), y: Math.max(0, box.y - 16), width: box.w + 32, height: box.h + 32 };
      await page.screenshot({ path: join(OUT, `t1495-${LABEL}-${what}-${shade}.png`), clip });
      console.log(`${shade} ${what}: tag ${Math.round(box.w)}x${Math.round(box.h)}`);
    };
    const settled = async () => {
      await demandFinal(page, { timeout: 30000 }).catch(() => {});
      await page.waitForTimeout(1500);
    };
    const hover = async (p, off = [0, 0], zoom = 16) => {
      await page.evaluate(([c, z]) => { window.tw.map.jumpTo({ center: c, zoom: z }); window.tw.overlay.invalidate(); }, [p, zoom]);
      await page.waitForTimeout(1200);
      await settled();
      const [x, y] = await xy(p);
      await page.evaluate(([a, b]) => { window.__px = a; window.__py = b; }, [x, y]);
      await page.mouse.move(x + off[0] + 30, y + off[1] + 30);
      await page.mouse.move(x + off[0], y + off[1], { steps: 4 });
    };

    // a station: its line bullets and its ridership dots
    await hover(t.station);
    await page.waitForTimeout(1500);
    await shot('station');

    // the track tool: a cost chip and the level and speed chips
    await page.mouse.move(5, 400);
    await page.keyboard.press('1');
    await page.waitForTimeout(500);
    await page.evaluate(() => { window.tw.map.jumpTo({ center: [-73.78, 40.74], zoom: 14.5 }); window.tw.overlay.invalidate(); });
    await page.waitForTimeout(1500);
    const [sx, sy] = await xy([-73.78, 40.74]);
    await page.mouse.move(sx + 40, sy + 40);
    await page.mouse.move(sx, sy, { steps: 3 });
    await page.mouse.click(sx, sy);
    await page.waitForTimeout(400);
    await page.mouse.move(sx + 120, sy - 90, { steps: 6 });
    await page.evaluate(([a, b]) => { window.__px = a; window.__py = b; }, [sx + 120, sy - 90]);
    await page.waitForTimeout(1500);
    await shot('track-tool');
    await page.keyboard.press('Escape');
    await page.keyboard.press('Escape');
    console.log(errors.length ? 'errors: ' + [...new Set(errors)].slice(0, 3).join(' / ') : 'no console errors');
  } finally {
    await close();
  }
}
