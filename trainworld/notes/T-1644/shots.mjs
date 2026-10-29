// T-1644: screenshots of the map canvas (basemap and overlay together) at fixed views, clock paused.
// argv: tag (before|after), CDP port env. Output: artifacts/trainworld/t1644/<tag>-<view>.png
import { mkdir, writeFile } from 'node:fs/promises';
import { playwright } from 'file:///C:/Users/anita/projects/maps/trainworld/checks/lib.mjs';
const PORT = process.env.PORT ?? '6124', CDP = process.env.CDP ?? '9344';
const tag = process.argv[2];
const OUT = 'C:/Users/anita/projects/maps/artifacts/trainworld/t1644/';
await mkdir(OUT, { recursive: true });
const { chromium } = await playwright();
const browser = await chromium.connectOverCDP(`http://127.0.0.1:${CDP}`);
const page = browser.contexts()[0].pages().find((p) => p.url().includes(PORT));
const VIEWS = [
  ['tokyo15', { center: [139.767, 35.681], zoom: 15 }],
  ['shinjuku16', { center: [139.7005, 35.6905], zoom: 16 }],
  ['osaka15', { center: [135.4959, 34.7025], zoom: 15 }],
  ['tokyo14', { center: [139.767, 35.681], zoom: 14 }],
  ['tokyo13', { center: [139.767, 35.681], zoom: 13 }],
  ['tokyo15sel', { center: [139.767, 35.681], zoom: 15, sel: true }],
  ['tokyo15dark', { center: [139.767, 35.681], zoom: 15, shade: 'dark' }],
];
await page.evaluate(() => { const tw = window.tw; tw.clock.setRunning(false); tw.display.value = { ...tw.display.value, projection: 'globe', hillshade: true }; });
for (const [name, v] of VIEWS) {
  await page.evaluate(async (v) => {
    const tw = window.tw;
    tw.shade.value = v.shade ?? 'light';
    tw.map.jumpTo({ center: v.center, zoom: v.zoom });
    tw.select(null);
    await new Promise((r) => setTimeout(r, 5000));
    if (v.sel) {
      const n = tw.renderer.net, s = n.lines.seg, w = tw.overlay.cssW, h = tw.overlay.cssH;
      let best = null, bd = Infinity;
      for (let i = 0; i < n.lines.count; i += 1) {
        const [px, py] = tw.overlay.toScreen((s[i * 4] + s[i * 4 + 2]) / 2, (s[i * 4 + 1] + s[i * 4 + 3]) / 2);
        const d = Math.hypot(px - w / 2, py - h / 2);
        if (d < bd) { bd = d; best = n.lines.colour[i]; }
      }
      tw.select({ kind: 'line', line: String(best) });
      await new Promise((r) => setTimeout(r, 1500));
    }
  }, v);
  const box = await page.evaluate(() => { const r = window.tw.map.getCanvas().getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; });
  await writeFile(`${OUT}${tag}-${name}.png`, await page.screenshot({ clip: box }));
  console.log(name, 'ok');
}
await page.evaluate(() => { window.tw.shade.value = 'light'; window.tw.select(null); });
process.exit(0);
