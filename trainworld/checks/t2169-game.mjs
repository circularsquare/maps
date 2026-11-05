// T-2169: the network diagram's water in the game itself: each save imported into a new game in headless Chrome, the
// diagram opened, and pictures taken once the water is in (light, and dark with --dark), with the water's times.
//   node trainworld/checks/t2169-game.mjs <url> <views.json> [--dark]
//   views: [{ save: "nyc" | "tor" | "tyo", name, at: "<station name>", px: <a pitch in CSS px>, w?, h? }]
//   (pictures in artifacts/trainworld/t2169/game-<name>-<light|dark>.png)
import { readFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { ARTIFACTS, REAL_SAVE, NOISE, launchChrome, collectErrors, importSave } from './lib.mjs';

const base = process.argv[2];
const views = JSON.parse(readFileSync(process.argv[3]));
const dark = process.argv.includes('--dark');
const FILES = { nyc: REAL_SAVE, tyo: fileURLToPath(new URL('../data/packs/real/tyo.save', import.meta.url)), tor: fileURLToPath(new URL('../data/packs/real/tor.save', import.meta.url)) };
const shade = dark ? 'dark' : 'light';
for (const save of [...new Set(views.map((v) => v.save))]) {
  const errors = [];
  const { ctx, close } = await launchChrome({ prefix: `tw-diagwater-${save}-`, size: [1600, 1100] });
  try {
    const page = await ctx.newPage();
    await page.addInitScript((s) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: s, v: 2 })); } catch {} }, shade);
    collectErrors(page, errors, NOISE);
    const sep = base.includes('?') ? '&' : '?';
    await page.goto(`${base}${sep}debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
    await importSave(page, await readFile(FILES[save]));
    await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 180000 });
    await page.waitForTimeout(1500);
    await page.keyboard.press('m');
    await page.waitForFunction(() => window.twDiagram?.water?.().ms, undefined, { timeout: 600000, polling: 1000 });
    const info = await page.evaluate(() => ({ layout: window.twDiagram.ms, water: window.twDiagram.water() }));
    console.log(`${save}: layout ${Math.round(info.layout.total ?? 0)} ms; water ${JSON.stringify(info.water.ms, (k, v) => (typeof v === 'number' ? Math.round(v) : v))}; rivers ${JSON.stringify(info.water.stats)}; sea ${JSON.stringify(info.water.sea)}`);
    for (const v of views.filter((q) => q.save === save)) {
      await page.setViewportSize({ width: v.w ?? 1600, height: v.h ?? 1100 });
      const ok = await page.evaluate(([n, px]) => window.twDiagram.centreOn(n, px), [v.at, v.px]);
      if (!ok) console.log(`${save}: no station ${v.at}`);
      await page.waitForTimeout(400);
      const file = `${process.env.WATER_OUT ?? `${ARTIFACTS}t2169/`}game-${v.name}-${shade}.png`;
      await page.addStyleTag({ content: '#debug-root, .busy-tag, .status-tag, .loading-card-wrap, .loading-screen { display: none !important; }' });
      const r = await page.locator('.diagram-canvas').boundingBox();
      await page.screenshot({ path: file, clip: { x: r.x, y: r.y + 50, width: r.width, height: r.height - 50 } });
      console.log(file);
    }
    if (errors.length) console.log(`${save}: page errors: ${errors.slice(0, 3).join(' | ')}`);
  } finally {
    await close();
  }
}
