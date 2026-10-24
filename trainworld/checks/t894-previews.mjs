// T-894: save previews. In a fresh browser profile, a game with lines in New York, Boston and
// Philadelphia, then the real New York save loaded from its file; each slot's picture and panels are
// read back, and the saves panel, the start menu and a new empty game's saves panel are screenshot to
// artifacts/trainworld/previews/ (SHADE=dark for the dark theme).
//   SHADE=dark node trainworld/checks/t894-previews.mjs [url]   (from the maps root; default localhost:6120)
// Prints PASS or FAIL lines and exits 1 on a failure. A few minutes.
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchChrome, buildLine, demandFinal, REAL_SAVE, ANGLE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6120/';
const SHADE = process.env.SHADE ?? 'light';
const ART = fileURLToPath(new URL('../../artifacts/trainworld/previews/', import.meta.url));
await mkdir(ART, { recursive: true });

let failed = false;
const report = (name, ok, detail) => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

const { ctx, close } = await launchChrome({ prefix: 'tw-t894-', size: [1300, 900], args: ANGLE });
try {
  await ctx.addInitScript((shade) => { try { if (!localStorage.getItem('t894')) { localStorage.setItem('trainworld-ui', JSON.stringify({ shade, display: { dayShown: 'weekday' }, v: 2 })); localStorage.setItem('t894', '1'); } } catch {} }, SHADE);
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  const wait = (ms) => page.waitForTimeout(ms);
  const solved = () => demandFinal(page, { strict: false, polling: 300 }).catch(() => {});
  const flush = async () => { await page.evaluate(() => window.tw.client.post({ kind: 'flush' })); await wait(2500); };
  const build = async (pts, name, colour) => (await buildLine(page, pts, { name, colour })).ok;
  const go = async (zone, center) => {
    await page.evaluate((c) => void window.tw.map.jumpTo({ center: c, zoom: 12 }), center);
    await page.waitForFunction((zone) => window.twDemand.activeCity.peek() === zone, zone, { timeout: 60000, polling: 100 }).catch(() => {});
    await wait(2500);
    await solved();
  };
  const slots = () => page.evaluate(async () => (await (await import('/src/workers/saveFile.ts')).listSlots())
    .map((s) => ({ key: s.key, name: s.name, lines: s.lines, kb: s.thumb ? Math.round(s.thumb.length * 0.75 / 1024) : 0, shade: s.shade, panels: (s.panels ?? []).map((p) => `${p.city}:${p.stations}`).join(' ') })));
  const panelShot = async (file) => {
    await page.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = true; });
    await page.waitForSelector('.saves-panel .save-files');
    await wait(800);
    await page.locator('.saves-panel').screenshot({ path: join(ART, file) });
    await page.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = false; });
  };

  // 1. a game in three cities, in the profile's first slot
  await page.goto(`${BASE}?debug=1&continue=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await solved();
  const built = [];
  await go('nyc', [-73.99, 40.74]);
  built.push(await build([[-74.0110, 40.7128], [-73.9969, 40.7306], [-73.9903, 40.7506], [-73.9819, 40.7681], [-73.9680, 40.7990]], 'Eighth', '#e4002b'));
  built.push(await build([[-73.9897, 40.6926], [-73.9776, 40.6840], [-73.9615, 40.6712], [-73.9442, 40.6501]], 'Flatbush', '#0060a8'));
  await go('bos', [-71.07, 42.355]);
  built.push(await build([[-71.0589, 42.3601], [-71.0846, 42.3626], [-71.1190, 42.3736]], 'Red', '#d62c2c'));
  built.push(await build([[-71.0600, 42.3510], [-71.0700, 42.3300], [-71.0960, 42.3100]], 'Orange', '#f08a00'));
  await go('phl', [-75.18, 39.955]);
  built.push(await build([[-75.2200, 39.9600], [-75.1900, 39.9550], [-75.1630, 39.9520], [-75.1450, 39.9500]], 'Market', '#2a7fd4'));
  await flush();
  let s = await slots();
  const three = s[0];
  report('three cities saved', built.every(Boolean) && three?.panels?.split(' ').length === 3, `(lines built ${built.join(',')}; ${JSON.stringify(three)})`);

  // 2. the real save, loaded from its file into a slot of its own
  const before = await page.evaluate(() => window.tw.lastSaved.peek());
  await page.evaluate(async () => { window.__cc = await import('/src/workers/clockClient.ts'); });
  const [fc] = await Promise.all([page.waitForEvent('filechooser'), page.evaluate(async () => {
    (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = true;
    await new Promise((r) => setTimeout(r, 500));
    [...document.querySelectorAll('.saves-panel .save-files button')].find((b) => b.textContent === 'Load a file').click();
  })]);
  await fc.setFiles(REAL_SAVE);
  await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 100, undefined, { timeout: 240000 });
  await solved();
  await flush();
  await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, before, { timeout: 60000 }).catch(() => {});
  s = await slots();
  const real = s.find((x) => x.name === 'T-007-real-nyc');
  // New Haven and the other fringe zones hold under 8% of its stations: New York alone
  report('real save pictured', real && real.kb > 0 && /^nyc:\d+$/.test(real.panels), `(${JSON.stringify(real)})`);
  report('pictures are small', s.every((x) => x.kb < 120), `(${s.map((x) => `${x.name} ${x.kb} KB`).join(', ')})`);
  report('drawn in the theme', s.every((x) => !x.kb || x.shade === SHADE), `(${s.map((x) => x.shade).join(', ')})`);
  await panelShot(`saves-${SHADE}.png`);

  // 3. the start menu, then a new empty game from it (not listed while empty, T-677)
  await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
  await page.waitForSelector('.start-menu .start-saves');
  await wait(800);
  const labels = await page.locator('.start-menu .thumb-label').count();
  await page.locator('.start-box').screenshot({ path: join(ART, `start-${SHADE}.png`) });
  report('start menu names the cities', labels === 4, `(${labels} names over pictures)`);
  await page.click('.start-menu .start-new');
  await page.click('.start-menu .new-game .start-new');
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.world.value.save.lines.length === 0, undefined, { timeout: 180000 });
  await wait(2000);
  await flush();
  await panelShot(`saves-newgame-${SHADE}.png`);
  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  console.log(`screenshots in ${ART}`);
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
