// T-734: on the real save, with the track tool, a press and drag on a blueprint node moves it and a click still places track.
//   node trainworld/checks/t734-track-drag.mjs [url]   (from the maps root; default localhost:6112)
import { mkdtemp, rm, readdir, readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const BASE = process.argv[2] ?? 'http://localhost:6112/';
async function playwright() {
  try { return await import('playwright'); } catch { /* */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright');
}
const { chromium } = await playwright();
const profile = await mkdtemp(join(tmpdir(), 'tw-t734-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1200, height: 800 }, args: ['--use-angle=d3d11'] });
let failed = false;
const check = (ok, msg) => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${msg}`); };
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const bytes = await readFile(fileURLToPath(new URL('../T-007-real-nyc.save', import.meta.url)));
  await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(3000);
  const setup = await page.evaluate(async () => {
    const tw = window.tw, map = tw.map, tools = tw.tools;
    const ui = await import('/src/game/ui.ts');
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const raf = () => new Promise((r) => requestAnimationFrame(() => r()));
    const st = [...tw.world.value.nodes.values()].filter((n) => n.platform > 0).sort((a, b) => a.id - b.id)[5];
    map.jumpTo({ center: [st.lng, st.lat], zoom: 15.5 });
    await raf(); await raf();
    const before = new Set(tw.world.value.nodes.keys());
    ui.tool.value = 'track';
    const c = map.project(map.getCenter());
    for (const [x, y] of [[c.x - 180, c.y - 160], [c.x - 60, c.y - 170]]) {
      tools.click({ point: { x, y }, lngLat: map.unproject([x, y]), originalEvent: { detail: 1 } });
      await sleep(1500);
    }
    const W = await import('/src/game/world.ts'); window.__note = W.notice.value?.text; tools.reset();
    await sleep(300);
    const w = tw.world.value;
    const ids = [...w.nodes.values()].filter((n) => !before.has(n.id)).map((n) => ({ id: n.id, lng: n.lng, lat: n.lat }));
    return { ids, tool: ui.tool.value, edges: w.edges.length, note: window.__note };
  });
  console.log('setup', JSON.stringify(setup));
  check(setup.ids.length >= 2, 'a blueprint segment was placed with the track tool');
  const nodePos = (id) => page.evaluate((id) => { const tw = window.tw; const n = tw.world.value.nodes.get(id); const p = tw.map.project([n.lng, n.lat]); const r = tw.map.getContainer().getBoundingClientRect(); return { x: p.x + r.left, y: p.y + r.top, lng: n.lng, lat: n.lat }; }, id);
  const state = () => page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); const tw = window.tw; return { tool: ui.tool.value, start: !!tw.tools.start, edges: tw.world.value.edges.length, nodes: tw.world.value.nodes.size }; });
  const target = setup.ids[1].id;
  const p0 = await nodePos(target);
  // drag
  await page.mouse.move(p0.x, p0.y);
  await page.mouse.down();
  await page.mouse.move(p0.x + 20, p0.y + 30, { steps: 5 });
  await page.mouse.move(p0.x + 60, p0.y + 70, { steps: 5 });
  await page.waitForTimeout(500);
  await page.mouse.up();
  await page.waitForTimeout(1500);
  const p1 = await nodePos(target);
  const s1 = await state();
  console.log('drag', JSON.stringify({ p0, p1, s1 }));
  const movedPx = Math.hypot(p1.x - p0.x, p1.y - p0.y);
  check(Math.abs(movedPx - Math.hypot(60, 70)) < 6, `dragging the node with the track tool moved it (${movedPx.toFixed(1)} px)`);
  check(s1.tool === 'track' && !s1.start && s1.edges === setup.edges, 'no track was placed by the drag, the track tool stays on');
  // click on the node: a start is set
  await page.mouse.click(p1.x, p1.y);
  await page.waitForTimeout(1000);
  const s2 = await state();
  console.log('click', JSON.stringify(s2));
  check(s2.start, 'a click on the node starts track from it');
  // a small wobble under 3 px is still a click: second click on free ground places the segment
  const c = { x: p1.x + 130, y: p1.y + 10 };
  await page.mouse.click(c.x, c.y);
  await page.waitForTimeout(2000);
  const s3 = await state();
  console.log('segment', JSON.stringify(s3));
  check(s3.edges > setup.edges, 'a second click places the segment');
  // press and drag on a start node mid-run (start not at the node): dragging the node does not drop the run
  const s4 = await state();
  check(s4.start, 'the run continues from the segment end');
  const t2 = setup.ids[0].id;
  const q0 = await nodePos(t2);
  await page.mouse.move(q0.x, q0.y);
  await page.mouse.down();
  await page.mouse.move(q0.x - 20, q0.y - 20, { steps: 5 });
  await page.mouse.move(q0.x - 40, q0.y - 40, { steps: 5 });
  await page.waitForTimeout(500);
  await page.mouse.up();
  await page.waitForTimeout(1500);
  const q1 = await nodePos(t2);
  const s5 = await state();
  console.log('drag2', JSON.stringify({ q0, q1, s5 }));
  check(Math.hypot(q1.x - q0.x, q1.y - q0.y) > 40, 'dragging another node with a run in progress moves it');
  check(s5.start, 'and the run stays');
} catch (e) {
  failed = true;
  console.log('FAIL  ' + e.message);
} finally {
  await ctx.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
process.exitCode = failed ? 1 : 0;
