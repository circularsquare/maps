// T-3319: a line's bullet shape. On the real save (no shapes: every line takes its mode's), a few lines
// are given their own shapes; a mode's default from Settings changes only the lines without one; picking
// the mode's shape gives a line back to it; a copy keeps its source's shape; and the shapes survive the
// save file and its load. Shots of the Lines tab, the line editor and Settings, light and dark, and the
// drawn sizes of each shape at each bullet size. From the maps root:
//   node trainworld/checks/t3319-line-shapes.mjs [url]   (about two minutes; default localhost:6172)
// Screenshots to artifacts/trainworld/t3319/.
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, GL, launchChrome, collectErrors, importSave } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6172/';
const OUT = join(ARTIFACTS, 't3319');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const chrome = await launchChrome({ prefix: 'tw-t3319-', size: [1400, 1000], args: GL, deviceScaleFactor: 2 });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { if (!sessionStorage.getItem('t3319')) { sessionStorage.setItem('t3319', '1'); localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 14 })); } } catch {} });
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 4, undefined, { timeout: 120000 });
  await page.waitForTimeout(3000);
  await page.evaluate(async () => {
    window.__m = { st: await import('/src/game/lineShapeState.ts'), ui: await import('/src/game/ui.ts'), w: await import('/src/game/world.ts') };
  });
  // the page's CSP refuses eval, so each step goes as an expression through the protocol
  const S = (f, arg) => page.evaluate(`(async () => (${f.toString()})({ ...window.__m, tw: window.tw }, ${JSON.stringify(arg ?? null)}))()`);
  const shade = async (s) => { await page.evaluate((s) => { window.tw.shade.value = s; }, s); await page.waitForTimeout(400); };

  // 1. an old save: no shapes, every rail line a circle
  const start = await S(({ st, tw }) => {
    const lines = tw.world.value.save.lines;
    return { own: Object.keys(st.lineShapes.value).length, shapes: [...new Set(lines.map((l) => st.shapeOf(l.id)))], ids: lines.slice(0, 6).map((l) => [l.id, l.num, l.name, l.letter]) };
  });
  console.log('lines', JSON.stringify(start.ids));
  check(start.own === 0, `old save loads with no shapes of its own (${start.own})`);
  check(start.shapes.length === 1 && start.shapes[0] === 'circle', `every line takes its mode's shape: ${start.shapes}`);
  const [a, b, c, d, e] = start.ids.map((x) => x[0]);

  // 2. shapes set by hand
  const want = { [a]: 'square', [b]: 'diamond', [c]: 'pill', [d]: 'diamond' };
  await S(({ st }, want) => { for (const [id, s] of Object.entries(want)) st.setLineShape(id, s); }, want);
  const set = await S(({ st }, ids) => ids.map((id) => st.shapeOf(id)), [a, b, c, d, e]);
  check(JSON.stringify(set) === JSON.stringify(['square', 'diamond', 'pill', 'diamond', 'circle']), `shapes set: ${set}`);

  // 3. a mode's default changes only the lines without their own
  const mode = await S(({ st, ui }, ids) => {
    const was = ui.display.peek().modeShapes;
    ui.setDisplay('modeShapes', { ...was, train: 'square' });
    const out = ids.map((id) => st.shapeOf(id));
    ui.setDisplay('modeShapes', was);
    return out;
  }, [b, e]);
  check(mode[0] === 'diamond' && mode[1] === 'square', `train default square: own diamond kept, plain line square (${mode})`);

  // 4. picking the mode's shape gives the line back to it
  const back = await S(({ st }, id) => { st.setLineShape(id, 'circle'); const r = id in st.lineShapes.peek(); st.setLineShape(id, 'diamond'); return r; }, d);
  check(!back, 'picking the mode\'s shape clears the line\'s own');

  // 5. a copy keeps its source's shape
  const copyId = await S(async ({ w, tw }, id) => {
    const l = tw.world.value.save.lines.find((x) => x.id === id);
    const r = await w.edit({ op: 'copyLine', line: l.num });
    return r.ok && r.line !== undefined ? String(r.line) : null;
  }, b);
  if (copyId) await page.waitForFunction((id) => window.__m.st.lineShapes.value[id] !== undefined, copyId, { timeout: 30000, polling: 200 }).catch(() => {});
  const copyShape = copyId ? await S(({ st }, id) => st.lineShapes.peek()[id] ?? null, copyId) : null;
  check(copyShape === 'diamond', `the copy (line ${copyId}) keeps the diamond: ${copyShape}`);
  if (copyId) { await S(async ({ w }, num) => w.edit({ op: 'removeLine', line: Number(num) }), copyId); await page.waitForTimeout(800); }

  // 6. shots and sizes, with a two-character diamond and square too
  await S(({ st, tw }) => { for (const [n, s] of [['6X', 'diamond'], ['7X', 'square']]) { const l = tw.world.value.save.lines.find((x) => x.name === n); if (l) st.setLineShape(l.id, s); } });
  await S(({ ui }) => { ui.tab.value = 'lines'; ui.selection.value = null; });
  await page.waitForTimeout(1200);
  const sizes = await page.evaluate(() => {
    const out = {};
    for (const el of document.querySelectorAll('.chip > svg')) {
      const o = el.firstElementChild, bb = o.getBBox(), sw = o.tagName === 'path' ? Number(o.getAttribute('stroke-width')) : 0;
      const size = el.parentElement.classList.contains('lg') ? 'lg' : el.parentElement.classList.contains('sm') ? 'sm' : 'md';
      const key = `${size} ${o.tagName === 'path' ? 'diamond' : Number(o.getAttribute('rx')) >= bb.height / 2 - 0.01 ? (bb.width > bb.height * 1.4 ? 'pill' : 'circle') : 'square'}`;
      out[key] ??= `${(bb.width + sw).toFixed(1)} x ${(bb.height + sw).toFixed(1)}`;
    }
    return out;
  });
  console.log('drawn sizes (w x h, px):', JSON.stringify(sizes));
  for (const s of ['light', 'dark']) {
    await shade(s);
    await page.mouse.move(700, 900);
    await page.screenshot({ path: join(OUT, `lines-tab-${s}.png`), clip: { x: 0, y: 0, width: 440, height: 560 } });
  }
  await shade('light');
  // a station panel's small bullets: the stop of line 2 with the most lines
  await S(({ w, tw }, id) => {
    const l = tw.world.value.save.lines.find((x) => x.id === id);
    const best = l.stops.map((sid) => [sid, w.linesAt(sid).length]).sort((p, q) => q[1] - p[1])[0];
    tw.selection.value = { kind: 'station', station: best[0] };
  }, b);
  await page.waitForTimeout(1500);
  for (const s of ['light', 'dark']) {
    await shade(s);
    await page.mouse.move(700, 900);
    await page.screenshot({ path: join(OUT, `station-${s}.png`), clip: { x: 0, y: 0, width: 440, height: 760 } });
  }
  await shade('light');
  await S(({ tw }, id) => { tw.selection.value = { kind: 'line', line: id }; }, b);
  await page.waitForSelector('button.badge-edit', { timeout: 30000 });
  await page.click('button.badge-edit');
  await page.waitForSelector('.line-shape .group.shapes', { timeout: 10000 });
  await page.waitForTimeout(600);
  const picked = await page.$$eval('.line-shape .btn', (bs) => bs.map((x) => (x.classList.contains('on') ? 'on' : '') + ':' + (x.getAttribute('data-tip') ?? '')));
  check(picked[2].startsWith('on') && picked[0].includes('Default for train lines'), `the editor shows the diamond picked and the circle as the train default: ${JSON.stringify(picked)}`);
  const box = await page.evaluate(() => { const r = document.querySelector('.line-identity').getBoundingClientRect(), t = document.querySelector('.title').getBoundingClientRect(); return { x: 0, y: Math.max(0, t.top - 8), width: 440, height: r.bottom - t.top + 16 }; });
  for (const s of ['light', 'dark']) {
    await shade(s);
    await page.hover('.line-shape .btn:first-child');
    await page.waitForTimeout(400);
    await page.screenshot({ path: join(OUT, `editor-${s}.png`), clip: { ...box, height: box.height + 40 } });
  }
  await shade('light');
  await page.keyboard.press('Escape');
  await S(({ ui, tw }) => { tw.selection.value = null; ui.tab.value = 'settings'; });
  await page.waitForTimeout(800);
  await page.evaluate(() => { const h = [...document.querySelectorAll('.h')].find((x) => x.textContent === 'Line shapes'); h?.scrollIntoView({ block: 'center' }); });
  await page.waitForTimeout(500);
  const sbox = await page.evaluate(() => { const h = [...document.querySelectorAll('.h')].find((x) => x.textContent === 'Line shapes'); const r = h.getBoundingClientRect(); return { x: 0, y: Math.max(0, r.top - 120), width: 440, height: 320 }; });
  for (const s of ['light', 'dark']) {
    await shade(s);
    await page.mouse.move(700, 900);
    await page.screenshot({ path: join(OUT, `settings-${s}.png`), clip: sbox });
  }
  await shade('light');

  // 7. the shapes are in the save file and come back with it
  // a deleted line's shape stays in hand (for an undo) but is not saved
  const expected = await S(({ st, tw }) => { const ids = new Set(tw.world.value.save.lines.map((l) => l.id)); return Object.fromEntries(Object.entries(st.lineShapes.peek()).filter(([id]) => ids.has(id))); });
  check(copyId === null || !(copyId in expected), 'the deleted copy is left out');
  const bytes = await page.evaluate(async () => { const b = await (await import('/src/workers/clockClient.ts')).exportForShare(); return b ? Array.from(b) : null; });
  check(!!bytes, 'save exported');
  const header = await page.evaluate(async (data) => (await import('/src/workers/saveFile.ts')).decodeSave(Uint8Array.from(data)), bytes).then((r) => r?.header ?? r, () => null);
  console.log('header shapes', JSON.stringify(header?.shapes));
  check(JSON.stringify(header?.shapes ?? null) === JSON.stringify(expected), `the save's header holds the shapes: ${JSON.stringify(header?.shapes)}`);
  await S(({ st }) => { st.takeLineShapes({}); });
  const loadsBefore = await page.evaluate(() => window.tw.client.lastLoad);
  await importSave(page, Buffer.from(bytes));
  await page.waitForFunction((was) => window.tw.client.lastLoad && window.tw.client.lastLoad !== was, loadsBefore, { timeout: 120000 });
  await page.waitForFunction((n) => Object.keys(window.__m.st.lineShapes.value).length === n, Object.keys(expected).length, { timeout: 30000, polling: 200 }).catch(() => {});
  const after = await S(({ st }) => ({ ...st.lineShapes.peek() }));
  check(JSON.stringify(after) === JSON.stringify(expected), `the shapes come back with the load: ${JSON.stringify(after)}`);
  await page.waitForTimeout(1500);
  await S(({ ui }) => { ui.tab.value = 'lines'; });
  await page.waitForTimeout(1000);
  await page.screenshot({ path: join(OUT, 'lines-tab-reloaded.png'), clip: { x: 0, y: 0, width: 440, height: 560 } });
} finally {
  await chrome.close();
}
const real = errors.filter((e) => !/Failed to load resource/.test(e));
check(real.length === 0, `no page errors${real.length ? ': ' + real.slice(0, 5).join(' | ') : ''}`);
console.log(fail ? `${fail} FAILED` : 'all passed');
process.exit(fail ? 1 : 0);
