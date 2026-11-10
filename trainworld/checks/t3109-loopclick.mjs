// T-3109/T-3110: one click per job while a line's stops are edited, on Anita's Amsterdam save (Line 1 turned round
// a loop at each end, Line 1a beside it). Line 1's stop list is one row per station with the stops being edited; a
// click on Bankwerkerij's marker takes the stop out (and nothing else); a right click on a marker that offers a loop
// turns or opens it; a right click where there is none says so; Line 1a's related-lines strip while its stops are
// edited (screenshot). Then a bus route drawn out, round a block and joined back partway with a right click: the
// line is made, runs out, round and back, and a station on the way out is a stop both ways. Screenshots
// artifacts/trainworld/t3109-*.png.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3109-loopclick.mjs [dev url]
// About two minutes.
import { ARTIFACTS, collectErrors, demandFinal, launchChrome } from './lib.mjs';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6183/';
const SAVE = process.env.SAVE ?? join(ARTIFACTS, 't3099-anita-amsterdam.save');
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const chrome = await launchChrome({ prefix: 'tw-t3109-', size: [1400, 900], deviceScaleFactor: 2 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);
const out = { when: new Date().toISOString() };
const ready = () => page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 300000 });

/** a line's stops by name, its loop and the rows its panel lists */
const info = (name) => page.evaluate((name) => {
  const w = window.tw.world.value, l = w.save.lines.find((x) => x.name === name), st = w.lineStats[l.id];
  const nm = new Map(w.save.stations.map((s) => [s.id, s.name]));
  return { num: l.num, id: l.id, stops: l.stops.map((s) => nm.get(s)), ids: l.stops.map(Number), loop: st.loop, oneWay: st.oneWay, hand: st.hand };
}, name);
const rows = () => page.evaluate(() => [...document.querySelectorAll('ul.stops > li[data-stop] .stop-name')].map((e) => e.textContent.trim()));
const editStops = (num) => page.evaluate(async (num) => {
  const tw = window.tw, ui = await import('/src/game/ui.ts');
  const l = tw.world.value.save.lines.find((x) => x.num === num);
  tw.selection.value = { kind: 'line', line: l.id };
  ui.tool.value = 'line';
  ui.lineDraft.value = { line: num, stops: l.stops.map(Number) };
}, num);
/** the screen point of a station's marker, by name */
const markerOf = (name) => page.evaluate((name) => {
  const tw = window.tw, w = tw.world.value, s = w.save.stations.find((x) => x.name === name);
  const m = tw.tools.stops.marks.get(s.id);
  const r = tw.map.getContainer().getBoundingClientRect();
  return m && m.shown ? { x: r.left + m.x, y: r.top + m.y, id: s.id } : null;
}, name);
const centre = (name, zoom) => page.evaluate(([name, zoom]) => {
  const tw = window.tw, s = tw.world.value.save.stations.find((x) => x.name === name), n = tw.world.value.nodes.get(Number(s.id));
  tw.map.jumpTo({ center: tw.packToLngLat ? tw.packToLngLat(n.x, n.y) : [n.lng, n.lat], zoom });
}, [name, zoom]);
const tagText = () => page.evaluate(() => { const t = document.querySelector('.stop-tag'); return t && t.style.display !== 'none' ? t.innerText : ''; });
const undo = () => page.evaluate(() => window.tw.client.edit({ op: 'undo' }));

try {
  await page.goto(`${BASE}?debug=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await ready();
  const bytes = await readFile(SAVE);
  await page.evaluate(async (data) => { const m = await import('/src/workers/clockClient.ts'); await m.importGame(new File([Uint8Array.from(data)], 'game.save')); }, Array.from(bytes));
  await page.waitForURL(/load=file/, { timeout: 120000 });
  await ready();
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000, polling: 100 });
  await demandFinal(page, { timeout: 600000 });
  await page.waitForTimeout(1000);

  // ---- Line 1: its loops as loops, Exact edits gone
  const l1 = await info('Line 1');
  out.line1 = l1;
  console.log('Line 1', l1.stops.length, 'stops', JSON.stringify(l1.stops));
  await editStops(l1.num);
  await page.waitForTimeout(2500);
  const r1 = await rows();
  const distinct = new Set(l1.stops).size;
  check(r1.length === distinct, 'Line 1 lists each station once while its stops are edited', `${r1.length} rows, ${distinct} stations, ${l1.stops.length} stops`);
  check(!(await page.evaluate(() => [...document.querySelectorAll('button')].some((b) => /Exact edits/.test(b.textContent)))), 'no Exact edits button');
  await page.screenshot({ path: join(ARTIFACTS, 't3109-line1-editing.png') });

  // ---- left click on Bankwerkerij: the stop goes, nothing loops
  await centre('Bankwerkerij', 14.5);
  await page.waitForTimeout(2500);
  let mk = await markerOf('Bankwerkerij');
  check(!!mk, 'Bankwerkerij has a marker');
  await page.mouse.move(mk.x, mk.y);
  await page.waitForTimeout(500);
  const hover = await tagText();
  out.hoverBankwerkerij = hover;
  check(/Click to remove the stop/.test(hover), 'its hover offers to remove the stop', hover.replace(/\n/g, ' / '));
  await page.screenshot({ path: join(ARTIFACTS, 't3109-bankwerkerij-hover.png') });
  await page.mouse.click(mk.x, mk.y);
  await page.waitForTimeout(2000);
  const after = await info('Line 1');
  out.afterLeft = after;
  check(!after.stops.includes('Bankwerkerij') && after.stops.length >= l1.stops.length - 2, 'a click takes Bankwerkerij out and nothing else', JSON.stringify(after.stops));
  await undo();
  await page.waitForTimeout(2000);
  check(JSON.stringify((await info('Line 1')).stops) === JSON.stringify(l1.stops), 'undo puts it back');

  // ---- right click: the stations whose marker offers a loop
  const tiers = await page.evaluate(() => Object.fromEntries([...window.tw.tools.stops.tiers].filter(([, t]) => ['close', 'open', 'turn', 'unturn'].includes(t)).map(([id, t]) => [window.tw.world.value.save.stations.find((s) => s.id === id)?.name, t])));
  out.loopTiers = tiers;
  console.log('loop tiers on Line 1', JSON.stringify(tiers));
  const loopName = Object.keys(tiers)[0];
  if (loopName) {
    await centre(loopName, 14.5);
    await page.waitForTimeout(2500);
    mk = await markerOf(loopName);
    await page.mouse.move(mk.x, mk.y);
    await page.waitForTimeout(500);
    const t2 = await tagText();
    check(/Right click to/.test(t2), `${loopName}'s hover names its right click`, t2.replace(/\n/g, ' / '));
    await page.screenshot({ path: join(ARTIFACTS, 't3109-rightclick-hover.png') });
    await page.mouse.click(mk.x, mk.y, { button: 'right' });
    await page.waitForTimeout(2500);
    const r = await info('Line 1');
    out.afterRight = r;
    check(JSON.stringify(r.stops) !== JSON.stringify(l1.stops), `a right click on ${loopName} (${tiers[loopName]}) changes the loop`, JSON.stringify(r.stops));
    await undo();
    await page.waitForTimeout(2000);
  }
  // a right click on a plain stop: no edit, a word why
  const plain = l1.stops.find((s) => s !== 'Bankwerkerij' && !(s in tiers) && s !== l1.stops[0]);
  await centre(plain, 14.5);
  await page.waitForTimeout(2500);
  mk = await markerOf(plain);
  await page.mouse.click(mk.x, mk.y, { button: 'right' });
  await page.waitForTimeout(1200);
  const t3 = await tagText();
  check(/Can't loop back here/.test(t3) && JSON.stringify((await info('Line 1')).stops) === JSON.stringify(l1.stops), `a right click on ${plain} changes nothing and says why`, t3.replace(/\n/g, ' / '));

  // ---- T-3110: each line's related-lines strip, its stops edited and not: the other line's column runs beside it
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.relatedLines.value = true; });
  const colourOf = (name) => page.evaluate((name) => window.tw.world.value.save.lines.find((x) => x.name === name).colour, name);
  const shot = async (file, other) => {
    const list = page.locator('ul.stops').first();
    await list.scrollIntoViewIfNeeded();
    const b = await list.boundingBox();
    if (b) await page.screenshot({ path: join(ARTIFACTS, file), clip: { x: 0, y: Math.max(0, b.y - 40), width: 310, height: Math.min(900 - Math.max(0, b.y - 40), b.height + 50) } });
    // the other line's marks beside the list: bars across rows count apart from its column
    return page.evaluate((colour) => {
      const svg = document.querySelector('ul.related-rows')?.parentElement?.querySelector('svg');
      if (!svg) return null;
      const mine = [...svg.querySelectorAll('path, line, rect')].filter((e) => (e.getAttribute('stroke') ?? e.getAttribute('fill') ?? '').toLowerCase() === colour.toLowerCase());
      const across = mine.filter((e) => e.tagName === 'line' && e.getAttribute('y1') === e.getAttribute('y2')).length;
      return { across, parts: mine.length };
    }, await colourOf(other));
  };
  for (const [name, other, file] of [['Line 1a', 'Line 1', 't3109-line1a-related'], ['Line 1', 'Line 1a', 't3109-line1-related']]) {
    const l = await info(name);
    await editStops(l.num);
    await page.waitForTimeout(2500);
    const editing = await shot(`${file}-editing.png`, other);
    await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.tool.value = 'select'; });
    await page.waitForTimeout(1500);
    const plain = await shot(`${file}.png`, other);
    out[name] = { editing, plain };
    console.log(name, 'strip', JSON.stringify(editing), JSON.stringify(plain));
    check(!!editing && editing.across === plain?.across, `${name}'s strip draws ${other} alike with its stops edited or not`, JSON.stringify([editing, plain]));
  }

  // ---- a bus route out, round a block and joined back partway with a right click
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.tool.value = 'select'; ui.selection.value = null; });
  await centre('Nes', 15.2);
  await page.waitForTimeout(2500);
  const before = await page.evaluate(() => window.tw.world.value.save.lines.map((l) => l.num));
  await page.evaluate(async () => { const m = await import('/src/game/lineActions.ts'); m.newBusLine(); });
  await page.waitForTimeout(800);
  const box = await page.evaluate(() => { const r = window.tw.map.getContainer().getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; });
  const cx = box.x + box.w / 2, cy = box.y + box.h / 2;
  const pts = [[-300, 120], [-60, 120], [180, 120], [180, -120], [-60, -120]].map(([dx, dy]) => [cx + dx, cy + dy]);
  const idle = () => page.waitForFunction(() => !window.tw.tools.bus.busy, undefined, { timeout: 20000 });
  for (const [x, y] of pts) {
    await page.mouse.move(x, y);
    await page.waitForTimeout(700);
    await page.mouse.click(x, y);
    await page.waitForTimeout(900);
    await idle();
  }
  const drawn = await page.evaluate(() => window.tw.tools.bus.pts.length);
  check(drawn === 5, 'five points of a bus route are drawn', String(drawn));
  // hover the second point: its tag offers the loop back
  const p1 = await page.evaluate(() => { const b = window.tw.tools.bus, p = b.pts[1], s = window.tw.map.project([p.lng, p.lat]), r = window.tw.map.getContainer().getBoundingClientRect(); return [r.left + s.x, r.top + s.y]; });
  await page.mouse.move(p1[0], p1[1]);
  await page.waitForTimeout(600);
  const busTag = await page.evaluate(() => { const t = document.querySelector('.build-tag'); return t && t.style.display !== 'none' ? t.innerText : ''; });
  out.busTag = busTag;
  check(/Click to delete/.test(busTag) && /Right click to loop back here/.test(busTag), "the bus route's second point offers to delete and to loop back", busTag.replace(/\n/g, ' / '));
  await page.screenshot({ path: join(ARTIFACTS, 't3109-bus-loopback-hover.png') });
  await page.mouse.click(p1[0], p1[1], { button: 'right' });
  await page.waitForFunction((n) => window.tw.world.value.save.lines.length > n, before.length, { timeout: 30000 }).catch(() => null);
  await page.waitForTimeout(1500);
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.renameLinePrompt.value = null; });
  const bus = await page.evaluate((before) => {
    const w = window.tw.world.value, l = w.save.lines.find((x) => !before.includes(x.num)), st = l && w.lineStats[l.id];
    if (!st?.bus) return null;
    const ll = st.bus.routeLL, n = ll.length >> 1, same = (i, j) => ll[2 * i] === ll[2 * j] && ll[2 * i + 1] === ll[2 * j + 1];
    let k = 0;
    while (2 * (k + 1) < n - 1 && same(k + 1, n - 2 - k)) k++;
    return { num: l.num, points: n, closed: same(0, n - 1), back: k, stops: st.bus.stops.length, loop: st.loop };
  }, before);
  out.bus = bus;
  check(!!bus && bus.closed && bus.back >= 1 && bus.loop === 'circle', 'a right click partway along makes the line, out, round the loop and back', JSON.stringify(bus));
  if (bus) {
    // its route opened for editing again: the way out once, the loop, the point it closes at
    await page.evaluate(async (num) => {
      const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num), st = w.lineStats[l.id];
      window.tw.tools.bus.begin(num, Array.from(st.bus.routeLL), l.colour, Array.from(st.bus.kinds ?? []));
    }, bus.num);
    await page.waitForTimeout(800);
    const again = await page.evaluate(() => window.tw.tools.bus.pts.length);
    check(again === 6, 'editing it again shows its five points and the join', String(again));
    await page.screenshot({ path: join(ARTIFACTS, 't3109-bus-loopback.png') });
    await page.keyboard.press('Escape');
  }
} catch (e) {
  console.log('FAIL', e.message);
  fail++;
}
check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
out.errors = errors;
await writeFile(join(ARTIFACTS, 't3109-loopclick.json'), JSON.stringify(out, null, 1));
await chrome.close();
console.log(fail ? `T-3109 FAILED (${fail})` : 'T-3109 PASSED');
process.exit(fail ? 1 : 0);
