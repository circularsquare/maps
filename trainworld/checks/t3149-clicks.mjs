// T-3149: the click standard (docs/ui-building.md "Clicks while drawing a route or editing stops"), walked on Anita's
// Amsterdam save. Bus route and ferry route, five points drawn out and a fresh draft per case: left and right click
// on the first, a middle, the point before the last, the last and a lone point, a double click on a point and on
// empty ground, a finger held on a point, the touch bar's Back and the Loop button; each case asserts the result and
// the tag text over the point. Then a ring (Edit route on a closed line): a right click on a point opens it, and
// the rail stops of Line 1a (an open line): left click removes the first, a middle and the last stop, right click on
// the last says it cannot loop, a finger held does the same, and the Loop button path.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3149-clicks.mjs [dev url]      ONLY=bus|ferry|rail picks one part
// About ten minutes.
import { ARTIFACTS, collectErrors, demandFinal, launchChrome } from './lib.mjs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6199/';
const SAVE = process.env.SAVE ?? join(ARTIFACTS, 't3099-anita-amsterdam.save');
const ONLY = process.env.ONLY ?? '';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = await launchChrome({ prefix: 'tw-t3149-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);
const ready = () => page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 300000 });

// ---------------------------------------------------------------- helpers
const lines = () => page.evaluate(() => window.tw.world.value.save.lines.length);
const centre = (lng, lat, zoom) => page.evaluate(([lng, lat, zoom]) => window.tw.map.jumpTo({ center: [lng, lat], zoom }), [lng, lat, zoom]);
const settle = async () => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await wait(500); };
const toolName = () => page.evaluate(async () => (await import('/src/game/ui.ts')).tool.value);
const clearPrompt = () => page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.renameLinePrompt.value = null; });
const touch = (on) => page.evaluate(async (on) => { (await import('/src/map/touch.ts')).touchInput.value = on; }, on);
/** the visible tag over the pointer, rows joined by " / " */
const tag = () => page.evaluate(() => [...document.querySelectorAll('.build-tag, .stop-tag')].filter((t) => t.style.display !== 'none').map((t) => t.innerText.replace(/\n/g, ' / ')).join(' // '));

/** the two route tools behind one set of calls: how many points, where one is on screen, how to start a draft */
const TOOLS = {
  bus: {
    count: () => page.evaluate(() => window.tw.tools.bus.pts.length),
    where: (i) => page.evaluate((i) => { const p = window.tw.tools.bus.pts[i], s = window.tw.map.project([p.lng, p.lat]), r = window.tw.map.getContainer().getBoundingClientRect(); return [r.left + s.x, r.top + s.y]; }, i),
    start: () => page.evaluate(async () => (await import('/src/game/lineActions.ts')).newBusLine()),
    idle: () => page.waitForFunction(() => !window.tw.tools.bus.busy, undefined, { timeout: 30000 }),
    closed: () => page.evaluate(() => window.tw.tools.bus.closed()),
    word: 'point',
  },
  ferry: {
    count: () => page.evaluate(() => window.tw.tools.ferry.ports.length),
    where: (i) => page.evaluate((i) => { const p = window.tw.tools.ferry.ports[i].landing, s = window.tw.map.project([p.lng, p.lat]), r = window.tw.map.getContainer().getBoundingClientRect(); return [r.left + s.x, r.top + s.y]; }, i),
    start: () => page.evaluate(async () => (await import('/src/game/lineActions.ts')).newFerryLine()),
    idle: () => page.waitForFunction(() => !window.tw.tools.ferry.busy, undefined, { timeout: 30000 }),
    closed: () => page.evaluate(() => window.tw.tools.ferry.closed()),
    word: 'port',
  },
};

/** a finger held on a screen point for ms, then lifted */
const hold = (x, y, ms = 700) => page.evaluate(async ([x, y, ms]) => {
  const c = window.tw.map.getCanvas();
  const ev = (type) => new PointerEvent(type, { pointerType: 'touch', pointerId: 7, isPrimary: true, clientX: x, clientY: y, bubbles: true, cancelable: true, composed: true });
  c.dispatchEvent(ev('pointerdown'));
  await new Promise((r) => setTimeout(r, ms));
  c.dispatchEvent(ev('pointerup'));
}, [x, y, ms]);

/** wait until `test()` holds, up to 4 s; the last answer */
async function until(read, test, ms = 4000) {
  const t0 = Date.now();
  let v = await read();
  while (!test(v) && Date.now() - t0 < ms) { await wait(150); v = await read(); }
  return v;
}

const DRAW = {
  // pixels from the map's centre, drawn out and round a block (as t3109)
  bus: [[-300, 120], [-60, 120], [180, 120], [180, -120], [-60, -120]],
  // set below once the harbour is known: lng, lat
  ferry: null,
};

/** a fresh draft of `n` points */
async function fresh(kind, n) {
  const T = TOOLS[kind];
  await page.evaluate(() => window.tw.tools.exit());
  await touch(false);
  await clearPrompt();
  await T.start();
  await wait(500);
  const box = await page.evaluate(() => { const r = window.tw.map.getContainer().getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; });
  const spots = kind === 'bus'
    ? DRAW.bus.map(([dx, dy]) => [box.x + box.w / 2 + dx, box.y + box.h / 2 + dy])
    : DRAW.ferry.map(([lng, lat]) => page.evaluate(([lng, lat]) => { const s = window.tw.map.project([lng, lat]), r = window.tw.map.getContainer().getBoundingClientRect(); return [r.left + s.x, r.top + s.y]; }, [lng, lat]));
  for (let i = 0; i < n; i++) {
    const [x, y] = await spots[i];
    await page.mouse.move(x - 4, y - 4);
    await page.mouse.move(x, y, { steps: 2 });
    await wait(kind === 'bus' ? 700 : 900);
    await page.mouse.click(x, y);
    await until(T.count, (c) => c >= i + 1, 8000);
    await T.idle();
    await wait(300);
  }
  return T.count();
}

/** hover a draft point and read the tag */
async function hoverTag(kind, i) {
  const [x, y] = await TOOLS[kind].where(i);
  await page.mouse.move(x + 30, y + 30);
  await page.mouse.move(x, y, { steps: 3 });
  await wait(600);
  return tag();
}

/** a click on a bar button by the page itself (Playwright's own click is intercepted by the page) */
const press = (loc) => loc.evaluate((e) => e.click());
const has = (s, re) => re.test(s);
const lacks = (s, re) => !re.test(s);

async function routeTool(kind) {
  const T = TOOLS[kind], W = T.word;
  const base = await lines();
  const made = (n = 1) => until(lines, (c) => c >= base + n, 30000);
  console.log(`---- ${kind}`);

  // tags over the first, a middle, the one before the last, the last
  let n = await fresh(kind, 5);
  check(n === 5, `${kind}: five ${W}s drawn`, String(n));
  let s = await hoverTag(kind, 4);
  check(has(s, /^Click to make the line/) && has(s, /Right click to delete/) && lacks(s, /Right click to (loop|close|open)/), `${kind}: over the last ${W}: make the line, right click deletes`, s);
  s = await hoverTag(kind, 2);
  check(has(s, /Click to delete/) && has(s, /Right click to loop back here/), `${kind}: over a middle ${W}: click deletes, right click loops back`, s);
  s = await hoverTag(kind, 1);
  check(has(s, /Click to delete/) && has(s, /Right click to loop back here/), `${kind}: over the second ${W}: click deletes, right click loops back`, s);
  s = await hoverTag(kind, 3);
  check(has(s, /Click to delete/) && lacks(s, /Right click/), `${kind}: over the ${W} before the last: click deletes, no right click`, s);
  s = await hoverTag(kind, 0);
  check(has(s, /Click to delete/) && has(s, /Right click to close loop/), `${kind}: over the first ${W}: click deletes, right click closes the loop`, s);

  // left click: a middle point goes, nothing is saved
  let [x, y] = await T.where(2);
  await page.mouse.click(x, y);
  n = await until(T.count, (c) => c === 4);
  check(n === 4 && (await lines()) === base, `${kind}: left click on a middle ${W} deletes it and saves nothing`, `${n} ${W}s`);

  // left click on the first point deletes it
  await fresh(kind, 5);
  [x, y] = await T.where(0);
  await page.mouse.click(x, y);
  n = await until(T.count, (c) => c === 4);
  check(n === 4 && (await lines()) === base, `${kind}: left click on the first ${W} deletes it`, `${n} ${W}s`);

  // left click on the last point saves the route
  await fresh(kind, 5);
  [x, y] = await T.where(4);
  await page.mouse.click(x, y);
  let c = await made();
  check(c === base + 1, `${kind}: left click on the last ${W} makes the line`, `${c - base} new`);
  await clearPrompt();

  // right click on the point before the last does nothing; on the last it deletes it
  await fresh(kind, 5);
  [x, y] = await T.where(3);
  await page.mouse.click(x, y, { button: 'right' });
  await wait(1200);
  check((await T.count()) === 5 && (await lines()) === base + 1, `${kind}: right click on the ${W} before the last does nothing`);
  [x, y] = await T.where(4);
  await page.mouse.click(x, y, { button: 'right' });
  n = await until(T.count, (c) => c === 4);
  check(n === 4 && (await lines()) === base + 1, `${kind}: right click on the last ${W} deletes it`, `${n} ${W}s`);

  // right click on empty ground takes back the last point
  const box = await page.evaluate(() => { const r = window.tw.map.getContainer().getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; });
  await page.mouse.click(box.x + box.w / 2 + 300, box.y + box.h / 2 - 300, { button: 'right' });
  n = await until(T.count, (c) => c === 3);
  check(n === 3, `${kind}: right click on empty ground takes back the last ${W}`, `${n} ${W}s`);

  // right click on a middle point loops back and saves
  await fresh(kind, 5);
  [x, y] = await T.where(1);
  await page.mouse.click(x, y, { button: 'right' });
  c = await made(2);
  check(c === base + 2, `${kind}: right click on a middle ${W} loops back and makes the line`, `${c - base} new`);
  await clearPrompt();

  // right click on the first point closes the loop and saves
  await fresh(kind, 5);
  [x, y] = await T.where(0);
  await page.mouse.click(x, y, { button: 'right' });
  c = await made(3);
  check(c === base + 3, `${kind}: right click on the first ${W} closes the loop and makes the line`, `${c - base} new`);
  const ring = await page.evaluate(() => { const w = window.tw.world.value, l = w.save.lines[w.save.lines.length - 1]; return { id: l.id, num: l.num, colour: l.colour, loop: w.lineStats[l.id].loop }; });
  check(ring.loop === 'circle' || ring.loop === true || !!ring.loop, `${kind}: that line is a loop`, JSON.stringify(ring.loop));
  await clearPrompt();

  // the ring, drawn again for editing: a right click on any point opens it, and says so
  await page.evaluate(async (ring) => {
    const a = await import('/src/game/lineActions.ts'), w = window.tw.world.value, l = w.save.lines.find((x) => x.num === ring.num);
    (ring.kind === 'bus' ? a.editBusRoute : a.editFerryRoute)(l);
  }, { ...ring, kind });
  await wait(1500);
  n = await T.count();
  check(await T.closed(), `${kind}: Edit route on it shows a ring`, `${n} ${W}s`);
  s = await hoverTag(kind, 2);
  check(has(s, /Right click to open loop/), `${kind}: over a ${W} of a ring the tag offers to open it`, s);
  [x, y] = await T.where(2);
  await page.mouse.click(x, y, { button: 'right' });
  await until(T.closed, (v) => !v);
  check(!(await T.closed()) && (await T.count()) === n - 1, `${kind}: right click on a ${W} of the ring opens it there`, `${await T.count()} ${W}s`);
  await page.evaluate(() => window.tw.tools.exit());

  // a double click acts on empty ground: it saves; on a middle point it deletes it and saves nothing
  const before = await lines();
  await fresh(kind, 5);
  [x, y] = await T.where(2);
  await page.mouse.dblclick(x, y);
  await wait(2500);
  n = await T.count();
  check(n === 4 && (await lines()) === before && (await toolName()) === kind, `${kind}: double click on a middle ${W} deletes it once and saves nothing`, `${n} ${W}s, ${(await lines()) - before} new`);

  await fresh(kind, 5);
  const empty = await (kind === 'bus' ? page.evaluate(() => { const r = window.tw.map.getContainer().getBoundingClientRect(); return [r.left + r.width / 2 + 330, r.top + r.height / 2 + 120]; }) : page.evaluate(() => { const r = window.tw.map.getContainer().getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; }));
  if (kind === 'bus') {
    await page.mouse.dblclick(empty[0], empty[1]);
    c = await until(lines, (v) => v >= before + 1, 30000);
    check(c === before + 1, 'bus: double click on empty ground adds a point and saves the line', `${c - before} new`);
    await clearPrompt();
  } else {
    // the ferry's empty ground is the shore: a point not near a placed port
    const [ex, ey] = await page.evaluate(() => { const p = window.tw.tools.ferry.ports.at(-1).landing, s = window.tw.map.project([p.lng, p.lat]), r = window.tw.map.getContainer().getBoundingClientRect(); return [r.left + s.x + 160, r.top + s.y]; });
    await page.mouse.dblclick(ex, ey);
    c = await until(lines, (v) => v >= before + 1, 30000);
    check(c === before + 1, 'ferry: double click on empty ground saves the line', `${c - before} new, ${await T.count()} ports`);
    await clearPrompt();
  }

  // a lone point: the tag has no click row, a click does nothing, a right click deletes it
  await fresh(kind, 1);
  s = await hoverTag(kind, 0);
  check(lacks(s, /Click to/), `${kind}: over a lone ${W} the tag has no click row`, s);
  check(has(s, /Right click to delete/), `${kind}: over a lone ${W} the tag offers a right click to delete`, s);
  [x, y] = await T.where(0);
  await page.mouse.click(x, y);
  await wait(1200);
  check((await T.count()) === 1, `${kind}: left click on a lone ${W} does nothing`);
  await page.mouse.click(x, y, { button: 'right' });
  n = await until(T.count, (v) => v === 0);
  check(n === 0 && (await toolName()) === kind, `${kind}: right click on a lone ${W} deletes it`, `${n} ${W}s`);

  // a finger: held on the last point deletes it; held on a middle point loops back
  const b2 = await lines();
  await fresh(kind, 5);
  [x, y] = await T.where(4);
  await hold(x, y);
  n = await until(T.count, (v) => v === 4);
  check(n === 4, `${kind}: a finger held on the last ${W} deletes it`, `${n} ${W}s`);
  [x, y] = await T.where(1);
  await hold(x, y);
  c = await until(lines, (v) => v >= b2 + 1, 30000);
  check(c === b2 + 1, `${kind}: a finger held on a middle ${W} loops back and makes the line`, `${c - b2} new`);
  await clearPrompt();

  // the touch bar: Back takes back the last point, and a lone one
  await fresh(kind, 3);
  await touch(true);
  await wait(500);
  const back = page.locator('.touch-bar button.back');
  check((await back.count()) === 1, `${kind}: on touch the bar has a Back button`);
  await press(back);
  n = await until(T.count, (v) => v === 2);
  check(n === 2, `${kind}: Back takes back the last ${W}`, `${n} ${W}s`);
  const loopBtn = page.locator('#tool-notes button', { hasText: /^Loop$/ });
  check((await loopBtn.count()) === 1, `${kind}: on touch the bar has a Loop button with two ${W}s`);
  await press(back);
  await until(T.count, (v) => v === 1);
  check((await loopBtn.count()) === 0, `${kind}: no Loop button with one ${W}`);
  await press(back);
  n = await until(T.count, (v) => v === 0);
  check(n === 0, `${kind}: Back takes back a lone ${W}`, `${n} ${W}s`);

  // the Loop button, then a tap on a point: the first closes the loop, a middle loops back, and a ring opens
  const b3 = await lines();
  await fresh(kind, 5);
  await touch(true);
  await wait(500);
  await press(loopBtn);
  s = await page.evaluate(() => document.querySelector('#tool-notes')?.innerText ?? '');
  check(has(s, new RegExp(`Tap a ${W} to loop there`)), `${kind}: the Loop button's hint says what a tap does`, s.replace(/\n/g, ' / '));
  [x, y] = await T.where(0);
  await page.mouse.click(x, y);
  c = await until(lines, (v) => v >= b3 + 1, 30000);
  check(c === b3 + 1, `${kind}: Loop, then a tap on the first ${W}, closes the loop and makes the line`, `${c - b3} new`);
  await clearPrompt();
  await touch(false);
}

async function railStops() {
  console.log('---- rail');
  const info = (name) => page.evaluate((name) => {
    const w = window.tw.world.value, l = w.save.lines.find((x) => x.name === name), nm = new Map(w.save.stations.map((s) => [s.id, s.name]));
    return { num: l.num, stops: l.stops.map((s) => nm.get(s)), loop: w.lineStats[l.id].loop };
  }, name);
  const editStops = (num) => page.evaluate(async (num) => {
    const tw = window.tw, ui = await import('/src/game/ui.ts'), l = tw.world.value.save.lines.find((x) => x.num === num);
    tw.selection.value = { kind: 'line', line: l.id };
    ui.tool.value = 'line';
    ui.lineDraft.value = { line: num, stops: l.stops.map(Number) };
  }, num);
  const markerOf = (name) => page.evaluate((name) => {
    const tw = window.tw, s = tw.world.value.save.stations.find((x) => x.name === name), m = tw.tools.stops.marks.get(s.id), r = tw.map.getContainer().getBoundingClientRect();
    return m && m.shown ? { x: r.left + m.x, y: r.top + m.y } : null;
  }, name);
  const goTo = (name, zoom = 14.5) => page.evaluate(([name, zoom]) => {
    const tw = window.tw, s = tw.world.value.save.stations.find((x) => x.name === name), n = tw.world.value.nodes.get(Number(s.id));
    tw.map.jumpTo({ center: tw.packToLngLat ? tw.packToLngLat(n.x, n.y) : [n.lng, n.lat], zoom });
  }, [name, zoom]);
  const undo = () => page.evaluate(() => window.tw.client.edit({ op: 'undo' }));
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

  const l = await info('Line 1a');
  console.log('Line 1a', JSON.stringify(l.stops), l.loop);
  await editStops(l.num);
  await wait(2500);
  const first = l.stops[0], last = l.stops.at(-1), mid = l.stops[Math.floor(l.stops.length / 2)];
  const stopsOf = async () => (await info('Line 1a')).stops;
  for (const [what, name] of [['the last', last], ['a middle', mid], ['the first', first]]) {
    await goTo(name);
    await wait(2500);
    let m = await markerOf(name);
    await page.mouse.move(m.x, m.y);
    await wait(500);
    const s = await tag();
    check(/Click to remove the stop/.test(s), `rail: over ${what} stop (${name}) the tag offers to remove it`, s);
    await page.mouse.click(m.x, m.y);
    await wait(2200);
    const after = await stopsOf();
    check(after.length === l.stops.length - 1, `rail: left click on ${what} stop removes it`, JSON.stringify(after));
    await undo();
    await wait(2200);
    check(same(await stopsOf(), l.stops), 'rail: undo puts it back');
  }
  // right click on the last stop: nothing happens to the line, a word says why
  await goTo(last);
  await wait(2500);
  let m = await markerOf(last);
  await page.mouse.click(m.x, m.y, { button: 'right' });
  await wait(1200);
  let s = await tag();
  check(/Can't loop back here/.test(s) && same(await stopsOf(), l.stops), 'rail: right click on the last stop changes nothing and says why', s);
  // a finger held does the same as a right click
  await page.mouse.move(5, 5);
  await wait(2500);
  await hold(m.x, m.y);
  await wait(800);
  s = await tag();
  check(/Can't loop back here/.test(s) && same(await stopsOf(), l.stops), 'rail: a finger held on the last stop does the same as a right click', s);
  // double click on a stop: it goes and comes back, or the reverse: the line is as it was
  await wait(2500);
  m = await markerOf(mid);
  await goTo(mid);
  await wait(2500);
  m = await markerOf(mid);
  await page.mouse.dblclick(m.x, m.y);
  await wait(3000);
  check(same(await stopsOf(), l.stops), 'rail: double click on a stop leaves the line as it was');
  // the Loop button: on, then a tap on a stop does a right click's work (the last stop: a word)
  await touch(true);
  await wait(500);
  const loopBtn = page.locator('#tool-notes button', { hasText: /^Loop$/ });
  check((await loopBtn.count()) === 1, 'rail: on touch the bar has a Loop button');
  await press(loopBtn);
  await goTo(last);
  await wait(2500);
  m = await markerOf(last);
  await page.mouse.click(m.x, m.y);
  await wait(1200);
  s = await tag();
  check(/Can't loop back here/.test(s) && same(await stopsOf(), l.stops), 'rail: Loop, then a tap on the last stop says it cannot loop', s);
  await touch(false);
  // a tier that does something: the first stop of Line 1 closes or opens its loop (t3109 walks the rest)
  const l1 = await info('Line 1');
  await editStops(l1.num);
  await goTo(l1.stops[1]);
  await wait(2500);
  m = await markerOf(l1.stops[1]);
  await page.mouse.move(m.x, m.y);
  await wait(500);
  s = await tag();
  check(/Click to remove the stop/.test(s) && /Right click to open loop/.test(s), 'rail: over a stop of a loop the tag offers to remove it and to open the loop', s);
  await page.mouse.click(m.x, m.y, { button: 'right' });
  await wait(2500);
  const opened = await info('Line 1');
  check(!opened.loop && opened.stops[0] !== opened.stops.at(-1), 'rail: right click on a stop of a loop opens it', JSON.stringify(opened.stops));
  await undo();
  await wait(2000);
  m = await markerOf(l1.stops[1]);
  await hold(m.x, m.y);
  await wait(2500);
  const heldOpen = await info('Line 1');
  check(!heldOpen.loop, 'rail: a finger held on a stop of a loop opens it', JSON.stringify(heldOpen.stops));
  await undo();
  await wait(1500);
}

try {
  await page.goto(`${BASE}?debug=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await ready();
  const bytes = await readFile(SAVE);
  await page.evaluate(async (data) => { const m = await import('/src/workers/clockClient.ts'); await m.importGame(new File([Uint8Array.from(data)], 'game.save')); }, Array.from(bytes));
  await page.waitForURL(/load=file/, { timeout: 120000 });
  await ready();
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000, polling: 100 });
  await demandFinal(page, { timeout: 600000 });
  await wait(1000);

  if (!ONLY || ONLY === 'bus') {
    await centre(4.8997, 52.3702, 15.2);
    await settle();
    await routeTool('bus');
  }
  if (!ONLY || ONLY === 'ferry') {
    // the IJ east of Centraal station: five ports along its two shores
    DRAW.ferry = JSON.parse(process.env.FERRY ?? '[[4.9010,52.3838],[4.9060,52.3790],[4.9160,52.3830],[4.9280,52.3830],[4.9380,52.3860]]');
    await centre(4.9170, 52.3830, 13.5);
    await settle();
    await routeTool('ferry');
  }
  if (!ONLY || ONLY === 'rail') await railStops();
} catch (e) {
  console.log('FAIL', e.stack ?? e.message);
  fail++;
}
check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
await chrome.close();
console.log(fail ? `T-3149 FAILED (${fail})` : 'T-3149 PASSED');
process.exit(fail ? 1 : 0);
