// T-3203: a right click closes an open line into a loop at its last stop as well as its first. On a new game in
// New York: an octagon of track with a station at five of its corners, a line over four-and-a-half sides whose two
// ends are the open stations next to each other. At each end, in stop editing, the marker says "Right click to close loop"
// and the right click closes the line into a circle (the first stop added at the end); undo opens it again.
// Screenshots artifacts/trainworld/t3203-*.png.
// From the maps root, on a Vite of your own:   node trainworld/checks/t3203-close-either-end.mjs [dev url]
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6115/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const C = [-73.93, 40.668];

const chrome = await launchChrome({ prefix: 'tw-t3203-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);
const edit = (op) => page.evaluate((op) => window.tw.client.edit(op), op);

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  const built = await page.evaluate(async (c) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const R = 2200, pts = Array.from({ length: 8 }, (_, k) => ({ x: o.x + R * Math.cos((k * Math.PI) / 4), y: o.y + R * Math.sin((k * Math.PI) / 4) }));
    const free = (p) => ({ kind: 'free', x: p.x, y: p.y, level: -1 });
    const at = (n) => ({ kind: 'node', node: n });
    const res = [];
    const go = async (from, to) => { res.push(await cl.edit({ op: 'route', from, to, pis: [], single: false })); await pause(300); };
    await go(free(pts[0]), free(pts[1]));
    const nodes = [near(pts[0]), near(pts[1])];
    for (let k = 2; k < 8; k++) { await go(at(nodes[k - 1]), free(pts[k])); nodes.push(near(pts[k])); }
    await go(at(nodes[7]), at(nodes[0]));
    const stops = nodes.slice(0, 5);
    for (const n of stops) { res.push(await cl.edit({ op: 'addStation', at: at(n), platform: 160, name: '' })); await pause(200); }
    res.push(await cl.edit({ op: 'constructAll' }));
    await pause(500);
    const line = await cl.edit({ op: 'addLine', stops, name: 'Either end', colour: '#7b3fb3' });
    await pause(800);
    return { stops, line: line.line, ok: line.ok, refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues ?? r).slice(0, 140)) };
  }, C);
  check(built.ok && !built.refused.length, 'the octagon and its line are built', built.refused.join(' | '));
  if (!built.ok) throw new Error('no line');
  const id = built.line, [first, , , , last] = built.stops;
  const info = () => page.evaluate((num) => {
    const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num), st = w.lineStats[l.id];
    return { stops: l.stops.map(Number), loop: st.loop };
  }, id);
  const s0 = await info();
  check(s0.loop === null && s0.stops.length === 5, 'the line starts open', JSON.stringify(s0));

  await page.evaluate(async ([num, c]) => {
    const tw = window.tw, ui = await import('/src/game/ui.ts');
    const l = tw.world.value.save.lines.find((x) => x.num === num);
    tw.selection.value = { kind: 'line', line: l.id };
    ui.tool.value = 'line';
    ui.lineDraft.value = { line: num, stops: l.stops.map(Number) };
    tw.map.jumpTo({ center: [c[0], c[1]], zoom: 12.6 });
  }, [id, C]);
  await page.waitForTimeout(2500);
  const markerAt = (n) => page.evaluate((n) => {
    const tw = window.tw, node = tw.world.value.nodes.get(String(n)) ?? [...tw.world.value.nodes.values()].find((x) => Number(x.id) === n);
    const [lng, lat] = tw.packToLngLat(node.x, node.y);
    const p = tw.map.project([lng, lat]);
    const rect = tw.map.getContainer().getBoundingClientRect();
    let best = null, bd = 1e9;
    for (const el of document.querySelectorAll('.stop-marks > div')) {
      if (el.style.display === 'none') continue;
      const m = /translate\((-?[\d.]+)px, (-?[\d.]+)px\)/.exec(el.style.transform);
      if (!m) continue;
      const d = Math.hypot(+m[1] - p.x, +m[2] - p.y);
      if (d < bd) { bd = d; best = { x: +m[1] + rect.left, y: +m[2] + rect.top, d }; }
    }
    return best;
  }, n);

  for (const [what, n] of [['last', last], ['first', first]]) {
    const mk = await markerAt(n);
    check(mk && mk.d < 30, `the ${what} stop has a marker on screen`, JSON.stringify(mk));
    await page.mouse.move(mk.x, mk.y);
    await page.waitForTimeout(500);
    const tag = await page.evaluate(() => document.querySelector('.stop-tag')?.innerText ?? '');
    check(/Right click to close loop/.test(tag), `the ${what} stop's hover offers the loop`, tag.replace(/\n/g, ' / '));
    if (what === 'last') await page.screenshot({ path: join(ARTIFACTS, 't3203-hover-last.png') });
    await page.mouse.click(mk.x, mk.y, { button: 'right' });
    await page.waitForTimeout(1800);
    const s1 = await info();
    check(s1.loop === 'circle' && s1.stops.length === 6 && s1.stops[5] === s0.stops[0], `a right click on the ${what} stop closes it into a circle`, JSON.stringify(s1));
    await page.screenshot({ path: join(ARTIFACTS, `t3203-closed-from-${what}.png`) });
    await edit({ op: 'undo' });
    await page.waitForTimeout(1500);
    const s2 = await info();
    check(JSON.stringify(s2.stops) === JSON.stringify(s0.stops) && s2.loop === null, 'undo opens it again', JSON.stringify(s2));
  }
  check(errors.length === 0, 'no page errors', errors.slice(0, 2).join(' | '));
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
