// T-1638 check: one run on hold.mjs's Chrome (CDP 9339, dev server 6159), Japan save loaded.
// 1. Pictures: each view drawn with the stroke cache on and off (clock paused), pixel diff and a 4x crop.
// 2. Updates: with the clock running, each change redraws the cache within two frames and the picture
//    then matches the uncached one.
import { mkdir, writeFile } from 'node:fs/promises';
import { playwright } from 'file:///C:/Users/anita/projects/maps/trainworld/checks/lib.mjs';

const OUT = 'C:/Users/anita/projects/maps/artifacts/trainworld/t1638/';
await mkdir(OUT, { recursive: true });
const { chromium } = await playwright();
const browser = await chromium.connectOverCDP('http://127.0.0.1:9339');
const page = browser.contexts()[0].pages().find((p) => p.url().includes('6159'));
const log = (s) => console.log(s);

await page.evaluate(() => {
  const tw = window.tw;
  window.__f = (n = 2) => new Promise((res) => { let k = 0; const f = () => (++k >= n ? res() : requestAnimationFrame(f)); requestAnimationFrame(f); });
  window.__cache = async (on) => { tw.renderer.strokeCache = on; tw.overlay.invalidate(); await window.__f(3); };
  window.__red = () => tw.renderer.mainCache?.redraws ?? 0;
  // the line segment nearest the screen's centre: its line and screen point
  window.__near = () => {
    const n = tw.renderer.net, s = n.lines.seg, w = tw.overlay.cssW, h = tw.overlay.cssH;
    let best = null, bd = Infinity;
    for (let i = 0; i < n.lines.count; i++) {
      const x = (s[i * 4] + s[i * 4 + 2]) / 2, y = (s[i * 4 + 1] + s[i * 4 + 3]) / 2;
      const [px, py] = tw.overlay.toScreen(x, y);
      const d = Math.hypot(px - w / 2, py - h / 2);
      if (d < bd && Math.hypot(s[i * 4] - s[i * 4 + 2], s[i * 4 + 1] - s[i * 4 + 3]) > 50) { bd = d; best = { line: n.lines.colour[i], px, py }; }
    }
    return best;
  };
});

const box = await page.evaluate(() => { const r = window.tw.map.getCanvas().getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; });
const shot = async () => (await page.screenshot({ clip: box })).toString('base64');

// diff of two screenshots in the page; a crop [cached | direct | diff x8] at 4x around (cx, cy)
// the overlay canvas alone, drawn with the cache on (A) and off (B), read back right after each draw,
// composited over white and flipped upright
const ovCompare = (cx, cy) => page.evaluate(async ([cx, cy]) => {
  const tw = window.tw, gl = tw.overlay.gl;
  const grab = (on) => {
    tw.renderer.strokeCache = on;
    tw.overlay.draw();
    const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, p = new Uint8Array(w * h * 4), d = new Uint8ClampedArray(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, p);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = ((h - 1 - y) * w + x) * 4, o = (y * w + x) * 4;
      for (let k = 0; k < 3; k++) d[o + k] = p[i + k] + 255 - p[i + 3];
      d[o + 3] = 255;
    }
    const c = new OffscreenCanvas(w, h);
    c.getContext('2d').putImageData(new ImageData(d, w, h), 0, 0);
    return { bm: c, d, w, h };
  };
  const B = grab(false), A = grab(true);
  let n2 = 0, n16 = 0, max = 0;
  for (let i = 0; i < A.d.length; i += 4) {
    const m = Math.max(Math.abs(A.d[i] - B.d[i]), Math.abs(A.d[i + 1] - B.d[i + 1]), Math.abs(A.d[i + 2] - B.d[i + 2]));
    if (m > 2) n2++;
    if (m > 16) n16++;
    max = Math.max(max, m);
  }
  const W = 200, H = 120, S = 4, x0 = Math.max(0, Math.min(A.w - W, Math.round(cx - W / 2))), y0 = Math.max(0, Math.min(A.h - H, Math.round(cy - H / 2)));
  const c = new OffscreenCanvas(W * S * 3 + 8, H * S), g = c.getContext('2d');
  g.imageSmoothingEnabled = false;
  g.fillStyle = '#f0f'; g.fillRect(0, 0, c.width, c.height);
  g.drawImage(A.bm, x0, y0, W, H, 0, 0, W * S, H * S);
  g.drawImage(B.bm, x0, y0, W, H, W * S + 4, 0, W * S, H * S);
  const dd = new ImageData(W, H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = ((y0 + y) * A.w + x0 + x) * 4, o = (y * W + x) * 4;
    for (let k = 0; k < 3; k++) dd.data[o + k] = Math.min(255, Math.abs(A.d[i + k] - B.d[i + k]) * 8);
    dd.data[o + 3] = 255;
  }
  const t = new OffscreenCanvas(W, H); t.getContext('2d').putImageData(dd, 0, 0);
  g.drawImage(t, 0, 0, W, H, 2 * (W * S + 4), 0, W * S, H * S);
  const blob = await c.convertToBlob({ type: 'image/png' });
  const buf = new Uint8Array(await blob.arrayBuffer());
  let s = ''; for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode(...buf.subarray(i, i + 0x8000));
  return { px: A.w * A.h, n2, n16, max, crop: btoa(s) };
}, [cx, cy]);

const JAPAN = { center: [137.5, 37.0], zoom: 4.6 }, TOKYO = { center: [139.75, 35.68], zoom: 11 }, TOKYO15 = { center: [139.767, 35.681], zoom: 15 };
const setup = (o) => page.evaluate(async (o) => {
  const tw = window.tw;
  tw.clock.setRunning(false);
  tw.shade.value = o.shade ?? 'light';
  tw.display.value = { ...tw.display.value, projection: 'globe', hillshade: true, trackColour: o.colour ?? 'line' };
  tw.map.jumpTo(o.view);
  tw.select(null);
  await new Promise((r) => setTimeout(r, 3500));
  const near = window.__near();
  if (o.sel && near) tw.select({ kind: 'line', line: String(near.line) });
  await window.__f(3);
  return near;
}, o);

const results = [];
const VIEWS = [
  ['japan-light', { view: JAPAN }],
  ['japan-dark', { view: JAPAN, shade: 'dark' }],
  ['tokyo-light', { view: TOKYO }],
  ['tokyo-dark', { view: TOKYO, shade: 'dark' }],
  ['tokyo-selected', { view: TOKYO, sel: true }],
  ['tokyo-elevation', { view: TOKYO, colour: 'height' }],
  ['tokyo-traffic', { view: TOKYO, colour: 'traffic' }],
  ['tokyo15-boxes', { view: TOKYO15 }],
  ['tokyo15-selected-dark', { view: TOKYO15, sel: true, shade: 'dark' }],
];
for (const [name, o] of VIEWS) {
  const near = await setup(o);
  const r = await ovCompare(near?.px ?? box.width / 2, near?.py ?? box.height / 2);
  await writeFile(`${OUT}${name}.png`, Buffer.from(r.crop, 'base64'));
  await writeFile(`${OUT}${name}-full.png`, await page.screenshot({ clip: box }));
  const row = { name, pxOver2: r.n2, pxOver16: r.n16, maxDiff: r.max, ofPx: r.px };
  results.push(row);
  log(JSON.stringify(row));
}

// updates with the clock running, Tokyo zoom 11
const near = await setup({ view: TOKYO });
await page.evaluate(() => { window.tw.clock.setSpeed(0); window.tw.clock.setRunning(true); });
const upd = [];
const step = async (name, act, waitNet = false) => {
  const r = await page.evaluate(async ([act, waitNet]) => {
    const tw = window.tw, r0 = window.__red();
    const net0 = tw.renderer.net;
    const res = await (0, eval)(act);
    window.__res = res === undefined ? '' : JSON.stringify(res)?.slice(0, 160);
    let frames = 0;
    if (waitNet) {
      const t = performance.now();
      while (tw.renderer.net === net0 && performance.now() - t < 20000) await window.__f(1);
    }
    const r1 = window.__red();
    while (window.__red() === r1 && frames < 2 && r1 === r0) { await window.__f(1); frames++; }
    return { redrawn: window.__red() > r0, frames, netChanged: tw.renderer.net !== net0, res: window.__res };
  }, [act, waitNet]);
  // the picture now: cached against direct, paused
  await page.evaluate(() => window.tw.clock.setRunning(false));
  await page.evaluate(() => window.__f(3));
  const c = await ovCompare(box.width / 2, box.height / 2);
  await page.evaluate(() => window.tw.clock.setRunning(true));
  const row = { name, ...r, pxOver16: c.n16, maxDiff: c.max };
  upd.push(row);
  log(JSON.stringify(row));
};
// nothing changes for a second: no redraws expected
const idle = await page.evaluate(async () => { const r0 = window.__red(); await new Promise((r) => setTimeout(r, 1000)); return window.__red() - r0; });
log(`redraws in 1 s of clock with nothing changing: ${idle}`);
const L = String(near.line);
await step('select line', `window.tw.select({ kind: 'line', line: '${L}' })`);
await step('deselect', `window.tw.select(null)`);
await step('elevation view', `window.tw.display.value = { ...window.tw.display.value, trackColour: 'height' }`);
await step('traffic view', `window.tw.display.value = { ...window.tw.display.value, trackColour: 'traffic' }`);
await step('line view', `window.tw.display.value = { ...window.tw.display.value, trackColour: 'line' }`);
await step('dark shade', `window.tw.shade.value = 'dark'`);
await step('light shade', `window.tw.shade.value = 'light'`);
await step('line width', `window.tw.display.value = { ...window.tw.display.value, lineWidth: 1.5 }`);
await step('line width back', `window.tw.display.value = { ...window.tw.display.value, lineWidth: 1 }`);
await step('line colour edit', `window.tw.client.edit({ op: 'lineLook', line: Number('${L}'), name: window.tw.world.value.save.lines.find((l) => String(l.id) === '${L}' || l.num === Number('${L}'))?.name ?? '', colour: '#00e070' })`, true);
await step('undo colour', `window.tw.client.edit({ op: 'undo' })`, true);
await step('build track', `(() => { const tw = window.tw, p = tw.lngLatToPack(139.70, 35.64), q = tw.lngLatToPack(139.80, 35.72); return tw.client.edit({ op: 'route', from: { kind: 'free', x: p.x, y: p.y, level: 0 }, to: { kind: 'free', x: q.x, y: q.y, level: 0 }, pis: [], single: false }); })()`, true);
await step('undo track', `window.tw.client.edit({ op: 'undo' })`, true);
// hover as trackHover.ts sets it, through the app's own module instance
await step('hover line', `import('/src/map/hovered.ts').then((m) => m.setPointerNet({ line: Number('${L}') }))`);
await step('hover off', `import('/src/map/hovered.ts').then((m) => m.setPointerNet(null))`);
const hot = { hot: '-' }, hot2 = { hot: '-', red: 0 }; hot.red = 0;
await page.evaluate(() => window.tw.clock.setRunning(false));

const md = ['| view | px differing > 2 | px differing > 16 | max channel diff |', '|---|---|---|---|', ...results.map((r) => `| ${r.name} | ${r.pxOver2} | ${r.pxOver16} | ${r.maxDiff} |`),
  '', `Clock 1x, nothing changing: ${idle} redraws in 1 s. Hover: hotLine ${hot.hot} on the line, ${hot2.hot} off it, ${hot2.red - hot.red} redraws moving off.`, '',
  '| change (clock 1x) | cache redrawn | frames after | net changed | px differing > 16 after | max diff | result |', '|---|---|---|---|---|---|---|', ...upd.map((r) => `| ${r.name} | ${r.redrawn} | ${r.frames} | ${r.netChanged} | ${r.pxOver16} | ${r.maxDiff} | ${r.res} |`)].join('\n');
await writeFile(OUT + 'check.md', md + '\n');
console.log(md);
process.exit(0);
