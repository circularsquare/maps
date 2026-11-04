// T-1644: the overlay as drawn (stroke cache, culling and gathering on) against every stroke drawn directly
// (no cache, no culling): pixels differing, over the overlay canvas, at many views. Page 2 (CDP 9345).
import { mkdir, writeFile } from 'node:fs/promises';
import { playwright } from 'file:///C:/Users/anita/projects/maps/trainworld/checks/lib.mjs';
const PORT = process.env.PORT ?? '6124', CDP = process.env.CDP ?? '9345';
const OUT = 'C:/Users/anita/projects/maps/artifacts/trainworld/t1644/';
await mkdir(OUT, { recursive: true });
const { chromium } = await playwright();
const browser = await chromium.connectOverCDP(`http://127.0.0.1:${CDP}`);
const page = browser.contexts()[0].pages().find((p) => p.url().includes(PORT));
await page.evaluate(() => {
  window.__f = (n = 2) => new Promise((res) => { let k = 0; const f = () => (++k >= n ? res() : requestAnimationFrame(f)); requestAnimationFrame(f); });
  window.__grab = (mode) => {
    const tw = window.tw, r = tw.renderer, gl = tw.overlay.gl;
    if (mode === 'direct') { r.strokeCache = false; r.strokeCull = false; r.strokeGather = false; }
    else if (mode === 'ranges') { r.strokeCache = false; r.strokeCull = true; r.strokeGather = false; }
    else { r.strokeCache = true; r.strokeCull = true; r.strokeGather = true; r.mainCache?.clear(); }
    tw.overlay.draw();
    const inst = r.strokeInstances;
    const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, p = new Uint8Array(w * h * 4), d = new Uint8ClampedArray(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, p);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = ((h - 1 - y) * w + x) * 4, o = (y * w + x) * 4;
      for (let k = 0; k < 3; k++) d[o + k] = p[i + k] + 255 - p[i + 3];
      d[o + 3] = 255;
    }
    r.strokeCache = true; r.strokeCull = true; r.strokeGather = true;
    return { d, w, h, inst };
  };
  window.__cmp = (a, b, cx, cy) => {
    const A = window.__grab(a), B = window.__grab(b);
    let n2 = 0, n16 = 0, max = 0;
    for (let i = 0; i < A.d.length; i += 4) {
      const m = Math.max(Math.abs(A.d[i] - B.d[i]), Math.abs(A.d[i + 1] - B.d[i + 1]), Math.abs(A.d[i + 2] - B.d[i + 2]));
      if (m > 2) n2++;
      if (m > 16) n16++;
      if (m > max) max = m;
    }
    return { n2, n16, max, px: A.w * A.h, instA: A.inst, instB: B.inst };
  };
});
const V = (c, z) => ({ center: c, zoom: z });
const TOKYO = [139.767, 35.681], SHINJUKU = [139.7005, 35.6905], OSAKA = [135.4959, 34.7025];
const LAYERS = process.env.LAYERS === '1';
const VIEWS = LAYERS ? [['tokyo15-layers', V(TOKYO, 15), { layers: true }], ['tokyo13-layers', V(TOKYO, 13), { layers: true }], ['tokyo11-layers', V([139.75, 35.68], 11), { layers: true }]] : [
  ['japan', V([137.5, 37.0], 4.6)], ['kanto9', V([139.75, 35.68], 9.5)], ['tokyo11', V([139.75, 35.68], 11)], ['tokyo12', V(TOKYO, 12)], ['tokyo13', V(TOKYO, 13)], ['tokyo14', V(TOKYO, 14)],
  ['tokyo15', V(TOKYO, 15)], ['tokyo16', V(TOKYO, 16)], ['shinjuku17', V(SHINJUKU, 17)], ['osaka15', V(OSAKA, 15)], ['tokyo15-sel', V(TOKYO, 15), { sel: true }], ['tokyo13-sel', V(TOKYO, 13), { sel: true }],
  ['tokyo15-dark', V(TOKYO, 15), { shade: 'dark' }], ['tokyo15-elevation', V(TOKYO, 15), { colour: 'height' }], ['tokyo15-traffic', V(TOKYO, 15), { colour: 'traffic' }], ['tokyo13-speed', V(TOKYO, 13), { colour: 'speed' }],
  ['tokyo15-edge', V([139.7, 35.62], 15)], ['tokyo15-edge2', V([139.84, 35.74], 15)],
];
const rows = [];
for (const [name, v, o = {}] of VIEWS) {
  await page.evaluate(async ([v, o]) => {
    const tw = window.tw;
    tw.clock.setRunning(false);
    tw.shade.value = o.shade ?? 'light';
    tw.display.value = { ...tw.display.value, projection: 'globe', hillshade: true, trackColour: o.colour ?? 'line' };
    tw.map.jumpTo(v);
    tw.select(null);
    await new Promise((r) => setTimeout(r, 3500));
    if (o.layers) {
      const r = tw.renderer, n = r.net, c = n.lines.count;
      const lw = new Float32Array(c), fu = new Float32Array(c);
      for (let i = 0; i < c; i++) { lw[i] = 0.5 + ((i * 2654435761) % 1000) / 300; fu[i] = i % 7 === 0 ? -1 : i % 11 === 0 ? 0.6 : 0; }
      r.setLayers({ lineWidth: lw, fill: new Float32Array(Math.max(1, n.samples.length / 2) * 2) });
      r.rewrite(r.lines.fullBuf, fu); r.linesFull = fu; r.perSegV++;
      tw.overlay.invalidate();
    }
    if (o.sel) {
      const n = tw.renderer.net, s = n.lines.seg, w = tw.overlay.cssW, h = tw.overlay.cssH;
      let best = null, bd = Infinity;
      for (let i = 0; i < n.lines.count; i++) {
        const [px, py] = tw.overlay.toScreen((s[i * 4] + s[i * 4 + 2]) / 2, (s[i * 4 + 1] + s[i * 4 + 3]) / 2);
        const d = Math.hypot(px - w / 2, py - h / 2);
        if (d < bd) { bd = d; best = n.lines.colour[i]; }
      }
      tw.select({ kind: 'line', line: String(best) });
    }
    await window.__f(3);
  }, [v, o]);
  const a = await page.evaluate(() => window.__cmp('cached', 'direct'));
  const b = await page.evaluate(() => window.__cmp('ranges', 'direct'));
  const row = { name, gatherVsDirect: [a.n2, a.n16, a.max], rangesVsDirect: [b.n2, b.n16, b.max], inst: [a.instA, b.instA, a.instB], px: a.px };
  rows.push(row);
  console.log(JSON.stringify(row));
}
await page.evaluate(() => { const tw = window.tw; tw.shade.value = 'light'; tw.select(null); tw.display.value = { ...tw.display.value, trackColour: 'line' }; });
const md = ['| view | gathered vs every stroke: px > 2 / > 16 / max | ranges vs every stroke: px > 2 / > 16 / max | stroke instances gathered / ranges / every stroke |', '|---|---|---|---|', ...rows.map((r) => `| ${r.name} | ${r.gatherVsDirect.join(' / ')} | ${r.rangesVsDirect.join(' / ')} | ${r.inst.join(' / ')} |`)].join('\n');
await writeFile(OUT + 'check.md', md + '\n');
console.log(md);
process.exit(0);
