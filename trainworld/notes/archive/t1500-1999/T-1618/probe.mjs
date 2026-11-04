// T-1618 probe: the stroke pass's counts, and GPU ms per overlay draw with parts of it cut (clock paused, forced draws).
import { playwright } from 'file:///C:/Users/anita/projects/maps/trainworld/checks/lib.mjs';
const { chromium } = await playwright();
const browser = await chromium.connectOverCDP('http://127.0.0.1:9338');
const page = browser.contexts()[0].pages().find((p) => p.url().includes('6158'));
const out = await page.evaluate(async () => {
  const tw = window.tw, r = tw.renderer, gl = tw.overlay.gl, m = window.__m;
  tw.clock.setRunning(false);
  const res = { track: r.track?.count, lines: r.lines?.count, anyFull: r.anyFull, hasBlueprint: r.hasBlueprint, highlight: !!r.highlight, buses: r.buses?.any };
  const wait = (ms) => new Promise((d) => setTimeout(d, ms));
  // time N forced draws in the current setup: average GPU ms of the overlay draws
  const timeDraws = async (n = 40) => {
    m.ovGpu.length = 0;
    for (let i = 0; i < n; i++) { tw.overlay.draw(); await new Promise((d) => requestAnimationFrame(d)); }
    await wait(300);
    return m.ovGpu.length ? +(m.ovGpu.reduce((a, b) => a + b, 0) / m.ovGpu.length).toFixed(2) : null;
  };
  // count draw calls and instances per overlay draw
  const calls = { n: 0, inst: 0 };
  const odi = gl.drawArraysInstanced.bind(gl);
  gl.drawArraysInstanced = (mode, first, count, inst) => { calls.n++; calls.inst += inst; return odi(mode, first, count, inst); };
  const views = { japan: { center: [137.5, 37.0], zoom: 4.6 }, kanto: { center: [139.75, 35.68], zoom: 8 }, tokyo: { center: [139.75, 35.68], zoom: 11 }, street: { center: [139.70, 35.69], zoom: 15 } };
  for (const [k, v] of Object.entries(views)) {
    tw.map.jumpTo(v);
    await wait(1500);
    calls.n = calls.inst = 0;
    tw.overlay.draw();
    const c = { calls: calls.n, instances: calls.inst };
    const all = await timeDraws();
    const t = r.track, l = r.lines;
    r.lines = null; const noLines = await timeDraws(); r.lines = l;
    r.track = null; const noTrack = await timeDraws(); r.track = t;
    // a viewport of 1x1 px: what is left is vertex cost
    const vp = gl.viewport.bind(gl); gl.viewport = () => vp(0, 0, 1, 1);
    const tinyVp = await timeDraws(); gl.viewport = vp;
    res[k] = { ...c, all, noLines, noTrack, tinyVp };
  }
  gl.drawArraysInstanced = odi;
  tw.map.jumpTo(views.japan);
  return res;
});
console.log(JSON.stringify(out, null, 1));
process.exit(0);
