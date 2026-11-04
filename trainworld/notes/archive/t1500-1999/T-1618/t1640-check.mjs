// T-1640 check: one run on hold.mjs's Chrome (CDP and PORT from the environment), Japan save loaded.
// The throttle on: the game minute still turns on time, trains appear at once on zooming in, selection,
// hover, display changes and edits draw within a frame, zoomed in stays at full rate.
import { playwright } from 'file:///C:/Users/anita/projects/maps/trainworld/checks/lib.mjs';

const PORT = process.env.PORT ?? '6158', CDP = process.env.CDP ?? '9338';
const { chromium } = await playwright();
const browser = await chromium.connectOverCDP(`http://127.0.0.1:${CDP}`);
const page = browser.contexts()[0].pages().find((p) => p.url().includes(PORT));

await page.evaluate(() => {
  const tw = window.tw;
  window.__f = (n = 1) => new Promise((res) => { let k = 0; const f = () => (++k >= n ? res() : requestAnimationFrame(f)); requestAnimationFrame(f); });
  window.__wait = (ms) => new Promise((r) => setTimeout(r, ms));
  window.__rate = async (ms) => {
    const f0 = tw.overlay.frames, g0 = [], orig = tw.overlay.draw.bind(tw.overlay);
    let last = performance.now(), gap = 0;
    tw.overlay.draw = () => { const t = performance.now(); gap = Math.max(gap, t - last); last = t; return orig(); };
    await window.__wait(ms);
    delete tw.overlay.draw;
    return { drawsPerS: (tw.overlay.frames - f0) / (ms / 1000), maxGapMs: Math.round(gap) };
  };
  // frames drawn within `n` animation frames after `act`
  window.__after = async (act, n = 2) => {
    await window.__wait(1300); // a pending throttle timer would be due by now
    const f0 = tw.overlay.frames;
    await (0, eval)(act);
    await window.__f(n);
    return tw.overlay.frames - f0;
  };
  window.__minutes = async (ms) => { let c = 0; const u = tw.clock.minute.subscribe(() => c++); c = 0; await window.__wait(ms); u(); return c / (ms / 1000); };
});

const run = (name, fn, arg) => page.evaluate(fn, arg).then((r) => { console.log(name, JSON.stringify(r)); return r; });
const JAPAN = { center: [137.5, 37.0], zoom: 4.6 }, TOKYO = { center: [139.75, 35.68], zoom: 11 }, Z13 = { center: [139.767, 35.681], zoom: 13 }, Z95 = { center: [139.75, 35.68], zoom: 9.5 };
const after = (name, act) => run(name, (a) => window.__after(a), act);
const at = (view, speed) => page.evaluate(async ([v, s]) => { const tw = window.tw; tw.clock.setSpeed(s); tw.clock.setRunning(true); tw.map.jumpTo(v); tw.select(null); await window.__wait(3500); }, [view, speed]);

// 1. the minute turns on time with nothing drawn
await at(JAPAN, 0);
await run('japan 1x draws', () => window.__rate(3000));
await run('japan 1x minute changes/s (expect 1)', () => window.__minutes(3000));
await at(JAPAN, 2);
await run('japan fastest draws', () => window.__rate(3000));
await run('japan fastest minute changes/s (expect 16)', () => window.__minutes(3000));

// 2. trains appear at once on zooming in
await run('zoom in to Tokyo z11', async () => {
  const tw = window.tw, f0 = tw.overlay.frames, t0 = performance.now();
  tw.map.jumpTo({ center: [139.75, 35.68], zoom: 11 });
  while (tw.overlay.frames === f0 && performance.now() - t0 < 2000) await window.__f(1);
  const ms = performance.now() - t0, hidden = tw.renderer.trainsHidden;
  await window.__f(3);
  return { msToFirstDraw: Math.round(ms), trainsHiddenAtFirstDraw: hidden, trainsHiddenAfter: tw.renderer.trainsHidden };
});
await run('Tokyo z11 fastest draws', () => window.__rate(2000));

// 3. zoomed in stays at full rate
await at(Z13, 0);
await run('Tokyo z13 1x draws', () => window.__rate(2000));

// 4. crowds only (z9.5)
await at(Z95, 0);
await run('Kanto z9.5 1x draws', () => window.__rate(2000));

// 5. updates at Japan zoom with the throttle holding (frames drawn within 2 animation frames)
await at(JAPAN, 0);
const L = await page.evaluate(() => { const n = window.tw.renderer.net, s = n.lines.seg; return String(n.lines.colour[0]); });
await after('select line', `window.tw.select({ kind: 'line', line: '${L}' })`);
await after('deselect', `window.tw.select(null)`);
await after('hover line', `import('/src/map/hovered.ts').then((m) => m.setPointerNet({ line: Number('${L}') }))`);
await after('hover off', `import('/src/map/hovered.ts').then((m) => m.setPointerNet(null))`);
await after('dark shade', `window.tw.shade.value = 'dark'`);
await after('light shade', `window.tw.shade.value = 'light'`);
await after('line width', `window.tw.display.value = { ...window.tw.display.value, lineWidth: 1.5 }`);
await after('line width back', `window.tw.display.value = { ...window.tw.display.value, lineWidth: 1 }`);
await run('line colour edit (frames after the net changes)', async (L) => {
  const tw = window.tw, net0 = tw.renderer.net;
  await window.__wait(1300);
  tw.client.edit({ op: 'lineLook', line: Number(L), name: tw.world.value.save.lines.find((l) => String(l.id) === L || l.num === Number(L))?.name ?? '', colour: '#00e070' });
  const t = performance.now();
  while (tw.renderer.net === net0 && performance.now() - t < 20000) await window.__f(1);
  const f0 = tw.overlay.frames;
  await window.__f(3);
  const r = { netChanged: tw.renderer.net !== net0, framesDrawnIn3: tw.overlay.frames - f0 };
  tw.client.edit({ op: 'undo' });
  await window.__wait(3000);
  return r;
}, L);

// 6. trains toggled off then on at Tokyo z11: back to full rate at once
await at(TOKYO, 0);
await run('trains off, then draws/s', async () => { const tw = window.tw; tw.display.value = { ...tw.display.value, trains: false }; await window.__wait(1500); return window.__rate(2000); });
await run('trains on: frames within 2 rAF, then draws/s', async () => {
  const tw = window.tw, f0 = tw.overlay.frames;
  tw.display.value = { ...tw.display.value, trains: true };
  await window.__f(2);
  const first = tw.overlay.frames - f0;
  return { framesWithin2: first, ...(await window.__rate(2000)) };
});
await page.evaluate(() => window.tw.clock.setRunning(false));
process.exit(0);
