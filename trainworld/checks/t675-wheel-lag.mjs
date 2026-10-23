// T-675: a mouse-wheel zoom with a main-thread stall as it starts travels as far as one without.
// A steady wheel (12 ticks of deltaY 100, one every 16 ms) with and without a 144 ms stall as it
// starts, twice over: replayed in the page (the first 9 ticks held and handled together, the same
// every run) and as real input through CDP behind a busy loop (Chrome's queueing varies). Before
// the fix MapLibre took the stalled scroll for a trackpad: about 3 times as far, up to a whole zoom
// level in one frame. PASS when each stalled scroll travels at most 1.25 times the calm one and no
// frame steps more than 0.5 zoom.
//   BASE=http://localhost:6101/ node trainworld/checks/t675-wheel-lag.mjs
// RAW=1 takes the fix off (the page's own MapLibre handler again): it should FAIL. OUT=<dir> writes
// t675-wheel-lag.json there.
import { rm, mkdir, writeFile, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { pathToFileURL } from 'node:url';

async function playwright() {
  if (process.env.PLAYWRIGHT_MODULE) return import(process.env.PLAYWRIGHT_MODULE);
  try { return await import('playwright'); } catch { /* not installed here */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright');
}
const { chromium } = await playwright();
const BASE = process.env.BASE ?? 'http://localhost:6101/';
const RAW = process.env.RAW === '1';
const REPS = Number(process.env.REPS ?? 2);
const profile = join(tmpdir(), `tw-t675-${process.pid}`);
await rm(profile, { recursive: true, force: true });
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
const cdp = await ctx.newCDPSession(page);
const X = 700, Y = 380;
const runs = [];

/** one scroll from zoom 11 over Manhattan; zoom travelled and the largest step in one frame.
 * kind 'replay': 12 wheel events made in the page one every 16 ms (so their timeStamps are 16 ms
 * apart), the first `held` of them held back and handled together, as a stall does; always the same.
 * kind 'real': 12 real ticks through CDP and a busy loop of `held` x 16 ms started with the first;
 * how Chrome queues them behind the loop varies, so it reproduces most runs, not all. */
const scroll = async (kind, held) => {
  await page.evaluate(() => void window.tw.map.jumpTo({ center: [-73.96, 40.75], zoom: 11 }));
  await page.waitForTimeout(1500); // past MapLibre's 400 ms, so this is a new scroll
  const replay = page.evaluate(([kind, held, X, Y]) => {
    const w = (window.__t675 = { fr: [], on: true });
    const loop = () => { w.fr.push(window.tw.map.getZoom()); if (w.on) requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
    if (kind === 'real') {
      if (held) setTimeout(() => { const t = performance.now(); while (performance.now() - t < held * 16); }, 0);
      return;
    }
    const canvas = window.tw.map.getCanvas(), queue = [];
    return new Promise((done) => {
      let n = 0;
      const id = setInterval(() => {
        queue.push(new WheelEvent('wheel', { deltaY: -100, deltaMode: 0, clientX: X, clientY: Y, bubbles: true, cancelable: true }));
        n++;
        if (n >= held) for (const e of queue.splice(0)) canvas.dispatchEvent(e);
        if (n === 12) { clearInterval(id); done(); }
      }, 16);
    });
  }, [kind, held, X, Y]);
  if (kind === 'real') {
    const sends = [];
    for (let i = 0; i < 12; i++) {
      sends.push(cdp.send('Input.dispatchMouseEvent', { type: 'mouseWheel', x: X, y: Y, deltaX: 0, deltaY: -100, modifiers: 0, pointerType: 'mouse' }));
      await new Promise((r) => setTimeout(r, 16));
    }
    await Promise.all(sends);
  }
  await replay;
  await page.waitForTimeout(1200);
  const fr = await page.evaluate(() => { window.__t675.on = false; return window.__t675.fr; });
  let maxStep = 0;
  for (let i = 1; i < fr.length; i++) maxStep = Math.max(maxStep, Math.abs(fr[i] - fr[i - 1]));
  const r = { kind, held, travelled: +(fr.at(-1) - 11).toFixed(3), maxStep: +maxStep.toFixed(3) };
  runs.push(r);
  return r;
};

let pass = false, line = '';
try {
  await page.goto(`${BASE}?new=1&opened=all&debug=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.map?.loaded?.(), undefined, { timeout: 120000 });
  await page.waitForTimeout(5000);
  if (RAW) await page.evaluate(() => { delete window.tw.map.scrollZoom.wheel; });
  await page.mouse.move(X, Y);
  const parts = [];
  pass = errors.length === 0;
  for (const kind of ['replay', 'real']) {
    const calm = [], stalled = [];
    for (let i = 0; i < REPS; i++) {
      calm.push(await scroll(kind, 0));
      stalled.push(await scroll(kind, 9)); // 9 ticks, a 144 ms stall
    }
    const base = Math.max(...calm.map((r) => r.travelled));
    const worst = Math.max(...stalled.map((r) => r.travelled));
    const step = Math.max(...stalled.map((r) => r.maxStep));
    pass &&= base > 0.5 && worst <= 1.25 * base && step <= 0.5;
    parts.push(`${kind}: calm ${base.toFixed(2)} zoom, stalled ${worst.toFixed(2)} (x${(worst / base).toFixed(2)}), largest step in one frame ${step.toFixed(2)}`);
  }
  line = `${parts.join('; ')}${RAW ? ' [RAW, fix off]' : ''}${errors.length ? `; page errors: ${errors.slice(0, 2).join('; ')}` : ''}`;
} catch (e) {
  line = `error: ${e.message}`;
} finally {
  await ctx.close();
  await rm(profile, { recursive: true, force: true });
}
if (process.env.OUT) {
  await mkdir(process.env.OUT, { recursive: true });
  await writeFile(join(process.env.OUT, `t675-wheel-lag${RAW ? '-raw' : ''}.json`), JSON.stringify({ runs, errors }, null, 1));
}
console.log(`${pass ? 'PASS' : 'FAIL'} t675 wheel zoom under a stall: ${line}`);
process.exit(pass ? 0 : 1);
