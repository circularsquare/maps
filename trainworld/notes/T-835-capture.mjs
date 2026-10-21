// T-835: load the real save on our Vite (6158), wait for demand's final, and save the riders
// message the main thread posts to the clock worker (weekday), as a Float32Array file.
import { readFile, writeFile, mkdtemp, rm, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const OUT = process.argv[2];
const SAVE = 'C:/Users/anita/projects/maps/trainworld/T-007-real-nyc.save';
const BASE = 'http://localhost:6158/';
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';

async function playwright() {
  try { return await import('playwright'); } catch {}
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no playwright');
}
const { chromium } = await playwright();
const profile = await mkdtemp(join(tmpdir(), 'tw-t835-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: CHROME, headless: true, viewport: { width: 1200, height: 800 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => {
    try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {}
    window.__riders = [];
    const orig = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function (msg, ...rest) {
      try {
        if (msg && msg.kind === 'riders' && msg.packed) window.__riders.push({ t: performance.now(), day: msg.day ?? 0, packed: msg.packed.slice() });
      } catch {}
      return orig.call(this, msg, ...rest);
    };
  });
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(2000);
  const bytes = await readFile(SAVE);
  await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.waitForFunction(() => {
    const d = window.twDemand, s = d?.log?.at(-1), v = d?.view?.();
    return s && v && v.version === s.version && v.final && !d.client.waiting && v.lines.size > 100;
  }, undefined, { timeout: 400000, polling: 500 });
  await page.waitForTimeout(5000);
  const got = await page.evaluate(() => {
    const v = window.twDemand.view();
    const r = window.__riders.filter((x) => x.day === 0).at(-1);
    return { n: window.__riders.length, days: window.__riders.map((x) => x.day), rail: v.railTripsPerDay, round: v.round, packed: r ? Array.from(r.packed) : null };
  });
  console.log('riders messages', got.n, got.days.join(','), 'rail trips', Math.round(got.rail), 'round', got.round, 'floats', got.packed?.length);
  if (got.packed) await writeFile(OUT, Buffer.from(new Float32Array(got.packed).buffer));
} finally {
  await ctx.close();
  await rm(profile, { recursive: true, force: true });
}
