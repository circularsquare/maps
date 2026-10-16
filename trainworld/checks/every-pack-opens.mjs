// T-484: every registered zone opens and solves without a demand error. For each zone in
// registry.json, a fresh ?new=1 game flies to the zone's centre (its pack opens for bubbles), then
// a three-station line is built across the centre and demand must reach a final result. One PASS or
// FAIL line per zone, with the worker's error; exits 1 if any failed. About 30 s a zone.
//   node trainworld/checks/every-pack-opens.mjs [url] [zone ...]      (from the maps root)
// With no url it starts its own Vite (checks/dev-nowatch.mjs, port PACKS_PORT or 6191). A pack being
// rewritten while the check reads it can fail spuriously: rerun that zone alone.
import { spawn } from 'node:child_process';
import { readFile, readdir, mkdtemp, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const REGISTRY = fileURLToPath(new URL('../app/src/game/registry.json', import.meta.url));
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const IGNORE = /favicon|openfreemap|AJAXError/;

async function playwright() {
  if (process.env.PLAYWRIGHT_MODULE) return import(process.env.PLAYWRIGHT_MODULE);
  try { return await import('playwright'); } catch { /* not installed here */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright: set PLAYWRIGHT_MODULE to a file URL of playwright/index.mjs');
}

async function startVite() {
  const port = Number(process.env.PACKS_PORT ?? 6191);
  const child = spawn(process.execPath, [join(HERE, 'dev-nowatch.mjs'), String(port)], { stdio: ['ignore', 'pipe', 'pipe'] });
  let out = '';
  child.stdout.on('data', (d) => { out += d; });
  child.stderr.on('data', (d) => { out += d; });
  const deadline = Date.now() + 60000;
  while (!/localhost:/.test(out)) {
    if (child.exitCode !== null || Date.now() > deadline) throw new Error('Vite did not start: ' + out.slice(-500));
    await new Promise((r) => setTimeout(r, 200));
  }
  return { child, url: `http://localhost:${port}/` };
}

/** a line of three stations across the zone's centre, east to west, `km` apart */
const buildLine = (page, [lng, lat], km) => page.evaluate(async ([lng, lat, km]) => {
  const tw = window.tw, c = tw.client;
  const dLng = km / (111.32 * Math.cos((lat * Math.PI) / 180));
  const xy = [-1, 0, 1].map((k) => { const p = tw.lngLatToPack(lng + k * dLng, lat); return [p.x, p.y]; });
  const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
  const wait = (ms) => new Promise((d) => setTimeout(d, ms));
  await c.edit({ op: 'route', from: { kind: 'free', x: xy[0][0], y: xy[0][1], level: -1 }, to: { kind: 'free', x: xy[1][0], y: xy[1][1], level: -1 }, pis: [], single: false });
  await wait(300);
  await c.edit({ op: 'route', from: { kind: 'node', node: near(xy[1]) }, to: { kind: 'free', x: xy[2][0], y: xy[2][1], level: -1 }, pis: [], single: false });
  await wait(300);
  for (const p of xy) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await wait(150); }
  await c.edit({ op: 'constructAll' });
  await wait(300);
  const r = await c.edit({ op: 'addLine', stops: xy.map(near), name: 'Check', colour: '#0a7d3b' });
  if (r?.line === undefined) return null;
  await c.edit({ op: 'setTrainCount', line: r.line, trains: [6, 4, 3] });
  return r.line;
}, [lng, lat, km]);

async function checkZone(ctx, base, city) {
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error' && !IGNORE.test(m.text())) errors.push(m.text()); });
  const failed = () => page.evaluate(() => (String(window.twDemand?.client?.status ?? '').startsWith('failed') ? window.twDemand.client.status : null));
  try {
    const sep = base.includes('?') ? '&' : '?';
    await page.goto(`${base}${sep}debug=1&new=1&paused=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 120000 });
    await page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z }), [city.center, Math.max(city.zoom, 11)]);
    await page.waitForFunction((id) => window.twDemand.activeCity.peek() === id, city.id, { timeout: 60000, polling: 100 });
    // its pack opened for the view (bubbles): give it time to download and open
    await page.waitForTimeout(6000);
    let f = await failed();
    if (f) return { ok: false, step: 'view', detail: f };
    const line = await buildLine(page, city.center, 2);
    if (line === null) return { ok: false, step: 'build', detail: 'no line built' };
    await page.waitForFunction(() => {
      const d = window.twDemand, s = d?.log?.at(-1), v = d?.view?.();
      if (String(d.client.status).startsWith('failed')) return true;
      return s && v && v.version === s.version && v.final && !d.client.waiting && v.lines.size > 0;
    }, undefined, { timeout: 240000, polling: 250 });
    f = await failed();
    if (f) return { ok: false, step: 'solve', detail: f };
    const v = await page.evaluate(() => { const d = window.twDemand, s = d.log.at(-1), v = d.view(); return { rail: Math.round(v.railTripsPerDay), cities: s.cities.join('+') }; });
    const errs = errors.filter((e) => !/^demand:/.test(e));
    if (errs.length) return { ok: false, step: 'console', detail: errs.slice(0, 2).join(' | ').slice(0, 300) };
    return { ok: true, detail: `${v.rail.toLocaleString('en-US')} rail trips a day, zones ${v.cities}` };
  } catch (e) {
    const f = await failed().catch(() => null);
    return { ok: false, step: 'timeout', detail: f ?? e.message.split('\n')[0] };
  } finally {
    await page.close();
  }
}

const args = process.argv.slice(2);
const given = args.find((a) => /^https?:/.test(a));
const only = args.filter((a) => a !== given);
const registry = JSON.parse(await readFile(REGISTRY, 'utf8'));
const cities = registry.cities.filter((c) => !only.length || only.includes(c.id));
let vite = null, failures = 0;
const t0 = Date.now();
const { chromium } = await playwright();
const profile = await mkdtemp(join(tmpdir(), 'tw-packs-'));
let ctx = null;
try {
  if (!given) vite = await startVite();
  const base = given ?? vite.url;
  console.log(`      ${cities.length} zones on ${base}`);
  ctx = await chromium.launchPersistentContext(profile, { executablePath: CHROME, headless: true, viewport: { width: 1200, height: 800 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
  for (const city of cities) {
    const r = await checkZone(ctx, base, city);
    if (!r.ok) failures++;
    console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${city.id} ${city.name}  ${r.ok ? '' : r.step + ': '}${r.detail}`);
  }
} finally {
  await ctx?.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
  if (vite) vite.child.kill();
}
const secs = ((Date.now() - t0) / 1000).toFixed(0);
console.log(failures ? `PACKS FAILED: ${failures} of ${cities.length} zones, ${secs} s` : `PACKS PASSED: ${cities.length} zones, ${secs} s`);
process.exit(failures ? 1 : 0);
