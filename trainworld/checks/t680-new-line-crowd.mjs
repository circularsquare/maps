// T-680: a new line's platforms start empty when its trains start and fill at the arrival rate.
// In a ?new=1 game, paused, the Mumbai Western line test (T-292's, as in t586) is built, given
// trains and constructed; once demand has its riders and the boarding account its hour, every
// platform of the line holds under one rider at the moment of the edit, and over the next 20 game
// minutes never more than the riders who arrive in that time. Writes
// artifacts/trainworld/t680-crowd.json.
//   node trainworld/checks/t680-new-line-crowd.mjs [url]   (from the maps root; default localhost:6145)
// Prints PASS or FAIL lines and exits 1 on failure. About two minutes.
import { mkdtemp, readdir, rm, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const BASE = (process.argv[2] ?? 'http://localhost:6145/').replace(/\/?$/, '/');
const OUT = 'artifacts/trainworld/t680-crowd.json';
const CENTRE = [72.8428, 19.0192];
const STOPS = [['Churchgate', 72.8272, 18.9355], ['Marine Lines', 72.8237, 18.9458], ['Charni Road', 72.8182, 18.9518],
  ['Grant Road', 72.8158, 18.9634], ['Mumbai Central', 72.8193, 18.9696], ['Mahalaxmi', 72.8242, 18.9825],
  ['Lower Parel', 72.8303, 18.9957], ['Prabhadevi', 72.8359, 19.0075], ['Dadar', 72.8428, 19.0192],
  ['Matunga Road', 72.8467, 19.0278], ['Mahim', 72.8466, 19.0407], ['Bandra', 72.8402, 19.0550]];

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
const profile = await mkdtemp(join(tmpdir(), 'tw-t680-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11'] });
const results = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

try {
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 240000 });
  await page.evaluate((ll) => window.tw.map.jumpTo({ center: ll, zoom: 11.2 }), CENTRE);
  await page.waitForFunction(() => window.twDemand.activeCity.value === 'mum', undefined, { timeout: 300000 });
  const before = await page.evaluate(() => window.twDemand.view()?.version ?? -1);
  const built = await page.evaluate(async (stops) => {
    const t = window.tw, run = async (op) => { const r = await t.client.edit(op); if (!r.ok) throw Error(JSON.stringify({ op, r })); return r; };
    const pts = stops.map((s) => t.lngLatToPack(s[1], s[2]));
    const tail = (a, b) => { const dx = a.x - b.x, dy = a.y - b.y, l = Math.hypot(dx, dy); return { x: a.x + 300 * dx / l, y: a.y + 300 * dy / l }; };
    const path = [tail(pts[0], pts[1]), ...pts, tail(pts.at(-1), pts.at(-2))];
    const nodes = [];
    for (let i = 1; i < path.length; i++) {
      const from = i === 1 ? { kind: 'free', x: path[0].x, y: path[0].y, level: -2 } : { kind: 'node', node: nodes.at(-1) };
      await run({ op: 'route', from, to: { kind: 'free', x: path[i].x, y: path[i].y, level: -2 }, pis: [], single: false });
      const e = t.world.value.edges.at(-1);
      if (i === 1) nodes.push(e.a);
      nodes.push(e.b);
    }
    const at = nodes.slice(1, -1);
    for (let i = 0; i < stops.length; i++) await run({ op: 'addStation', at: { kind: 'node', node: at[i] }, platform: 160, name: stops[i][0] });
    const line = (await run({ op: 'addLine', stops: at, name: 'Western line test', colour: '#e4002b' })).line;
    await run({ op: 'setTrainCount', line, trains: [10, 10, 5] });
    await run({ op: 'constructAll' });
    return { line: String(line), at: t.clock.now() };
  }, STOPS);
  await page.waitForFunction((v) => { const d = window.twDemand, dv = d?.view?.(); return dv && dv.final && dv.version > v && !d.client.waiting; }, before, { timeout: 900000, polling: 500 });
  // the account's hour with the new line's platforms in it
  const ready = await page.waitForFunction((line) => {
    const b = window.tw.renderer.board;
    if (!b) return false;
    for (let q = 0; q < b.queueCount; q++) if (String(b.queues[q * 12 + 1]) === line && b.queues[q * 12 + 4] > 0) return true;
    return false;
  }, built.line, { timeout: 120000, polling: 250 }).then(() => true, () => false);
  report('the new line has platforms in the boarding account', ready);
  const rows = !ready ? [] : await page.evaluate(async ({ line, at }) => {
    // the app's own crowd, or on a production build (8800, no source modules) a copy of it
    const m = await import('/src/game/boarding.ts').catch(() => ({
      QUEUE_STRIDE: 12,
      pulsesBetween: (h, q, t0, t) => { let s = 0; const a = h.queues[q * 12 + 10], n = h.queues[q * 12 + 11]; for (let p = a; p < a + n; p++) if (h.pulses[p * 2] > t0 && h.pulses[p * 2] <= t) s += h.pulses[p * 2 + 1]; return s; },
      crowdAt(h, q, t) {
        const first = h.queues[q * 12 + 8], count = h.queues[q * 12 + 9];
        let j = first;
        for (let e = first; e < first + count; e++) if (h.events[e * 2] <= t) j = e;
        const t0 = h.events[j * 2];
        return { waiting: h.events[j * 2 + 1] + Math.max(0, t - t0) * h.queues[q * 12 + 4] + this.pulsesBetween(h, q, t0, t) };
      },
    }));
    const t = window.tw, r = t.renderer, b = r.board, now = t.clock.now(), out = [];
    for (let q = 0; q < b.queueCount; q++) {
      const base = q * m.QUEUE_STRIDE;
      if (String(b.queues[base + 1]) !== line) continue;
      const rate = b.queues[base + 4];
      // seconds after the edit, while still in the hour the board covers
      const s0 = now - r.tripsEpoch, steps = [];
      for (let s = 0; s <= 1200 && s0 + s <= 3600; s += 30) {
        const w = m.crowdAt(b, q, s0 + s).waiting, arrived = rate * (s + now - at) + m.pulsesBetween(b, q, at - r.tripsEpoch, s0 + s);
        steps.push([s, Math.round(w), Math.round(arrived)]);
      }
      const ev = b.events.subarray(b.queues[base + 8] * 2, (b.queues[base + 8] + b.queues[base + 9]) * 2);
      out.push({ node: b.queues[base], run: b.queues[base + 2], perMinute: Math.round(rate * 60), editAt: s0, trains: Array.from(ev.slice(0, 8), Math.round), steps });
    }
    return out;
  }, built);
  await writeFile(OUT, JSON.stringify({ built, rows }, null, 1));
  const atEdit = rows.map((x) => x.steps[0]?.[1] ?? 0);
  report('every platform is empty when the trains start', rows.length > 0 && atEdit.every((w) => w < 1), `(${rows.length} platforms, most waiting ${Math.max(0, ...atEdit)})`);
  const over = rows.flatMap((x) => x.steps.filter(([, w, a]) => w > a * 1.01 + 1).map(([s, w, a]) => ({ node: x.node, run: x.run, s, w, a })));
  const grows = rows.some((x) => x.steps.some(([, w]) => w >= 1));
  report('the crowds grow no faster than riders arrive', rows.length > 0 && over.length === 0 && grows, over.length ? JSON.stringify(over[0]) : `(most after 20 min ${Math.max(0, ...rows.map((x) => x.steps.at(-1)?.[1] ?? 0))})`);
  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await ctx.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
process.exit(results.length && results.every(Boolean) ? 0 : 1);
