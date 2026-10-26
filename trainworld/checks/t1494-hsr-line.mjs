// T-1494: an ordinary line on high-speed track is drawn as an ordinary line; a high-speed line and
// the bare high-speed track keep the high-speed look. New game: one high-speed route with two
// stations and a line (shot with high-speed trains, then ordinary), and a second bare high-speed
// route. Screenshots into artifacts/trainworld/t1491/ (t1494-<label>-*.png).
//   node trainworld/checks/t1494-hsr-line.mjs <label> [url]     (from the maps root; a running dev server)
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, NOISE, collectErrors, launchChrome } from './lib.mjs';

const LABEL = process.argv[2] ?? 'after';
const BASE = process.argv[3] ?? 'http://localhost:6171/';
const OUT = join(ARTIFACTS, 't1491');
await mkdir(OUT, { recursive: true });

const { ctx, close } = await launchChrome({ prefix: 'tw-t1494-', size: [1300, 800], deviceScaleFactor: 3 });
const errors = [];
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
  collectErrors(page, errors, NOISE);
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools, undefined, { timeout: 180000 });
  await page.waitForTimeout(2000);
  const made = await page.evaluate(async () => {
    const tw = window.tw, c = tw.client;
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const near = (p) => [...tw.world.value.nodes.values()].sort((m, n) => Math.hypot(m.x - p.x, m.y - p.y) - Math.hypot(n.x - p.x, n.y - p.y))[0];
    const route = async (lat, lng0, lng1) => {
      const p0 = tw.lngLatToPack(lng0, lat), p1 = tw.lngLatToPack(lng1, lat);
      const r = await c.edit({ op: 'route', from: { kind: 'free', x: p0.x, y: p0.y, level: 1 }, to: { kind: 'free', x: p1.x, y: p1.y, level: 1 }, pis: [], single: false, through: true, hsr: true });
      await sleep(600);
      return { r, a: near(p0), b: near(p1) };
    };
    const one = await route(40.700, -73.95, -73.85);
    const stn = [];
    for (const n of [one.a, one.b]) { stn.push(await c.edit({ op: 'addStation', at: { kind: 'node', node: n.id }, platform: 200, name: '' })); await sleep(300); }
    const bare = await route(40.640, -73.95, -73.85);
    await c.edit({ op: 'constructAll' });
    await sleep(800);
    const line = await c.edit({ op: 'addLine', stops: [one.a.id, one.b.id], name: 'Check', colour: '#d33682' });
    await sleep(500);
    return { ok: one.r.ok, bare: bare.r.ok, mid: [(one.a.lng + one.b.lng) / 2, (one.a.lat + bare.a.lat) / 2], a: [one.a.lng, one.a.lat], lines: JSON.stringify(tw.world.value.save.lines.map((l) => Object.keys(l))), stn: stn.map((s) => s.ok), line: line?.line, hsr: tw.world.value.edges.filter((e) => e.hsr).length };
  });
  console.log(JSON.stringify(made));
  const shoot = async (name, center, zoom, close = false) => {
    await page.evaluate(([c, z]) => { window.tw.map.jumpTo({ center: c, zoom: z }); window.tw.overlay.invalidate(); }, [center, zoom]);
    await page.waitForTimeout(1500);
    await page.screenshot({ path: join(OUT, `t1494-${LABEL}-${name}.png`), clip: close ? { x: 281 + 509 - 180, y: 90 + 355 - 80, width: 360, height: 160 } : { x: 281, y: 90, width: 1019, height: 710 } });
  };
  for (const fast of [true, false]) {
    await page.evaluate(([l, f]) => window.tw.client.edit({ op: 'lineFast', line: l, fast: f }), [made.line, fast]);
    await page.waitForTimeout(1200);
    const isFast = await page.evaluate((l) => window.tw.world.value.lineStats[l]?.fast, made.line);
    console.log(`line fast=${fast} (state ${isFast})`);
    const tag = fast ? 'fast' : 'ordinary';
    await shoot(tag + '-z11', made.mid, 11.2);
    await shoot(tag + '-z15', [-73.90, 40.70], 15, true);
    if (fast) await shoot('bare-z15', [-73.90, 40.64], 15, true);
  }
  console.log(errors.length ? 'errors: ' + errors.join('\n') : 'no console errors');
} finally {
  await close();
}
