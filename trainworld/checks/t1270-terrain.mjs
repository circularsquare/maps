// T-1270 in a new game through the edit API: track priced by the ground's terrain class. A
// kilometre or so at ground level on flat ground in Brooklyn costs 45 US$M a km, the same at
// ground level on mountain ground above Innsbruck 2.5x that, and a bored tunnel (-2) there its flat
// 99; Construct charges the quote. The spots come from the zones' terrain files (a run of cells of
// one class). Then screenshots: the hillshade (Settings, Terrain) at Innsbruck in both themes, and
// the track tool's tag over a mountain segment. About a minute.
// Usage: node trainworld/checks/t1270-terrain.mjs [url] (default http://localhost:6184/)

import { mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6184/';
const sep = BASE.includes('?') ? '&' : '?';
const OUT = join(ARTIFACTS, 't1270');
mkdirSync(OUT, { recursive: true });
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) failed++; };

/** A zone's terrain file (sim `cost::ClassGrid`). */
function grid(id) {
  const b = readFileSync(new URL(`../data/terrain/${id}.terrain`, import.meta.url));
  if (b.toString('latin1', 0, 4) !== 'TWT1') throw new Error(`${id}.terrain is not a terrain file`);
  const [w, h] = [b.readUInt32LE(4), b.readUInt32LE(8)], [c0, r0] = [b.readInt32LE(12), b.readInt32LE(16)], per = b.readUInt32LE(20);
  const cls = new Uint8Array(w * h);
  let i = 24, k = 0;
  while (i < b.length && k < cls.length) {
    const c = b[i++];
    let n = 0, shift = 0;
    for (;;) { const x = b[i++]; n += (x & 0x7f) * 2 ** shift; if (!(x & 0x80)) break; shift += 7; }
    cls.fill(c, k, Math.min(cls.length, k + n));
    k += n;
  }
  return { w, h, c0, r0, per, cls, at: (lon, lat) => {
    const c = Math.floor((lon + 180) * per) - c0, r = Math.floor((lat + 90) * per) - r0;
    return c < 0 || r < 0 || c >= w || r >= h ? -1 : cls[r * w + c];
  } };
}

/** Lon, lat of both ends of the run of `n` cells of `cls` along a row nearest (lon, lat). */
function runNear(g, cls, n, lon, lat) {
  let best = null;
  for (let r = 1; r < g.h - 1; r++) {
    let run = 0;
    for (let c = 0; c < g.w; c++) {
      run = g.cls[r * g.w + c] === cls ? run + 1 : 0;
      if (run < n) continue;
      const a = [(g.c0 + c - n + 1.5) / g.per - 180, (g.r0 + r + 0.5) / g.per - 90], b = [(g.c0 + c - 0.5) / g.per - 180, a[1]];
      const d = Math.hypot((a[0] + b[0]) / 2 - lon, a[1] - lat);
      if (!best || d < best.d) best = { a, b, d };
    }
  }
  return best;
}

const ins = grid('ins'), nyc = grid('nyc');
const mtn = runNear(ins, 3, 6, 11.39, 47.30);
check(!!mtn, `a run of mountain cells near Innsbruck (${mtn?.a.map((v) => v.toFixed(4))} to ${mtn?.b.map((v) => v.toFixed(4))})`);
// inland Brooklyn (Flatbush), where a run of flat cells has no water on it
const flat = runNear(nyc, 0, 5, -73.955, 40.64);
check(!!flat, `a run of flat cells in Brooklyn (${flat?.a.map((v) => v.toFixed(4))} to ${flat?.b.map((v) => v.toFixed(4))})`);
const [flatA, flatB] = [flat.a, flat.b];

const { ctx, close } = await launchChrome({ prefix: 'tw-t1270-' });
try {
  const page = await ctx.newPage();
  const errors = [];
  collectErrors(page, errors);
  // the Terrain setting on, as a player would have stored it
  await page.addInitScript(() => {
    try {
      const s = JSON.parse(localStorage.getItem('trainworld-ui') || '{}');
      s.display = { ...(s.display ?? {}), hillshade: true };
      s.v = s.v ?? 10;
      localStorage.setItem('trainworld-ui', JSON.stringify(s));
    } catch { /* private window */ }
  });
  await page.goto(`${BASE}${sep}debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });

  /** Build a straight blueprint segment between two lon, lat points at `level`; wait for its price
   * to settle (the terrain lands after the edit); its US$M a km of its length. */
  const segment = (a, b, level) => page.evaluate(async ([a, b, level]) => {
    const tw = window.tw, c = tw.client;
    tw.map.jumpTo({ center: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], zoom: 13 });
    await new Promise((d) => setTimeout(d, 1500));
    const before = new Set(tw.world.value.edges.map((e) => e.id));
    const p = tw.lngLatToPack(a[0], a[1]), q = tw.lngLatToPack(b[0], b[1]);
    const r = await c.edit({ op: 'route', from: { kind: 'free', x: p.x, y: p.y, level }, to: { kind: 'free', x: q.x, y: q.y, level }, pis: [], single: false });
    if (!r.ok) return { error: r.issues.join(', ') };
    const mine = () => tw.world.value.edges.filter((e) => !before.has(e.id));
    let last = -1;
    for (let k = 0; k < 40; k++) {
      const cost = mine().reduce((s, e) => s + e.cost, 0);
      if (cost === last && k > 5) break;
      last = cost;
      await new Promise((d) => setTimeout(d, 250));
    }
    const es = mine();
    const cost = es.reduce((s, e) => s + e.cost, 0) / 1e6, km = es.reduce((s, e) => s + e.lengthM, 0) / 1000;
    return { perKm: cost / km, km, quote: tw.world.value.money.blueprintCost / 1e6, rows: tw.world.value.money.blueprintItems.filter((r) => r.kind === 6).map((r) => [r.level, Math.round(r.cost / 1e6)]) };
  }, [a, b, level]);

  const plain = await segment(flatA, flatB, 0);
  check(!plain.error && Math.abs(plain.perKm - 45) < 0.5, `flat ground at 0: ${plain.perKm?.toFixed(2) ?? plain.error} US$M a km (45)`);
  const ground = await segment(mtn.a, mtn.b, 0);
  check(!ground.error && Math.abs(ground.perKm - 112.5) < 2, `mountain ground at 0: ${ground.perKm?.toFixed(2) ?? ground.error} US$M a km (112.5)`);
  check(ground.rows?.some(([cls]) => cls === 3), `the blueprint breakdown has a mountain row (${JSON.stringify(ground.rows)})`);
  const dy = 0.004; // a parallel run 450 m south, for the tunnel
  const tunnel = await segment([mtn.a[0], mtn.a[1] - dy], [mtn.b[0], mtn.b[1] - dy], -2);
  check(!tunnel.error && Math.abs(tunnel.perKm - 99) < 0.5, `a bored tunnel at -2 there: ${tunnel.perKm?.toFixed(2) ?? tunnel.error} US$M a km (99)`);

  // Construct charges what the quote says
  const built = await page.evaluate(async () => {
    const tw = window.tw;
    const quote = tw.world.value.money.blueprintCost / 1e6;
    const r = await tw.client.edit({ op: 'constructAll' });
    return { ok: r.ok, charge: r.charge, quote };
  });
  check(built.ok && Math.abs(built.charge - built.quote) < 0.01, `Construct charged ${built.charge?.toFixed(1)} against the quote ${built.quote?.toFixed(1)} US$M`);

  // screenshots: the hillshade in both themes, then the tool's tag over a mountain segment
  for (const theme of ['light', 'dark']) {
    await page.evaluate(([lng, lat, theme]) => {
      window.tw.shade.value = theme;
      window.tw.map.jumpTo({ center: [lng, lat], zoom: 11.5 });
    }, [11.39, 47.28, theme]);
    await page.waitForTimeout(3500);
    await page.screenshot({ path: join(OUT, `hillshade-${theme}.png`) });
  }
  await page.evaluate(() => { window.tw.shade.value = 'light'; });
  const shade = await page.evaluate(() => window.tw.map.getLayer('hillshade') && window.tw.map.getLayoutProperty('hillshade', 'visibility'));
  check(shade === 'visible', `the hillshade layer is shown with Terrain on (${shade})`);
  // a run 450 m north of the one built, the same width
  const lng = mtn.a[0], lat = mtn.a[1] + 0.004, len = mtn.b[0] - mtn.a[0];
  await page.evaluate(([lng, lat, len]) => void window.tw.map.jumpTo({ center: [lng + len / 2, lat], zoom: 14 }), [lng, lat, len]);
  await page.waitForTimeout(1500);
  await page.keyboard.press('1');
  await page.waitForTimeout(400);
  const px = async (x, y) => page.evaluate(([x, y]) => { const r = window.tw.map.getCanvas().getBoundingClientRect(), p = window.tw.map.project([x, y]); return [r.left + p.x, r.top + p.y]; }, [x, y]);
  const [x0, y0] = await px(lng, lat), [x1, y1] = await px(lng + len, lat);
  await page.mouse.click(x0, y0);
  await page.waitForTimeout(500);
  await page.mouse.move(x1, y1, { steps: 6 });
  await page.waitForTimeout(2500);
  await page.mouse.move(x1 + 1, y1, { steps: 1 });
  await page.waitForTimeout(1200);
  const tag = await page.evaluate(() => [...document.querySelectorAll('.tool-tag')].map((t) => t.textContent).join(' | '));
  check(/Mountains 2\.5x|Hills 1\.6x|Gentle slope 1\.25x/.test(tag), `the track tool's tag names the ground (${tag})`);
  await page.screenshot({ path: join(OUT, 'mountain-tag.png') });
  await page.keyboard.press('Escape');

  check(errors.length === 0, `no page errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
