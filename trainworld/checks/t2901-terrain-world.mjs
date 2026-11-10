// T-2901 stage 3: the world's terrain read by degree block, in a new game paused.
//   1. a segment on mountain cells in the Atlas (no zone) prices as mountain ground (262.5 US$M a km)
//   2. the tint shows there at zoom 10 and is gone below zoom 9, with Terrain on and a build tool; the
//      hillshade over the plain near Marrakech is flattened from block data
//   3. a view over open sea fetches no terrain block and no 404
//   4. a block that fails (404) or answers with bytes that are not a block never prices flat: Construct
//      is refused with TerrainMissing until the block lands, then charges the true price
// Screenshots to artifacts/trainworld/t2901/. About a minute.
// Usage: node trainworld/checks/t2901-terrain-world.mjs [url] (default http://localhost:6240/)
import { readFileSync, mkdirSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6240/';
const sep = BASE.includes('?') ? '&' : '?';
const OUT = join(ARTIFACTS, 't2901');
mkdirSync(OUT, { recursive: true });

let failed = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) failed++; };
const hold = (ms) => new Promise((d) => setTimeout(d, ms));

/** the classes of a block file */
function block(la, lo) {
  let b = readFileSync(new URL(`../data/terrain-world/${la}_${lo}.terrain`, import.meta.url));
  if (b[0] === 0x1f && b[1] === 0x8b) b = gunzipSync(b);
  const [w, h] = [b.readUInt32LE(4), b.readUInt32LE(8)], [c0, r0] = [b.readInt32LE(12), b.readInt32LE(16)], per = b.readUInt32LE(20);
  const cls = new Uint8Array(w * h);
  let i = 24, k = 0;
  while (i < b.length && k < cls.length) {
    const c = b[i++];
    let m = 0, shift = 0;
    for (;;) { const x = b[i++]; m += (x & 0x7f) * 2 ** shift; if (!(x & 0x80)) break; shift += 7; }
    cls.fill(c, k, Math.min(cls.length, k + m));
    k += m;
  }
  return { w, h, c0, r0, per, cls };
}

/** a run of `n` cells of `want` along a row, nearest (lon, lat) */
function runNear(g, want, n, lon, lat) {
  let best = null;
  for (let r = 1; r < g.h - 1; r++) {
    let run = 0;
    for (let c = 0; c < g.w; c++) {
      run = g.cls[r * g.w + c] === want ? run + 1 : 0;
      if (run < n) continue;
      const a = [(g.c0 + c - n + 1.5) / g.per - 180, (g.r0 + r + 0.5) / g.per - 90], e = [(g.c0 + c - 0.5) / g.per - 180, a[1]];
      const d = Math.hypot((a[0] + e[0]) / 2 - lon, a[1] - lat);
      if (!best || d < best.d) best = { a, b: e, d };
    }
  }
  return best;
}

const atlas = runNear(block(31, -8), 3, 8, -7.9, 31.1);
check(!!atlas, `mountain cells in the High Atlas, block 31_-8 (${atlas?.a.map((v) => v.toFixed(4))} to ${atlas?.b.map((v) => v.toFixed(4))})`);
const alps = runNear(block(47, 11), 3, 6, 11.39, 47.3);

const { ctx, close } = await launchChrome({ prefix: 'tw-t2901-', size: [1300, 900] });
try {
  // block 47_11: "pass", "gone" (a 404), or "junk" (a 200 that is not a block)
  const net = { alps: 'pass' };
  await ctx.route(/terrain-world\/47_11\.terrain/, async (route) => {
    try {
      if (net.alps === 'gone') return await route.fulfill({ status: 404, body: '' });
      if (net.alps === 'junk') return await route.fulfill({ status: 200, body: '<html></html>' });
      await route.continue();
    } catch { /* the page went */ }
  });
  const page = await ctx.newPage();
  const errors = [];
  collectErrors(page, errors, /favicon|openfreemap|AJAXError|ERR_INTERNET_DISCONNECTED|Failed to load resource/);
  const terrainResponses = [];
  page.on('response', (r) => { if (r.url().includes('/terrain-world/')) terrainResponses.push([r.url().split('/').pop(), r.status()]); });
  await page.addInitScript(() => {
    try {
      const s = JSON.parse(localStorage.getItem('trainworld-ui') || '{}');
      s.display = { ...(s.display ?? {}), hillshade: true };
      s.v = s.v ?? 10;
      localStorage.setItem('trainworld-ui', JSON.stringify(s));
    } catch { /* private window */ }
  });
  await page.goto(`${BASE}${sep}debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.addStyleTag({ content: '#debug-root { display: none !important; }' });

  /** a straight blueprint segment; its US$M a km once the price has settled */
  const segment = (a, b, level = 0) => page.evaluate(async ([a, b, level]) => {
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
    return { perKm: es.reduce((s, e) => s + e.cost, 0) / 1e6 / (es.reduce((s, e) => s + e.lengthM, 0) / 1000) };
  }, [a, b, level]);

  // 1. the Atlas
  const ground = await segment(atlas.a, atlas.b);
  check(!ground.error && Math.abs(ground.perKm - 262.5) < 12, `Atlas track at 0 prices as mountain: ${ground.perKm?.toFixed(1) ?? ground.error} US$M a km (262.5)`);

  // 2. the tint
  const setTool = (t) => page.evaluate(async (t) => { (await import('/src/game/ui.ts')).tool.value = t; }, t);
  await setTool('track');
  const tintAt = async (zoom) => {
    await page.evaluate(([lng, lat, zoom]) => window.tw.map.jumpTo({ center: [lng, lat], zoom }), [(atlas.a[0] + atlas.b[0]) / 2, atlas.a[1], zoom]);
    await hold(2500);
    return page.evaluate(() => {
      const m = window.tw.map;
      return { vis: m.getLayer('terrain-tint') ? m.getLayoutProperty('terrain-tint', 'visibility') : 'none' };
    });
  };
  const z10 = await tintAt(10);
  await page.screenshot({ path: join(OUT, 'atlas-z10.png') });
  check(z10.vis === 'visible', `tint shows at zoom 10 over the Atlas (${z10.vis})`);
  const grids = await page.evaluate(async () => {
    const t = await import('/src/game/terrain.ts');
    return { size: t.terrainGrids.size, here: t.terrainGrids.has(t.blockKey(-7.9, 31.1)) };
  });
  check(grids.here, `the block under the view reached the page (${JSON.stringify(grids)})`);
  // the hillshade flattened by block: the plain north of the Atlas, near Marrakech
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-8.0, 31.6], zoom: 11.5 }));
  let flat = null;
  for (let k = 0; k < 30 && !(flat?.flattened > 0); k++, await hold(1000)) flat = await page.evaluate(() => window.twDemFlat);
  check(flat?.flattened > 0, `hillshade tiles flattened on block data (${JSON.stringify(flat)})`);
  const z8 = await tintAt(8);
  check(z8.vis !== 'visible', `tint is off below zoom 9 (${z8.vis})`);

  // 3. open sea: no block fetched, no error
  const before = terrainResponses.length;
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-35, 30], zoom: 9 }));
  await hold(4000);
  const seaFetches = terrainResponses.slice(before);
  check(seaFetches.length === 0, `mid-Atlantic view at zoom 9 fetches no terrain block (${JSON.stringify(seaFetches)})`);
  const failedFetches = terrainResponses.filter(([, s]) => s >= 400);
  check(failedFetches.length === 0, `no terrain response failed so far (${JSON.stringify(failedFetches.slice(0, 5))})`);

  // 4. a listed block that fails
  await setTool('hand');
  await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 7 }), [-35, 30]);
  await page.evaluate(() => window.tw.client.edit({ op: 'constructAll' }).catch(() => {}));
  for (const mode of ['gone', 'junk']) {
    net.alps = mode;
    if (mode === 'gone') {
      const seg = await segment(alps.a, alps.b);
      check(!seg.error && seg.perKm < 60, `block 47_11 ${mode}: its blueprint is priced flat for now (${seg.perKm?.toFixed(1) ?? seg.error} US$M a km), not trusted`);
    }
    const r = await page.evaluate(async () => {
      const b0 = window.tw.world.value.money.builtValue;
      const r = await window.tw.client.edit({ op: 'constructAll' });
      await new Promise((d) => setTimeout(d, 400));
      return { ok: r.ok, issues: r.issues, charge: r.charge, built: window.tw.world.value.money.builtValue, b0 };
    });
    check(!r.ok && r.issues.join() === 'TerrainMissing' && r.charge === 0 && r.built === r.b0, `block 47_11 ${mode}: Construct refused, nothing paid (${JSON.stringify(r)})`);
    const tag = await page.evaluate(() => document.querySelector('.status-tag .demand-busy')?.textContent ?? '');
    check(/terrain/i.test(tag), `block 47_11 ${mode}: the tag says so (${tag})`);
  }
  net.alps = 'pass';
  const landed = await page.evaluate(async () => {
    const tw = window.tw;
    const r = await tw.client.edit({ op: 'constructAll' });
    return { ok: r.ok, charge: r.charge, issues: r.issues };
  });
  check(landed.ok && landed.charge > 60, `block 47_11 back: Construct charges mountain ground (${JSON.stringify(landed)})`);

  check(errors.length === 0, `no errors on the page (${errors.slice(0, 3).join(' | ')})`);
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
