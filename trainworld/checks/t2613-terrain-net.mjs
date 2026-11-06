// T-2613: terrain on a bad connection, in a new game paused at Innsbruck (mountain ground, `ins`).
//   1. every `.terrain` download fails: the status tag says "Terrain could not load" with its hover;
//      a blueprint on mountain ground at ground level is priced flat, and Construct is refused with
//      "The terrain here did not load. Try again." (nothing paid);
//   2. the downloads come back, 6 s each: the retry shows "Downloading terrain…", the tag goes once
//      they land, the blueprint is priced on the mountain, and Construct charges that quote;
//   3. Terrain (hill shading) on with its tiles 5 s each: "Downloading terrain…" while only they load.
// Screenshots to artifacts/trainworld/terrainnet/. About two minutes.
// Usage: node trainworld/checks/t2613-terrain-net.mjs [url] (default http://localhost:6491/)

import { mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { gunzipSync } from 'node:zlib';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6491/';
const sep = BASE.includes('?') ? '&' : '?';
const OUT = join(ARTIFACTS, 'terrainnet');
mkdirSync(OUT, { recursive: true });
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) failed++; };

/** a run of `n` mountain cells along a row of Innsbruck's terrain grid, nearest (lon, lat) (as t1270-terrain.mjs) */
function mountainRun(n, lon, lat) {
  let b = readFileSync(new URL('../data/terrain/ins.terrain', import.meta.url));
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
  let best = null;
  for (let r = 1; r < h - 1; r++) {
    let run = 0;
    for (let c = 0; c < w; c++) {
      run = cls[r * w + c] === 3 ? run + 1 : 0;
      if (run < n) continue;
      const a = [(c0 + c - n + 1.5) / per - 180, (r0 + r + 0.5) / per - 90], e = [(c0 + c - 0.5) / per - 180, a[1]];
      const d = Math.hypot((a[0] + e[0]) / 2 - lon, a[1] - lat);
      if (!best || d < best.d) best = { a, b: e, d };
    }
  }
  return best;
}
const mtn = mountainRun(6, 11.39, 47.30);

// what the network does to each kind of download: "block", "slow" (held, then sent on) or "pass"
const net = { terrain: 'block', shade: 'pass' };
const hold = (ms) => new Promise((d) => setTimeout(d, ms));

const { ctx, close } = await launchChrome({ prefix: 'tw-t2613-', size: [1300, 900] });
try {
  await ctx.route(/\/terrain\/[a-z0-9_-]+\.terrain/, async (route) => {
    try {
      if (net.terrain === 'block') return await route.abort('internetdisconnected');
      if (net.terrain === 'slow') await hold(6000);
      await route.continue();
    } catch { /* the page went */ }
  });
  await ctx.route(/elevation-tiles-prod/, async (route) => {
    try {
      if (net.shade === 'slow') await hold(5000);
      await route.continue();
    } catch { /* the page went */ }
  });
  const page = await ctx.newPage();
  const errors = [];
  collectErrors(page, errors, /favicon|openfreemap|AJAXError|ERR_INTERNET_DISCONNECTED|Failed to load resource/);
  await page.goto(`${BASE}${sep}debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.addStyleTag({ content: '#debug-root { display: none !important; }' });
  const tag = () => page.evaluate(() => {
    const t = document.querySelector('.status-tag');
    return t ? { text: t.querySelector('.demand-busy')?.textContent ?? '', tip: t.getAttribute('data-tip'), spinner: !!t.querySelector('svg, .bunny-train, [class*=bunny]') } : null;
  });
  const tagIs = async (text, ms) => {
    for (const end = Date.now() + ms; Date.now() < end; await hold(250)) if ((await tag())?.text === text) return true;
    return false;
  };
  const shot = (name) => page.screenshot({ path: join(OUT, name), clip: { x: 0, y: 0, width: 1300, height: 220 } });

  // 1. terrain blocked
  // (Innsbruck's realm is not shown in a game started in New York, so the view fetches nothing
  // there ahead: the blueprint is what needs its terrain)
  await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 13 }), [(mtn.a[0] + mtn.b[0]) / 2, mtn.a[1]]);
  await hold(3000);
  const segment = () => page.evaluate(async ([a, b]) => {
    const tw = window.tw, c = tw.client;
    const before = new Set(tw.world.value.edges.map((e) => e.id));
    const p = tw.lngLatToPack(a[0], a[1]), q = tw.lngLatToPack(b[0], b[1]);
    const r = await c.edit({ op: 'route', from: { kind: 'free', x: p.x, y: p.y, level: 0 }, to: { kind: 'free', x: q.x, y: q.y, level: 0 }, pis: [], single: false });
    if (!r.ok) return { error: r.issues.join(', ') };
    await new Promise((d) => setTimeout(d, 1500));
    const es = tw.world.value.edges.filter((e) => !before.has(e.id));
    return { perKm: es.reduce((s, e) => s + e.cost, 0) / 1e6 / (es.reduce((s, e) => s + e.lengthM, 0) / 1000) };
  }, [mtn.a, mtn.b]);
  const perKm = () => page.evaluate(() => {
    const es = window.tw.world.value.edges.filter((e) => !e.built);
    return es.reduce((s, e) => s + e.cost, 0) / 1e6 / (es.reduce((s, e) => s + e.lengthM, 0) / 1000);
  });
  const seg = await segment();
  // flat ground's 22.5 a km plus the roads it crosses
  check(!seg.error && seg.perKm < 60, `blocked: the mountain blueprint is priced as flat ground (${seg.perKm?.toFixed(1) ?? seg.error} US$M a km)`);
  const failedTag = await tagIs('Terrain could not load', 90000);
  const ground = await page.evaluate(async () => (await import('/src/game/terrain.ts')).groundNet.value);
  check(failedTag, `blocked: the tag says "Terrain could not load" (${JSON.stringify(await tag())}, ${JSON.stringify(ground)})`);
  const t1 = await tag();
  check(t1?.tip === 'Check your internet connection. It is tried again within 30 s.', `its hover (${t1?.tip})`);
  check(!t1?.spinner, 'no spinner on the failure');
  await shot('t2613-failed.png');
  const cash0 = await page.evaluate(() => window.tw.world.value.money.builtValue);
  const refused = await page.evaluate(async () => {
    const r = await (await import('/src/game/world.ts')).edit({ op: 'constructAll' });
    await new Promise((d) => setTimeout(d, 400));
    return { ok: r.ok, issues: r.issues, charge: r.charge, notice: document.body.innerText.includes('The terrain here did not load. Try again.') };
  });
  const cash1 = await page.evaluate(() => window.tw.world.value.money.builtValue);
  check(!refused.ok && refused.issues.join() === 'TerrainMissing' && refused.charge === 0 && cash1 === cash0, `blocked: Construct is refused, nothing built or paid (${JSON.stringify(refused)}, built ${cash0} -> ${cash1})`);
  check(refused.notice, 'blocked: the notice says "The terrain here did not load. Try again."');
  await page.screenshot({ path: join(OUT, 't2613-refused.png') });

  // 2. slow but working: the next retry downloads, the tag says so, then goes
  net.terrain = 'slow';
  check(await tagIs('Downloading terrain…', 45000), `slow: the tag says "Downloading terrain…" (${JSON.stringify(await tag())})`);
  await shot('t2613-downloading.png');
  let gone = false;
  for (const end = Date.now() + 60000; Date.now() < end && !gone; await hold(500)) gone = !/terrain/i.test((await tag())?.text ?? '');
  check(gone, `slow: the tag goes once the terrain is in (${JSON.stringify(await tag())})`);
  const after = await perKm();
  check(after > 200, `the blueprint is priced on the mountain once it lands (${after.toFixed(1)} US$M a km, about 247.5)`);
  const built = await page.evaluate(async () => {
    const quote = window.tw.world.value.money.blueprintCost / 1e6;
    const r = await window.tw.client.edit({ op: 'constructAll' });
    return { ok: r.ok, charge: r.charge, quote };
  });
  check(built.ok && Math.abs(built.charge - built.quote) < 0.01, `Construct now charges the quote (${built.charge?.toFixed(1)} against ${built.quote?.toFixed(1)} US$M)`);

  // 3. the hillshade's tiles slow, the grids in hand: the tag still says it
  net.terrain = 'pass';
  net.shade = 'slow';
  await page.evaluate(async () => {
    const ui = await import('/src/game/ui.ts');
    ui.display.value = { ...ui.display.value, hillshade: true };
    window.tw.map.jumpTo({ center: [11.25, 47.25], zoom: 11 });
  });
  const shadeShown = await tagIs('Downloading terrain…', 15000);
  const state = await page.evaluate(async () => { const t = await import('/src/game/terrain.ts'); return { grids: t.groundNet.value, shade: t.shadeDownloads.value }; });
  check(shadeShown && state.shade > 0, `hillshade tiles slow: "Downloading terrain…" (${JSON.stringify(state)})`);
  await shot('t2613-shade.png');
  net.shade = 'pass';
  gone = false;
  for (const end = Date.now() + 40000; Date.now() < end && !gone; await hold(500)) gone = !/terrain/i.test((await tag())?.text ?? '');
  check(gone, 'hillshade: the tag goes once its tiles are in');

  check(errors.length === 0, `no page errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
