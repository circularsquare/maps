// T-2890: track is charged the terrain class of the hex drawn under it. In a new game on the Tyrol's
// mixed ground: (1) short straight blueprint segments, the metres the blueprint breakdown charges per
// class against the same 25 m samples classed by the map's own hex lookup; (2) pixels of the tint's
// image against the class of the hex at that pixel. Dev server only (it imports the app's module).
// Usage: node trainworld/checks/t2890-hex.mjs [url] (default http://localhost:6170/)

import { collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6170/';
const sep = BASE.includes('?') ? '&' : '?';
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) failed++; };

const { ctx, close } = await launchChrome({ prefix: 'tw-t2890-' });
try {
  const page = await ctx.newPage();
  const errors = [];
  collectErrors(page, errors);
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
  await page.evaluate(() => window.tw.map.jumpTo({ center: [11.385, 47.2735], zoom: 11 }));
  await page.waitForTimeout(6000);
  await page.keyboard.press('1');
  await page.waitForTimeout(1000);

  const priced = await page.evaluate(async () => {
    const tw = window.tw, { hexClass } = await import('/src/game/terrain.ts');
    let seed = 2890;
    const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
    const quoted = async () => {
      const q = await tw.client.query({ what: 'blueprintQuote' }), by = [0, 0, 0, 0];
      let wet = 0;
      for (let i = 0; i + 6 < q.length; i += 7) if (q[i] === 0) { if (q[i + 3] === 1) wet += q[i + 5]; else by[q[i + 4]] += q[i + 5]; }
      return { by, wet };
    };
    let segments = 0, samples = 0, exact = 0, skipped = 0, mixed = 0;
    const misses = [], issues = {};
    for (let n = 0; n < 150; n++) {
      // a 300-500 m segment; the box is the slope north of Innsbruck, hills and mountains and flat beside
      const lon = 11.28 + rnd() * 0.22, lat = 47.25 + rnd() * 0.05, ang = rnd() * 2 * Math.PI, len = 300 + rnd() * 200;
      const p = tw.lngLatToPack(lon, lat), q = { x: p.x + Math.cos(ang) * len, y: p.y + Math.sin(ang) * len };
      const before = await quoted();
      const r = await tw.client.edit({ op: 'route', from: { kind: 'free', x: p.x, y: p.y, level: 0 }, to: { kind: 'free', x: q.x, y: q.y, level: 0 }, pis: [], single: false });
      if (!r.ok) { skipped++; issues[r.issues?.[0]] = (issues[r.issues?.[0]] ?? 0) + 1; continue; }
      const after = await quoted();
      if (after.wet > before.wet) { skipped++; continue; }
      const got = after.by.map((v, k) => v - before.by[k]);
      // the pack frame is not metres; the charged metres are the fitted length
      const L = got.reduce((s, v) => s + v, 0), m = Math.ceil(L / 25), step = L / m, want = [0, 0, 0, 0];
      for (let j = 0; j < m; j++) {
        const t = (j + 0.5) / m, [lo, la] = tw.packToLngLat(p.x + (q.x - p.x) * t, p.y + (q.y - p.y) * t);
        want[hexClass(lo, la)] += step;
      }
      segments++, samples += m;
      if (want.filter((v) => v > 0).length > 1) mixed++;
      const off = want.reduce((s, v, k) => s + Math.abs(v - got[k]), 0);
      if (off < 0.5 * step) exact++; else misses.push({ lon, lat, len, want: want.map(Math.round), got: got.map(Math.round) });
    }
    return { segments, samples, exact, skipped, mixed, issues, misses: misses.slice(0, 5) };
  });
  check(priced.segments >= 100, `${priced.segments} segments priced, ${priced.samples} points sampled, ${priced.mixed} crossing more than one class (${priced.skipped} on water or refused, skipped ${JSON.stringify(priced.issues)})`);
  check(priced.exact === priced.segments, `every segment charges the classes of the hexes drawn under it: ${priced.exact} of ${priced.segments}${priced.misses.length ? ' ' + JSON.stringify(priced.misses) : ''}`);

  // the image itself, pixel by pixel, against the lookup
  for (const zoom of [15, 13]) {
    await page.evaluate((zoom) => window.tw.map.jumpTo({ center: [11.385, 47.2735], zoom }), zoom);
    await page.waitForTimeout(3500);
    const drawn = await page.evaluate(async () => {
      const { hexClass } = await import('/src/game/terrain.ts');
      const src = window.tw.map.getSource('terrain-tint');
      const img = await createImageBitmap(await (await fetch(src.url)).blob());
      const c = new OffscreenCanvas(img.width, img.height), g = c.getContext('2d');
      g.drawImage(img, 0, 0);
      const data = g.getImageData(0, 0, img.width, img.height).data;
      const [[lonW, north], [lonE], , [, south]] = src.coordinates;
      const my = (lat) => 0.5 - Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360)) / (2 * Math.PI);
      const y0 = my(north), y1 = my(south), W = img.width, H = img.height;
      const alpha = [0, 20, 38, 61];
      let seed = 7, checked = 0, ok = 0;
      const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
      const counts = [0, 0, 0, 0];
      for (let n = 0; n < 2000; n++) {
        const x = Math.floor(rnd() * W), y = Math.floor(rnd() * H);
        const lon = lonW + (lonE - lonW) * (x + 0.5) / W, m = y0 + (y1 - y0) * (y + 0.5) / H;
        const lat = (Math.atan(Math.sinh(Math.PI * (1 - 2 * m))) * 180) / Math.PI;
        const k = hexClass(lon, lat);
        checked++, counts[k]++;
        if (Math.abs(data[(y * W + x) * 4 + 3] - alpha[k]) <= 2) ok++;
      }
      return { checked, ok, counts, W, H };
    });
    check(drawn.ok === drawn.checked, `z${zoom}: ${drawn.ok} of ${drawn.checked} pixels of the image (${drawn.W}x${drawn.H}) show their hex's class ${JSON.stringify(drawn.counts)}`);
  }
  check(errors.length === 0, `no page errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
