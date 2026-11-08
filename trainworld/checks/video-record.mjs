// T-2774: records the launch video of the game (trainsworld): the station zoomed in, then a smooth zoom out to the whole world as the landing screen shows it,
// then the landing screen's animated logo fades in over it. Headed Chrome with its own temp profile, 1920x1080, the page's own
// tab recorded with its sound (getDisplayMedia, preferCurrentTab) into a webm; convert it with the ffmpeg line at the end.
//   node trainworld/checks/dev-nowatch.mjs 6173          (your own Vite, first)
//   TW_CORES=4 node trainworld/checks/video-record.mjs --port 6173 --out <dir>
// Options: --zoom 17 (start zoom) --lng/--lat (the station) --hold 5 --travel 10 --accel 1.5 --stop 1 --fade 0.4 --tail 3
//   --save <file> --endzoom <z> --endlng/--endlat --dry (no recording: camera pass only) --nowarm
// The recorded view is set up from outside the app (tw.display, tw.map, CSS, the layers of map/cityBoundaries.ts hidden); the
// logo is app/src/videoLogo.tsx, loaded through the dev server. Nothing here changes how the game plays.
import { launchChrome, importSave, demandFinal, GL } from './lib.mjs';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';

const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i < 0 ? d : (process.argv[i + 1]?.startsWith('--') || i + 1 >= process.argv.length) ? true : process.argv[i + 1]; };
const PORT = arg('port', 6173);
const OUT = arg('out', '.');
const SAVE = arg('save', 'C:/Users/anita/Downloads/trainsworld-north-america-day-473.save');
const P = {
  zoom: +arg('zoom', 18), lng: +arg('lng', -73.9948), lat: +arg('lat', 40.7507),
  endZoom: +arg('endzoom', 2.986), endLng: +arg('endlng', -73.9948), endLat: +arg('endlat', 20),
  hold: +arg('hold', 5), travel: +arg('travel', 14), accel: +arg('accel', 1.5), stop: +arg('stop', 1), fade: +arg('fade', 0.4), tail: +arg('tail', 3),
};
await mkdir(OUT, { recursive: true });

const { ctx, close } = await launchChrome({
  prefix: 'tw-video-', size: [1920, 1080], headless: false, viewport: null,
  args: [...GL, '--autoplay-policy=no-user-gesture-required', '--auto-accept-this-tab-capture', '--kiosk', '--disable-features=CalculateNativeWinOcclusion', '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding'],
});
const page = ctx.pages()[0] ?? await ctx.newPage();
page.on('console', (m) => { if (m.type() === 'error' && !/favicon|openfreemap|AJAXError/.test(m.text())) console.log('ERR', m.text().slice(0, 200)); });
page.on('pageerror', (e) => console.log('PAGEERR', e.message.slice(0, 200)));
const chunks = [];
await page.exposeFunction('twSaveChunk', (b64) => { chunks.push(Buffer.from(b64, 'base64')); });

try {
  await page.goto(`http://localhost:${PORT}/?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.map, null, { timeout: 120000 });
  await importSave(page, await readFile(SAVE));
  const t0 = Date.now();
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { timeout: 900000, polling: 1000, lines: 0 });
  console.log(`game loaded and solved in ${((Date.now() - t0) / 1000).toFixed(0)} s`);

  const near = await page.evaluate(([lng, lat]) => window.tw.world.value.save.stations.map((s) => ({ name: s.name, lng: s.lng, lat: s.lat, m: Math.round(Math.hypot((s.lng - lng) * 84300, (s.lat - lat) * 110950)) })).sort((a, b) => a.m - b.m).slice(0, 12), [P.lng, P.lat]);
  console.log('nearest stations', JSON.stringify(near));

  // the view: no UI, no labels, no region shade or outlines, light map, flat world, line colours, platform fill but not train fill
  await page.addStyleTag({ content: '#dock-root,#topbar-root,#tools-root,#debug-root,.maplibregl-ctrl-bottom-right,.maplibregl-ctrl-bottom-left,.maplibregl-ctrl-top-right,.maplibregl-ctrl-top-left,.toast,#toasts,.map-top-left,.map-top-row,.map-find,.demand-selector,.map-sky,.realm-buttons,.zone-loads,.intercity-labels,.touch-bar-host{display:none!important}#game{display:block!important}#main{position:fixed!important;inset:0!important;width:100vw!important;height:100vh!important}#map{position:absolute!important;inset:0!important}*{cursor:none!important}' });
  await page.evaluate(async (P) => {
    const tw = window.tw, map = tw.map;
    tw.display.value = { ...tw.display.value, trackColour: 'line', trains: true, trainLoad: false, platformCrowds: 'patch', sound: true, faces: true, basemapLabels: false, stationNames: false, stationUse: 'none', commuters: false, projection: 'flat', foldedRows: [] };
    tw.clock.setSpeed(0); // normal speed, running
    await new Promise((r) => setTimeout(r, 500)); map.resize(); await new Promise((r) => setTimeout(r, 500));
    const hideLayers = () => { for (const l of map.getStyle().layers) if (/^tw-(city|zone)/.test(l.id) && map.getLayoutProperty(l.id, 'visibility') !== 'none') map.setLayoutProperty(l.id, 'visibility', 'none'); };
    hideLayers();
    setInterval(hideLayers, 300);
    const V = window.__video = { P, frames: [], logo: null };
    // zoom distance covered by time t into the zoom-out: constant acceleration over `accel`, constant rate, constant braking over `stop`
    const Z = P.zoom - P.endZoom, T = P.travel, a = P.accel, d = P.stop, v = Z / (T - a / 2 - d / 2);
    V.dist = (t) => t <= 0 ? 0 : t >= T ? Z : t < a ? v * t * t / (2 * a) : t < T - d ? v * a / 2 + v * (t - a) : Z - v * (T - t) * (T - t) / (2 * d);
    const sm = (x) => { x = Math.max(0, Math.min(1, x)); return x * x * (3 - 2 * x); };
    V.view = (t) => { // t seconds into the zoom-out
      const dz = V.dist(t), w = sm((dz / Z - 0.9) / 0.1);
      return { zoom: P.zoom - dz, center: [P.lng + (P.endLng - P.lng) * w, P.lat + (P.endLat - P.lat) * w] };
    };
    V.set = (t) => { const s = V.view(t); map.jumpTo({ center: s.center, zoom: s.zoom, bearing: 0, pitch: 0 }); };
    V.tiles = () => new Promise((res) => { let ok = 0; const t0 = performance.now(); const poll = () => { if (map.areTilesLoaded()) ok++; else ok = 0; if (ok >= 3 || performance.now() - t0 > 20000) res(ok >= 3); else setTimeout(poll, 100); }; poll(); });
    // a pass over the whole path in small steps, waiting for every tile at each step
    V.warm = async () => {
      const n = 80;
      for (let i = n; i >= 0; i--) { V.set(P.travel * i / n); await V.tiles(); await new Promise((r) => setTimeout(r, 120)); }
      for (let i = 0; i <= n; i++) { V.set(P.travel * i / n); await V.tiles(); await new Promise((r) => setTimeout(r, 120)); }
      V.set(0); await V.tiles();
    };
    // the run in real time: holds, zooms out, holds; frame gaps and frames with unloaded tiles are kept
    V.run = (withLogo) => new Promise((res) => {
      const total = P.hold + P.travel + P.fade + P.tail;
      V.frames = []; let unloaded = 0, last = 0;
      const t0 = performance.now();
      const step = (now) => {
        const t = (now - t0) / 1000;
        if (last) V.frames.push(now - last);
        last = now;
        V.set(Math.max(0, t - P.hold));
        if (!map.areTilesLoaded()) unloaded++;
        if (withLogo && V.logo) V.logo.fade(Math.max(0, Math.min(1, (t - P.hold - P.travel) / P.fade)));
        if (t < total) requestAnimationFrame(step); else res({ seconds: t, frames: V.frames.length, unloaded });
      };
      requestAnimationFrame(step);
    });
    V.stats = () => { const f = [...V.frames].sort((x, y) => x - y); const q = (p) => f[Math.min(f.length - 1, Math.floor(p * f.length))]; return { n: f.length, p50: q(0.5), p95: q(0.95), p99: q(0.99), max: f[f.length - 1], over25: f.filter((x) => x > 25).length, over40: f.filter((x) => x > 40).length }; };
    const m = await import('/src/videoLogo.tsx');
    V.logo = m.mountLogo();
    // recorder: the tab with its sound, saved in 1 s chunks through twSaveChunk
    V.record = async () => {
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: { ideal: 60, max: 60 }, width: 1920, height: 1080 }, audio: true, preferCurrentTab: true });
      const types = ['video/webm;codecs=avc1,opus', 'video/webm;codecs=vp9,opus', 'video/webm'];
      const mimeType = types.find((t) => MediaRecorder.isTypeSupported(t));
      const rec = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 24_000_000, audioBitsPerSecond: 192_000 });
      rec.ondataavailable = async (e) => { if (e.data.size) { const b = new Uint8Array(await e.data.arrayBuffer()); let s = ''; for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode(...b.subarray(i, i + 0x8000)); V.pending = (V.pending ?? Promise.resolve()).then(() => window.twSaveChunk(btoa(s))); } };
      rec.start(1000);
      V.rec = rec; V.stream = stream;
      const vt = stream.getVideoTracks()[0], at = stream.getAudioTracks()[0];
      return { mimeType, video: vt.getSettings(), audio: !!at };
    };
    V.stopRecord = () => new Promise((res) => { V.rec.onstop = async () => { await V.pending; V.stream.getTracks().forEach((t) => t.stop()); res(true); }; V.rec.stop(); });
  }, P);

  if (!arg('nowarm', false)) { console.log('warm-up pass'); await page.evaluate(() => window.__video.warm()); }
  await page.evaluate(() => { window.__video.set(0); return window.__video.tiles(); });
  await page.waitForTimeout(1500);
  const dry = await page.evaluate(() => window.__video.run(false));
  console.log('dry run', JSON.stringify(dry), JSON.stringify(await page.evaluate(() => window.__video.stats())));
  await page.screenshot({ path: join(OUT, 'last-dry.png') });
  if (arg('dry', false)) { await close(); process.exit(0); }

  await page.evaluate(() => { window.__video.set(0); return window.__video.tiles(); });
  await page.waitForTimeout(1500);
  const info = await page.evaluate(() => window.__video.record());
  console.log('recording', JSON.stringify(info));
  const res = await page.evaluate(() => window.__video.run(true));
  console.log('run', JSON.stringify(res), JSON.stringify(await page.evaluate(() => window.__video.stats())));
  await page.screenshot({ path: join(OUT, 'last-end.png') });
  await page.evaluate(() => window.__video.stopRecord());
  const file = join(OUT, 'trainsworld.webm');
  await writeFile(file, Buffer.concat(chunks));
  console.log('saved', file, (Buffer.concat(chunks).length / 1e6).toFixed(1), 'MB');
} finally {
  await close().catch(() => {});
}
