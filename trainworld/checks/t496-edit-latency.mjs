// T-496: click-to-drawn for station placement on the real save (notes/T-496.md), and a visual and
// buffer comparison against an older network.ts.
//   PLAYWRIGHT_MODULE=file:///.../playwright/index.mjs  MODE=measure|visual  BASE=http://localhost:6140/
//   LABEL=<name>  RATE=paused|960  N=12     node trainworld/checks/t496-edit-latency.mjs
// visual needs the old code at app/src/map/network_t496old.ts (git show <rev>:trainworld/app/src/map/network.ts),
// removed again afterwards. Output: ../artifacts/trainworld/t496/.
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE);
const BASE = process.env.BASE ?? 'http://localhost:6140/';
const MODE = process.env.MODE ?? 'measure';
const LABEL = process.env.LABEL ?? 'run';
const RATE = process.env.RATE ?? 'paused';
const N = Number(process.env.N ?? 12);
const OUT = 'C:/Users/anita/projects/maps/artifacts/trainworld/t496';
const SAVE = 'C:/Users/anita/projects/maps/trainworld/T-007-real-nyc.save';
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const profile = await mkdtemp(join(tmpdir(), 'tw-t496-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
await ctx.addInitScript(() => {
  try { localStorage.setItem('trainworld-ui', JSON.stringify({ display: { dayShown: 'weekday' }, v: 2 })); } catch {}
  window.__t = { down: 0, world: 0, frame: 0 };
  addEventListener('mousedown', () => { window.__t = { down: performance.now(), world: 0, frame: 0 }; }, true);
  const hook = setInterval(() => {
    const tw = window.tw;
    if (!tw?.world) return;
    clearInterval(hook);
    let last = tw.world.peek();
    tw.world.subscribe((w) => {
      if (w === last) return;
      last = w;
      const t = window.__t;
      if (!t.down || t.world) return;
      t.world = performance.now();
      // the frame after the state was taken: rAF, then a task after it (the frame's work done)
      requestAnimationFrame(() => { const ch = new MessageChannel(); ch.port1.onmessage = () => { t.frame = performance.now(); }; ch.port2.postMessage(0); });
    });
  }, 20);
});
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.text().startsWith('t496')) log(m.text()); if (m.type() === 'error' && !/openfreemap|AJAXError|favicon/.test(m.text())) errors.push(m.text()); });
const out = { label: LABEL, rate: RATE, errors };
try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(1500);
  await page.evaluate(() => { const c = window.tw.client, e = c.edit.bind(c); c.edit = (op) => { if (window.__t && !window.__t.sent) window.__t.sent = performance.now(); return e(op); }; });
  await page.evaluate(() => { const c = window.tw.client, o = c.onState; c.onState = (s) => { if (!s.service) window.__s = s; o(s); }; });
  const bytes = await readFile(SAVE);
  await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  log('loaded');
  await page.waitForFunction(() => { const d = window.twDemand, s = d?.log?.at(-1), v = d?.view?.(); return s && v && v.version === s.version && v.final && !d.client.waiting; }, undefined, { timeout: 300000, polling: 500 });
  log('demand final');
  await page.evaluate((rate) => {
    const tw = window.tw;
    tw.clock.setRunning(false);
    tw.clock.set(Math.floor(tw.clock.now() / 86400) * 86400 + 8.5 * 3600);
    if (rate !== 'paused') { tw.clock.setSpeed(2); tw.clock.setRunning(true); }
  }, RATE);
  // straight edges with no line, longer than 400 m, spread through the list
  const cands = await page.evaluate((N) => {
    const tw = window.tw, w = tw.world.value, net = tw.renderer.net;
    const ok = w.edges.filter((e) => e.lengthM > 600 && net.edgePts.get(e.id)?.length >= 6);
    const straight = ok.filter((e) => { const p = net.edgePts.get(e.id), n = p.length / 3; const d = Math.hypot(p[(n - 1) * 3] - p[0], p[(n - 1) * 3 + 1] - p[1]); let L = 0; for (let i = 1; i < n; i++) L += Math.hypot(p[i * 3] - p[i * 3 - 3], p[i * 3 + 1] - p[i * 3 - 2]); return d > 0.995 * L; });
    const pick = [];
    for (let k = 0; k < N; k++) pick.push(straight[Math.floor((straight.length * (k + 0.5)) / N)]);
    return pick.filter(Boolean).map((e) => { const p = net.edgePts.get(e.id), n = p.length / 3; const x = (p[0] + p[(n - 1) * 3]) / 2, y = (p[1] + p[(n - 1) * 3 + 1]) / 2; const [lng, lat] = tw.toLngLat(x, y, net.anchor ?? tw.anchor()); return { id: e.id, lng, lat }; });
  }, N);
  log('candidates', cands.length);
  if (MODE === 'measure') {
    await page.keyboard.press('2');
    await page.waitForTimeout(300);
    out.trials = [];
    for (const c of cands) {
      await page.evaluate((c) => window.tw.map.jumpTo({ center: [c.lng, c.lat], zoom: 16 }), c);
      await page.waitForTimeout(1200);
      const box = await page.evaluate((c) => { const r = window.tw.map.getContainer().getBoundingClientRect(), p = window.tw.map.project([c.lng, c.lat]); return [r.left + p.x, r.top + p.y]; }, c);
      await page.mouse.move(box[0] - 3, box[1]);
      await page.waitForTimeout(150);
      await page.mouse.move(box[0], box[1]);
      await page.waitForTimeout(500);
      const before = await page.evaluate(() => window.tw.world.value.save.stations.length);
      await page.mouse.down();
      await page.mouse.up();
      try {
        await page.waitForFunction(() => window.__t.frame > 0, undefined, { timeout: 5000 });
      } catch { log('no update for', c.id); continue; }
      await page.waitForTimeout(400);
      const r = await page.evaluate((before) => { const t = window.__t, c = window.tw.client; return { placed: window.tw.world.value.save.stations.length - before, sent: t.sent - t.down, world: t.world - t.down, frame: t.frame - t.down, editLog: c.editLog.at(-1) }; }, before);
      r.edge = c.id;
      out.trials.push(r);
      log(JSON.stringify(r));
    }
    const fr = out.trials.filter((t) => t.placed > 0).map((t) => t.frame).sort((a, b) => a - b);
    const wr = out.trials.filter((t) => t.placed > 0).map((t) => t.world).sort((a, b) => a - b);
    out.summary = { n: fr.length, frameMedian: fr[fr.length >> 1], frameMin: fr[0], frameMax: fr.at(-1), worldMedian: wr[wr.length >> 1], worldMin: wr[0], worldMax: wr.at(-1) };
    log('summary', JSON.stringify(out.summary));
  } else {
    // visual: a series of edits through the client, an undo, then screenshots and a buffer digest
    const edits = await page.evaluate(async (cands) => {
      const tw = window.tw, c = tw.client, res = [];
      const near = (lng, lat) => { const p = tw.lngLatToPack(lng, lat); let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
      for (const k of [0, 1, 2, 3]) {
        const p = tw.lngLatToPack(cands[k].lng, cands[k].lat);
        res.push((await c.edit({ op: 'addStation', at: { kind: 'edge', edge: cands[k].id, x: p.x, y: p.y }, platform: 200, name: '' })).ok);
      }
      // a new stretch of track from a free point to another
      const a = tw.lngLatToPack(cands[4].lng + 0.003, cands[4].lat + 0.003), b = tw.lngLatToPack(cands[4].lng + 0.012, cands[4].lat + 0.006);
      const r = await c.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: -1 }, to: { kind: 'free', x: b.x, y: b.y, level: -1 }, pis: [], single: false });
      res.push(r.ok);
      // drag its end node
      if (r.node !== undefined) { const n = tw.world.value.nodes.get(r.node); res.push((await c.edit({ op: 'moveNode', node: r.node, x: n.x + 150, y: n.y - 80 })).ok); }
      // a station on it, then undo that station
      const mid = tw.lngLatToPack(cands[4].lng + 0.0075, cands[4].lat + 0.0045);
      res.push((await c.edit({ op: 'addStation', at: { kind: 'node', node: near(cands[4].lng + 0.012, cands[4].lat + 0.006) }, platform: 200, name: '' })).ok);
      res.push((await c.edit({ op: 'undo' })).ok);
      // a station removed
      const st = tw.world.value.save.stations[5];
      res.push((await c.edit({ op: 'removeStation', node: st.num })).ok);
      return res;
    }, cands);
    out.edits = edits;
    log('edits', JSON.stringify(edits));
    await page.waitForTimeout(1500);
    // digest of every buffer the renderer holds, and of a full rebuild from the same state
    // the renderer's (incremental) buffers against the old full build from the same state
    out.digest = await page.evaluate(async () => {
      const old = await import('/src/map/network_t496old.ts');
      const h = (a) => { if (!a) return 'none'; const u = new Uint8Array(a.buffer, a.byteOffset, a.byteLength); let x = 2166136261; for (let i = 0; i < u.length; i++) { x ^= u[i]; x = Math.imul(x, 16777619); } return (x >>> 0).toString(16) + ':' + a.length; };
      const dig = (n) => {
        const r = {};
        for (const k of ['track', 'lines']) for (const f of ['seg', 'colour', 'edge', 'level', 'flags', 'dist', 'speed', 'info']) r[k + '.' + f] = h(n[k][f]);
        for (const f of ['lineColours', 'stations', 'stationAxis', 'stationShape', 'stationLine', 'stationLength', 'stationLinkIdx', 'stationLinks', 'junctions', 'phases', 'meta', 'samples', 'lineTable', 'lineCars', 'sampleStroke']) r[f] = h(n[f]);
        r.lineIds = n.lineIds.join(',');
        r.stationIds = n.stationIds.join(',');
        r.junctionIds = n.junctionIds.join(',');
        r.edgePts = [...n.edgePts.entries()].sort((a, b) => a[0] - b[0]).map(([k, v]) => k + '=' + h(v)).join(';');
        return r;
      };
      const a = dig(window.tw.renderer.net);
      window.__oldNet = old.networkFromState(window.__s);
      const b = dig(window.__oldNet);
      const diff = Object.keys(b).filter((k) => a[k] !== b[k]);
      return { keys: Object.keys(b).length, diff, segs: [window.tw.renderer.net.track.count, window.tw.renderer.net.lines.count, window.__oldNet.track.count, window.__oldNet.lines.count] };
    });
    log('digest', JSON.stringify(out.digest));
    const views = [[cands[0].lng, cands[0].lat, 15], [cands[4].lng + 0.007, cands[4].lat + 0.004, 14], [-73.98, 40.75, 12], [-73.95, 40.7, 10]];
    out.shots = [];
    const { readFile: rf } = await import('node:fs/promises');
    for (let i = 0; i < views.length; i++) {
      const [lng, lat, z] = views[i];
      await page.evaluate(([lng, lat, z]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: z }), [lng, lat, z]);
      await page.waitForTimeout(3000);
      await page.screenshot({ path: `${OUT}/visual-${LABEL}-${i}-new.png` });
      // the same view drawn from the old full build
      await page.evaluate(() => { const tw = window.tw, cur = tw.renderer.net; window.__newNet = cur; tw.renderer.setNetwork(window.__oldNet); tw.overlay.invalidate(); tw.map.triggerRepaint(); });
      await page.waitForTimeout(1000);
      await page.screenshot({ path: `${OUT}/visual-${LABEL}-${i}-old.png` });
      await page.evaluate(() => { const tw = window.tw; tw.renderer.setNetwork(window.__newNet); tw.overlay.invalidate(); tw.map.triggerRepaint(); });
      await page.waitForTimeout(1000);
      await page.screenshot({ path: `${OUT}/visual-${LABEL}-${i}-new2.png` });
      const [x, y, z2] = await Promise.all(['new', 'old', 'new2'].map((k) => rf(`${OUT}/visual-${LABEL}-${i}-${k}.png`)));
      const px = await page.evaluate(async (urls) => {
        const data = await Promise.all(urls.map(async (u) => { const im = new Image(); im.src = u; await im.decode(); const c = document.createElement('canvas'); c.width = im.width; c.height = im.height; const g = c.getContext('2d'); g.drawImage(im, 0, 0); return g.getImageData(0, 0, c.width, c.height).data; }));
        const cnt = (a, b) => { let n = 0; for (let i = 0; i < a.length; i += 4) if (Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]) > 6) n++; return n; };
        return { newOld: cnt(data[0], data[1]), oldNew2: cnt(data[1], data[2]), newNew2: cnt(data[0], data[2]) };
      }, [x, y, z2].map((b) => 'data:image/png;base64,' + b.toString('base64')));
      out.shots.push({ view: i, ...px });
    }
    log('shots', JSON.stringify(out.shots));
  }
} finally {
  await ctx.close();
}
await writeFile(`${OUT}/${MODE}-${LABEL}-${RATE}.json`, JSON.stringify(out, null, 1));
log('errors', errors.slice(0, 5));
