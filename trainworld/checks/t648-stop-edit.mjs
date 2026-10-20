// T-648: adding a stop while a line's stops are edited, on the real save (notes/perf-survey.md).
// Click to the world taken and the frame after it, the toggle markers' state on every frame for
// 300 ms after the click (a marker changing class and back is the flash), and a frame strip.
//   PLAYWRIGHT_MODULE=file:///.../playwright/index.mjs  BASE=http://localhost:6155/  LABEL=<name>  N=8
//   SHOTS=0 (no screencast; it slows the page, so latency runs use it)  PROFILE=1 (main thread)
//   FRAMES=1 (the toggle markers' classes every frame; costs the page some ms a frame)
//   BUILT=0 (the new station left a blueprint, so the line turns planned)  RATE=paused|<speed index>
//   node trainworld/checks/t648-stop-edit.mjs
// Output: ../artifacts/trainworld/t648-<label>.json and t648-<label>-strip.png.
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE);
const BASE = process.env.BASE ?? 'http://localhost:6155/';
const LABEL = process.env.LABEL ?? 'run';
const N = Number(process.env.N ?? 8);
const SHOTS = process.env.SHOTS !== '0';
const RATE = process.env.RATE ?? 'paused';
const OUT = 'C:/Users/anita/projects/maps/artifacts/trainworld';
const SAVE = 'C:/Users/anita/projects/maps/trainworld/T-007-real-nyc.save';
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const profile = await mkdtemp(join(tmpdir(), 'tw-t648-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
await ctx.addInitScript(([built, frames]) => { window.__built = built; window.__frames = frames; }, [process.env.BUILT !== '0', process.env.FRAMES === '1']);
await ctx.addInitScript(() => {
  try { localStorage.setItem('trainworld-ui', JSON.stringify({ display: { dayShown: 'weekday' }, v: 2 })); } catch {}
  window.__t = { down: 0, world: 0, frame: 0, frames: [] };
  // every frame after a click: the markers' classes and the world version, until 300 ms
  const marks = () => { const c = {}; for (const el of document.querySelectorAll('.stop-marks .stop-mark')) if (el.style.display !== 'none') c[el.className] = (c[el.className] ?? 0) + 1; return c; };
  const sig = () => { const out = []; for (const el of document.querySelectorAll('.stop-marks .stop-mark')) if (el.style.display !== 'none') out.push(el.className.slice(10, 11) + el.style.transform); return out.join('|'); };
  const tick = () => {
    const t = window.__t;
    if (!t.down || !window.__frames) return;
    const now = performance.now();
    t.frames.push({ at: now - t.down, v: window.tw?.world?.peek()?.version, marks: marks(), sig: sig() });
    if (now - t.down < 300) requestAnimationFrame(tick);
  };
  addEventListener('mousedown', () => { window.__t = { down: performance.now(), world: 0, frame: 0, frames: [] }; requestAnimationFrame(tick); }, true);
  const hook = setInterval(() => {
    const tw = window.tw;
    if (!tw?.world) return;
    clearInterval(hook);
    // the new network reaching the renderer, and the frame drawn with it
    const sn = tw.renderer.setNetwork.bind(tw.renderer);
    tw.renderer.setNetwork = (n) => {
      sn(n);
      const t = window.__t;
      if (!t.down || t.net) return;
      t.net = performance.now();
      requestAnimationFrame(() => { const ch = new MessageChannel(); ch.port1.onmessage = () => { t.drawn = performance.now(); }; ch.port2.postMessage(0); });
    };
    let last = tw.world.peek();
    tw.world.subscribe((w) => {
      if (w === last) return;
      last = w;
      const t = window.__t;
      if (!t.down || t.world) return;
      t.world = performance.now();
      requestAnimationFrame(() => { const ch = new MessageChannel(); ch.port1.onmessage = () => { t.frame = performance.now(); }; ch.port2.postMessage(0); });
    });
  }, 20);
});
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error' && !/openfreemap|AJAXError|favicon/.test(m.text())) errors.push(m.text()); });
const out = { label: LABEL, rate: RATE, errors, trials: [] };
try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(1500);
  const bytes = await readFile(SAVE);
  await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  log('loaded');
  await page.waitForFunction(() => { const d = window.twDemand, s = d?.log?.at(-1), v = d?.view?.(); return s && v && v.version === s.version && v.final && !d.client.waiting; }, undefined, { timeout: 300000, polling: 500 });
  log('demand final');
  await page.evaluate((rate) => { const tw = window.tw; tw.clock.setRunning(false); tw.clock.set(Math.floor(tw.clock.now() / 86400) * 86400 + 8.5 * 3600); if (rate !== 'paused') { tw.clock.setSpeed(Number(rate)); tw.clock.setRunning(true); } }, RATE);
  // The real save's lines stop at every station they pass, so each trial first builds a station
  // on a long straight stretch of one line's track (not timed), then adds it as a stop with a click.
  const picks = await page.evaluate((N) => {
    const tw = window.tw, w = tw.world.value, net = tw.renderer.net;
    const ok = w.edges.filter((e) => e.lines.length === 1 && e.lengthM > 800 && net.edgePts.get(e.id)?.length >= 6);
    const straight = ok.filter((e) => { const p = net.edgePts.get(e.id), n = p.length / 3; const d = Math.hypot(p[(n - 1) * 3] - p[0], p[(n - 1) * 3 + 1] - p[1]); let L = 0; for (let i = 1; i < n; i++) L += Math.hypot(p[i * 3] - p[i * 3 - 3], p[i * 3 + 1] - p[i * 3 - 2]); return d > 0.995 * L; });
    const seen = new Set(), pick = [];
    for (let k = 0; k < straight.length && pick.length < N; k++) {
      const e = straight[Math.floor((straight.length * ((k * 7919) % straight.length + 0.5)) / straight.length)];
      if (seen.has(e.lines[0])) continue;
      seen.add(e.lines[0]);
      pick.push(e);
    }
    return pick.map((e) => { const p = net.edgePts.get(e.id), n = p.length / 3; const x = (p[0] + p[(n - 1) * 3]) / 2, y = (p[1] + p[(n - 1) * 3 + 1]) / 2; const [lng, lat] = tw.toLngLat(x, y, net.anchor ?? tw.anchor()); return { edge: e.id, line: Number(e.lines[0]), lng, lat }; });
  }, N);
  log('picks', picks.length);
  // PROFILE=1: a CPU profile of the page from each click to its drawn frame, self times summed over
  // the trials (Playwright opens no CDP session on a worker: the clock worker's parts are its `ms`)
  const prof = process.env.PROFILE === '1' ? await ctx.newCDPSession(page) : null;
  const self = new Map();
  if (prof) {
    await prof.send('Profiler.enable');
    await prof.send('Profiler.setSamplingInterval', { interval: 100 });
  }
  // when the click's insertStop query and its setStops edit go out and come back
  await page.evaluate(() => {
    const c = window.tw.client, q = c.query.bind(c), e = c.edit.bind(c);
    const at = (k) => { const t = window.__t; if (t?.down && t[k] === undefined) t[k] = performance.now() - t.down; };
    c.query = (x) => { if (x.what === 'insertStop') at('qSent'); const p = q(x); if (x.what === 'insertStop') p.then(() => at('qBack')); return p; };
    const stopOp = (op) => op.op === 'setStops' || op.op === 'insertStop';
    c.edit = (op) => { if (stopOp(op)) at('eSent'); const p = e(op); if (stopOp(op)) p.then(() => at('eBack')); return p; };
  });
  for (const p of picks) {
    const built = await page.evaluate(async (p) => {
      const la = await import('/src/game/lineActions.ts');
      const ui = await import('/src/game/ui.ts');
      if (ui.tool.peek() === 'line') window.tw.tools.exit?.();
      const q = window.tw.lngLatToPack(p.lng, p.lat);
      const a = await window.tw.client.edit({ op: 'addStation', at: { kind: 'edge', edge: p.edge, x: q.x, y: q.y }, platform: 320, name: '' });
      if (!a.ok) return a.issues.join('; ') || 'refused';
      // BUILT=0 leaves it a blueprint, which makes the line a planned one when it stops there
      if (window.__built !== false) { const b = await window.tw.client.edit({ op: 'constructAll' }); if (!b.ok) return 'construct: ' + b.issues.join('; '); }
      await new Promise((r) => setTimeout(r, 300));
      const l = window.tw.world.value.save.lines.find((x) => x.num === p.line);
      la.toggleEditStops(l);
      window.tw.map.jumpTo({ center: [p.lng, p.lat], zoom: 15 });
      return true;
    }, p);
    if (built !== true) { log('no station on', p.edge, built); continue; }
    await page.waitForTimeout(1500);
    const at = await page.evaluate((p) => { const r = window.tw.map.getContainer().getBoundingClientRect(), q = window.tw.map.project([p.lng, p.lat]); return [r.left + q.x, r.top + q.y]; }, p);
    // the marker nearest the station's point
    const target = await page.evaluate(([x, y]) => {
      let best = null, bd = 1e9;
      for (const el of document.querySelectorAll('.stop-marks .stop-mark.route')) { if (el.style.display === 'none') continue; const r = el.getBoundingClientRect(); const cx = r.left + r.width / 2, cy = r.top + r.height / 2, d = Math.hypot(cx - x, cy - y); if (d < bd) { bd = d; best = [cx, cy]; } }
      return best && bd < 60 ? best : null;
    }, at);
    if (!target) {
      log('no marker for', p.line, p.edge, await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); return JSON.stringify({ tool: ui.tool.peek(), draft: ui.lineDraft.peek().line, editing: document.documentElement.classList.contains('stop-editing'), marks: document.querySelectorAll('.stop-mark').length, net: !!window.tw.renderer.net, ver: window.tw.world.peek()?.version }); }), await page.evaluate(([x, y]) => [...document.querySelectorAll('.stop-marks .stop-mark')].filter((el) => el.style.display !== 'none').map((el) => { const r = el.getBoundingClientRect(); return el.className.slice(10) + '@' + Math.round(Math.hypot(r.left + r.width / 2 - x, r.top + r.height / 2 - y)); }).join(' '), at)); await page.evaluate(async () => { window.tw.tools.exit?.(); await window.tw.client.edit({ op: 'undo' }); }); continue;
    }
    await page.mouse.move(target[0] - 2, target[1]);
    await page.waitForTimeout(100);
    await page.mouse.move(target[0], target[1]);
    await page.waitForTimeout(400);
    const before = await page.evaluate((id) => window.tw.world.value.save.lines.find((l) => l.num === id).stops.length, p.line);
    // a screencast of the click's frames (every trial; SHOTS=0 turns it off, since it slows the page)
    let cdp = null, shots = [];
    if (SHOTS) {
      cdp = await ctx.newCDPSession(page);
      cdp.on('Page.screencastFrame', async (f) => { shots.push({ ts: f.metadata.timestamp, data: f.data }); try { await cdp.send('Page.screencastFrameAck', { sessionId: f.sessionId }); } catch {} });
      await cdp.send('Page.startScreencast', { format: 'png', everyNthFrame: 1 });
      await page.waitForTimeout(300);
    }
    if (prof) await prof.send('Profiler.start');
    const t0 = Date.now() / 1000;
    await page.mouse.down();
    await page.mouse.up();
    try { await page.waitForFunction(() => window.__t.frame > 0 && window.__t.drawn > 0, undefined, { timeout: 5000, polling: 'raf' }); } catch { log('no update for', p.line); continue; }
    if (prof) {
      const { profile: pr } = await prof.send('Profiler.stop');
      const byId = new Map(pr.nodes.map((n) => [n.id, n]));
      for (let i = 0; i < pr.samples.length; i++) { const f = byId.get(pr.samples[i]).callFrame; const k = `${f.functionName || '(anon)'} ${f.url.split('/').pop().split('?')[0]}:${f.lineNumber + 1}`; self.set(k, (self.get(k) ?? 0) + (pr.timeDeltas[i] ?? 0)); }
    }
    await page.waitForTimeout(800);
    let flash = null;
    if (cdp) {
      await cdp.send('Page.stopScreencast');
      await cdp.detach();
      // the last frame before the click, then every frame to 300 ms after it, then the settled one
      const pre = shots.filter((s) => s.ts < t0).at(-1);
      const win = shots.filter((s) => s.ts >= t0 && s.ts <= t0 + 0.3);
      const fin = shots.at(-1);
      const seq = [pre, ...win, fin].filter(Boolean).map((s) => ({ at: Math.round((s.ts - t0) * 1000), data: s.data }));
      // a flash: pixels the same before and after the edit that some frame between draws otherwise
      flash = await page.evaluate(async (seq) => {
        const px = await Promise.all(seq.map(async (s) => { const im = new Image(); im.src = 'data:image/png;base64,' + s.data; await im.decode(); const c = document.createElement('canvas'); c.width = im.width; c.height = im.height; const g = c.getContext('2d'); g.drawImage(im, 0, 0); return { w: im.width, h: im.height, d: g.getImageData(0, 0, im.width, im.height).data }; }));
        const a = px[0], z = px.at(-1), diff = (p, q, i) => Math.abs(p[i] - q[i]) + Math.abs(p[i + 1] - q[i + 1]) + Math.abs(p[i + 2] - q[i + 2]);
        const per = [];
        for (let k = 1; k < px.length - 1; k++) {
          const f = px[k].d; let n = 0, x0 = 1e9, y0 = 1e9, x1 = -1, y1 = -1;
          for (let i = 0; i < f.length; i += 4) if (diff(a.d, z.d, i) < 12 && diff(f, a.d, i) > 40) { n++; const p = i / 4, x = p % a.w, y = (p / a.w) | 0; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
          per.push({ at: seq[k].at, n, box: n ? [x0, y0, x1, y1] : null });
        }
        return { w: a.w, h: a.h, frames: per };
      }, seq);
      flash.worst = flash.frames.reduce((m, f) => (f.n > (m?.n ?? 0) ? f : m), null);
      // STRIP=<trial index> picks the strip's click, so two runs show the same one; else the worst
      const pickIt = process.env.STRIP !== undefined ? out.trials.length === Number(process.env.STRIP) : !out.strip || (flash.worst?.n ?? 0) > (out.strip.flash.worst?.n ?? 0);
      if (pickIt) out.strip = { line: p.line, target, flash, seq };
    }
    const r = await page.evaluate(([id, before]) => {
      const t = window.__t, c = window.tw.client;
      // marker states in order, merged runs; a flash is a state that comes and goes again
      const runs = [];
      for (const f of t.frames) if (!runs.length || runs.at(-1).sig !== f.sig) runs.push({ at: Math.round(f.at), sig: f.sig, marks: f.marks, v: f.v });
      return { added: window.tw.world.value.save.lines.find((l) => l.num === id).stops.length - before, world: t.world - t.down, frame: t.frame - t.down, drawn: (t.drawn ?? NaN) - t.down, qSent: t.qSent, qBack: t.qBack, eSent: t.eSent, eBack: t.eBack, frames: t.frames.length, states: runs.map((x) => ({ at: x.at, v: x.v, marks: x.marks })), editLog: c.editLog.at(-1) };
    }, [p.line, before]);
    r.line = p.line;
    if (flash) r.flash = { worst: flash.worst, frames: flash.frames.map((f) => `${f.at}:${f.n}`).join(' ') };
    log(JSON.stringify({ line: p.line, added: r.added, world: r.world.toFixed(0), frame: r.frame.toFixed(0), drawn: r.drawn.toFixed(0), flash: r.flash?.frames, markerStates: r.states.map((s) => `${s.at}:${Object.entries(s.marks).map(([k, n]) => k.slice(10) + n).join(',')}`).join(' / ') }));
    out.trials.push(r);
    await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); if (ui.tool.peek() === 'line') window.tw.tools.exit?.(); for (let k = window.__built !== false ? 3 : 2; k > 0; k--) await window.tw.client.edit({ op: 'undo' }); });
    await page.waitForTimeout(500);
  }
  if (prof) {
    const n = Math.max(1, out.trials.length);
    const top = [...self].filter(([k]) => !/^\(idle\)/.test(k)).sort((a, b) => b[1] - a[1]).slice(0, 30).map(([k, us]) => `${(us / 1000 / n).toFixed(2)} ms/click  ${k}`);
    out.profile = top;
    log('profile\n  ' + top.join('\n  '));
  }
  const ok = out.trials.filter((t) => t.added > 0);
  const med = (k) => { const v = ok.map((t) => t[k]).sort((a, b) => a - b); return v[v.length >> 1]; };
  ok.forEach((t) => (t.statesN = t.states.length));
  out.summary = { n: ok.length, worldMedian: med('world'), frameMedian: med('frame'), drawnMedian: med('drawn'), drawnMax: Math.max(...ok.map((t) => t.drawn)), frameMax: Math.max(...ok.map((t) => t.frame)), markerStatesMedian: med('statesN') };
  log('summary', JSON.stringify(out.summary));
  // the strip: the trial with the biggest flash, each frame cropped round it (else round the click);
  // first frame before the click, last the settled one
  if (out.strip) {
    const s = out.strip, b = s.flash.worst?.box;
    // CROP=x,y,w,h (frame px) fixes the crop, so two runs' strips show the same place
    const crop = process.env.CROP ? process.env.CROP.split(',').map(Number) : null;
    const strip = await page.evaluate(async ({ seq, b, target, w, h, crop }) => {
      const W = crop ? crop[2] : 420, H = crop ? crop[3] : 280, cols = Math.min(5, seq.length), rows = Math.ceil(seq.length / cols);
      const k = w / 1400; // screencast frames may be scaled
      const cx = b ? (b[0] + b[2]) / 2 : target[0] * k, cy = b ? (b[1] + b[3]) / 2 : target[1] * k;
      const sx = crop ? crop[0] : Math.max(0, Math.min(w - W, cx - W / 2)), sy = crop ? crop[1] : Math.max(0, Math.min(h - H, cy - H / 2));
      const c = document.createElement('canvas'); c.width = cols * W; c.height = rows * (H + 18);
      const g = c.getContext('2d'); g.fillStyle = '#fff'; g.fillRect(0, 0, c.width, c.height); g.font = '14px sans-serif';
      for (let i = 0; i < seq.length; i++) {
        const im = new Image(); im.src = 'data:image/png;base64,' + seq[i].data; await im.decode();
        const x = (i % cols) * W, y = Math.floor(i / cols) * (H + 18);
        g.drawImage(im, sx, sy, W, H, x, y + 18, W, H);
        g.fillStyle = '#000'; g.fillText(i === 0 ? 'before' : i === seq.length - 1 ? `settled (${seq[i].at} ms)` : `${seq[i].at} ms`, x + 4, y + 14);
      }
      return c.toDataURL('image/png');
    }, { seq: s.seq, b, target: s.target, w: s.flash.w, h: s.flash.h, crop });
    await writeFile(`${OUT}/t648-${LABEL}-strip.png`, Buffer.from(strip.split(',')[1], 'base64'));
    // the worst frame in full, the flash's pixels in red over a faded copy of it
    const wi = s.seq.findIndex((f) => f.at === s.flash.worst?.at);
    if (wi > 0) {
      const mask = await page.evaluate(async ([a, f, z]) => {
        const load = async (d) => { const im = new Image(); im.src = 'data:image/png;base64,' + d; await im.decode(); return im; };
        const [A, F, Z] = await Promise.all([a, f, z].map(load));
        const c = document.createElement('canvas'); c.width = A.width; c.height = A.height; const g = c.getContext('2d');
        const get = (im) => { g.drawImage(im, 0, 0); return g.getImageData(0, 0, c.width, c.height); };
        const da = get(A).data, dz = get(Z).data, img = get(F), df = img.data;
        const diff = (p, q, i) => Math.abs(p[i] - q[i]) + Math.abs(p[i + 1] - q[i + 1]) + Math.abs(p[i + 2] - q[i + 2]);
        for (let i = 0; i < df.length; i += 4) {
          if (diff(da, dz, i) < 12 && diff(df, da, i) > 40) { df[i] = 255; df[i + 1] = 0; df[i + 2] = 0; }
          else for (let j = 0; j < 3; j++) df[i + j] = 200 + df[i + j] * 0.2;
        }
        g.putImageData(img, 0, 0);
        return c.toDataURL('image/png');
      }, [s.seq[0].data, s.seq[wi].data, s.seq.at(-1).data]);
      await writeFile(`${OUT}/t648-${LABEL}-flashmask.png`, Buffer.from(mask.split(',')[1], 'base64'));
      await writeFile(`${OUT}/t648-${LABEL}-flashframe.png`, Buffer.from(s.seq[wi].data, 'base64'));
      await writeFile(`${OUT}/t648-${LABEL}-flashbefore.png`, Buffer.from(s.seq[0].data, 'base64'));
    }
    out.strip = { line: s.line, worst: s.flash.worst, frames: s.seq.map((f) => f.at) };
  }
} finally {
  await ctx.close();
}
await writeFile(`${OUT}/t648-${LABEL}.json`, JSON.stringify(out, null, 1));
log('errors', errors.slice(0, 5));
