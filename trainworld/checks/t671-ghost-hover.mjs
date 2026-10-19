// The station ghost's first frame against the worker's answer (T-695), the browser stage of
// t671-station-snap-check.mjs. On the real save: hovers over overlapping track pairs (spread over
// all the pairs found) and near the ends of new blueprint drawn beside a few stations, recording the
// ghost (its DOM transform and class) every frame until the worker has answered. A correction is a
// ghost that was showing and then changed; appearing after a wait is not one. Then a slide 2 px a
// frame along each pair. PASS when corrections are at most 5% of hovers, overlaps and ends apart.
//   node trainworld/checks/t671-ghost-hover.mjs [url] [pairs] [chains]   (from the maps root)
// Writes artifacts/trainworld/t671-ghost-hover.json. About 5 minutes with 40 pairs and 6 chains.
import { mkdtemp, rm, readdir, readFile, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const BASE = process.argv[2] ?? 'http://localhost:6130/';
const MAXPAIRS = Number(process.argv[3] ?? 40);
const CHAINS = Number(process.argv[4] ?? 6);
const SAVE = fileURLToPath(new URL('../T-007-real-nyc.save', import.meta.url));
const OUT = fileURLToPath(new URL('../../artifacts/trainworld/', import.meta.url));

async function playwright() {
  try { return await import('playwright'); } catch { /* */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright');
}
const { chromium } = await playwright();
await mkdir(OUT, { recursive: true });
const profile = await mkdtemp(join(tmpdir(), 'tw-t671-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1200, height: 800 }, args: ['--use-angle=d3d11'] });
let failed = false;
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const bytes = await readFile(SAVE);
  await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(3000);
  // new blueprint track to have ends to hover: a chain of 3 segments beside each of a few stations
  const drawn = await page.evaluate(async (opts) => {
    const tw = window.tw, map = tw.map, tools = tw.tools;
    const ui = await import('/src/game/ui.ts');
    const W = await import('/src/game/world.ts');
    const notes = [];
    const raf = () => new Promise((r) => requestAnimationFrame(() => r()));
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const w0 = tw.world.value, before = new Set(w0.nodes.keys());
    const stations = [...w0.nodes.values()].filter((n) => n.platform > 0).sort((a, b) => a.id - b.id);
    ui.tool.value = 'track';
    for (let s = 0; s < opts.chains; s++) {
      const st = stations[Math.floor((s * stations.length) / opts.chains)];
      map.jumpTo({ center: [st.lng, st.lat], zoom: 15.5 });
      await raf(); await raf();
      const c = map.project([st.lng, st.lat]);
      const P = [[c.x - 180, c.y - 160], [c.x - 60, c.y - 170], [c.x + 60, c.y - 155], [c.x + 180, c.y - 160]];
      for (const [i, [x, y]] of P.entries()) {
        tools.click({ point: { x, y }, lngLat: map.unproject([x, y]), originalEvent: { shiftKey: i < P.length - 1 } });
        await sleep(700);
      }
      notes.push(W.notice.value?.text);
      ui.tool.value = 'select';
      await sleep(100);
      ui.tool.value = 'track';
    }
    await sleep(1000);
    const w = tw.world.value;
    return { notes, ids: [...w.nodes.values()].filter((n) => !before.has(n.id) && n.ports === 1 && n.platform === 0).map((n) => n.id) };
  }, { chains: CHAINS });
  console.log(`${drawn.ids.length} new track ends to hover`);
  const endIds = drawn.ids;
  const scan = await page.evaluate(async (opts) => {
    const tw = window.tw, map = tw.map, tools = tw.tools;
    const ui = await import('/src/game/ui.ts');
    ui.tool.value = 'station';
    const Z = opts.Z;
    const w = tw.world.value, pts = tw.renderer.net.edgePts;
    const raf = () => new Promise((r) => requestAnimationFrame(() => r()));
    const any = [...pts.values()][0];
    map.jumpTo({ center: tw.toLngLat(any[0], any[1]), zoom: Z });
    await raf(); await raf();
    const p0 = map.project(tw.toLngLat(any[0], any[1])), p1 = map.project(tw.toLngLat(any[0] + 100, any[1]));
    const k = Math.hypot(p1.x - p0.x, p1.y - p0.y) / 100;
    const step = 2 / k, cell = 4 / k, near = 3 / k, endGap = 20 / k;
    const grid = new Map(), samples = [];
    for (const [e, p] of pts) {
      let total = 0;
      for (let i = 0; i + 5 < p.length; i += 3) total += Math.hypot(p[i + 3] - p[i], p[i + 4] - p[i + 1]);
      let at = 0;
      for (let i = 0; i + 5 < p.length; i += 3) {
        const dx = p[i + 3] - p[i], dy = p[i + 4] - p[i + 1], l = Math.hypot(dx, dy);
        for (let s = Math.ceil(at / step) * step - at; s < l; s += step) {
          const along = at + s;
          if (along < endGap || total - along < endGap) continue;
          const x = p[i] + (dx * s) / l, y = p[i + 1] + (dy * s) / l;
          const sm = { e, x, y, along };
          samples.push(sm);
          const key = Math.floor(x / cell) + ',' + Math.floor(y / cell);
          (grid.get(key) ?? grid.set(key, []).get(key)).push(sm);
        }
        at += l;
      }
    }
    const pairs = new Map();
    for (const s of samples) {
      const cx = Math.floor(s.x / cell), cy = Math.floor(s.y / cell);
      const seen = new Set();
      for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (const o of grid.get(cx + i + ',' + (cy + j)) ?? []) {
        if (o.e <= s.e || seen.has(o.e) || Math.hypot(o.x - s.x, o.y - s.y) > near) continue;
        seen.add(o.e);
        const key = s.e + ' ' + o.e;
        (pairs.get(key) ?? pairs.set(key, []).get(key)).push(s);
      }
    }
    const runs = [...pairs.entries()].filter(([, v]) => v.length >= 5).map(([key, v]) => { v.sort((a, b) => a.along - b.along); const m = v[Math.floor(v.length / 2)]; return { kind: 'pair', key, n: v.length, x: m.x, y: m.y }; });
    const stepP = Math.max(1, runs.length / opts.maxPairs);
    const pairTargets = runs.filter((_, i) => Math.floor(i / stepP) !== Math.floor((i - 1) / stepP)).slice(0, opts.maxPairs);
    const endTargets = [];
    const pick = opts.endIds.map((id) => w.nodes.get(id)).filter(Boolean);
    for (const n of pick) {
      const e = w.edges.find((x) => x.a === n.id || x.b === n.id);
      const p = e && pts.get(e.id);
      if (!p) continue;
      const P = [];
      for (let i = 0; i < p.length; i += 3) P.push([p[i], p[i + 1]]);
      if (e.b === n.id) P.reverse();
      let total = 0;
      for (let i = 1; i < P.length; i++) total += Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]);
      const perM = total / e.lengthM;
      for (const d of [8, 30, 60, 90, 120, e.lengthM - 60, e.lengthM - 20]) {
        if (d > e.lengthM - 10) continue;
        let want = d * perM, x = P[0][0], y = P[0][1];
        for (let i = 1; i < P.length; i++) {
          const l = Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]);
          if (want <= l) { x = P[i - 1][0] + ((P[i][0] - P[i - 1][0]) * want) / l; y = P[i - 1][1] + ((P[i][1] - P[i - 1][1]) * want) / l; break; }
          want -= l;
        }
        endTargets.push({ kind: 'end', node: n.id, edge: e.id, d, x, y });
      }
    }
    return { code: 'guess' in tools ? 'new' : 'old', k, pairsFound: runs.length, targets: [...pairTargets, ...endTargets] };
  }, { endIds, maxPairs: MAXPAIRS, Z: 15.5 });
  console.log(`${scan.pairsFound} overlapping pairs, ${scan.targets.length} hovers`);
  const results = [];
  for (const [i, t] of scan.targets.entries()) {
    const h = await page.evaluate(async (t) => {
      const tw = window.tw, map = tw.map, tools = tw.tools;
      const raf = () => new Promise((r) => requestAnimationFrame(() => r()));
      const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
      const ghostState = () => { const g = tools.ghost; return g && g.style.display !== 'none' ? { t: g.style.transform, c: g.getAttribute('class') } : null; };
      const spotOf = () => { const s = tools.stationSpot(); return s ? (s.kind === 'node' ? 'n' + s.node : s.kind === 'edge' ? 'e' + s.edge : 'free') : 'none'; };
      const nums = (t) => (t.match(/-?[\d.]+/g) ?? []).map(Number);
      const same = (a, b) => (!a && !b) || (!!a && !!b && a.c === b.c && nums(a.t).every((v, i) => Math.abs(v - nums(b.t)[i]) < 0.6));
      const ev = (x, y) => ({ point: { x, y }, lngLat: map.unproject([x, y]), originalEvent: {} });
      map.jumpTo({ center: tw.toLngLat(t.x, t.y), zoom: 15.5 });
      await raf(); await raf();
      const q = map.project(tw.toLngLat(t.x, t.y));
      tools.move(ev(q.x + 40, q.y + 40));
      await raf();
      await sleep(60);
      const t0 = performance.now();
      tools.move(ev(q.x, q.y));
      const first = ghostState(), firstSpot = spotOf();
      const alts = tools.alts.map((s) => (s.kind === 'node' ? 'n' + s.node : 'e' + s.edge));
      // a correction: the ghost was showing and then changed; appearing from nothing is not one
      let changedAt = null, changes = 0, appearedAt = first ? 0 : null, last = first;
      const step = (g) => {
        if (!same(g, last)) {
          if (last) { changes++; if (changedAt === null) changedAt = performance.now() - t0; }
          else if (appearedAt === null) appearedAt = performance.now() - t0;
        }
        last = g;
      };
      for (let i = 0; i < 20; i++) { await raf(); step(ghostState()); }
      await sleep(200);
      const fin = ghostState();
      step(fin);
      // then slide 2 px a frame along the track for 12 frames: a jump is the ghost moving more than
      // 3 px off the pointer's own step, or changing colour
      let slideJumps = 0;
      if (t.kind === 'pair') {
        const e = Number(t.key.split(' ')[0]), p = tw.renderer.net.edgePts.get(e);
        let best = Infinity, dx = 1, dy = 0;
        for (let k = 0; k + 5 < p.length; k += 3) {
          const a = map.project(tw.toLngLat(p[k], p[k + 1])), b = map.project(tw.toLngLat(p[k + 3], p[k + 4]));
          const d = Math.hypot((a.x + b.x) / 2 - q.x, (a.y + b.y) / 2 - q.y);
          if (d < best) { best = d; const l = Math.hypot(b.x - a.x, b.y - a.y) || 1; dx = (b.x - a.x) / l; dy = (b.y - a.y) / l; }
        }
        let prev = ghostState();
        for (let i = 1; i <= 12; i++) {
          tools.move(ev(q.x + dx * 2 * i, q.y + dy * 2 * i));
          await raf();
          const g = ghostState();
          if (prev && g) {
            const a = nums(prev.t), b = nums(g.t);
            if (g.c !== prev.c || Math.hypot(b[0] - a[0] - dx * 2, b[1] - a[1] - dy * 2) > 3) slideJumps++;
          }
          prev = g;
        }
        await sleep(200);
        const g = ghostState();
        if (prev && g) { const a = nums(prev.t), b = nums(g.t); if (g.c !== prev.c || Math.hypot(b[0] - a[0], b[1] - a[1]) > 3) slideJumps++; }
      }
      return { slideJumps, alts, firstSpot, finalSpot: spotOf(), first, fin, moved: changes > 0, spotMoved: firstSpot !== spotOf(), colour: !!first && first.c !== fin?.c, changedAt, appearedAt };
    }, t);
    results.push({ ...t, ...h });
  }
  const sum = (xs) => ({ n: xs.length, corrected: xs.filter((x) => x.moved).length, spotMoved: xs.filter((x) => x.spotMoved).length, colour: xs.filter((x) => x.colour).length, waited: xs.filter((x) => x.appearedAt > 0).length, worstWaitMs: Math.round(Math.max(0, ...xs.map((x) => x.appearedAt ?? 0))), slidesWithJumps: xs.filter((x) => x.slideJumps > 0).length });
  const summary = { pairsFound: scan.pairsFound, overlaps: sum(results.filter((r) => r.kind === 'pair')), ends: sum(results.filter((r) => r.kind === 'end')) };
  // the first frame before T-695 was the first candidate: how often the worker's spot differs from it
  const oldWrong = (xs) => xs.filter((r) => r.fin?.c?.includes('ok') && r.alts[0] !== r.finalSpot).length;
  await writeFile(join(OUT, 't671-ghost-hover.json'), JSON.stringify({ summary, results }, null, 1));
  for (const [name, s, kind] of [['over overlapping tracks', summary.overlaps, 'pair'], ['near track ends', summary.ends, 'end']]) {
    const ok = s.n > 0 && s.corrected <= 0.05 * s.n;
    failed ||= !ok;
    console.log(`${ok ? 'PASS' : 'FAIL'}  ghost corrected after its first frame ${name}  (${s.corrected} of ${s.n}; the first candidate was not the worker's spot in ${oldWrong(results.filter((r) => r.kind === kind))}; waited for the worker ${s.waited}, at most ${s.worstWaitMs} ms; slides with a jump ${s.slidesWithJumps})`);
  }
} catch (e) {
  failed = true;
  console.log('FAIL  ghost hover check  ' + e.message);
} finally {
  await ctx.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
process.exitCode = failed ? 1 : 0;
