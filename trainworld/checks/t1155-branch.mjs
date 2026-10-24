// T-1155: track drawn from a station branches off just past its platform, so the station keeps no
// junction. On the real save and on a fresh game (double, quad, high-speed track; a through
// station and an end station), through the track tool's own clicks: the branch is placed, the
// station keeps two track ends, the junction sits 20 m past the platform, and one undo restores
// the track. Screenshots of the ghost and the placed branch in artifacts/trainworld/t1155/.
//   node trainworld/checks/t1155-branch.mjs [url]   (from the maps root; default localhost:6122)
import { readFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { launchChrome, importSave, demandFinal, REAL_SAVE, ANGLE, ARTIFACTS } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6122/';
const OUT = join(ARTIFACTS, 't1155');
await mkdir(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t1155-', size: [1200, 800], args: ANGLE });
let failed = false;
const check = (ok, msg) => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${msg}`); };

/** In the page: helpers on window.__b for the cases below. */
const HELPERS = async () => {
  const tw = window.tw, map = tw.map, tools = tw.tools;
  const ui = await import('/src/game/ui.ts');
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const raf = () => new Promise((r) => requestAnimationFrame(() => r()));
  const w = () => tw.world.value;
  const ev = (x, y) => ({ point: { x, y }, lngLat: map.unproject([x, y]), originalEvent: { detail: 1, shiftKey: false } });
  const settle = async (pred, ms = 8000) => { const t = performance.now(); while (performance.now() - t < ms) { if (pred()) return true; await sleep(100); } return false; };
  const snapshot = () => ({ edges: w().edges.length, nodes: [...w().nodes.keys()].sort((a, b) => a - b).join(',') });
  /** Start at the station's node (or at a point on its platform), hover `to` px off it, return the ghost. */
  const ghost = async (start, dx, dy) => {
    tools.reset();
    ui.tool.value = 'track';
    await raf();
    const p = map.project([start.lng, start.lat]);
    tools.move(ev(p.x, p.y));
    tools.click(ev(p.x, p.y));
    await sleep(200);
    tools.move(ev(p.x + dx, p.y + dy));
    const ok = await settle(() => tools.lastPreview && tools.wantKey === tools.previewKey, 8000);
    await raf(); await raf();
    const pv = tools.lastPreview;
    return { asked: ok, ok: !!pv?.ok, issues: pv?.issues ?? [], start: pv ? [pv.verts[0], pv.verts[1]] : null, hover: tools.hover?.kind };
  };
  /** The ghost's click: places the branch. */
  const place = async (station, dx, dy) => {
    const before = snapshot();
    const p = map.project([station.lng, station.lat]);
    tools.click(ev(p.x + dx, p.y + dy));
    const placed = await settle(() => w().edges.length !== before.edges, 8000);
    await sleep(300);
    tools.reset();
    return { placed, before };
  };
  /** What the station has round it now: its track ends, and each junction next to it with the track's length to it. */
  const around = (id) => {
    const n = w().nodes.get(id);
    const out = [];
    for (const e of w().edges) {
      if (e.a !== id && e.b !== id) continue;
      const o = w().nodes.get(e.a === id ? e.b : e.a);
      if (o && o.ports >= 3) out.push({ node: o.id, m: e.lengthM, ports: o.ports });
    }
    return { ports: n?.ports, platform: n?.platform, junctions: out };
  };
  const undo = async (before) => {
    await tw.client.edit({ op: 'undo' });
    await settle(() => snapshot().nodes === before.nodes && snapshot().edges === before.edges, 8000);
    return snapshot().nodes === before.nodes && snapshot().edges === before.edges;
  };
  window.__b = { ghost, place, around, undo, snapshot, sleep, raf, ui, w };
};

/** One branch: ghost (screenshot), place (screenshot), the checks, undo. */
async function branch(page, label, station, dx, dy, { want = 'junction', shots = false, reach = null } = {}) {
  // the paused page's loading card sits over the map's middle, where the station is: hidden for the shots
  if (shots) await page.addStyleTag({ content: '.t1155-hide{visibility:hidden!important}' }).then(() => page.evaluate(() => {
    for (const el of document.querySelectorAll('body *')) if (el.childElementCount === 0 && el.textContent === 'Loading data') { let p = el; for (let k = 0; k < 3 && p.parentElement && p.parentElement !== document.body; k++) p = p.parentElement; if (p.getBoundingClientRect().width < 500) p.classList.add('t1155-hide'); else el.parentElement.classList.add('t1155-hide'); }
  }));
  const g = await page.evaluate(([s, dx, dy]) => window.__b.ghost(s, dx, dy), [station, dx, dy]);
  check(g.ok, `${label}: the ghost is buildable (${g.issues.join(' ') || 'ok'})`);
  if (shots) await page.screenshot({ path: join(OUT, `${label}-ghost.png`) });
  if (!g.ok) { await page.evaluate(() => window.tw.tools.reset()); return false; }
  const r = await page.evaluate(([s, dx, dy]) => window.__b.place(s, dx, dy), [station, dx, dy]);
  check(r.placed, `${label}: the click placed the track`);
  if (shots) {
    await page.evaluate(() => window.__b.raf());
    await page.screenshot({ path: join(OUT, `${label}-placed.png`) });
  }
  const a = await page.evaluate((id) => window.__b.around(id), station.id);
  console.log(label, JSON.stringify(a));
  const plat = reach ?? a.platform / 2;
  // a quad station's crossover (its nearest end 50 m past the platform) is a junction too
  const branchAt = a.junctions.filter((j) => Math.abs(j.m - (plat + 20)) < 1.5);
  if (want === 'junction') {
    check(a.ports === 2 && branchAt.length === 1, `${label}: the station keeps two track ends, and the junction is 20 m past the platform's end (platform reach ${plat} m)`);
  } else {
    check(a.ports === 2 && branchAt.length === 0, `${label}: the line is extended, no junction (${a.ports} track ends)`);
  }
  const u = await page.evaluate((b) => window.__b.undo(b), r.before);
  check(u, `${label}: one undo restores the track`);
  return true;
}

try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  page.on('console', (m) => m.type() === 'error' && console.log('console', m.text()));

  // ---- a fresh game: double, quad and high-speed track, each with a through and an end station
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools, undefined, { timeout: 180000 });
  await page.waitForTimeout(2000);
  await page.evaluate(HELPERS);
  for (const [kind, lat] of [['double', 40.715], ['quad', 40.700], ['hsr', 40.685]]) {
    const made = await page.evaluate(async ([kind, lat]) => {
      const tw = window.tw, b = window.__b;
      b.ui.keepPlacing.value = false;
      b.ui.buildLevel.value = 1;
      b.ui.trackCount.value = kind === 'quad' ? 4 : 2;
      b.ui.highSpeed.value = kind === 'hsr';
      const p0 = tw.lngLatToPack(-73.90, lat), p1 = tw.lngLatToPack(-73.86, lat);
      const r = await tw.client.edit({ op: 'route', from: { kind: 'free', x: p0.x, y: p0.y, level: 1 }, to: { kind: 'free', x: p1.x, y: p1.y, level: 1 }, pis: [], single: false, quad: kind === 'quad', through: true, hsr: kind === 'hsr' });
      if (!r.ok) return { err: 'route ' + JSON.stringify(r) };
      await b.sleep(500);
      const near = (p) => [...b.w().nodes.values()].sort((m, n) => Math.hypot(m.x - p.x, m.y - p.y) - Math.hypot(n.x - p.x, n.y - p.y))[0];
      const a = near(p0), e = near(p1);
      const edge = b.w().edges.find((x) => (x.a === a.id && x.b === e.id) || (x.a === e.id && x.b === a.id));
      const mid = { x: (p0.x + p1.x) / 2, y: (p0.y + p1.y) / 2 };
      const s1 = await tw.client.edit({ op: 'addStation', at: { kind: 'edge', edge: edge.id, ...mid }, platform: 200, name: 'Through ' + kind });
      await b.sleep(300);
      const s2 = await tw.client.edit({ op: 'addStation', at: { kind: 'node', node: e.id }, platform: 200, name: 'End ' + kind });
      await b.sleep(500);
      const st = near(mid);
      const end = b.w().nodes.get(e.id);
      tw.map.jumpTo({ center: [st.lng, st.lat], zoom: 15 });
      await b.raf(); await b.raf();
      return { ok: s1.ok && s2.ok, through: { id: st.id, lng: st.lng, lat: st.lat }, end: { id: end.id, lng: end.lng, lat: end.lat }, platform: b.w().nodes.get(st.id).platform };
    }, [kind, lat]);
    if (made.err || !made.ok) { check(false, `${kind}: set up the track and stations (${made.err ?? 'station refused'})`); continue; }
    await page.waitForTimeout(800);
    // from the through station's node, the branch heading east and north (the ghost shown)
    await branch(page, `fresh-${kind}-through`, made.through, 190, -170, { shots: true });
    // the other way, west and south
    await branch(page, `fresh-${kind}-through-west`, made.through, -190, 170, { shots: kind === 'quad' });
    // the end station's node: the line is extended
    await page.evaluate((s) => { window.tw.map.jumpTo({ center: [s.lng, s.lat], zoom: 15 }); }, made.end);
    await page.waitForTimeout(500);
    await branch(page, `fresh-${kind}-end`, made.end, 220, -120, { want: 'extend' });
    if (kind === 'double') {
      // the end station's platform, drawn back the way the track comes: a branch behind its far end
      const onPlat = await page.evaluate((s) => { const p = window.tw.map.project([s.lng, s.lat]); const q = window.tw.map.unproject([p.x - 25, p.y]); return { id: s.id, lng: q.lng, lat: q.lat }; }, made.end);
      const g = await page.evaluate(([s, dx, dy]) => window.__b.ghost(s, dx, dy), [onPlat, -260, -160]);
      check(g.ok, `fresh-double-end-platform: the ghost is buildable (${g.issues.join(' ') || 'ok'})`);
      if (g.ok) {
        const r = await page.evaluate(([s, dx, dy]) => window.__b.place(s, dx, dy), [onPlat, -260, -160]);
        const a = await page.evaluate((id) => window.__b.around(id), made.end.id);
        console.log('fresh-double-end-platform', JSON.stringify(a));
        check(r.placed && a.ports === 1 && a.junctions.length === 1 && Math.abs(a.junctions[0].m - 220) < 1.5, 'fresh-double-end-platform: a branch 20 m behind the end station\'s 200 m platform');
        check(await page.evaluate((b) => window.__b.undo(b), r.before), 'fresh-double-end-platform: one undo restores the track');
      } else await page.evaluate(() => window.tw.tools.reset());
    }
  }

  // ---- the real save: the first through stations where a branch fits
  const bytes = await readFile(REAL_SAVE);
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { strict: false }).catch(() => console.log('demand did not finish; carrying on'));
  await page.waitForTimeout(1000);
  await page.evaluate(HELPERS);
  await page.evaluate(() => { const b = window.__b; b.ui.keepPlacing.value = false; b.ui.buildLevel.value = 0; b.ui.trackCount.value = 2; b.ui.highSpeed.value = false; });
  const stations = await page.evaluate(() => [...window.tw.world.value.nodes.values()].filter((n) => n.platform > 0 && n.ports === 2).sort((a, b) => a.id - b.id).map((n) => ({ id: n.id, lng: n.lng, lat: n.lat, level: n.level })));
  let done = 0;
  for (const s of stations.slice(0, 40)) {
    if (done >= 2) break;
    // the branch at the station's own level: most are underground, and a ramp needs 200 m a level
    await page.evaluate((s) => { window.__b.ui.buildLevel.value = s.level; window.tw.map.jumpTo({ center: [s.lng, s.lat], zoom: 16 }); }, s);
    await page.waitForTimeout(800);
    for (const [dx, dy] of [[200, -160], [-200, 160], [160, 200], [-160, -200]]) {
      const g = await page.evaluate(([s, dx, dy]) => window.__b.ghost(s, dx, dy), [s, dx, dy]);
      await page.evaluate(() => window.tw.tools.reset());
      if (process.env.T1155_DEBUG) console.log('try', s.id, dx, dy, JSON.stringify(g));
      if (!g.ok || g.hover !== 'free') continue;
      await branch(page, `real-${s.id}`, s, dx, dy, { shots: done === 0 });
      done++;
      break;
    }
  }
  check(done === 2, `real save: branched from ${done} through stations of the first 40 tried`);
} catch (e) {
  failed = true;
  console.log('FAIL  ' + e.message);
} finally {
  await close();
}
process.exitCode = failed ? 1 : 0;
