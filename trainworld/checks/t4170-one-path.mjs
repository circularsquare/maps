// T-4170: every edit shown at its click goes through one path (game/pending.ts `pendingSent`), so an edit has at most one
// pending entry, and none is left once the final state is in. A real game on a real save: a station placed with the station
// tool (its ghost's box), a line copied (drawn from the worker's word), Construct (from the op), a line panel switch and a
// platform length (from the op), a station deleted (from the op), a rename (no guess: answered by rows alone), edits the
// worker refuses (each drops its entry in its answer, with at most one notice), undo and redo, several sent at once. All
// along, no two entries share a seq; renames answered by their rows alone (no network follows) leave nothing; after the last answer, with the network the world's and a second for late states,
// no entry is left. Then the final review's cases (notes/T-4144.md "Final review fixes"): a copy's colour drawn at its word,
// the cash at the click exactly the charge or untouched (a free edit, Construct of a pick, Construct all quoted, Rebuild),
// nothing shown for a Construct the cash cannot pay, no undo drawn after a copy or a Construct, an edit out 40 s keeping
// its guess, a piece's look then its delete composed, nothing left bound once a removal is in, and in a new game the
// first station's box drawn at its click. A settle also fails while nothing is drawn ahead but its textures stay bound.
//   node trainworld/checks/t4170-one-path.mjs [url] [nyc|asia]      (from the maps root; default http://localhost:6171/ nyc)
//   --self-test: one edit sent through two `ahead()` handles, as T-4148's hook in world.ts did under a caller's own (the
//     outer one never keyed): the check must fail.
//   --shots: screenshots at the click (full networks held back a moment) and after the final state, in artifacts/trainworld/t4170/
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { openGame } from './lib/play.mjs';
import { ARTIFACTS, REAL_SAVE } from './lib.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const URL_ = args[0] ?? 'http://localhost:6171/';
const SETUP = args[1] ?? 'nyc';
const SELF = process.argv.includes('--self-test');
const SHOTS = process.argv.includes('--shots');
const SAVE = SETUP === 'asia' ? join(ARTIFACTS, 't3600-asianita-1852.save') : REAL_SAVE;
const SHOT_DIR = join(ARTIFACTS, 't4170');
if (SHOTS) mkdirSync(SHOT_DIR, { recursive: true });

const fails = [];
const fail = (m) => { fails.push(m); console.log('FAIL  ' + m); };
const g = await openGame({ url: URL_, save: SAVE, saveVia: 'post', size: [1400, 900], prefix: 'tw-t4170-' });
try {
  const page = g.page;
  // page-side helpers: the modules the game runs (the same instances: Vite serves one URL each), a sampler of the entries
  await page.evaluate(async () => {
    const pm = await import('/src/game/pending.ts'), wm = await import('/src/game/world.ts'), rm = await import('/src/game/remove.ts');
    const la = await import('/src/map/lineAhead.ts');
    const T = (window.t4170 = { pm, wm, rm, most: 0, unkeyed: 0, shared: [], held: null, empty: la.aheadEmpty, la });
    const sample = () => {
      const es = pm.pendingNow().entries.filter((e) => e.seq >= 0), seen = new Map();
      for (const e of es) seen.set(e.seq, (seen.get(e.seq) ?? 0) + 1);
      for (const [seq, n] of seen) if (n > 1 && T.shared.length < 20) T.shared.push(seq);
      T.most = Math.max(T.most, es.length);
      T.unkeyed = Math.max(T.unkeyed, pm.pendingNow().entries.filter((e) => e.seq < 0).length);
      T.raf = requestAnimationFrame(sample);
    };
    sample();
    // --shots: full networks held until `release()`, so the frame after a click is what the click drew
    const c = window.tw.client, recv = c.receive.bind(c);
    c.receive = (msg) => { if (T.held && msg.kind === 'state' && !msg.state.rows) { T.held.push(msg); return; } return recv(msg); };
    T.hold = () => { T.held = []; };
    /** what the renderer draws ahead now, counted */
    T.ahead = () => { const m = window.tw.renderer.ahead; return { lines: m.lines.size, edges: m.edges.size, hideEdges: m.hideEdges.size, stations: m.stations.length, hideStations: m.hideStations.size, rawStations: m.rawStations.length, colours: m.colours.size, entries: pm.pendingNow().entries.length }; };
    T.release = () => { const q = T.held ?? []; T.held = null; for (const m of q) recv(m); };
    /** the final state is in: no edit unanswered, no stop click out, the network drawn is the world's (`rowsOnly`: edits
     * answered by their rows, after which no network comes) */
    T.settled = (rowsOnly) => c.pending.size === 0 && wm.stopsQueued.peek() === 0 && (rowsOnly || window.tw.renderer.netVersion === wm.world.peek()?.version);
  });
  const settle = async (what, rowsOnly = false) => {
    const t = Date.now();
    await page.waitForFunction((r) => window.t4170.settled(r), rowsOnly, { timeout: 120000, polling: 50 });
    await page.waitForTimeout(rowsOnly ? 2000 : 1000); // (a state deferred behind the busy tag, a money record after it)
    const left = await page.evaluate(() => window.t4170.pm.pendingNow());
    if (left.entries.length) fail(`${what}: ${left.entries.length} entries left after the final state (${JSON.stringify(left.entries).slice(0, 300)})`);
    // nothing drawn ahead, and nothing of it left bound (a stale per-edge texture hid a new edge that took a removed one's id)
    const stale = await page.evaluate(() => { const R = window.tw.renderer; return window.t4170.empty(R.ahead) && (R.aheadEdgesTex || R.aheadLinesTex || R.aheadN[0] || R.aheadN[1]) ? R.aheadN : null; });
    if (stale) fail(`${what}: nothing drawn ahead, but its textures still bound (ids covered ${stale})`);
    return { ms: Date.now() - t, left: left.entries.length };
  };

  // sites: the biggest rail line, a built station in its middle, another for the platform, a built edge on screen
  const site = await page.evaluate(() => {
    const w = window.tw.world.value;
    const rail = w.save.lines.filter((l) => !w.lineStats[l.id]?.bus && l.stops.length >= 4).sort((a, b) => b.stops.length - a.stops.length);
    const line = rail[0], stops = line.stops.map(Number);
    const byNum = new Map(w.save.stations.map((s) => [s.num, s]));
    const mid = stops.slice(1, -1).map((n) => byNum.get(n)).filter((s) => s?.built);
    const nd = (s) => w.nodes.get(s.num);
    return { line: line.num, fast: !!w.lineStats[line.id]?.fast, del: mid[Math.floor(mid.length / 2)]?.num, plat: mid[1]?.num, platLen: mid[1] ? nd(mid[1]).platform : 0, name: mid[1]?.name, at: [nd(mid[0]).lng, nd(mid[0]).lat] };
  });
  console.log('site', JSON.stringify(site));
  /** `at` a quarter of the way in from the left, clear of the panels and the loading card */
  const look = async (at) => { await g.act.camera(at, 15); await page.evaluate(() => { const m = window.tw.map, c = m.getCanvas(); m.jumpTo({ center: m.unproject([c.clientWidth * 0.75, c.clientHeight / 2]) }); }); };
  await look(site.at);
  await page.waitForTimeout(1500);
  await settle('the save loaded');

  // 1. a station with the station tool, on built track near the centre (its ghost's box drawn at the click)
  const spot = await page.evaluate(() => {
    const tw = window.tw, n = tw.renderer.net, w = tw.world.value, built = new Set(w.edges.filter((e) => e.built && !e.crossover).map((e) => e.id));
    const W = tw.map.getCanvas().clientWidth, H = tw.map.getCanvas().clientHeight;
    let best = null;
    for (const [edge, pts] of n.edgePts) {
      if (!built.has(edge)) continue;
      const e = w.edges.find((x) => x.id === edge);
      if (!e || e.lengthM < 400) continue;
      const k = 3 * Math.floor(pts.length / 6);
      const [x, y] = tw.overlay.toScreen(pts[k], pts[k + 1]);
      // (left of the panels and the loading card)
      if (x < 250 || x > W / 2 - 50 || y < 200 || y > H - 200) continue;
      const d = Math.hypot(x - W / 2, y - H / 2);
      if (!best || d < best.d) best = { d, x, y, edge };
    }
    return best;
  });
  if (!spot) fail('no built edge on screen for the station');
  const stationAt = spot ? await page.evaluate(([x, y]) => { const p = window.tw.map.unproject([x, y]); return [p.lng, p.lat]; }, [spot.x, spot.y]) : null;
  if (spot) {
    // as a player points before clicking: the ghost (its box is what the click draws)
    await g.act.tool('station');
    const [{ x: px, y: py, cover }] = await g.toScreen([stationAt]);
    if (cover) fail('the station spot is under ' + cover);
    await page.mouse.move(px - 20, py - 10);
    await page.mouse.move(px, py, { steps: 4 });
    await page.waitForTimeout(1500);
  }
  let placed = false;
  const clipAt = async (lngLat) => { const [{ x, y }] = await g.toScreen([lngLat]); return { x: Math.max(0, x - 210), y: Math.max(0, y - 160), width: 420, height: 320 }; };
  if (spot) {
    if (SHOTS) await g.shot(join(SHOT_DIR, `${SETUP}-station-before.png`), { clip: await clipAt(stationAt) });
    const n0 = await page.evaluate(() => window.tw.world.value.save.stations.length);
    if (SHOTS) await page.evaluate(() => window.t4170.hold());
    const [{ x: px, y: py }] = await g.toScreen([stationAt]);
    await page.mouse.click(px, py);
    if (SHOTS) {
      await page.waitForTimeout(120);
      await g.shot(join(SHOT_DIR, `${SETUP}-station-click.png`), { clip: await clipAt(stationAt) });
      console.log('station at the click, drawn ahead:', JSON.stringify(await page.evaluate(() => window.t4170.ahead())));
      await page.evaluate(() => window.t4170.release());
    }
    placed = await page.waitForFunction((n0) => window.tw.world.value.save.stations.length > n0, n0, { timeout: 30000, polling: 50 }).then(() => true, () => false);
  }
  if (!placed) fail('the station tool placed no station');
  console.log('station', placed, JSON.stringify(await settle('a station placed')));
  if (SHOTS && placed) {
    await g.shot(join(SHOT_DIR, `${SETUP}-station-final.png`), { clip: await clipAt(stationAt) });
    // a copied line and Construct of the new station, each on its own: before, at the click (networks held), after;
    // the station tool put away and the pointer off the map, so its ghost is not in the pictures
    await g.act.key('Escape');
    await page.mouse.move(5, 895);
    for (const [name, at, op] of [['copy', site.at, { op: 'copyLine', line: site.line }], ['construct', stationAt, null]]) {
      await look(at);
      await page.waitForTimeout(800);
      await g.shot(join(SHOT_DIR, `${SETUP}-${name}-before.png`), { clip: await clipAt(at) });
      await page.evaluate(() => window.t4170.hold());
      const p = page.evaluate((op) => { const { wm } = window.t4170, w = wm.world.peek(); return wm.edit(op ?? { op: 'construct', edges: [], stations: w.save.stations.filter((s) => !s.built).map((s) => s.num) }).then((a) => a.ok); }, op);
      await page.waitForTimeout(op ? 400 : 120); // (a copy is drawn from the worker's word)
      await g.shot(join(SHOT_DIR, `${SETUP}-${name}-click.png`), { clip: await clipAt(at) });
      console.log(name + ' at the click, drawn ahead:', JSON.stringify(await page.evaluate(() => window.t4170.ahead())));
      await page.evaluate(() => window.t4170.release());
      console.log(name, await p, JSON.stringify(await settle(name)));
      await g.shot(join(SHOT_DIR, `${SETUP}-${name}-final.png`), { clip: await clipAt(at) });
      console.log(name + ' after the final state, drawn ahead:', JSON.stringify(await page.evaluate(() => window.t4170.ahead())));
    }
    await look(site.at);
  }

  // 2. several at once: a copy, Construct of the new station, a panel switch, a platform, a rename, refused ones
  const burst = await page.evaluate(async ([site, self]) => {
    const { wm, pm } = window.t4170, w = wm.world.peek();
    const st = w.save.stations.filter((s) => !s.built).map((s) => s.num);
    const ps = [
      wm.edit({ op: 'copyLine', line: site.line }),
      wm.edit({ op: 'construct', edges: [], stations: st }),
      wm.edit({ op: 'lineFast', line: site.line, fast: !site.fast }),
      site.plat !== undefined ? wm.edit({ op: 'stationProps', node: site.plat, platform: site.platLen + 20, name: site.name }) : null,
      site.plat !== undefined ? wm.edit({ op: 'stationProps', node: site.plat, platform: site.platLen + 20, name: site.name + ' X' }) : null,
      wm.edit({ op: 'lineFast', line: 987654, fast: true }),
      wm.edit({ op: 'removeStation', node: 98765432 }),
      // T-4148's old hook under a caller's own: the outer handle is never keyed
      self ? pm.ahead({ cost: 0.001 }).send(() => pm.ahead({ fields: new Map([['line:1', { t4170: 1 }]]) }).send(() => wm.edit({ op: 'lineFast', line: site.line, fast: site.fast }))) : null,
    ];
    const at = pm.pendingNow().entries.length;
    const as = await Promise.all(ps);
    return { atClick: at, answers: as.map((a) => (a ? a.ok : null)) };
  }, [site, SELF]);
  console.log('burst', JSON.stringify(burst), JSON.stringify(await settle('a burst of edits')));
  if (burst.answers[5] !== false || burst.answers[6] !== false) fail('the edits meant to be refused were taken: ' + JSON.stringify(burst.answers));
  // (in --shots the new station was built on its own already: the burst's Construct has nothing left)
  if (burst.answers.slice(0, 5).some((a, i) => a !== true && !(SHOTS && i === 1))) fail('an edit of the burst was refused: ' + JSON.stringify(burst.answers));

  // 3. refused edits shown ahead, a panel's and a map's: each entry goes in its answer, with one notice
  const refused = await page.evaluate(async () => {
    const { wm, pm } = window.t4170, R = window.tw.renderer;
    let bad = 0;
    const stop = wm.notice.subscribe((n) => { if (n?.bad) bad++; });
    bad = 0;
    const p = wm.edit({ op: 'lineFast', line: 987654, fast: true });
    const shownAtClick = pm.shown('line:987654', 'fast', 'none');
    const a = await p;
    const shownAfter = pm.shown('line:987654', 'fast', 'none'), notices = bad;
    bad = 0;
    const q = wm.edit({ op: 'removeStation', node: 98765432 });
    const hiddenAtClick = R.ahead.hideStations.has(98765432);
    const b = await q;
    const hiddenAfter = R.ahead.hideStations.has(98765432), notices2 = bad;
    stop();
    return { ok: a.ok, shownAtClick, shownAfter, notices, mapOk: b.ok, hiddenAtClick, hiddenAfter, notices2 };
  });
  console.log('refused', JSON.stringify(refused));
  if (refused.ok !== false || refused.shownAtClick !== true || refused.shownAfter !== 'none' || refused.notices !== 1) fail('a refused panel edit shown ahead: ' + JSON.stringify(refused));
  if (refused.mapOk !== false || !refused.hiddenAtClick || refused.hiddenAfter || refused.notices2 > 1) fail('a refused map edit shown ahead: ' + JSON.stringify(refused));

  // 4. a station deleted mid-line; once its network is in, an undo drawn at its click; then undo and redo at once
  const del = await page.evaluate(async (site) => { const a = await window.t4170.rm.removeThing({ kind: 'station', node: site.del }, true); return a ?? null; }, site);
  await settle('a station deleted');
  const dels = await page.evaluate(async () => {
    const { wm, pm } = window.t4170, m0 = window.tw.renderer;
    const kept = pm.pendingNow().undo;
    const p = wm.edit({ op: 'undo' });
    const m = m0.ahead, at = pm.pendingNow().entries.length;
    const drawn = m.hideEdges.size + m.hideLines.size + m.hideStations.size + m.track.length + m.lineStrokes.length + m.rawStations.length, miss = pm.pendingNow().undoMiss;
    const a = await p;
    const [b, c] = await Promise.all([wm.edit({ op: 'undo' }), wm.edit({ op: 'redo' })]);
    return { undo: a.ok, drawn, miss, kept, at, undo2: b.ok, redo: c.ok };
  });
  console.log('delete, undo, undo, redo', JSON.stringify(del), JSON.stringify(dels), JSON.stringify(await settle('a delete and undos')));
  if (!dels.undo || !dels.undo2 || !dels.redo || dels.miss !== '' || !dels.drawn) fail('the undo of a delete was not drawn at its click, or an undo or redo was refused: ' + JSON.stringify(dels));

  // 5. edits answered by their rows alone (no network follows): a rename through the platform op, a line's name
  const rowsOnly = await page.evaluate(async (site) => {
    const { wm } = window.t4170, w = wm.world.peek(), l = w.save.lines.find((x) => x.num === site.line);
    const a = site.plat !== undefined ? await wm.edit({ op: 'stationProps', node: site.plat, platform: w.nodes.get(site.plat).platform, name: site.name + ' Y' }) : null;
    const b = await wm.edit({ op: 'lineLook', line: site.line, name: l.name + ' Y' });
    return [a?.ok, b.ok];
  }, site);
  console.log('renames', JSON.stringify(rowsOnly), JSON.stringify(await settle('renames answered by rows alone', true)));
  if (rowsOnly.some((a) => a !== true)) fail('a rename was refused: ' + JSON.stringify(rowsOnly));

  // 6. edits no network follows, with something drawn ahead (the review of 11-24): a platform change refused and the
  // old length sent again before the refusal lands (the second is a rename to the worker); a train count shown with a
  // stand-in. Their entries must go with their rows, not wait for a network.
  const noNet = await page.evaluate(async (site) => {
    const { wm, pm } = window.t4170, w = wm.world.peek(), p0 = w.nodes.get(site.plat).platform, l = w.save.lines.find((x) => x.num === site.line);
    const a = wm.edit({ op: 'stationProps', node: site.plat, platform: 5000, name: site.name });
    const b = wm.edit({ op: 'stationProps', node: site.plat, platform: p0, name: site.name });
    const t = wm.trainsShown(l.id);
    const c = pm.ahead({ hideStations: [site.plat] }).send(() => wm.edit({ op: 'setTrainCount', line: site.line, trains: [t[0] + 1, t[1], t[2]] }));
    const r = await Promise.all([a, b, c]);
    return r.map((x) => x.ok);
  }, site);
  console.log('no network follows', JSON.stringify(noNet), JSON.stringify(await settle('edits no network follows', true)));
  if (noNet[0] !== false || noNet[1] !== true || noNet[2] !== true) fail('the platform pair or the train count was answered otherwise: ' + JSON.stringify(noNet));

  // 7. the view travels far enough to move the draw anchor, then undo: an undo is never drawn against the network at
  // the old anchor (that hid every edge, line and station), for a train count and for a deleted line
  for (const [what, far] of [['a train count', [-0.12, 51.5]], ['a deleted line', site.at]]) {
    const r = await page.evaluate(async ([what, far, site]) => {
      const { wm, pm } = window.t4170, c = window.tw.client, R = window.tw.renderer, map = window.tw.map, w = wm.world.peek();
      const l = w.save.lines.find((x) => x.num === site.line);
      const t = wm.trainsShown(l.id);
      const e = await wm.edit(what === 'a train count' ? { op: 'setTrainCount', line: site.line, trains: [t[0] + 1, t[1], t[2]] } : { op: 'removeLine', line: site.line });
      const idle = async (rows) => { const t0 = performance.now(); while (!window.t4170.settled(rows) && performance.now() - t0 < 60000) await new Promise((r) => setTimeout(r, 50)); await new Promise((r) => setTimeout(r, 1500)); };
      await idle(what === 'a train count');
      const v0 = R.net.anchor?.version;
      map.jumpTo({ center: far, zoom: 10 });
      c.followView(far[0], far[1], 10);
      const t0 = performance.now();
      while (R.net.anchor?.version === v0 && performance.now() - t0 < 30000) await new Promise((r) => setTimeout(r, 100));
      const moved = R.net.anchor?.version !== v0;
      await idle();
      const p = wm.edit({ op: 'undo' });
      const m = R.ahead, hidden = m.hideEdges.size + m.hideLines.size + m.hideStations.size;
      const u = await p;
      return { edit: e.ok, moved, hidden, miss: pm.pendingNow().undoMiss, undo: u.ok };
    }, [what, far, site]);
    console.log('undo after the anchor moved,', what, JSON.stringify(r), JSON.stringify(await settle('an undo after the anchor moved', what === 'a train count')));
    if (!r.edit || !r.undo || !r.moved) fail(`${what}, anchor moved, undo: not as set up: ${JSON.stringify(r)}`);
    if (r.hidden > 0) fail(`${what}: an undo after the anchor moved hid ${r.hidden} edges, lines and stations at its click`);
  }
  await g.act.camera(site.at, 15);

  // 8. the final review of 11-24 (notes/T-4144.md "Final review fixes"), each a case that went wrong
  const f8 = await page.evaluate(async (site) => {
    const { wm, pm, la } = window.t4170, R = window.tw.renderer, c = window.tw.client;
    const mm = await import('/src/game/money.ts'), qm = await import('/src/game/quotes.ts'), rb = await import('/src/game/rebuild.ts'), ui = await import('/src/game/ui.ts');
    const idle = async () => { const t0 = performance.now(); while (!window.t4170.settled() && performance.now() - t0 < 60000) await new Promise((r) => setTimeout(r, 50)); await new Promise((r) => setTimeout(r, 1200)); };
    const cash = () => mm.liveMoney.peek().cash, near = (a, b) => Math.abs(a - b) < 1e-6;
    const out = { fails: [], notes: {} };
    const bad = (m) => out.fails.push(m);
    const recv = c.receive.bind(c);
    /** messages held but those `pass` lets through, until the returned release */
    const holdAll = (pass) => { const q = []; c.receive = (m) => (pass?.(m) ? recv(m) : void q.push(m)); return () => { c.receive = recv; for (const m of q) recv(m); }; };
    /** blueprint track no line runs on: an express pair beside built double track */
    const blueprint = async () => {
      const w = wm.world.peek(), had = new Set(w.edges.map((e) => e.id));
      for (const e of w.edges.filter((e) => e.built && !e.crossover && e.tracks === 2 && !e.kind).slice(0, 40)) {
        const a = await wm.edit({ op: 'doubleUp', edge: e.id }, true);
        if (!a.ok) continue;
        await idle();
        const now = wm.world.peek().edges.filter((x) => !had.has(x.id) && !x.built).map((x) => x.id);
        if (now.length) return now;
      }
      return [];
    };
    // a copied line drawn at its word: its colour reaches the stroke shader (its id was past the colours, alpha 0)
    {
      const release = holdAll((m) => !(m.kind === 'state' && !m.state.rows));
      const p = wm.edit({ op: 'copyLine', line: site.line });
      const t0 = performance.now();
      while (!R.ahead.lines.size && performance.now() - t0 < 10000) await new Promise((r) => setTimeout(r, 20));
      const id = [...R.ahead.lines.keys()][0], col = id === undefined ? null : R.shownColours(R.net);
      const alpha = col && id * 4 + 3 < col.length ? col[id * 4 + 3] : 0;
      release();
      await p;
      out.notes.copy = { id, alpha };
      if (!(alpha > 0)) bad('a copied line drawn at its word has no colour to draw with (alpha ' + alpha + ', line ' + id + ')');
      await idle();
      // an undo right after takes back only its badge: nothing drawn at that undo's click
      const u = wm.edit({ op: 'undo' });
      const m = R.ahead, drawn = m.hideEdges.size + m.hideLines.size + m.hideStations.size + m.track.length + m.lineStrokes.length + m.rawStations.length;
      await u;
      if (drawn) bad('an undo right after a copy drew ' + drawn + ' parts at its click (it takes back only the copy\'s badge)');
      await idle();
    }
    // a quoted price on an edit the worker charges nothing for: gone with the word, before any money comes
    {
      const l = wm.world.peek().lineStats[String(site.line)];
      const release = holdAll((m) => m.kind === 'applied');
      const p = pm.ahead({ cost: 7 }).send(() => wm.edit({ op: 'lineOneWay', line: site.line, oneWay: !l?.oneWay }));
      const t0 = performance.now();
      while (!pm.pendingNow().entries.some((e) => e.armed >= 0) && performance.now() - t0 < 10000) await new Promise((r) => setTimeout(r, 10));
      const taken = cash() - pm.cashShown(cash());
      release();
      await p;
      if (!near(taken, 0)) bad('a free edit still took ' + taken + ' from the cash after its word');
      await idle();
      await wm.edit({ op: 'lineOneWay', line: site.line, oneWay: !!l?.oneWay });
      await idle();
    }
    // blueprint the cash cannot pay: Construct shows nothing at its click
    const bp = await blueprint();
    out.notes.blueprint = bp.length;
    if (!bp.length) bad('no blueprint made for the money cases');
    else {
      const c0 = cash();
      c.debugCash(0.001);
      const t0 = performance.now();
      while (cash() > 0.01 && performance.now() - t0 < 5000) await new Promise((r) => setTimeout(r, 20));
      const p = wm.edit({ op: 'constructAll' }, true);
      const built = [...R.ahead.edges.values()].filter((x) => x.built).length, shown = pm.cashShown(cash());
      const a = await p;
      c.debugCash(c0);
      await idle();
      if (a.ok) bad('Construct with no cash was taken');
      if (built || shown < 0) bad('Construct the cash cannot pay drew ' + built + ' pieces built and showed cash ' + shown + ' at its click');
      // a pick: at its click the cash takes its price or nothing (its track's share alone was shown before)
      const c1 = cash();
      const pp = wm.edit({ op: 'construct', edges: [bp[0]], stations: [] });
      const tPick = c1 - pm.cashShown(c1);
      const ap = await pp;
      if (!ap.ok) bad('Construct of one blueprint piece was refused');
      else if (!near(tPick, 0) && !near(tPick, ap.charge)) bad('Construct of a pick took ' + tPick + ' at its click and charged ' + ap.charge);
      await idle();
      // all of it, quoted first as the Build tab does: at its click the cash takes exactly the charge
      const rest = wm.world.peek().edges.filter((e) => !e.built && !e.off);
      if (rest.length) {
        await qm.blueprintQuote(wm.world.peek().version, true);
        const c2 = cash();
        const pa = wm.edit({ op: 'constructAll' });
        const tAll = c2 - pm.cashShown(c2);
        const aa = await pa;
        out.notes.constructAll = { atClick: tAll, charge: aa.charge };
        if (aa.ok && !near(tAll, aa.charge)) bad('Construct all, quoted, took ' + tAll + ' at its click and charged ' + aa.charge);
        await idle();
        // no undo takes a construct back: nothing drawn at that undo's click
        const u = wm.edit({ op: 'undo' });
        const m = R.ahead, drawn = m.hideEdges.size + m.hideLines.size + m.hideStations.size + m.track.length + m.lineStrokes.length + m.rawStations.length + m.edges.size;
        await u;
        if (drawn) bad('an undo right after Construct drew ' + drawn + ' parts at its click');
        await idle();
      }
    }
    // a look and then a delete on one blueprint piece, both out: the deleted piece is not drawn; once in, nothing left
    // bound (the settle after this checks it)
    const bp2 = await blueprint();
    out.notes.blueprint2 = bp2.length;
    if (!bp2.length) bad('no second blueprint made for the look-then-delete case');
    else {
      out.notes.blueprint2Strokes = R.net.track.edge.filter((e) => bp2.includes(e)).length;
      // an edit still out after 15 s (its ground downloading): its guess stays until its answer
      {
        const release = holdAll(() => false);
        const p = wm.edit({ op: 'blueprintOff', edges: bp2, stations: [], off: true });
        const seq = Math.max(...pm.pendingNow().entries.map((e) => e.seq));
        const t0 = performance.now();
        let goneAt = null;
        const ages = [];
        while (performance.now() - t0 < 40000) {
          await new Promise((r) => setTimeout(r, 250));
          if (goneAt === null && !pm.pendingNow().entries.some((e) => e.seq === seq)) goneAt = Math.round(performance.now() - t0);
          if (ages.length < Math.floor((performance.now() - t0) / 10000)) ages.push(pm.pendingNow().entries.map((e) => [e.seq, e.ageMs, e.armed]));
        }
        out.notes.unanswered = { seq, goneAt, ages };
        release();
        await p;
        if (goneAt !== null) bad('an edit unanswered for 40 s lost what it showed at its click after ' + goneAt + ' ms');
        await idle();
        await wm.edit({ op: 'blueprintOff', edges: bp2, stations: [], off: false });
        await idle();
      }
      // a look and then a delete on one piece, composed (two shows, as two edits out at once): the piece is not drawn
      const look = pm.ahead({ edges: new Map([[bp2[0], { off: true }]]) }), gone = pm.ahead({ hideEdges: [bp2[0]] });
      const parts = la.aheadParts(R.net, R.ahead);
      let drawn = 0;
      for (const s of [parts.track, parts.lines]) if (s) for (let i = 0; i < s.count; i++) if (s.edge[i] === bp2[0]) drawn++;
      out.notes.lookThenDelete = { look: R.ahead.edges.has(bp2[0]), hidden: R.ahead.hideEdges.has(bp2[0]), drawn };
      look.drop();
      gone.drop();
      // then deleted for real: once its network is in, nothing of it left bound
      await wm.edit({ op: 'removePieces', edges: bp2, stations: [] });
      if (drawn) bad('a blueprint piece deleted after a change of look was still drawn (' + drawn + ' strokes)');
      await idle();
      // its network in, nothing drawn ahead: no per-edge texture left bound (one hid a new piece given a removed id)
      if (la.aheadEmpty(R.ahead) && (R.aheadEdgesTex || R.aheadN[1])) bad('blueprint removed: its hidden pieces still bound after the network came (ids covered ' + R.aheadN[1] + ')');
    }
    // Rebuild as the select bar sends it: what the quote refuses is not drawn, what it changes is charged as quoted
    {
      const w = wm.world.peek(), lineEdges = new Set(w.save.lines.find((l) => l.num === site.line)?.path?.map((p) => p >> 1) ?? []);
      const pick = w.edges.filter((e) => e.built && !e.crossover && (lineEdges.size ? lineEdges.has(e.id) : true)).slice(0, 4).map((e) => e.id);
      for (const ch of [{ level: 1, kind: null, tracks: null }, { level: null, kind: null, tracks: 1 }]) {
        const q = await rb.fetchRebuildQuote({ edges: pick, stations: [] }, ch);
        if (!q) continue;
        const c0 = cash();
        rb.rebuildPieces({ edges: pick, stations: [] }, ch, q);
        const asked = ui.ask.peek();
        const before = c.pending.size;
        asked?.run();
        const looks = [...R.ahead.edges.keys()], taken = c0 - pm.cashShown(c0);
        out.notes['rebuild' + (ch.level !== null ? 'Level' : 'Single')] = { changed: q.changed.length, refused: q.refused.length, price: q.price, looks: looks.length, taken, sent: c.pending.size > before || !asked };
        if (looks.some((id) => q.refused.includes(id))) bad('Rebuild drew pieces its quote refuses');
        if (q.changed.length && !near(taken, q.price)) bad('Rebuild took ' + taken + ' at its click, quoted ' + q.price);
        await idle();
      }
    }
    return out;
  }, site);
  console.log('final review cases', JSON.stringify(f8.notes), JSON.stringify(await settle('the final review cases')));
  for (const m of f8.fails) fail(m);

  const end = await page.evaluate(() => { cancelAnimationFrame(window.t4170.raf); const n = window.t4170.pm.pendingNow(); return { doubled: n.doubled, swept: n.swept, shared: window.t4170.shared, most: window.t4170.most, unkeyed: window.t4170.unkeyed }; });
  if (end.doubled || end.shared.length) fail(`two entries for one edit: ${end.doubled} at the send, seqs ${end.shared.join(', ')} seen shared`);
  if (end.swept) fail(`${end.swept} armed entries were left for the backstop to take`);
  // (the station tool's tries are the one unkeyed entry a real edit has; two at once is a second handle)
  if (end.unkeyed > 1) fail(`${end.unkeyed} unkeyed entries at once`);
  const errs = g.pageErrors().filter((e) => /T-41[47]\d|pending/i.test(String(e)));
  if (errs.length) fail('page errors: ' + errs.slice(0, 3).join(' | '));
  console.log(`entries at most ${end.most} at once`);
} finally {
  await g.close();
}

// 9. a new game's first station drawn at its click (the station pass was skipped while the network had no station)
if (!SELF) {
  const g2 = await openGame({ url: URL_, city: 'nyc', size: [1400, 900], prefix: 'tw-t4170b-' });
  try {
    const page = g2.page, P = [-73.985, 40.748];
    await g2.act.camera(P, 15);
    await page.waitForTimeout(1000);
    await page.evaluate(() => {
      const R = window.tw.renderer, gl = R.gl, draw = gl.drawArraysInstanced.bind(gl);
      window.t4170b = { stands: 0, held: null };
      gl.drawArraysInstanced = (m, f, n, k) => { if (R.standVao && gl.getParameter(gl.VERTEX_ARRAY_BINDING) === R.standVao) window.t4170b.stands++; return draw(m, f, n, k); };
      const c = window.tw.client, recv = c.receive.bind(c);
      c.receive = (msg) => { if (window.t4170b.held && msg.kind === 'state' && !msg.state.rows) { window.t4170b.held.push(msg); return; } return recv(msg); };
      window.t4170b.release = () => { const q = window.t4170b.held ?? []; window.t4170b.held = null; for (const m of q) recv(m); };
      window.t4170b.held = [];
    });
    await g2.act.tool('station');
    await g2.act.press(P);
    await page.waitForTimeout(300);
    await g2.act.press([P[0] + 0.004, P[1] - 0.002]);
    await page.waitForFunction(() => window.tw.renderer.standCount > 0, null, { timeout: 15000 }).catch(() => {});
    await page.evaluate(() => new Promise((r) => { window.tw.overlay.invalidate(); requestAnimationFrame(() => requestAnimationFrame(r)); }));
    const b = await page.evaluate(() => ({ stands: window.t4170b.stands, standCount: window.tw.renderer.standCount, net: window.tw.renderer.net?.stationCount }));
    await page.evaluate(() => window.t4170b.release());
    console.log('first station of a new game, at its click', JSON.stringify(b));
    if (!b.standCount) fail('the first station of a new game: nothing drawn ahead at its click');
    else if (!b.stands) fail('the first station of a new game: its box drawn ahead was never drawn (network stations ' + b.net + ')');
  } finally {
    await g2.close();
  }
}
if (SELF) {
  console.log(`${fails.length ? 'PASS' : 'FAIL'}  t4170 one path self-test: ${fails.length ? 'the check fails with a second handle on one edit' : 'the check did not notice the second handle'}`);
  process.exit(fails.length ? 0 : 1);
}
console.log(`${fails.length ? 'FAIL' : 'PASS'}  t4170 one path (${SETUP}): no edit had two entries; none was left after the final state; a refusal dropped its entry in its answer`);
process.exit(fails.length ? 1 : 0);
