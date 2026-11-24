// T-4214: a map click made right after an edit acts on the map that edit left: the click waits only until the edit's
// patch of the clickable map lands (stage C, with the edit's word), never for its network. Anita's three quick sequences
// on a new game in New York, each with the full states held back in the page so the network is as late as on a big save:
// track then delete it (the track goes), track then a station on it (a train station on the new track), a station then a
// delete of the track past it (that piece goes; the station and the piece before it stay). Each second click must be
// carried out and answered while the states are still held (so not through the network's wait), and the map must be
// right then (the tools' world, map/clickMap.ts) and again once they are let go. Then two stations clicked quickly and a
// switch to the delete tool with a delete click: the stations go in as stations, the switch after them, then the delete.
// While the states are held `world` is never set: a patch goes to the tools' world only (setting `world` ran every
// panel and layer on it once more per edit). Prints how long after the first edit's word each second edit went. About a
// minute.
//   node trainworld/checks/t4214-waiting-clicks.mjs [url]      (from the maps root; default http://localhost:6173/)
//   --self-test: clicks carried out at once, as before T-4214: every sequence must fail.
//   --switch-drops: a tool switch goes at once and drops the clicks waiting (the first waiting clicks' way): the last
//     case must fail.
//   --no-patch: the edits' patches are dropped in the page, so clicks wait for the held networks: every sequence must fail.
import { openGame } from './lib/play.mjs';

const URL_ = process.argv.slice(2).find((a) => !a.startsWith('--')) ?? 'http://localhost:6173/';
const SELF = process.argv.includes('--self-test');
const DROPS = process.argv.includes('--switch-drops');
const NOPATCH = process.argv.includes('--no-patch');
let failed = false;
const report = (ok, what, detail) => {
  failed ||= !ok;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}  ${JSON.stringify(detail)}`);
};
const along = (a, b, f) => [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];

const g = await openGame({ url: URL_, city: 'nyc', mode: 'sandbox', paused: true, size: [1400, 900], prefix: 'tw-t4214-' });
try {
  const page = g.page;
  // the page's own checks (T-4214's patched rows against the state's; T-1029, T-1030) and errors
  page.on('console', (m) => { if (m.type() === 'error' || /T-4214/.test(m.text())) console.log(`page ${m.type()}: ${m.text().slice(0, 500)}`); });
  await page.evaluate(async ([self, drops, nopatch]) => {
    const pm = await import('/src/game/pending.ts'), cm = await import('/src/map/clickMap.ts');
    if (self) Object.assign(window.tw.tools, { click: window.tw.tools.clickNow, tap: window.tw.tools.tapNow });
    if (drops) (await import('/src/game/ui.ts')).setSwitchGate(() => false);
    const tw = window.tw, c = tw.client, recv = c.receive.bind(c);
    const T = (window.t4214 = { pm, held: null, sends: [], applied: new Map(), worldSets: 0 });
    // `world` set while the states are held (a patch must not set it)
    tw.world.subscribe(() => { if (T.held) T.worldSets++; });
    // when each edit went and each word came (with a patch or not), for the lag from an edit to the next click's edit
    const post = c.post.bind(c);
    c.post = (m, tr) => { if (m.kind === 'edit') T.sends.push({ seq: m.seq, op: m.op.op, t: performance.now() }); return post(m, tr); };
    // when each press on the map was made
    T.presses = [];
    document.addEventListener('pointerdown', () => T.presses.push(performance.now()), true);
    // full states held until release(): the map the tools pick on stays behind the edits, as on a big save
    c.receive = (msg) => {
      if (msg.kind === 'applied' && nopatch) delete msg.patch;
      if (msg.kind === 'applied') T.applied.set(msg.seq, { t: performance.now(), patch: !!msg.patch });
      if (T.held && msg.kind === 'state' && !msg.state.rows) { T.held.push(msg); return; }
      return recv(msg);
    };
    T.hold = () => { T.held = []; };
    T.release = () => { const q = T.held ?? []; T.held = null; for (const m of q) recv(m); return q.length; };
    T.waiting = () => ({ clicks: tw.tools.waitingClicks.length, out: pm.pendingNow().out.length, unanswered: c.pending.size, held: T.held?.length ?? 0 });
    /** every click carried out and every edit answered (with states held, only a patched edit is answered) */
    T.acted = () => c.pending.size === 0 && !tw.tools.waitingClicks.length && !tw.tools.clicksOut && !tw.tools.placing;
    T.mark = () => T.sends.length;
    /** the demand layers built so far (T-3159) */
    T.layers = () => window.twViews?.stats.layerBuilds ?? -1;
    /** from the word of the first edit sent since `from` to the first edit sent after it, ms, and whether that word had a patch */
    T.lag = (from) => {
      const a = T.sends[from], b = T.sends[from + 1], w = a && T.applied.get(a.seq);
      if (!a || !b || !w) return null;
      const press = T.presses.filter((p) => p <= b.t).pop() ?? b.t;
      // waited: from the press to its edit going; afterWord: from the first edit's word to the next edit going, when the
      // press came before that word (the click waited for it), else 0
      return { first: a.op, next: b.op, waited: Math.round(b.t - press), afterWord: press < w.t ? Math.round(b.t - w.t) : 0, patch: w.patch };
    };
    T.settled = () => c.pending.size === 0 && pm.pendingNow().out.length === 0 && !tw.tools.waitingClicks.length && !tw.tools.clicksOut && !tw.tools.placing && tw.renderer.netVersion === tw.world.value?.version && cm.clickWorld() === tw.world.value;
    /** the counts of the world the tools pick on (as lib/reads.mjs `counts` reads `world`) */
    T.counts = () => { const w = cm.clickWorld(); return { lines: w.save.lines.length, stations: w.save.stations.length, edges: w.edges.length, nodes: w.nodes.size }; };
    /** nodes within `m` metres of a point in the world the tools pick on, with their track degree and whether a train station */
    T.near = ([lng, lat], m) => {
      const w = cm.clickWorld(), deg = new Map();
      for (const e of w.edges) for (const n of [e.a, e.b]) deg.set(n, (deg.get(n) ?? 0) + 1);
      const k = 111320, out = [];
      for (const n of w.nodes.values()) {
        const d = Math.hypot((n.lng - lng) * k * Math.cos((lat * Math.PI) / 180), (n.lat - lat) * k);
        if (d <= m) out.push({ id: n.id, d: Math.round(d), deg: deg.get(n.id) ?? 0, station: n.platform > 0 && !n.bus && !n.port });
      }
      return out;
    };
  }, [SELF, DROPS, NOPATCH]);
  const settle = async (what) => {
    try { await page.waitForFunction(() => window.t4214.settled(), undefined, { timeout: 30000, polling: 100 }); }
    catch { report(false, `${what}: settles`, await page.evaluate(() => window.t4214.waiting())); }
    await page.waitForTimeout(300);
  };
  const counts = () => g.state.counts();
  /** the tools' world's counts, while the states are held */
  const heldCounts = () => page.evaluate(() => window.t4214.counts());
  /** the clicks carried out and answered while the states are held (below the 5 s backstop) */
  const acted = async () => {
    try { await page.waitForFunction(() => window.t4214.acted(), undefined, { timeout: 3000, polling: 20 }); return true; } catch { return false; }
  };
  await g.act.camera([-73.9525, 40.64], 14);
  await g.act.tool('track', { type: 'regular', snap: { real: false }, keepPlacing: false });

  // 1. track, then delete it at once (the trips final first, so no new solve builds the demand layers meanwhile)
  const A = [-73.96, 40.646], B = [-73.945, 40.646];
  await g.waitDemand({ timeout: 90000 }).catch(() => {});
  const c0 = await counts();
  await page.evaluate(() => window.t4214.hold());
  const m1 = await page.evaluate(() => window.t4214.mark()), lb1 = await page.evaluate(() => window.t4214.layers());
  await g.act.press(A);
  await g.act.press(B);
  await g.act.tool('delete');
  await g.act.press(along(A, B, 0.5));
  const a1 = await acted(), w1 = await page.evaluate(() => window.t4214.waiting()), h1 = await heldCounts(), l1 = await page.evaluate((m) => window.t4214.lag(m), m1);
  const lb1b = await page.evaluate(() => window.t4214.layers()), ws1 = await page.evaluate(() => window.t4214.worldSets);
  await page.evaluate(() => window.t4214.release());
  await settle('track then delete');
  const c1 = await counts();
  report(a1 && h1.edges === c0.edges && h1.nodes === c0.nodes && c1.edges === c0.edges && c1.nodes === c0.nodes, 'track then delete: the delete took the new track before its network', { acted: a1, lag: l1, waiting: w1, before: c0, held: h1, after: c1 });
  // (the layers are for the record: a solve that ends meanwhile builds them too)
  report(ws1 === 0, 'the patches set no `world`', { worldSets: ws1, layersBefore: lb1, layersHeld: lb1b });

  // 2. track, then a station on it at once
  const C = [-73.96, 40.64], D = [-73.945, 40.64], N = along(C, D, 0.5);
  await g.act.tool('track');
  await page.evaluate(() => window.t4214.hold());
  const m2 = await page.evaluate(() => window.t4214.mark());
  await g.act.press(C);
  await g.act.press(D);
  await g.act.tool('station');
  await g.act.press(N);
  const a2 = await acted(), w2 = await page.evaluate(() => window.t4214.waiting()), h2 = await heldCounts(), l2 = await page.evaluate((m) => window.t4214.lag(m), m2);
  const heldOn = (await page.evaluate((p) => window.t4214.near(p, 150), N)).filter((n) => n.station);
  await page.evaluate(() => window.t4214.release());
  await settle('track then station');
  const c2 = await counts(), onTrack = (await page.evaluate((p) => window.t4214.near(p, 150), N)).filter((n) => n.station);
  report(a2 && h2.stations === c1.stations + 1 && heldOn.length === 1 && heldOn[0].deg === 2 && c2.edges === c1.edges + 2 && c2.stations === c1.stations + 1 && onTrack.length === 1 && onTrack[0].deg === 2, 'track then station: the station went on the new track before its network', { acted: a2, lag: l2, waiting: w2, before: c1, held: h2, after: c2, station: onTrack });

  // 3. a station, then a delete of the track past it at once
  const E = [-73.96, 40.634], F = [-73.945, 40.634], S = along(E, F, 0.4);
  await g.act.tool('track');
  const run = await g.act.drawTrack([E, F]);
  await settle('the third piece');
  const c3 = await counts();
  await page.evaluate(() => window.t4214.hold());
  const m3 = await page.evaluate(() => window.t4214.mark());
  await g.act.tool('station');
  await g.act.press(S);
  await g.act.tool('delete');
  await g.act.press(along(E, F, 0.85));
  const a3 = await acted(), w3 = await page.evaluate(() => window.t4214.waiting()), l3 = await page.evaluate((m) => window.t4214.lag(m), m3);
  const heldF = await page.evaluate(([p, m]) => window.t4214.near(p, m), [F, 60]);
  await page.evaluate(() => window.t4214.release());
  await settle('station then delete');
  const c4 = await counts();
  const [atE, atF, st] = await Promise.all([[E, 40], [F, 60], [S, 150]].map(([p, m]) => page.evaluate(([p, m]) => window.t4214.near(p, m), [p, m])));
  report(run.segments?.[0]?.ok !== false && a3 && heldF.length === 0 && c4.stations === c3.stations + 1 && st.some((n) => n.station) && atE.length > 0 && atF.length === 0,
    'station then delete past it: the delete took only the piece past the station, before the network', { acted: a3, lag: l3, waiting: w3, heldAtF: heldF.length, before: c3, after: c4, atE, atF, station: st.filter((n) => n.station) });

  // 4. two stations clicked quickly, then the delete tool and a delete: the waiting station is not lost to the switch
  const P = [-73.941, 40.646], Q = [-73.929, 40.646], G = [-73.941, 40.64], H = [-73.929, 40.64];
  await g.act.tool('track');
  await g.act.drawTrack([P, Q]);
  await g.act.drawTrack([G, H]);
  await settle('the fourth pieces');
  const c5 = await counts();
  await page.evaluate(() => window.t4214.hold());
  await g.act.tool('station');
  await g.act.press(along(P, Q, 0.3));
  await g.act.press(along(P, Q, 0.7));
  await g.act.tool('delete');
  await g.act.press(along(G, H, 0.5));
  const a4 = await acted(), w4 = await page.evaluate(() => window.t4214.waiting());
  await page.evaluate(() => window.t4214.release());
  await settle('stations then a switch to delete');
  const c6 = await counts();
  const onP = (await Promise.all([0.3, 0.7].map((f) => page.evaluate(([p, m]) => window.t4214.near(p, m), [along(P, Q, f), 150])))).map((ns) => ns.filter((n) => n.station && n.deg === 2).length);
  const atGH = (await Promise.all([G, H].map((p) => page.evaluate(([p, m]) => window.t4214.near(p, m), [p, 60])))).map((ns) => ns.length);
  report(a4 && c6.stations === c5.stations + 2 && onP.every((n) => n === 1) && atGH.every((n) => n === 0), 'stations then a switch to delete: both stations went in, then the delete, before the network', { acted: a4, waiting: w4, before: c5, after: c6, stationsOnTrack: onP, leftAtDeleted: atGH });
  const wsAll = await page.evaluate(() => window.t4214.worldSets);
  report(wsAll === 0, 'no `world` set while the states were held, in all four', { worldSets: wsAll });
  await g.shot('t4214_after.png');
} finally {
  await g.close();
}
if (SELF) console.log(failed ? 'self-test PASS: the old clicks fail' : 'self-test FAIL: the check passes without waiting clicks');
if (NOPATCH) console.log(failed ? 'no-patch PASS: without patches the clicks wait for the networks' : 'no-patch FAIL: the check passes without patches');
if (DROPS) console.log(failed ? 'switch-drops PASS: dropping on a switch fails' : 'switch-drops FAIL: the check passes when a switch drops clicks');
process.exit(failed !== (SELF || DROPS || NOPATCH) ? 1 : 0);
