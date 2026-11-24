// T-4214: a map click made while an earlier edit has not reached the map the tools pick on waits for it, then acts on
// the map that holds it. Anita's three quick sequences on a new game in New York, each with the full states held back
// in the page so the window is as wide as on a big save: track then delete it (the track goes), track then a station on
// it (a train station on the new track), a station then a delete of the track past it (that piece goes; the station and
// the piece before it stay). Each second click must be seen waiting first. Then two stations clicked quickly and a
// switch to the delete tool with a delete click: the waiting station still goes in as a station, the switch after it,
// then the delete. About a minute.
//   node trainworld/checks/t4214-waiting-clicks.mjs [url]      (from the maps root; default http://localhost:6173/)
//   --self-test: clicks carried out at once, as before T-4214: every sequence must fail.
//   --switch-drops: a tool switch goes at once and drops the clicks waiting (the first waiting clicks' way): the last
//     case must fail.
import { openGame } from './lib/play.mjs';

const URL_ = process.argv.slice(2).find((a) => !a.startsWith('--')) ?? 'http://localhost:6173/';
const SELF = process.argv.includes('--self-test');
const DROPS = process.argv.includes('--switch-drops');
let failed = false;
const report = (ok, what, detail) => {
  failed ||= !ok;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}  ${JSON.stringify(detail)}`);
};
const along = (a, b, f) => [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];

const g = await openGame({ url: URL_, city: 'nyc', mode: 'sandbox', paused: true, size: [1400, 900], prefix: 'tw-t4214-' });
try {
  const page = g.page;
  await page.evaluate(async ([self, drops]) => {
    const pm = await import('/src/game/pending.ts');
    if (self) Object.assign(window.tw.tools, { click: window.tw.tools.clickNow, tap: window.tw.tools.tapNow });
    if (drops) (await import('/src/game/ui.ts')).setSwitchGate(() => false);
    const tw = window.tw, c = tw.client, recv = c.receive.bind(c);
    const T = (window.t4214 = { pm, held: null });
    // full states held until release(): the map the tools pick on stays behind the edits, as on a big save
    c.receive = (msg) => { if (T.held && msg.kind === 'state' && !msg.state.rows) { T.held.push(msg); return; } return recv(msg); };
    T.hold = () => { T.held = []; };
    T.release = () => { const q = T.held ?? []; T.held = null; for (const m of q) recv(m); return q.length; };
    T.waiting = () => ({ clicks: tw.tools.waitingClicks.length, out: pm.pendingNow().out.length });
    T.settled = () => c.pending.size === 0 && pm.pendingNow().out.length === 0 && !tw.tools.waitingClicks.length && !tw.tools.clicksOut && !tw.tools.placing && tw.renderer.netVersion === tw.world.value?.version;
    /** nodes within `m` metres of a point, with their track degree and whether a train station */
    T.near = ([lng, lat], m) => {
      const w = tw.world.value, deg = new Map();
      for (const e of w.edges) for (const n of [e.a, e.b]) deg.set(n, (deg.get(n) ?? 0) + 1);
      const k = 111320, out = [];
      for (const n of w.nodes.values()) {
        const d = Math.hypot((n.lng - lng) * k * Math.cos((lat * Math.PI) / 180), (n.lat - lat) * k);
        if (d <= m) out.push({ id: n.id, d: Math.round(d), deg: deg.get(n.id) ?? 0, station: n.platform > 0 && !n.bus && !n.port });
      }
      return out;
    };
  }, [SELF, DROPS]);
  const settle = async (what) => {
    try { await page.waitForFunction(() => window.t4214.settled(), undefined, { timeout: 30000, polling: 100 }); }
    catch { report(false, `${what}: settles`, await page.evaluate(() => window.t4214.waiting())); }
    await page.waitForTimeout(300);
  };
  const counts = () => g.state.counts();
  await g.act.camera([-73.9525, 40.64], 14);
  await g.act.tool('track', { type: 'regular', snap: { real: false }, keepPlacing: false });

  // 1. track, then delete it at once
  const A = [-73.96, 40.646], B = [-73.945, 40.646];
  const c0 = await counts();
  await page.evaluate(() => window.t4214.hold());
  await g.act.press(A);
  await g.act.press(B);
  await g.act.tool('delete');
  await g.act.press(along(A, B, 0.5));
  const w1 = await page.evaluate(() => window.t4214.waiting());
  await page.evaluate(() => window.t4214.release());
  await settle('track then delete');
  const c1 = await counts();
  report(w1.clicks === 1 && c1.edges === c0.edges && c1.nodes === c0.nodes, 'track then delete: the delete waited, then took the new track', { waiting: w1, before: c0, after: c1 });

  // 2. track, then a station on it at once
  const C = [-73.96, 40.64], D = [-73.945, 40.64], N = along(C, D, 0.5);
  await g.act.tool('track');
  await page.evaluate(() => window.t4214.hold());
  await g.act.press(C);
  await g.act.press(D);
  await g.act.tool('station');
  await g.act.press(N);
  const w2 = await page.evaluate(() => window.t4214.waiting());
  await page.evaluate(() => window.t4214.release());
  await settle('track then station');
  const c2 = await counts(), onTrack = (await page.evaluate((p) => window.t4214.near(p, 150), N)).filter((n) => n.station);
  report(w2.clicks === 1 && c2.edges === c1.edges + 2 && c2.stations === c1.stations + 1 && onTrack.length === 1 && onTrack[0].deg === 2, 'track then station: the station waited, then went on the new track', { waiting: w2, before: c1, after: c2, station: onTrack });

  // 3. a station, then a delete of the track past it at once
  const E = [-73.96, 40.634], F = [-73.945, 40.634], S = along(E, F, 0.4);
  await g.act.tool('track');
  const run = await g.act.drawTrack([E, F]);
  await settle('the third piece');
  const c3 = await counts();
  await page.evaluate(() => window.t4214.hold());
  await g.act.tool('station');
  await g.act.press(S);
  await g.act.tool('delete');
  await g.act.press(along(E, F, 0.85));
  const w3 = await page.evaluate(() => window.t4214.waiting());
  await page.evaluate(() => window.t4214.release());
  await settle('station then delete');
  const c4 = await counts();
  const [atE, atF, st] = await Promise.all([[E, 40], [F, 60], [S, 150]].map(([p, m]) => page.evaluate(([p, m]) => window.t4214.near(p, m), [p, m])));
  report(run.segments?.[0]?.ok !== false && w3.clicks === 1 && c4.stations === c3.stations + 1 && st.some((n) => n.station) && atE.length > 0 && atF.length === 0,
    'station then delete past it: the delete waited, then took only the piece past the station', { waiting: w3, before: c3, after: c4, atE, atF, station: st.filter((n) => n.station) });

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
  const w4 = await page.evaluate(() => window.t4214.waiting());
  await page.evaluate(() => window.t4214.release());
  await settle('stations then a switch to delete');
  const c6 = await counts();
  const onP = (await Promise.all([0.3, 0.7].map((f) => page.evaluate(([p, m]) => window.t4214.near(p, m), [along(P, Q, f), 150])))).map((ns) => ns.filter((n) => n.station && n.deg === 2).length);
  const atGH = (await Promise.all([G, H].map((p) => page.evaluate(([p, m]) => window.t4214.near(p, m), [p, 60])))).map((ns) => ns.length);
  report(w4.clicks >= 2 && c6.stations === c5.stations + 2 && onP.every((n) => n === 1) && atGH.every((n) => n === 0), 'stations then a switch to delete: both stations went in, then the delete', { waiting: w4, before: c5, after: c6, stationsOnTrack: onP, leftAtDeleted: atGH });
  await g.shot('t4214_after.png');
} finally {
  await g.close();
}
if (SELF) console.log(failed ? 'self-test PASS: the old clicks fail' : 'self-test FAIL: the check passes without waiting clicks');
if (DROPS) console.log(failed ? 'switch-drops PASS: dropping on a switch fails' : 'switch-drops FAIL: the check passes when a switch drops clicks');
process.exit(failed !== (SELF || DROPS) ? 1 : 0);
