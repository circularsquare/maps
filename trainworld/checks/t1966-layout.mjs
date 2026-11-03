// T-1966: a test layout for waits at conflicts, built in a new game with the edit API: a double-track
// Y (lines A to B and A to C through junction J, eight trains each) and a third line D to E across
// A - J on the level (a flat crossing). Used by t1966-cross.mjs and t1966-pics.mjs (save `test`).

const LAT = 40.652, LNG0 = -73.985, KM = 1 / (111.32 * Math.cos((LAT * Math.PI) / 180)), KMLAT = 1 / 111.32;
export const Y = { a: [LNG0, LAT], j: [LNG0 + 2 * KM, LAT], b: [LNG0 + 4 * KM, LAT], c: [LNG0 + 4 * KM, LAT + 1.5 * KMLAT], d: [LNG0 + KM, LAT - KMLAT], e: [LNG0 + KM, LAT + KMLAT] };
/** the camera on J and the crossing */
export const TEST_SCENE = [LNG0 + 1.5 * KM, LAT + 0.2 * KMLAT, 14.6];

/** Build the layout; its lines and how each route edit went. */
export const buildTest = (page) => page.evaluate(async (Y) => {
  const tw = window.tw, c = tw.client, pause = (ms) => new Promise((d) => setTimeout(d, ms));
  const xy = (p) => { const q = tw.lngLatToPack(p[0], p[1]); return [q.x, q.y]; };
  const near = (p) => { const [x, y] = xy(p); let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - x, n.y - y); if (d < bd) { bd = d; best = n.id; } } return best; };
  const free = (p) => { const [x, y] = xy(p); return { kind: 'free', x, y, level: -1 }; };
  const steps = [];
  steps.push(await c.edit({ op: 'route', from: free(Y.a), to: free(Y.j), pis: [], single: false }));
  await pause(300);
  steps.push(await c.edit({ op: 'route', from: { kind: 'node', node: near(Y.j) }, to: free(Y.b), pis: [], single: false }));
  await pause(300);
  steps.push(await c.edit({ op: 'route', from: { kind: 'node', node: near(Y.j) }, to: free(Y.c), pis: [], single: false }));
  await pause(300);
  steps.push(await c.edit({ op: 'route', from: free(Y.d), to: free(Y.e), pis: [], single: false }));
  await pause(500);
  for (const p of [Y.a, Y.b, Y.c, Y.d, Y.e]) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await pause(200); }
  await c.edit({ op: 'constructAll' });
  await pause(500);
  const lines = [];
  for (const [s, t, colour] of [[Y.a, Y.b, '#0a7d3b'], [Y.a, Y.c, '#d9467a'], [Y.d, Y.e, '#2a6fdb']]) {
    const r = await c.edit({ op: 'addLine', stops: [near(s), near(t)], name: '', colour });
    if (r?.line !== undefined) await c.edit({ op: 'setTrainCount', line: r.line, trains: [8, 8, 8] });
    lines.push(r?.line);
    await pause(300);
  }
  await pause(1500);
  return { lines, ok: steps.map((s) => s?.ok), issues: steps.flatMap((s) => s?.issues ?? []) };
}, Y);
