// T-3429: a stop with no way back on one-way track is refused with its own message. One-way single track
// west to east with a station at each end: a line over the two is refused "NoWayBack", and the player-facing
// text is read back. Picture in artifacts/trainworld/t3429-noway.png. From the maps root, on a Vite of your own
// (node trainworld/checks/dev-nowatch.mjs 6150):
//   node trainworld/checks/t3429-noway.mjs [dev url]
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6150/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const RC = [-73.93, 40.675];

const chrome = await launchChrome({ prefix: 'tw-t3429-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false } })); } catch {} });
const errors = [];
collectErrors(page, errors);
try {
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  const out = await page.evaluate(async ([c]) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + 3000 + dy });
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const r = {};
    r.route = (await cl.edit({ op: 'route', from: { kind: 'free', ...P(-2000, 0), level: 0 }, to: { kind: 'free', ...P(2000, 0), level: 0 }, pis: [], single: true, quad: false })).ok;
    await pause(800);
    const near = (x, y) => { const w0 = tw.world.value; let best = null, bd = 1e9; for (const e of w0.edges) { const a = w0.nodes.get(e.a), b = w0.nodes.get(e.b); const dx = b.x - a.x, dy = b.y - a.y, t = Math.max(0, Math.min(1, ((x - a.x) * dx + (y - a.y) * dy) / (dx * dx + dy * dy || 1))); const d = Math.hypot(a.x + t * dx - x, a.y + t * dy - y); if (d < bd) { bd = d; best = e; } } return best; };
    const ids = [];
    for (const [name, dx] of [['West', -1500], ['East', 1500]]) {
      const p = P(dx, 0), e = near(p.x, p.y);
      await cl.edit({ op: 'addStation', at: { kind: 'edge', edge: e.id, x: p.x, y: p.y }, platform: 200, name });
      await pause(900);
      ids.push([...tw.world.value.nodes.values()].filter((n) => n.platform > 0).sort((a, b) => Math.hypot(a.x - p.x, a.y - p.y) - Math.hypot(b.x - p.x, b.y - p.y))[0].id);
    }
    r.ids = ids;
    const before = await cl.edit({ op: 'addLine', stops: ids, name: 'Two way', colour: '#d6336c' });
    r.twoWayOk = before?.ok;
    if (before?.ok) await cl.edit({ op: 'removeLine', line: before.line });
    for (const e of tw.world.value.edges) r.oneWay = ((await cl.edit({ op: 'oneWay', edge: e.id, dir: 1 }))?.ok) && (r.oneWay ?? true);
    await pause(500);
    const refused = await cl.edit({ op: 'addLine', stops: ids, name: 'One way', colour: '#1c7ed6' });
    r.refused = refused;
    const { describeIssue } = await import('/src/game/issues.ts');
    r.text = (refused?.issues ?? []).map((k) => describeIssue(k));
    return r;
  }, [RC]);
  console.log(JSON.stringify(out));
  check(out.route && out.ids?.length === 2 && out.ids[0] !== out.ids[1], 'one-way single track with two stations');
  check(out.twoWayOk, 'a line over them is accepted while the track runs both ways');
  check(out.oneWay && out.refused && !out.refused.ok && (out.refused.issues ?? []).join() === 'NoWayBack', 'the line is refused as NoWayBack once all the track is one way', JSON.stringify(out.refused?.issues));
  check(out.text?.[0] === 'No way back from this station on one-way track.', 'the message reads', out.text?.[0]);
  await page.screenshot({ path: join(ARTIFACTS, 't3429-noway.png') });
  check(errors.length === 0, 'no page errors', errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
