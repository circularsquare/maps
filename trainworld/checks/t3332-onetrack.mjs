// T-3332: one-track stations. A 6 km double track west to east with West, Mid and East, Mid placed with its
// platform on the south track (eastbound where trains keep right); a line over the three stops at Mid eastbound
// only (its half dot), and one ending at Mid from the east is refused. A 4 km single track to the north with a
// no-passing Mid. The station bar's icons, the ghost, the station panel's switch and Reverse direction.
// Pictures in artifacts/trainworld/t3332-*.png. From the maps root, on a Vite of your own
// (node trainworld/checks/dev-nowatch.mjs 6150):
//   node trainworld/checks/t3332-onetrack.mjs [dev url]
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6150/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const RC = [-73.93, 40.675];

const chrome = await launchChrome({ prefix: 'tw-t3332-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false } })); } catch {} });
const errors = [];
collectErrors(page, errors);
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `t3332-${name}.png`) });
const edit = (op) => page.evaluate(async (op) => { const r = await window.tw.client.edit(op); await new Promise((d) => setTimeout(d, 500)); return r; }, op);
const look = async (dx, dy, zoom) => {
  await page.evaluate(async ([dx, dy, zoom]) => {
    const tw = window.tw, { P } = window.__q;
    const q = tw.packToLngLat(P(dx, dy).x, P(dx, dy).y), [lng, lat] = Array.isArray(q) ? q : [q.lng, q.lat];
    tw.map.jumpTo({ center: [lng, lat], zoom });
  }, [dx, dy, zoom]);
  await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {});
  await wait(2000);
};
const select = (sel) => page.evaluate(async (sel) => { (await import('/src/game/ui.ts')).selection.value = sel; }, sel);

try {
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  await page.evaluate(([c]) => {
    const tw = window.tw, o = tw.lngLatToPack(c[0], c[1]);
    window.__q = {
      P: (dx, dy) => ({ x: o.x + dx, y: o.y + 3000 + dy }),
      near: (x, y) => {
        const w = tw.world.value;
        let best = null, bd = 1e9;
        for (const e of w.edges) {
          const a = w.nodes.get(e.a), b = w.nodes.get(e.b);
          const lo = (u, v) => Math.min(u, v) - 1, hi = (u, v) => Math.max(u, v) + 1;
          if (x < lo(a.x, b.x) || x > hi(a.x, b.x) || y < lo(a.y, b.y) || y > hi(a.y, b.y)) continue;
          const dx = b.x - a.x, dy = b.y - a.y, l2 = dx * dx + dy * dy || 1, t = ((x - a.x) * dx + (y - a.y) * dy) / l2;
          const d = Math.hypot(a.x + t * dx - x, a.y + t * dy - y);
          if (d < bd) { bd = d; best = e; }
        }
        return best;
      },
      station: (p) => [...tw.world.value.nodes.values()].filter((n) => n.platform > 0).sort((a, b) => Math.hypot(a.x - p.x, a.y - p.y) - Math.hypot(b.x - p.x, b.y - p.y))[0],
    };
  }, [RC]);
  // double track west to east, Mid's platform on the south track; single track to the north, Mid no passing
  const made = await page.evaluate(async () => {
    const tw = window.tw, cl = tw.client, { P, near, station } = window.__q;
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const out = {};
    out.route = (await cl.edit({ op: 'route', from: { kind: 'free', ...P(-3000, 0), level: 0 }, to: { kind: 'free', ...P(3000, 0), level: 0 }, pis: [], single: false, quad: false })).ok;
    await pause(800);
    out.single = (await cl.edit({ op: 'route', from: { kind: 'free', ...P(-2000, 1500), level: 0 }, to: { kind: 'free', ...P(2000, 1500), level: 0 }, pis: [], single: true, quad: false })).ok;
    await pause(800);
    const ids = {};
    for (const [name, dx, dy, one] of [['West', -2000, 0, null], ['Mid', 0, 0, -20], ['East', 2000, 0, null], ['N West', -1500, 1500, null], ['N Mid', 0, 1500, -20], ['N East', 1500, 1500, null]]) {
      const p = P(dx, dy), e = near(p.x, p.y);
      const r = await cl.edit({ op: 'addStation', at: { kind: 'edge', edge: e.id, x: p.x, y: p.y, ...(one !== null ? { one: P(dx, dy + one) } : {}) }, platform: 200, name });
      await pause(300);
      ids[name] = r.ok ? station(p).id : null;
    }
    out.ids = ids;
    const w = tw.world.value, st = (id) => w.save.stations.find((s) => s.num === id);
    out.one = [st(ids.Mid)?.one, st(ids['N Mid'])?.one, st(ids.West)?.one ?? 0];
    out.heading = st(ids.Mid)?.oneHeading;
    return out;
  });
  check(made.route && made.single && Object.values(made.ids).every((x) => x !== null), 'track and six stations placed', JSON.stringify(made));
  check(made.one[0] === 1 && made.one[1] === 2 && made.one[2] === 0, 'Mid one-track on double, N Mid no passing on single, West both', JSON.stringify(made.one));
  check(Math.cos(made.heading) > 0.99, 'south of the centre line on right-hand track serves eastbound', String(made.heading));
  const I = made.ids;
  // a line ending at Mid from the east is refused; through it both ways it stops eastbound only
  const refused = await edit({ op: 'addLine', stops: [I.East, I.Mid], name: 'R', colour: '#2f9e44' });
  check(refused && !refused.ok && (refused.issues ?? []).includes('OneTrackStop'), 'a line ending at Mid from the east is refused', JSON.stringify(refused?.issues));
  const line = await edit({ op: 'addLine', stops: [I.West, I.Mid, I.East], name: 'A', colour: '#d6336c' });
  const nline = await edit({ op: 'addLine', stops: [I['N West'], I['N Mid'], I['N East']], name: 'N', colour: '#1c7ed6' });
  check(line?.ok && nline?.ok, 'lines over both', JSON.stringify([line, nline]));
  await edit({ op: 'setTrainCount', line: line.line, trains: [6, 4, 2] });
  await edit({ op: 'setTrainCount', line: nline.line, trains: [3, 2, 1] });
  await wait(800);
  const ways = await page.evaluate((l) => window.tw.world.value.lineStats[String(l)]?.oneWayAt, line.line);
  check(ways && Object.keys(ways).length === 1 && Math.cos(ways[String(I.Mid)]) > 0.99, 'the line stops one way at Mid only, eastbound', JSON.stringify(ways));
  await edit({ op: 'constructAll' });
  await page.evaluate(() => { const tw = window.tw; tw.clock.setRunning(false); tw.clock.set(Math.floor(tw.clock.now() / 86400) * 86400 + 9.2 * 3600); });
  await select({ kind: 'line', line: String(line.line) });
  await look(0, 0, 14.6);
  await shot('1-mid-z14');
  const half = await page.evaluate((l) => ({ at: window.tw.world.value.lineStats[String(l)]?.oneWayAt, li: [...document.querySelectorAll('.stops li')].map((x) => `${x.dataset.station}:${x.className}:${getComputedStyle(x, '::after').backgroundImage}`) }), line.line);
  check(half.li.some((x) => x.startsWith(`${I.Mid}:`) && x.includes('oneway')), 'the stop list shows Mid half filled', JSON.stringify(half));
  await page.locator('.stops').first().locator('..').screenshot({ path: join(ARTIFACTS, 't3332-8-stops.png') }).catch(() => {});
  await look(0, 0, 16.2);
  await shot('2-mid-z16');
  await look(0, 0, 17.6);
  await shot('3-mid-z17');
  await select({ kind: 'line', line: String(nline.line) });
  await look(0, 1500, 16.5);
  await shot('4-single-z16');
  // the station panel: the switch and Reverse direction
  await select({ kind: 'station', station: String(I.Mid) });
  await look(0, 0, 16.2);
  const panel = await page.evaluate(() => document.querySelector('.inspector, aside')?.innerText ?? document.body.innerText);
  check(/Station type/.test(panel) && !/Reverse direction/.test(panel), 'the station panel has the selector and no Reverse direction button');
  const sel = await page.evaluate(() => [...document.querySelectorAll('.inspector .tb-count button, aside .tb-count button')].map((b) => b.getAttribute('aria-label') + '|' + b.getAttribute('data-tip')));
  check(sel.length === 3 && sel.some((x) => x.startsWith("One way station (eastbound)")) && sel.some((x) => x.startsWith("One way station (westbound)")) && sel.some((x) => x.startsWith("Two way station")), "three buttons: one way eastbound, two way, one way westbound", JSON.stringify(sel));
  const heading = () => page.evaluate((id) => window.tw.world.value.save.stations.find((s) => s.num === id)?.oneHeading, I.Mid);
  const press = async (label) => { await page.evaluate((l) => [...document.querySelectorAll(".tb-count button")].find((b) => b.getAttribute("aria-label")?.startsWith(l))?.click(), label); await wait(900); };
  await press("One way station (westbound)");
  check(Math.cos(await heading()) < -0.99, "the westbound button moves the platform to the other track");
  await press("One way station (eastbound)");
  check(Math.cos(await heading()) > 0.99, "the eastbound button moves it back");
  await shot('5-panel');
  const rev = await edit({ op: 'reverseStation', node: I.Mid });
  const after = await page.evaluate((id) => window.tw.world.value.save.stations.find((s) => s.num === id)?.oneHeading, I.Mid);
  check(rev?.ok && Math.cos(after) < -0.99, 'Reverse direction: Mid now serves westbound', `${JSON.stringify(rev)} ${after}`);
  await look(0, 0, 17.6);
  await shot('6-reversed-z17');
  const both = await edit({ op: 'stationTracks', node: I.Mid, one: false });
  const one0 = await page.evaluate((id) => window.tw.world.value.save.stations.find((s) => s.num === id)?.one ?? 0, I.Mid);
  check(both?.ok && one0 === 0, 'switched to both tracks', JSON.stringify(both));
  // the station tool's ghost with the one-track icon on, the pointer south of the centre line
  await select(null);
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.tool.value = 'station'; ui.stationOneTrack.value = true; });
  await look(1000, 0, 17.2);
  const pt = await page.evaluate(() => {
    const tw = window.tw, { P } = window.__q, q = tw.packToLngLat(P(1000, -5).x, P(1000, -5).y), [lng, lat] = Array.isArray(q) ? q : [q.lng, q.lat];
    const s = tw.map.project([lng, lat]), r = tw.map.getCanvas().getBoundingClientRect();
    return [r.left + s.x, r.top + s.y];
  });
  await page.mouse.move(pt[0], pt[1] - 2);
  await page.mouse.move(pt[0], pt[1]);
  await wait(1500);
  await shot('7-ghost');
  const bar = await page.evaluate(() => [...document.querySelectorAll('.station-bar button')].map((b) => b.getAttribute('aria-label')));
  check(bar.includes('One way station') && bar.includes('Two way station'), 'the station bar has the two icons', JSON.stringify(bar));
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.stationOneTrack.value = false; ui.tool.value = 'select'; });

  const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  await shot('died').catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3332 FAILED: ${fail}` : 'T-3332 PASSED');
process.exit(fail ? 1 : 0);
