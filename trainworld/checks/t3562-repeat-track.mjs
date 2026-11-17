// T-3562: a line whose run goes over a piece of track twice in the same direction gets an alert in its panel and a
// loop mark on the piece. Three lines built through addLine on a new game, each on its own straight double track:
//  plain      stops 1 2 3 4            -> nothing
//  out + back stops 1 2 3 4 3 2 1      -> nothing (opposite directions)
//  repeats    stops 1 3 2 4            -> (runs 2 to 3 twice, forward) alert "Runs the same track twice" and one mark (shape 7) on a piece of it
// Screenshot artifacts/trainworld/t3562-repeat.png.
// From the maps root, on a Vite of your own:   node trainworld/checks/t3562-repeat-track.mjs [dev url]
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6143/';
const M_REPEAT = 7;
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const chrome = await launchChrome({ prefix: 'tw-t3562-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7 })); } catch {} });
collectErrors(page, errors);

try {
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 300000 });
  await page.evaluate(() => {
    const tw = window.tw, c = tw.client;
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
    const at = (lng, lat) => { const p = tw.lngLatToPack(lng, lat); return [p.x, p.y]; };
    window.__build = async (lng, lat, order, name) => {
      const dLng = 1 / (111.32 * Math.cos((lat * Math.PI) / 180));
      const xy = [0, 3, 6, 9].map((km) => at(lng + km * dLng, lat));
      for (let i = 0; i < 3; i++) {
        const from = i === 0 ? { kind: 'free', x: xy[0][0], y: xy[0][1], level: 0 } : { kind: 'node', node: near(xy[i]) };
        await c.edit({ op: 'route', from, to: { kind: 'free', x: xy[i + 1][0], y: xy[i + 1][1], level: 0 }, pis: [], single: false });
        await pause(400);
      }
      for (const [k, p] of xy.entries()) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: `${name} ${k + 1}` }); await pause(250); }
      await c.edit({ op: 'constructAll' });
      await pause(600);
      // `order` may be a list of orders to try: the game refuses some stop orders
      let r = null;
      for (const o of Array.isArray(order[0]) ? order : [order]) {
        r = await c.edit({ op: 'addLine', stops: o.map((i) => near(xy[i])), name, colour: '#0a7d3b' });
        if (r.ok) { r.order = o; break; }
      }
      await pause(1500);
      const l = tw.world.value.save.lines.find((x) => x.num === r.line);
      return { ok: r.ok, issues: r.issues, order: r.order, num: r.line, id: l?.id, lng: lng + 3 * dLng, lat };
    };
    // the marks of shape 7, and the path's repeated entries, for a line
    window.__info = (id) => {
      const st = tw.world.value.lineStats[id], m = tw.renderer.net?.marks ?? [];
      let marks = 0;
      for (let k = 0; k < m.length; k += 6) if (m[k + 2] === 7) marks++;
      const seen = new Set(); let dup = 0;
      for (const p of st.path) { if (seen.has(p)) dup++; seen.add(p); }
      return { marks, dup, path: st.path.length };
    };
  });
  const lines = [];
  for (const [name, lat, order, want] of [['Plain', 42.30, [0, 1, 2, 3], false], ['Back', 42.40, [0, 3], false], ['Repeat', 42.50, [[0, 3, 1, 3], [0, 2, 1, 3], [0, 3, 2, 3], [0, 2, 3, 1, 2, 3], [0, 1, 3, 2, 3]], true]]) {
    const b = await page.evaluate(([lng, lat, order, name]) => window.__build(lng, lat, order, name), [-71.06, lat, order, name]);
    console.log(`  ${name}: ${JSON.stringify(b)}`);
    check(b.ok && b.id, `${name}: line built`, JSON.stringify(b.issues ?? ''));
    lines.push({ ...b, name, want });
  }
  await page.waitForTimeout(1500);
  for (const l of lines) {
    await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, l.id);
    await page.waitForTimeout(1200);
    const info = await page.evaluate((id) => window.__info(id), l.id);
    const text = await page.evaluate(() => document.body.innerText);
    const alert = text.includes('Runs the same track twice');
    console.log(`  ${l.name}: ${JSON.stringify(info)} alert ${alert}`);
    check(alert === l.want, `${l.name}: alert ${l.want ? 'shown' : 'absent'}`);
    check(l.want ? info.dup > 0 : info.dup === 0, `${l.name}: path ${l.want ? 'repeats' : 'does not repeat'} an entry`, `${info.dup}`);
  }
  // marks: only the repeating line has one
  const total = await page.evaluate((id) => window.__info(id).marks, lines[2].id);
  check(total === 1, 'one loop mark on the map, from the repeating line', `${total}`);
  // the picture
  const r = lines[2];
  await page.evaluate((r) => window.tw.map.jumpTo({ center: [r.lng + 4.5 / (111.32 * Math.cos((r.lat * Math.PI) / 180)), r.lat], zoom: 17 }), r);
  await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {});
  await page.waitForTimeout(1200);
  await page.screenshot({ path: join(ARTIFACTS, 't3562-repeat.png') });
  check(errors.filter((e) => !/favicon|404/.test(String(e))).length === 0, 'no page errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  console.log('ERROR', e.stack ?? e);
  fail++;
} finally {
  await chrome.close();
}
console.log(fail ? `${fail} FAILED` : 'ALL PASS');
process.exit(fail ? 1 : 0);
