// T-2992: bus-to-bus transfers at a short distance. On a new game in New York, two bus lines crossing square
// in Brooklyn, then the north-south one switched to Manual with one stop placed by hand at a few distances from
// the crossing: each time the stops near the crossing, whether demand counts them one place (the transfers at
// each), and the dashed transfer lines drawn. From the maps root, on a Vite of your own:
//   node trainworld/checks/t2992-bus-transfers.mjs [dev url]
// About 6 minutes (a demand solve per placing).
import { ARTIFACTS, collectErrors, demandFinal, launchChrome } from './lib.mjs';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6480/';
const DISTS = (process.env.TW_DISTS ?? '30,60,90,140').split(',').map(Number);
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const C = [-73.9535, 40.6505], HALF = 2000;

const chrome = await launchChrome({ prefix: 'tw-t2992-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);
const edit = (op) => page.evaluate((op) => window.tw.client.edit(op), op);
const version = () => page.evaluate(() => window.twDemand.view()?.version ?? 0);
const settle = async (ms = 1200) => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await page.waitForTimeout(ms); };
/** the two lines' stops within 300 m of the crossing, with the transfers demand counts at each */
const near = (a, b) => page.evaluate(([a, b, C]) => {
  const tw = window.tw, w = tw.world.value, v = window.twDemand.view();
  const m = (p, q) => Math.hypot((p[0] - q[0]) * 111320 * Math.cos((q[1] * Math.PI) / 180), (p[1] - q[1]) * 111320);
  const out = {};
  for (const [name, num] of [['A', a], ['B', b]]) {
    const l = w.save.lines.find((x) => x.num === num), bus = w.lineStats[l.id].bus;
    out[name] = bus.stops.map((s) => ({ id: s.id, kind: s.kind, d: Math.round(m([s.lng, s.lat], C)), xfer: Math.round(v.stations.get(String(s.id))?.transfers ?? -1), board: Math.round(v.stations.get(String(s.id))?.boardings ?? -1) })).filter((s) => s.d <= 300);
    out[name + 'riders'] = Math.round(v.lines.get(l.id)?.ridersPerDay ?? 0);
  }
  out.drawn = tw.renderer.buses.transfers;
  return out;
}, [a, b, C]);
const out = { when: new Date().toISOString(), runs: [] };

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && window.tw.tools?.bus, undefined, { timeout: 180000 });
  await demandFinal(page, { timeout: 400000 });
  const made = await page.evaluate(async ([C, H]) => {
    const tw = window.tw, c = tw.lngLatToPack(C[0], C[1]);
    const A = await tw.client.edit({ op: 'addBusLine', route: [c.x - H, c.y, 1, c.x + H, c.y, 1], name: 'Check A', colour: '#c0392b' });
    const B = await tw.client.edit({ op: 'addBusLine', route: [c.x, c.y - H, 1, c.x, c.y + H, 1], name: 'Check B', colour: '#2d7dd2' });
    const con = await tw.client.edit({ op: 'construct', edges: [], stations: [], buses: [A.line, B.line] });
    return { ok: A.ok && B.ok && con.ok, a: A.line, b: B.line, c: [c.x, c.y] };
  }, [C, HALF]);
  check(made.ok, 'two crossing bus lines');
  let v = await version();
  await demandFinal(page, { timeout: 400000, after: v });
  let r = await near(made.a, made.b);
  console.log('auto both', JSON.stringify(r));
  out.runs.push({ case: 'auto', ...r });
  await edit({ op: 'busHand', line: made.b });
  for (const d of DISTS) {
    v = await version();
    await edit({ op: 'busHandStop', line: made.b, x: made.c[0], y: made.c[1] + d, on: true });
    await demandFinal(page, { timeout: 400000, after: v });
    r = await near(made.a, made.b);
    console.log(`hand stop ${d} m`, JSON.stringify(r));
    out.runs.push({ case: `hand ${d}`, ...r });
    if (d === DISTS[1]) {
      await page.evaluate((C) => { window.tw.map.jumpTo({ center: C, zoom: 16.5 }); window.tw.overlay.invalidate(); }, C);
      await settle();
      await page.screenshot({ path: join(ARTIFACTS, 't2992-hand-light.png'), clip: { x: 400, y: 150, width: 800, height: 600 } });
      await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'dark'; window.tw.overlay.invalidate(); });
      await settle();
      await page.screenshot({ path: join(ARTIFACTS, 't2992-hand-dark.png'), clip: { x: 400, y: 150, width: 800, height: 600 } });
      await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'light'; });
    }
    await edit({ op: 'busHandStop', line: made.b, x: made.c[0], y: made.c[1] + d, on: false });
  }
  for (const x of out.runs.slice(1)) {
    const b = x.B.find((s) => s.kind === 2), m = b ? b.d : -1;
    check(m >= 0 && (m <= 100) === (x.drawn === 1), `a hand stop ${m} m from A's crossing stop: ${m <= 100 ? 'a transfer line' : 'none'}`, `drawn ${x.drawn}`);
  }
  check(out.runs[0].drawn === 0, 'two Auto stops on one spot draw no line', `drawn ${out.runs[0].drawn}`);

  // Manual editing in the panel: a click on B's route 60 m from the crossing adds its stop where B passes A's
  await page.evaluate((C) => { window.tw.map.jumpTo({ center: C, zoom: 16 }); }, C);
  await settle();
  const id = await page.evaluate((n) => window.tw.world.value.save.lines.find((x) => x.num === n).id, made.b);
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, id);
  await page.waitForTimeout(1200);
  const mode = (name) => page.locator('.h .group button').filter({ hasText: new RegExp(`^${name}$`) });
  await mode('Auto').click();
  await page.waitForTimeout(1500);
  await mode('Manual').click();
  await page.waitForTimeout(1500);
  const at = await page.evaluate(([C]) => { const k = 1 / 111320; const p = window.tw.map.project([C[0], C[1] + 60 * k]); const r = window.tw.map.getContainer().getBoundingClientRect(); return [p.x + r.left, p.y + r.top]; }, [C]);
  await page.mouse.move(at[0], at[1]);
  await page.waitForTimeout(300);
  const tag = await page.evaluate(() => [...document.querySelectorAll('.stop-tag')].map((e) => e.innerText).join(' '));
  await page.screenshot({ path: join(ARTIFACTS, 't2992-snap-light.png'), clip: { x: 400, y: 150, width: 800, height: 600 } });
  await page.mouse.click(at[0], at[1]);
  await page.waitForTimeout(2500);
  r = await near(made.a, made.b);
  const snapped = r.B.find((s) => s.kind === 2);
  check(!!snapped && snapped.d <= 5, 'Manual: a click 60 m off snaps the stop to where the route passes the other line\'s stop', `${JSON.stringify(r.B)} tag "${tag.replace(/\n/g, ' / ')}"`);
  check(/Check A/.test(tag), 'and the tag names that line', tag.replace(/\n/g, ' / '));
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  await page.screenshot({ path: join(ARTIFACTS, 't2992-died.png') }).catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
await writeFile(join(ARTIFACTS, 't2992-bus-transfers.json'), JSON.stringify(out, null, 1)).catch(() => {});
console.log(fail ? `T-2992 FAILED: ${fail}` : 'T-2992 PASSED');
process.exit(fail ? 1 : 0);
