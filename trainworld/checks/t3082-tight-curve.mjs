// T-3082: rail track can be drawn down to a 25 m radius; high-speed keeps 50 m.
//   node trainworld/checks/t3082-tight-curve.mjs [url]      (from the maps root; about a minute)
// On a fresh game, a 90-degree corner is placed with a set radius of 27 m (accepted, with the slowest
// speed on its track), 20 m (refused as too tight), and as high-speed track 40 m (refused) and 50 m
// (accepted). One PASS or FAIL line each.
import { launchChrome, GL } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6172/';
const { ctx, close } = await launchChrome({ prefix: 'tw-t3082-', size: [1200, 800], args: GL });
let failed = false;
const report = (ok, what, detail = '') => {
  failed ||= !ok;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}${detail ? '  (' + detail + ')' : ''}`);
};
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools, undefined, { timeout: 180000 });
  await page.waitForTimeout(2000);
  const corner = (dy, radius, hsr) => page.evaluate(async ([dy, radius, hsr]) => {
    const tw = window.tw, o = tw.lngLatToPack(-73.968, 40.778);
    const at = (x, y) => ({ kind: 'free', x: o.x + x, y: o.y + y, level: 0 });
    const before = new Set(tw.world.value.edges.map((e) => e.id));
    const r = await tw.client.edit({ op: 'route', from: at(0, dy), to: at(500, dy + 500), pis: [{ x: o.x + 500, y: o.y + dy, level: 0, radius }], single: false, hsr, design: hsr ? 320 : undefined });
    const e = tw.world.value.edges.filter((x) => !before.has(x.id));
    return { ok: !!r?.ok, issues: JSON.stringify(r?.issues ?? r?.error ?? null), vmin: e.length ? Math.min(...e.map((x) => x.vminKmh)) : null, edges: e.length };
  }, [dy, radius, hsr]);
  let r = await corner(0, 27, false);
  report(r.ok && r.edges > 0, 'rail corner of 27 m radius is accepted', JSON.stringify(r));
  report(r.vmin > 15 && r.vmin < 22, 'its slowest speed is 15-22 km/h', `${r.vmin} km/h`);
  r = await corner(1000, 20, false);
  report(!r.ok && /RadiusTooSmall/.test(r.issues), 'rail corner of 20 m radius is refused as too tight', JSON.stringify(r));
  r = await corner(2000, 40, true);
  report(!r.ok && /RadiusTooSmall/.test(r.issues), 'high-speed corner of 40 m radius is refused as too tight', JSON.stringify(r));
  r = await corner(3000, 50, true);
  report(r.ok && r.edges > 0, 'high-speed corner of 50 m radius is accepted', JSON.stringify(r));
} catch (e) {
  failed = true;
  console.log(`FAIL  tight curve check  ${e.message}`);
} finally {
  await close();
}
process.exitCode = failed ? 1 : 0;
