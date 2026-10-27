// T-1544: the draw anchor moves and the zone in view changes in one task (as a loaded game's first
// state does: its anchor, then the camera fit), with the focused zone's bubbles already built for the
// result shown. The ask the anchor started found its worker gone and threw "Invalid value used as
// weak map key" (commuterView's `sync` given null). Passes on no page error and the bubbles back.
//   BASE=http://localhost:6101/ node trainworld/checks/t1544-focus-race.mjs      (about a minute)
import { launchChrome, demandFinal, collectErrors, ANGLE } from './lib.mjs';

const BASE = process.env.BASE ?? 'http://localhost:6101/';
const { ctx, close } = await launchChrome({ prefix: 'tw-t1544-', args: ANGLE });
const errors = [];
let back = false;
try {
  const page = await ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?new=1&debug=1`);
  await page.waitForFunction(() => !!window.twDemand && !!window.twViews && !!window.tw?.map, undefined, { timeout: 120000 });
  await demandFinal(page, { held: true, timeout: 300000 });
  await page.evaluate(() => { const d = window.tw.display; d.value = { ...d.value, commuters: true, commuterEnd: 'work' }; });
  const own = () => { const o = window.twViews.overlay; return (o.bubbles[o.lastLevel]?.count ?? 0) > 0; };
  await page.waitForFunction(own, undefined, { timeout: 120000 });
  await page.waitForTimeout(1500);
  await page.evaluate(async () => {
    const geo = await import('/src/map/geo.ts');
    const av = await import('/src/map/anchorVersion.ts');
    const cities = await import('/src/game/cities.ts');
    for (let i = 0; i < 3; i++) {
      const focus = cities.activeCity.peek();
      const other = cities.CITIES.find((c) => c.id !== focus).id;
      const a = geo.anchor();
      geo.setAnchor(geo.makeAnchor(a.lng + 0.001, a.lat, a.version + 1));
      av.anchorVersion.value = a.version + 1;
      cities.activeCity.value = other;
      await new Promise((r) => setTimeout(r, 2000));
      cities.activeCity.value = focus;
      await new Promise((r) => setTimeout(r, 3000));
    }
  });
  back = await page.waitForFunction(own, undefined, { timeout: 60000 }).then(() => true, () => false);
} finally {
  await close();
}
if (errors.length) console.log('  errors: ' + JSON.stringify(errors.slice(0, 3)));
const ok = !errors.length && back;
console.log(`${ok ? 'PASS' : 'FAIL'} t1544 focus race (${errors.length} page errors, bubbles ${back ? 'back' : 'missing'})`);
process.exitCode = ok ? 0 : 1;
