// T-2404: a run of track closed on its own first point while the run's state is still held for the
// pointer's rest: the last click must join the first node (the run ends there, no second node on top).
// About 30 s.
//   node trainworld/checks/t2404-loop.mjs [url]
import { launchChrome, ANGLE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6191/';
const { ctx, close } = await launchChrome({ prefix: 'tw-loop-', size: [1400, 900], args: ANGLE });
let ok = false;
try {
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${BASE}?debug=1&new=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  await page.waitForTimeout(3000);
  const r = await page.evaluate(async () => {
    const tw = window.tw, ui = await import('/src/game/ui.ts');
    ui.keepPlacing.value = true;
    ui.tool.value = 'track';
    tw.map.jumpTo({ center: [-74.13, 40.585], zoom: 14 });
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    await sleep(1500);
    const cv = tw.map.getCanvas(), W = cv.clientWidth, H = cv.clientHeight;
    const ev = (x, y) => ({ point: { x, y }, lngLat: tw.map.unproject([x, y]), originalEvent: { ctrlKey: false, metaKey: false, shiftKey: false, detail: 1, button: 0, buttons: 0 } });
    const edges0 = tw.world.value.edges.length;
    // the hand never rests 220 ms until the last click
    const go = async (from, to, n = 12) => { for (let k = 1; k <= n; k++) { tw.tools.move(ev(from[0] + ((to[0] - from[0]) * k) / n, from[1] + ((to[1] - from[1]) * k) / n)); await sleep(16); } };
    const A = [W / 2 - 200, H / 2], B = [W / 2 + 150, H / 2 - 120], C = [W / 2 + 150, H / 2 + 160];
    tw.tools.move(ev(...A)); tw.tools.click(ev(...A));
    await go(A, B); tw.tools.click(ev(...B));
    await go(B, C); tw.tools.click(ev(...C));
    const heldBack = tw.world.value.edges.length - edges0;
    // the click comes at once on arriving, before the state the approach asked for can land
    await go(C, A, 20);
    tw.tools.move(ev(...A)); tw.tools.click(ev(...A));
    await sleep(2500);
    const w = tw.world.value, added = w.edges.slice(edges0);
    const ends = new Map();
    for (const e of added) for (const n of [e.a, e.b]) ends.set(n, (ends.get(n) ?? 0) + 1);
    return { heldBack, added: added.length, degrees: [...ends.values()].sort(), drawing: tw.tools.drawing };
  });
  // three pieces, every end shared by two of them: a closed loop
  ok = r.added === 3 && r.degrees.length === 3 && r.degrees.every((d) => d === 2) && !r.drawing && !errors.length;
  console.log(`${ok ? 'PASS' : 'FAIL'}  a run closed on its first point while held  ${JSON.stringify(r)}${errors.length ? ' errors ' + JSON.stringify(errors.slice(0, 3)) : ''}`);
} finally {
  await close();
}
process.exit(ok ? 0 : 1);
