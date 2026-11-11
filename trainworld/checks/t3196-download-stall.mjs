// T-3196: a download that never answers must not stop the game. The clock worker answers messages one at a
// time, and Construct waits there for the ground under the blueprint (road tiles, terrain, water masks). A
// road tile request the network never answered held Construct, and every edit, quote and hour of trains
// behind it, until a reload: players saw Construct never come back, new lines refused and all the trains
// stopped. Here the basemap's zoom 14 tiles are held unanswered after a line is built; then a blueprint is
// drawn and Construct pressed. It must be refused as the roads not loading within GROUND_WAIT_MS plus a
// margin, a quote and a new line must be answered after it, and the trains' hour must follow the clock at
// speed while Construct waits.
//   node trainworld/checks/t3196-download-stall.mjs [url]
import { collectErrors, launchChrome, acrossLine } from './lib.mjs';

const URL = process.argv[2] ?? 'http://localhost:6119/?new=1&debug=1&acct=off&notrack';
/** clock.worker.ts GROUND_WAIT_MS, and the time allowed past it */
const GROUND_WAIT_MS = 20_000, MARGIN_MS = 8_000;
const TILE = /\/14\/\d+\/\d+(\.pbf|\.mvt)?(\?|$)/;

const { ctx, close } = await launchChrome({ prefix: 'tw-t3196-' });
let fails = 0;
const check = (ok, what) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`); if (!ok) fails++; };
try {
  let hold = false, held = 0;
  await ctx.route((u) => TILE.test(u.href), (route) => { if (!hold) return route.continue(); held++; /* never answered */ });
  const page = ctx.pages()[0] ?? await ctx.newPage();
  const errors = [];
  collectErrors(page, errors);
  await page.goto(URL);
  await page.waitForFunction(() => window.tw?.client && window.tw.world?.value, null, { timeout: 120000 });
  await page.waitForTimeout(3000);
  const line = await acrossLine(page, [-73.98, 40.75]);
  check(line !== null, `a line built while tiles load (line ${line})`);
  await page.waitForTimeout(2000);
  hold = true;
  // a blueprint where no tile is in yet
  const route = await page.evaluate(async () => {
    const tw = window.tw, a = tw.lngLatToPack(-73.90, 40.70), b = tw.lngLatToPack(-73.86, 40.72);
    return tw.client.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: -1 }, to: { kind: 'free', x: b.x, y: b.y, level: -1 }, pis: [], single: false });
  });
  check(route.ok, 'blueprint drawn');
  await page.evaluate(() => window.tw.clock.setMultiple(100));
  // Construct, with the trains' hour sampled while it waits
  const r = await page.evaluate(async (limit) => {
    const tw = window.tw, t0 = performance.now(), lags = [];
    const timer = setInterval(() => lags.push(Math.floor(tw.clock.now() / 3600) - tw.renderer.tripsEpoch / 3600), 1000);
    const a = await Promise.race([tw.client.edit({ op: 'constructAll' }), new Promise((d) => setTimeout(() => d(null), limit))]);
    clearInterval(timer);
    return { a: a && { ok: a.ok, issues: a.issues }, ms: Math.round(performance.now() - t0), lags };
  }, GROUND_WAIT_MS + MARGIN_MS);
  console.log(`      construct after ${r.ms} ms: ${JSON.stringify(r.a)}; ${held} tiles held; hours behind each second: ${r.lags.join(' ')}`);
  check(!!r.a, `Construct answered within ${(GROUND_WAIT_MS + MARGIN_MS) / 1000} s`);
  check(r.a && !r.a.ok && r.a.issues.includes('RoadsMissing'), 'refused as the roads not loading');
  check(r.lags.length > 5 && Math.max(...r.lags.slice(2)) <= 1, 'the trains\' hour kept up with the clock meanwhile');
  const after = await page.evaluate(async () => {
    const c = window.tw.client, wait = (p) => Promise.race([p, new Promise((d) => setTimeout(() => d(null), 10000))]);
    const q = await wait(c.query({ what: 'blueprintQuote' }));
    const l = await wait(c.edit({ op: 'addLine', stops: [], name: 'After', colour: '#336699' }));
    return { quote: q ? q.length : null, line: l ? l.ok : null };
  });
  check(after.quote > 0, 'the blueprint quote answered after it');
  check(after.line === true, 'a new line added after it');
  const real = errors.filter((e) => !/net::ERR|Failed to fetch|AbortError|TimeoutError|signal timed out|not loaded/.test(e));
  check(!real.length, `no page errors${real.length ? ': ' + real.slice(0, 3).join(' | ') : ''}`);
} finally {
  await close();
}
console.log(fails ? `${fails} failed` : 'all passed');
process.exit(fails ? 1 : 0);
