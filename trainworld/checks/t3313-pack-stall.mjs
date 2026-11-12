// T-3313: a demand download that never answers must not freeze the figures for good. The demand pool waits
// for every pack a solve needs to open, and a pack fetch had no time limit: a request the network dropped
// without an error (a laptop waking on another network, a mobile connection switching) held every solve
// behind it, so riders, shares and the money they make stopped changing until a reload ("values don't
// update", "station riders stay at 0"). Here a new game loads, then the first request for each demand pack
// file after that (city .bin, market packs, the market's drives) is held unanswered for good while later
// requests pass, and a line is built. Its riders must still arrive, within the stall limit plus the retry
// and a margin.
//   node trainworld/checks/t3313-pack-stall.mjs [url]
import { collectErrors, launchChrome, demandFinal, acrossLine } from './lib.mjs';

const URL = process.argv[2] ?? 'http://localhost:6165/?new=1&debug=1&acct=off&notrack';
/** workers/patientFetch.ts STALL_MS, and the time allowed past it for the retry, the open and the solve */
const STALL_MS = 45_000, MARGIN_MS = 75_000;
// the demand pool's pack files; terrain and the clock worker's range reads have their own limits
const PACK = /\/packs\/(?!terrain-world\/)[^?]*(\.bin|\.market\.json)(\?|$)/;

const { ctx, close } = await launchChrome({ prefix: 'tw-t3313-' });
let fails = 0;
const check = (ok, what) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`); if (!ok) fails++; };
try {
  let hold = false;
  const seen = new Set(), held = [];
  await ctx.route((u) => PACK.test(u.href) && !/\/real\//.test(u.href), (route) => {
    const req = route.request(), key = req.url().replace(/\?.*$/, '');
    if (!hold || req.headers().range || seen.has(key)) return route.continue();
    seen.add(key);
    held.push(key.replace(/^.*\/packs\//, ''));
    // never answered
  });
  const page = ctx.pages()[0] ?? await ctx.newPage();
  const errors = [];
  collectErrors(page, errors);
  await page.goto(URL);
  await page.waitForFunction(() => window.tw?.client && window.tw.world?.value, null, { timeout: 120000 });
  await demandFinal(page, { strict: false, timeout: 120000 });
  hold = true;
  const t0 = Date.now();
  const line = await acrossLine(page, [-73.98, 40.75]);
  check(line !== null, `a line built (line ${line})`);
  let ok = true;
  await page.waitForFunction(() => [...(window.twDemand.view()?.lines.values() ?? [])].some((l) => l.ridersPerDay > 0), null, { timeout: STALL_MS + MARGIN_MS, polling: 1000 }).catch(() => { ok = false; });
  const s = Math.round((Date.now() - t0) / 1000);
  console.log(`      held once: ${held.join(', ') || 'nothing'}`);
  check(held.length > 0, 'a pack download was held');
  check(ok, ok ? `the line's riders arrived ${s} s after it was built` : `no riders within ${(STALL_MS + MARGIN_MS) / 1000} s of building it`);
  const real = errors.filter((e) => !/net::ERR|Failed to fetch|AbortError|TimeoutError|signal timed out|No data for|not loaded/.test(e));
  check(!real.length, `no page errors${real.length ? ': ' + real.slice(0, 3).join(' | ') : ''}`);
} finally {
  await close();
}
console.log(fails ? `${fails} failed` : 'all passed');
process.exit(fails ? 1 : 0);
