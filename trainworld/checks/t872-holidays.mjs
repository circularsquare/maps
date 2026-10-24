// T-872, T-262: a public holiday runs as a Sunday for the lines in its country. Imports the real save
// (New York) into a ?new=1 game and moves the clock to the end of Wednesday, Thursday (the next
// Thanksgiving), Saturday and Sunday of that week, reading each day's boarding account (every
// line's past 24 hours) and, after Monday's start, the money ledger. Thanksgiving's riders and fares
// should match Sunday's, not Wednesday's or Saturday's, and fares should follow riders. The calendar
// itself: the same day is a weekday in Canada and the UK. T-262: each country's own week (a plain
// week in six countries), China's substitute day, Hungary's worked Saturday and a US working holiday
// in the calendar; then a new game with a line in Hong Kong (Saturday a weekday) and one in
// Shanghai, read in the boarding account.
//   node trainworld/checks/t872-holidays.mjs [url]     (from the maps root; default localhost:6145)
// Prints PASS or FAIL lines and exits 1 on failure. About two minutes, most of it demand.
import { readFile } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, acrossLine, REAL_SAVE, ANGLE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6145/';

const { ctx, close } = await launchChrome({ prefix: 'tw-t872-', size: [1200, 800], args: ANGLE });
let ok = true;
const check = (pass, text) => { console.log(`${pass ? 'PASS' : 'FAIL'}  ${text}`); ok &&= pass; };
try {
  const page = await ctx.newPage();
  const wait = (ms) => page.waitForTimeout(ms);
  await page.goto(`${BASE}?debug=1&new=1&paused=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });

  const bytes = await readFile(REAL_SAVE);
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { strict: false, held: true });
  await page.evaluate(() => window.tw.clock.setRunning(false));

  // the next Thanksgiving after the save's clock, and the same day in Canada and the UK
  const cal = await page.evaluate(async () => {
    const c = await import('/src/game/holidayCalendar.ts');
    const d = (await import('/src/game/holidays.json')).default;
    let thu = Math.floor(window.tw.clock.now() / 86400) + 1;
    while (!/Thanksgiving/.test(c.holidayName(d, 'US', thu) ?? '')) thu++;
    return { thu, kinds: ['US', 'CA', 'GB'].map((k) => c.kindOfIn(d, k)(thu)), name: c.holidayName(d, 'US', thu) };
  });
  const THU = cal.thu, WED = THU - 1, SAT = THU + 2, SUN = THU + 3;
  check(cal.kinds.join() === '2,0,0', `game day ${THU} is ${cal.name}: a Sunday in the US, a weekday in CA, GB: ${cal.kinds.join(', ')}`);

  // every line's past 24 hours at 23:50 of `day`: riders and fares summed (the account and the money)
  const dayAt = (day) => page.evaluate(async (day) => {
    window.tw.clock.set(day * 86400 + 23 * 3600 + 3000);
    await new Promise((r) => setTimeout(r, 1500));
    const { queryClock } = await import('/src/workers/clockClient.ts');
    const { readToday } = await import('/src/game/today.ts');
    const t = readToday(await queryClock({ what: 'today' }));
    let riders = 0, fares = 0;
    for (const v of t.values()) { riders += v.riders; fares += v.fares; }
    return { riders, fares };
  }, day);
  const wed = await dayAt(WED), thu = await dayAt(THU), sat = await dayAt(SAT), sun = await dayAt(SUN);
  for (const [k, v] of Object.entries({ wed, thu, sat, sun })) console.log(`  ${k}: riders ${Math.round(v.riders)}, fares $${Math.round(v.fares)}`);
  const near = (a, b, tol) => Math.abs(a / b - 1) < tol;
  check(near(thu.riders, sun.riders, 0.02) && !near(thu.riders, wed.riders, 0.1) && !near(sat.riders, sun.riders, 0.01), `Thanksgiving's boarding account is Sunday's: riders ${Math.round(thu.riders)} against ${Math.round(sun.riders)} (Wednesday ${Math.round(wed.riders)}, Saturday ${Math.round(sat.riders)})`);
  check(near(thu.fares / thu.riders, sun.fares / sun.riders, 0.03) && near(thu.fares / wed.fares, thu.riders / wed.riders, 0.05), `fares follow the account: $/rider Thursday ${(thu.fares / thu.riders).toFixed(3)}, Sunday ${(sun.fares / sun.riders).toFixed(3)}, Wednesday ${(wed.fares / wed.riders).toFixed(3)}`);

  await page.evaluate((t) => window.tw.clock.set(t), (SUN + 1) * 86400 + 1800);
  await wait(1500);
  const led = await page.evaluate(() => window.tw.moneyState.value.ledger.map((d) => ({ day: d.day, fares: d.fares, rides: d.rides })));
  const by = new Map(led.map((d) => [d.day, d]));
  for (let d = WED; d <= SUN; d++) { const e = by.get(d); if (e) console.log(`  ledger day ${d}: fares ${e.fares.toFixed(3)} US$M, rides ${Math.round(e.rides)}`); }
  const [lw, lt, lf, lsun] = [WED, THU, THU + 1, SUN].map((d) => by.get(d));
  check(!!(lw && lt && lf && lsun) && near(lt.fares, lsun.fares, 0.03) && near(lw.fares, lf.fares, 0.03) && !near(lt.fares, lw.fares, 0.1), `the ledger's Thanksgiving is a Sunday, Wednesday and Friday weekdays`);

  // T-262: each country's own weekend, substitute days and weekend days worked, in the calendar
  const own = await page.evaluate(async () => {
    const c = await import('/src/game/holidayCalendar.ts');
    const d = (await import('/src/game/holidays.json')).default;
    const days = await import('/src/game/days.ts');
    // the first whole week from a Monday with no holiday in any of them: kinds Monday to Sunday
    const CCS = ['US', 'SA', 'IL', 'IR', 'HK', 'BN'];
    let mon = Math.floor(window.tw.clock.now() / 86400) + 1;
    while (days.weekdayOf(mon) !== 0 || CCS.some((cc) => Array.from({ length: 7 }, (_, k) => c.holidayName(d, cc, mon + k)).some(Boolean))) mon++;
    const week = (cc) => Array.from({ length: 7 }, (_, k) => c.kindOfIn(d, cc)(mon + k)).join('');
    const first = (cc, test) => { for (let x = 1; x < 3000; x++) if (test(x)) return x; return null; };
    const cnSub = first('CN', (x) => /Spring Festival.*observed/.test(c.holidayName(d, 'CN', x) ?? '') && days.weekdayOf(x) < 5);
    const huSat = first('HU', (x) => days.weekdayOf(x) === 5 && c.kindOfIn(d, 'HU')(x) === 0);
    const usCol = first('US', (x) => /^Columbus Day/.test(c.holidayName(d, 'US', x) ?? ''));
    const name = (cc, x) => (x === null ? null : `${days.dateName(x)} ${c.holidayName(d, cc, x) ?? ''}`.trim());
    return { weeks: Object.fromEntries(CCS.map((cc) => [cc, week(cc)])), cn: name('CN', cnSub), cnKind: cnSub && c.kindOfIn(d, 'CN')(cnSub), hu: name('HU', huSat), huKind: huSat && c.kindOfIn(d, 'HU')(huSat), us: name('US', usCol), usKind: usCol && c.kindOfIn(d, 'US')(usCol) };
  });
  // a week with no holidays in these six, Monday to Sunday (0 a weekday, 1 a Saturday, 2 a Sunday)
  const WANT = { US: '0000012', SA: '0000210', IL: '0000120', IR: '0000200', HK: '0000002', BN: '0000201' };
  for (const [cc, w] of Object.entries(WANT)) check(own.weeks[cc] === w, `${cc}'s week runs ${own.weeks[cc]} (want ${w})`);
  check(own.cnKind === 2, `China's substitute day off is a Sunday: ${own.cn}`);
  check(own.huKind === 0, `Hungary's worked Saturday is a weekday: ${own.hu}`);
  check(own.usKind === 0, `a US working holiday is a weekday: ${own.us}`);

  // T-262 in the boarding account: a line in Hong Kong (a Sunday-only weekend) and one in Shanghai, in a
  // new game; each line's riders over its own local Wednesday, Saturday and Sunday, and Shanghai's
  // over a substitute day off
  const far = await ctx.newPage();
  await far.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await far.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  for (const at of [[114.17, 22.32], [121.47, 31.23]]) {
    await far.evaluate((c) => void window.tw.map.jumpTo({ center: c, zoom: 12 }), at);
    if ((await acrossLine(far, at)) === null) throw new Error(`no line built at ${at}`);
  }
  await demandFinal(far, { strict: false, lines: 1, timeout: 180000 });
  await far.evaluate(() => window.tw.clock.setRunning(false));
  const local = await far.evaluate(async () => {
    const h = await import('/src/game/holidays.ts');
    const c = await import('/src/game/holidayCalendar.ts');
    const d = (await import('/src/game/holidays.json')).default;
    const days = await import('/src/game/days.ts');
    const { shiftAt } = await import('/src/game/localTime.ts');
    const { queryClock } = await import('/src/workers/clockClient.ts');
    const { readToday } = await import('/src/game/today.ts');
    const w = window.tw.world.value.save, out = {};
    const now = Math.floor(window.tw.clock.now() / 86400) + 1;
    for (const l of w.lines) {
      const cc = h.lineCountry(l.stops), s = w.stations.find((x) => x.id === l.stops[0]), shift = shiftAt(s.lng, s.lat);
      let wed = now;
      while (days.weekdayOf(wed) !== 2 || [0, 3, 4].some((k) => c.holidayName(d, cc, wed + k))) wed++;
      const pick = { wed, sat: wed + 3, sun: wed + 4 };
      if (cc === 'CN') { let x = now; while (!(/observed/.test(c.holidayName(d, cc, x) ?? '') && days.weekdayOf(x) < 5)) x++; pick.sub = x; }
      const riders = {};
      for (const [k, day] of Object.entries(pick)) {
        window.tw.clock.set(day * 86400 + 23 * 3600 + 3000 - shift);
        await new Promise((r) => setTimeout(r, 1500));
        riders[k] = Math.round(readToday(await queryClock({ what: 'today' })).get(String(l.id))?.riders ?? 0);
      }
      out[cc] = { riders, sub: pick.sub && days.dateName(pick.sub) };
    }
    return out;
  });
  await far.close();
  for (const [cc, v] of Object.entries(local)) console.log(`  ${cc}: ${JSON.stringify(v)}`);
  const hk = local.HK?.riders, cn = local.CN?.riders;
  check(!!hk && near(hk.sat, hk.wed, 0.02) && !near(hk.sun, hk.wed, 0.05), `Hong Kong's Saturday runs as a weekday: Wednesday ${hk?.wed}, Saturday ${hk?.sat}, Sunday ${hk?.sun}`);
  check(!!cn && !near(cn.sat, cn.sun, 0.002) && !near(cn.sat, cn.wed, 0.05) && near(cn.sub, cn.sun, 0.02), `Shanghai's ${local.CN?.sub} runs as a Sunday: ${cn?.sub} against Sunday ${cn?.sun}, Saturday ${cn?.sat}, Wednesday ${cn?.wed}`);
} finally {
  await close();
}
process.exit(ok ? 0 : 1);
