// T-3578: a line's own fare multiple scales the base too, a trip paying it times the largest among the lines it
// rides. On the real save, line 14's fare multiple is typed in its panel: at 3.00x its fare a ride is above
// T-3567's $2.076 (when only the part above the base was scaled) and its riders fewer; at 0.00x its fare a ride is
// below T-3567's $0.820 (the base no longer paid on trips riding only it) and its revenue still above 0 in every
// period. A trip found riding line 14 after another line: with that line at 1.50x and line 14 at 2.00x the route's
// fare is its distance fares times each line's multiple plus twice the base; line 14 alone at 2.00x, twice the base
// too. Back at 1.00x every figure is as at the start, to the bit (rail trips, each line's riders and revenue).
// T-3581: line 14 at 1.50x deleted from its panel and the delete undone: it comes back at 1.50x.
//   node trainworld/checks/t3578-line-base.mjs [url] [line number]        (about eight minutes; default localhost:6163, line 14)
//   node trainworld/checks/t3578-line-base.mjs [url] [line] --dump <file>  the start's figures only, as JSON (for a
//                                                                         comparison against another build)
import { writeFile } from 'node:fs/promises';
import { REAL_SAVE } from './lib.mjs';
import { openGame } from './lib/play.mjs';

const args = process.argv.slice(2);
const dumpAt = args.indexOf('--dump');
const dump = dumpAt >= 0 ? args.splice(dumpAt, 2)[1] : null;
const BASE = args[0] ?? 'http://localhost:6163/';
const NUM = Number(args[1] ?? 14);
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

const g = await openGame({ url: BASE, save: REAL_SAVE, saveVia: 'post', prefix: 'tw-t3578-', ui: { shade: 'light', v: 15 } });
const page = g.page;
/** rail trips a day, each line's riders and fare revenue a day (US$M, and by period), and line NUM's */
const figures = () => page.evaluate(async (num) => {
  const money = await import('/src/game/money.ts'), periods = await import('/src/game/periods.ts');
  const hours = periods.PERIODS.map((p) => p.to - p.from);
  const v = window.twDemand.view(), w = window.tw.world.value, inc = money.incomeByDay.value[0];
  const day = (m) => (m ? m.perHour.reduce((a, x, p) => a + x * hours[p], 0) : null);
  const lines = {};
  for (const l of w.save.lines) lines[l.num] = [v.lines.get(l.id)?.ridersPerDay ?? null, day(inc?.lines.get(l.id))];
  const [riders, revenue] = lines[num];
  const id = w.save.lines.find((l) => l.num === num)?.id;
  const byPeriod = inc?.lines.get(id)?.perHour ?? [];
  return { version: v.version, rail: v.railTripsPerDay, lines, riders, revenue, byPeriod, perRide: revenue * 1e6 / 100 / riders, own: money.moneyState.value?.fares?.lines ?? null };
}, NUM);
const row = async () => (await g.see.panel())?.rows['Fare multiple'] ?? null;
const settle = async (v) => { await g.waitDemand({ after: v }); await page.waitForTimeout(2000); };
const version = () => page.evaluate(() => window.twDemand.view().version);
/** type into the open panel's fare multiple, then wait for demand's final on the new fares */
const typeFare = async (text) => {
  const v = await version();
  await g.act.type('Type fare multiple', text, { within: '#inspector' });
  await settle(v);
};
/** set lines' multiples as the panel does (world.setLineFare), then wait for demand */
const setFares = async (muls) => {
  const v = await version();
  await page.evaluate(async (m) => { const w = await import('/src/game/world.ts'); for (const [num, mul] of m) await w.setLineFare(num, mul); }, muls);
  // the last edit's network sent to demand before the wait for its newest solve
  await page.waitForTimeout(1500);
  await settle(v);
};
const fmt = (f) => `${Math.round(f.riders).toLocaleString('en-US')} riders, $${f.revenue.toFixed(2)}M a day, $${f.perRide.toFixed(3)} a ride`;

/** a trip between two stations, by station ids: each period's base (perceived) and routes as ride legs */
const trip = (a, b) => page.evaluate(async ([a, b]) => {
  const t = await import('/src/game/trip.ts');
  const r = await t.askTrip(t.stationPoint(a), t.stationPoint(b), -1);
  return r?.periods.map((p) => p && { base: p.baseFarePerceived, perUsd: p.minPerUsd, routes: p.routes.map((x) => ({ share: x.share, baseMul: x.baseMul ?? 1, rides: x.legs.filter((l) => l.kind === 'ride').map((l) => ({ line: l.line, from: l.from, to: l.to, fare: l.farePerceived })) })) }) ?? null;
}, [a, b]);
const sig = (route) => route.rides.map((l) => `${l.line}:${l.from}>${l.to}`).join(' ');

try {
  await g.waitDemand({ lines: 50 });
  await page.waitForTimeout(2000);
  const f0 = await figures();
  console.log(`  line ${NUM} at 1.00x: ${fmt(f0)}; rail trips ${Math.round(f0.rail).toLocaleString('en-US')}`);
  if (dump) {
    await writeFile(dump, JSON.stringify({ rail: f0.rail, lines: f0.lines }));
    console.log(`  wrote ${dump}`);
  } else {
    // a two-leg trip: another rail line X into line NUM at a shared station, from X's stop a few stops away to
    // NUM's stop a few stops the other side
    const cand = await page.evaluate((num) => {
      const w = window.tw.world.value, L = w.save.lines.find((l) => l.num === num), out = [];
      const rail = (l) => { const s = w.lineStats[l.id]; return s && s.status === 'running' && !s.bus; };
      const at = new Map(w.save.stations.map((s) => [s.id, s]));
      // a change between the lines: a station of both, or two of a complex (within 200 m)
      const near = (a, b) => { const p = at.get(a), q = at.get(b); return !!p && !!q && Math.hypot((p.lng - q.lng) * 84000, (p.lat - q.lat) * 111000) < 200; };
      for (const X of w.save.lines) {
        if (X.num === num || !rail(X)) continue;
        for (const [j, s] of X.stops.entries()) {
          const i = L.stops.findIndex((t) => t === s || near(s, t));
          if (i < 0) continue;
          for (const dj of [-4, 4]) for (const di of [-4, 4]) {
            const a = X.stops[j + dj], b = L.stops[i + di];
            if (a && b && !L.stops.includes(a) && !X.stops.includes(b)) out.push({ x: X.num, xid: X.id, lid: L.id, a, b });
          }
        }
      }
      return out.slice(0, 40);
    }, NUM);
    let found = null;
    console.log(`  ${cand.length} candidate trips`);
    for (const [k, c] of cand.entries()) {
      const t = await trip(c.a, c.b);
      if (k < 3) console.log(`  line ${c.x} (${c.xid}) into ${NUM} (${c.lid}), ${c.a} to ${c.b}: ${t ? t.map((p) => p ? p.routes.slice(0, 3).map((r) => r.rides.map((l) => l.line).join('+')).join(' | ') : '-').join(' / ') : 'no answer'}`);
      const q = t?.findIndex((p) => p?.routes.some((r) => r.rides.length === 2 && r.rides[0].line === c.xid && r.rides[1].line === c.lid));
      if (q !== undefined && q >= 0) { found = { ...c, q, before: t[q] }; break; }
    }
    check(!!found, `a trip riding another line into line ${NUM}${found ? `: line ${found.x} then ${NUM}, ${found.a} to ${found.b}` : ''}`);
    if (found) {
      const route0 = found.before.routes.find((r) => r.rides.length === 2 && r.rides[0].line === found.xid && r.rides[1].line === found.lid);
      await setFares([[found.x, 1.5], [NUM, 2]]);
      const after = (await trip(found.a, found.b))[found.q];
      const route1 = after.routes.find((r) => sig(r) === sig(route0));
      const mul = [1.5, 2];
      const legsOk = !!route1 && route1.rides.every((l, k) => Math.abs(l.fare - route0.rides[k].fare * mul[k]) < 1e-4 * Math.max(1, l.fare));
      const usd = (m) => `$${(m / after.perUsd).toFixed(3)}`;
      check(route1 && route1.baseMul === 2 && legsOk,
        `lines at 1.50x and 2.00x: the route's base is ${route1?.baseMul}x (${usd(after.base * (route1?.baseMul ?? 0))} against one base ${usd(after.base)}), each leg's distance fare its line's multiple times before`);
      // the trip's riders: the share on the route through line NUM falls
      const share = (p) => p.routes.filter((r) => r.rides.some((l) => l.line === found.lid)).reduce((s, r) => s + r.share, 0);
      console.log(`  the trip's share on routes through line ${NUM}: ${share(found.before).toFixed(3)} at 1.00x, ${share(after).toFixed(3)} after`);
      await setFares([[found.x, 1], [NUM, 1]]);
      const back = await figures();
      check(back.rail === f0.rail && JSON.stringify(back.lines) === JSON.stringify(f0.lines), `both back at 1.00x: every figure as at the start (rail ${back.rail} vs ${f0.rail})`);
    }

    const name = await page.evaluate((num) => window.tw.world.value.save.lines.find((l) => l.num === num)?.name, NUM);
    await g.act.openLine(name);
    check((await row()) === '1.00x', `line ${NUM} (${name}): the row reads ${await row()}`);
    await typeFare('3');
    const f3 = await figures();
    check((await row()) === '3.00x' && f3.perRide > 2.076 && f3.riders < f0.riders, `at 3.00x, the base scaled too (T-3567's above-the-base only: $2.076): ${fmt(f3)}`);
    await typeFare('0');
    const fz = await figures();
    check((await row()) === '0.00x' && fz.perRide < 0.82 && fz.riders > f0.riders, `at 0.00x, the base gone on trips riding only it (T-3567: $0.820): ${fmt(fz)}`);
    check(fz.byPeriod.every((x) => x >= 0), `at 0.00x no period's revenue is below 0: ${fz.byPeriod.map((x) => x.toFixed(4)).join(', ')} $M an hour`);
    const v = await version();
    await g.act.click('Back to 1.00x', { within: '#inspector' });
    await settle(v);
    const f1 = await figures();
    check((await row()) === '1.00x' && f1.own === null, `the undo button: back to ${await row()} (${JSON.stringify(f1.own)})`);
    check(f1.rail === f0.rail && JSON.stringify(f1.lines) === JSON.stringify(f0.lines), `back at 1.00x every figure is as it started (rail ${f1.rail} vs ${f0.rail})`);

    // T-3581: delete the line at 1.50x, undo
    await typeFare('1.5');
    const before = await figures();
    const lines0 = await page.evaluate(() => window.tw.world.value.save.lines.length);
    await g.act.click('Delete line', { within: '#inspector' });
    await page.waitForFunction((n) => window.tw.world.value.save.lines.length === n - 1, lines0, { timeout: 30000 });
    await page.waitForTimeout(3000);
    await g.act.key('Control+Z');
    await page.waitForFunction((n) => window.tw.world.value.save.lines.length === n, lines0, { timeout: 30000 });
    await page.waitForFunction((num) => window.tw.moneyState.value?.fares?.lines?.[String(num)] !== undefined, NUM, { timeout: 30000, polling: 500 }).catch(() => {});
    const own = await page.evaluate(() => window.tw.moneyState.value?.fares?.lines ?? null);
    check(own?.[NUM] === 1.5, `deleted at 1.50x and undone: the line's multiple is ${JSON.stringify(own)} (was ${JSON.stringify(before.own)})`);
    await g.act.openLine(name);
    check((await row()) === '1.50x', `its panel reads ${await row()}`);

    // every line at 0x, then 0, 3 and 0.5 by line number: no line's revenue in any period below 0, and no leg fare
    const nums = await page.evaluate(() => window.tw.world.value.save.lines.map((l) => l.num));
    const income = () => page.evaluate(async () => {
      const money = await import('/src/game/money.ts'), periods = await import('/src/game/periods.ts');
      const hours = periods.PERIODS.map((p) => p.to - p.from), inc = money.incomeByDay.value[0];
      let min = Infinity, at = null, below = 0;
      for (const l of window.tw.world.value.save.lines) {
        const m = inc?.lines.get(l.id);
        if (!m) continue;
        if (m.perHour.some((v) => v < 0)) below++;
        for (const v of m.perHour) if (!(v >= min)) { min = v; at = l.num; }
      }
      return { min, at, below, total: inc.perHour.reduce((a, x, p) => a + x * hours[p], 0) };
    });
    const legs = async () => {
      const t = found ? await trip(found.a, found.b) : null;
      const fares = (t ?? []).flatMap((p) => (p ? p.routes.flatMap((r) => r.rides.map((l) => l.fare)) : []));
      return fares.length ? Math.min(...fares) : null;
    };
    const i1 = await income();
    for (const [what, mulOf] of [['every line at 0x', () => 0], ['lines at 0, 3 and 0.5x by number', (n) => [0, 3, 0.5][n % 3]]]) {
      await setFares(nums.map((n) => [n, mulOf(n)]));
      const i = await income(), leg = await legs();
      check(i.below === 0 && Number.isFinite(i.min), `${what}: ${i.below} lines below 0 in some period, the lowest $${i.min.toFixed(4)}M an hour (line ${i.at}); income $${i.total.toFixed(2)}M a day against $${i1.total.toFixed(2)}M`);
      check(leg === null || leg >= 0, `${what}: the trip's lowest leg fare ${leg?.toFixed(3)}`);
      if (mulOf(1) === 0 && mulOf(4) === 0) check(Math.abs(i.total) < 0.01 * i1.total, `every line at 0x: income about 0 ($${i.total.toFixed(3)}M a day)`);
    }
    await setFares(nums.map((n) => [n, 1]));
    const fb = await figures();
    check(fb.rail === f0.rail && JSON.stringify(fb.lines) === JSON.stringify(f0.lines), `every line back at 1.00x: every figure as at the start`);
  }
  const errs = g.pageErrors().filter((e) => !/WebGL|GPU|ResizeObserver/.test(e));
  check(errs.length === 0, `no page errors${errs.length ? ': ' + errs.slice(0, 3).join(' | ') : ''}`);
} catch (e) {
  check(false, `ran to the end: ${e.stack ?? e.message}`);
} finally {
  await g.close();
}
console.log(fail ? `${fail} FAILED` : 'all passed');
process.exit(fail ? 1 : 0);
