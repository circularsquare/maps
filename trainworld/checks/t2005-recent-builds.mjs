// T-2005, T-2153: the zones of the last two builds are kept from the memory drops of the levels of
// detail (workers/demandLod.ts `builtKeep`, `planFine`'s `keep`): not coarsened for being left alone,
// and the last to go when the budget forces zones coarse, the latest build's last of all; and they
// survive a reload (the save's `demandBuilt`). The second half plays sessions minute by minute
// under T-2005's rule and this one: the cases where T-2005 dropped them, and every other zone alike.
//   node trainworld/checks/t2005-recent-builds.mjs      (from the maps root; no browser, about 5 s)
// Prints PASS or FAIL lines and exits 1 on failure.
import { loadApp } from './lib.mjs';

globalThis.location ??= { search: '', hash: '', href: 'http://localhost/' };

const { builtKeep, planFine, zoneGB, EDIT_KEEP_MS, LEFT_ALONE_MS } = await loadApp("export { builtKeep, planFine, zoneGB, EDIT_KEEP_MS, LEFT_ALONE_MS } from '@app/src/workers/demandLod.ts';");
const { CITIES } = await loadApp("export { CITIES } from '@app/src/game/cities.ts';");
const { currentZones, headerUnread } = await loadApp("export { currentZones, headerUnread } from '@app/src/workers/saveFile.ts';");

const ids = CITIES.filter((c) => (c.localTrips ?? 0) > 1e6).slice(0, 10).map((c) => c.id);
let failed = 0;
const check = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const same = (a, b) => [...a].sort().join() === [...b].sort().join();
const sameBuilds = (a, b) => a.length === b.length && a.every((x, i) => same(x, b[i]));
check('ten zones with trips to test with', ids.length === 10, ids.join(','));
const [A, B, C, D, E, F] = ids;
const MIN = 60_000, now = 100 * MIN;
const all = new Set(ids);

// builtKeep: builds, latest first
check('no builds, nothing kept', builtKeep(new Map(), all).length === 0);
check('the last build is kept however old', sameBuilds(builtKeep(new Map([[A, now - 500 * MIN]]), all), [[A]]));
check('the second last is kept when recent', sameBuilds(builtKeep(new Map([[A, now - 8 * MIN], [B, now - 5 * MIN]]), all), [[B], [A]]));
check('the second last is kept however old', sameBuilds(builtKeep(new Map([[A, now - 500 * MIN], [B, now - MIN]]), all), [[B], [A]]));
check('only the last two count', sameBuilds(builtKeep(new Map([[A, now - 3 * MIN], [B, now - 2 * MIN], [C, now - MIN]]), all), [[C], [B]]));
{
  const m = new Map([[A, now - 3 * MIN], [B, now - 2 * MIN]]);
  m.set(A, now - MIN);
  check('a zone built again is last, the other second', sameBuilds(builtKeep(m, all), [[A], [B]]));
}
check('a zone the solves no longer cover is skipped', sameBuilds(builtKeep(new Map([[A, now - 3 * MIN], [B, now - MIN]]), new Set([A, C])), [[A]]));
check('zones of one edit are one build', sameBuilds(builtKeep(new Map([[A, now - MIN], [B, now - MIN], [C, now - 2 * MIN], [D, now - 3 * MIN]]), all), [[A, B], [C]]));

// planFine: every zone fine and left alone for 30 minutes, nothing held
const six = ids.slice(0, 6);
const long = now - 30 * MIN;
const base = (over) => ({ zones: six, held: [], near: [], fine: new Set(six), touched: new Map(six.map((id) => [id, long])), madeFine: new Map(six.map((id) => [id, long])), now, budget: 1e9, ...over });
check('left alone 30 min with no keep: all go coarse', planFine(base({})).size === 0);
check('left alone 30 min: the kept stay, the rest go', same(planFine(base({ keep: [[A], [B]] })), [A, B]));
const coarseGB = six.reduce((s, id) => s + zoneGB(id, true), 0);
const extra = (id) => zoneGB(id, false) - zoneGB(id, true);
const lastTouched = new Map(six.map((id, i) => [id, now - (15 + i) * MIN]));
const tight = planFine(base({ touched: lastTouched, budget: coarseGB + extra(A) + extra(B) + 1e-9, keep: [[A], [B]], madeFine: new Map() }));
check('over budget the kept zones are dropped last', same(tight, [A, B]), [...tight].join(','));
// recent zones (touched 1 minute ago, so left-alone does not drop them) and room for one more fine zone
const warm = new Map(six.map((id) => [id, now - MIN]));
const oneOf = (first, second) => planFine({ ...base({}), touched: warm, madeFine: new Map(), budget: coarseGB + Math.max(extra(first), extra(second)) + 1e-9, keep: [[first], [second]] });
check('under heavy pressure the latest build stays, the one before goes', same(oneOf(A, B), [A]) && same(oneOf(B, A), [B]), `${[...oneOf(A, B)]} ${[...oneOf(B, A)]}`);
check('no budget pressure and touched recently: everything stays', planFine({ ...base({}), touched: warm }).size === 6);
check('a held zone stays whatever', planFine(base({ held: [F], keep: [[A]] })).has(F));
check('a zone made fine under 5 minutes ago is not dropped for budget', planFine({ ...base({}), touched: warm, madeFine: new Map(six.map((id) => [id, now - MIN])), budget: 0 }).size === 6);

// the save's header carries the last two builds
{
  const h = { game: 'anitabuilder', level: 0, demandBuilt: [[A, 'no-such-zone'], [B]] };
  check('a header with demandBuilt is read whole', !headerUnread(h, 0));
  check('demandBuilt keeps its zones, unknown ids dropped', sameBuilds(currentZones(h).demandBuilt, [[A], [B]]), JSON.stringify(currentZones(h).demandBuilt));
}

// ---- sessions played minute by minute (demandClient.ts `lodCheck` every 30 s) ----------------------
/** T-2005's rule, as it was: the last build however old, the one before while under 10 minutes old,
 * one group; reloads forgot both. */
function oldKeep(built, t, among) {
  const rows = [...built].filter(([id]) => among.has(id));
  const times = [...new Set(rows.map(([, x]) => x))].sort((a, b) => b - a);
  const k = rows.filter(([, x]) => x === times[0] || (x === times[1] && t - x < 10 * MIN)).map(([id]) => id);
  return k.length ? [k] : [];
}
/**
 * `builds`: [minute, zone]; `view(minute)`: the zone in view (settled); `reloadAt`: minutes at which
 * the game is saved and loaded again. Returns, per zone, the minutes at which it was coarse.
 */
function session({ rule, zones, builds, view, reloadAt = [], minutes, budget = 1e9 }) {
  const among = new Set(zones);
  let fine = new Set(zones), touched = new Map(), madeFine = new Map(), edited = new Map(), built = new Map();
  for (const id of zones) { touched.set(id, 0); madeFine.set(id, 0); }
  const coarseAt = new Map(zones.map((id) => [id, []]));
  let saved = null;
  for (let step = 0; step <= minutes * 2; step++) {
    const m = step / 2, t = m * MIN;
    if (reloadAt.includes(m)) {
      // the page reloads: the save's demandFine is the fine set; T-2153's demandBuilt the last builds
      fine = new Set(fine);
      touched = new Map([...fine].map((id) => [id, t]));
      madeFine = new Map([...fine].map((id) => [id, t]));
      edited = new Map();
      built = new Map();
      if (rule === 'new' && saved) saved.forEach((b, i) => { for (const id of b) if (!built.has(id)) built.set(id, 2 - i); });
    }
    for (const [bm, z] of builds) if (bm === m) { edited.set(z, t); built.set(z, t); }
    const held = [];
    const v = view(m);
    if (v) held.push(v);
    for (const [id, x] of [...edited].sort((a, b) => b[1] - a[1])) if (t - x < EDIT_KEEP_MS && !held.includes(id)) held.push(id);
    for (const id of held) touched.set(id, t);
    const keep = rule === 'new' ? builtKeep(built, among) : oldKeep(built, t, among);
    const next = planFine({ zones, held, near: [], fine, touched, madeFine, now: t, budget, keep });
    for (const id of next) if (!fine.has(id)) madeFine.set(id, t);
    for (const id of [...edited.keys()]) if (t - edited.get(id) >= EDIT_KEEP_MS) edited.delete(id);
    for (const id of [...touched.keys()]) if (!next.has(id) && t - touched.get(id) >= LEFT_ALONE_MS) touched.delete(id);
    fine = next;
    if (rule === 'new') saved = keep;
    for (const id of zones) if (!fine.has(id)) coarseAt.get(id).push(m);
  }
  return coarseAt;
}
const firstCoarse = (r, id) => r.get(id)[0] ?? null;
const never = (r, id) => r.get(id).length === 0;
const sameOthers = (a, b, kept) => [...a.keys()].filter((id) => !kept.includes(id)).every((id) => a.get(id).join() === b.get(id).join());

// 1. built in A, then B, then looking at C for two hours (a desktop, far under the budget)
{
  const s = { zones: [A, B, C, D, E], builds: [[0, A], [5, B]], view: (m) => (m < 6 ? null : C), minutes: 120 };
  const o = session({ ...s, rule: 'old' }), n = session({ ...s, rule: 'new' });
  check('session 1, T-2005: the city built in before the last went coarse', firstCoarse(o, A) !== null, `A coarse from minute ${firstCoarse(o, A)}`);
  check('session 1: both cities built in stay fine for two hours', never(n, A) && never(n, B));
  check('session 1: every other zone goes coarse as before', sameOthers(o, n, [A, B]), `D coarse from ${firstCoarse(n, D)} (was ${firstCoarse(o, D)})`);
}
// 2. the same, saved and loaded again at 30 minutes
{
  const s = { zones: [A, B, C, D, E], builds: [[0, A], [5, B]], view: (m) => (m < 6 ? null : C), reloadAt: [30], minutes: 120 };
  const o = session({ ...s, rule: 'old' }), n = session({ ...s, rule: 'new' });
  check('session 2, T-2005: after a reload the last city built in went coarse', firstCoarse(o, B) !== null, `B coarse from minute ${firstCoarse(o, B)}`);
  check('session 2: after a reload both cities built in stay fine', never(n, A) && never(n, B));
  check('session 2: every other zone goes coarse as before', sameOthers(o, n, [A, B]));
}
// 3. a game over the budget: built in A, then B, then a look at a new zone every 2 minutes, with room
// for A and B fine beside the zones looked at in the last 5 minutes (FINE_HOLD_MS: three of them)
{
  const zones = ids;
  const tour = [C, D, E, F, ids[6], ids[7], ids[8], ids[9]];
  const coarse = zones.reduce((s, id) => s + zoneGB(id, true), 0);
  const budget = coarse + extra(A) + extra(B) + tour.map(extra).sort((a, b) => b - a).slice(0, 3).reduce((s, x) => s + x, 0) + 1e-9;
  const s = { zones, builds: [[0, A], [5, B]], view: (m) => (m < 6 ? null : tour[Math.floor((m - 6) / 2) % tour.length]), minutes: 120, budget };
  const o = session({ ...s, rule: 'old' }), n = session({ ...s, rule: 'new' });
  check('session 3, T-2005: over the budget the city built in before the last went coarse', firstCoarse(o, A) !== null, `A coarse from minute ${firstCoarse(o, A)}`);
  check('session 3: over the budget both cities built in stay fine while others go', never(n, A) && never(n, B) && tour.some((id) => n.get(id).length > 0));
}
// 4. built in A, then B, then C: only the last two are spared
{
  const s = { zones: [A, B, C, D], builds: [[0, A], [5, B], [10, C]], view: () => null, minutes: 90 };
  const n = session({ ...s, rule: 'new' });
  check('session 4: the city built in three builds ago goes coarse once left alone', firstCoarse(n, A) !== null && never(n, B) && never(n, C), `A coarse from minute ${firstCoarse(n, A)}`);
}
// 5. no builds at all: the new rule changes nothing
{
  const s = { zones: [A, B, C, D, E], builds: [], view: (m) => (m < 20 ? A : C), reloadAt: [40], minutes: 90 };
  check('session 5: a game with no builds is unchanged', sameOthers(session({ ...s, rule: 'old' }), session({ ...s, rule: 'new' }), []));
}

console.log(failed ? `FAIL  ${failed} failed` : 'PASS  all');
process.exit(failed ? 1 : 0);
