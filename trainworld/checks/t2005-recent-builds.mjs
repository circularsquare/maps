// T-2005: the zone built in last, and the one before it while it is recent, are kept from the memory
// drops of the levels of detail (workers/demandLod.ts `builtKeep`, `planFine`'s `keep`): not coarsened
// for being left alone, and the last to go when the budget forces zones coarse.
//   node trainworld/checks/t2005-recent-builds.mjs      (from the maps root; no browser, about 5 s)
// Prints PASS or FAIL lines and exits 1 on failure.
import { loadApp } from './lib.mjs';

globalThis.location ??= { search: '', hash: '', href: 'http://localhost/' };

const { builtKeep, planFine, zoneGB, SECOND_BUILT_KEEP_MS } = await loadApp("export { builtKeep, planFine, zoneGB, SECOND_BUILT_KEEP_MS } from '@app/src/workers/demandLod.ts';");
const { CITIES } = await loadApp("export { CITIES } from '@app/src/game/cities.ts';");

const ids = CITIES.filter((c) => (c.localTrips ?? 0) > 1e6).slice(0, 6).map((c) => c.id);
let failed = 0;
const check = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const same = (a, b) => [...a].sort().join() === [...b].sort().join();
check('six zones with trips to test with', ids.length === 6, ids.join(','));
const [A, B, C, D, E, F] = ids;
const MIN = 60_000, now = 100 * MIN;
const all = new Set(ids);

// builtKeep
check('no builds, nothing kept', builtKeep(new Map(), now, all).length === 0);
check('the last build is kept however old', same(builtKeep(new Map([[A, now - 500 * MIN]]), now, all), [A]));
check('the second last is kept when recent', same(builtKeep(new Map([[A, now - 8 * MIN], [B, now - 5 * MIN]]), now, all), [A, B]));
check('the second last goes when older than the recency', same(builtKeep(new Map([[A, now - SECOND_BUILT_KEEP_MS - MIN], [B, now - MIN]]), now, all), [B]));
check('only the last two count', same(builtKeep(new Map([[A, now - 3 * MIN], [B, now - 2 * MIN], [C, now - MIN]]), now, all), [B, C]));
{
  const m = new Map([[A, now - 3 * MIN], [B, now - 2 * MIN]]);
  m.set(A, now - MIN);
  check('a zone built again is last, the other second', same(builtKeep(m, now, all), [A, B]));
}
check('a zone the solves no longer cover is skipped', same(builtKeep(new Map([[A, now - 3 * MIN], [B, now - MIN]]), now, new Set([A, C])), [A]));
check('zones of one edit are all last', same(builtKeep(new Map([[A, now - MIN], [B, now - MIN], [C, now - 2 * MIN]]), now, all), [A, B, C]));

// planFine: every zone fine and left alone for 30 minutes, nothing held
const long = now - 30 * MIN;
const base = (over) => ({ zones: ids, held: [], near: [], fine: new Set(ids), touched: new Map(ids.map((id) => [id, long])), madeFine: new Map(ids.map((id) => [id, long])), now, budget: 1e9, ...over });
check('left alone 30 min with no keep: all go coarse', planFine(base({})).size === 0);
check('left alone 30 min: the kept stay, the rest go', same(planFine(base({ keep: [A, B] })), [A, B]));
const fineGB = ids.reduce((s, id) => s + zoneGB(id, false), 0), coarseGB = ids.reduce((s, id) => s + zoneGB(id, true), 0);
// a budget between all coarse and all coarse plus the two kept zones fine: the kept two survive
const keptExtra = [A, B].reduce((s, id) => s + zoneGB(id, false) - zoneGB(id, true), 0);
const lastTouched = new Map(ids.map((id, i) => [id, now - (15 + i) * MIN]));
const tight = planFine(base({ fine: new Set(ids), touched: lastTouched, budget: coarseGB + keptExtra + 1e-9, keep: [A, B], madeFine: new Map(), held: [] }));
check('over budget the kept zones are dropped last', same(tight, [A, B]), [...tight].join(','));
// the same with a budget that holds only one extra fine zone, recent zones (touched 1 minute ago, so left-alone does not drop them)
const warm = new Map(ids.map((id) => [id, now - MIN]));
const oneExtra = planFine({ ...base({}), touched: warm, madeFine: new Map(), budget: coarseGB + zoneGB(A, false) - zoneGB(A, true) + 1e-9, keep: [A, B] });
check('under heavy pressure only one kept zone stays, nothing else', oneExtra.size === 1 && (oneExtra.has(A) || oneExtra.has(B)), [...oneExtra].join(','));
check('no budget pressure and touched recently: everything stays', planFine({ ...base({}), touched: warm }).size === 6);
check('a held zone stays whatever', planFine(base({ held: [F], keep: [A] })).has(F));
check('a zone made fine under 5 minutes ago is not dropped for budget', planFine({ ...base({}), touched: warm, madeFine: new Map(ids.map((id) => [id, now - MIN])), budget: 0 }).size === 6);

console.log(failed ? `FAIL  ${failed} failed` : 'PASS  all');
process.exit(failed ? 1 : 0);
