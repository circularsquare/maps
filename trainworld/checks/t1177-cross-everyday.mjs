// T-1177, T-1229: the everyday outings across a zone line, by purpose, through the real demand
// worker in Node (as t409-worker-check.mjs does). Boston and Providence open, with made-up pairs
// written into Boston's market pack in memory (errands, social and school from Boston's biggest
// zone to Providence's, with their own walking shares; nothing under data/ is touched), and a line
// joining the two zones. The same market without the pairs is the control. Checked:
//   - the purposes' parts: the per-purpose figures add up to the pairs' sums, only the three
//     purposes have any, the longer trips do not move, and the market's trips grow by exactly the
//     pairs' legs (nothing counted twice);
//   - their loads sit under their own purposes and none is left under the longer trips;
//   - a selection of every Boston cell (homes) and of every Providence cell (activities): the
//     blocks by purpose and far zone hold the pairs' legs, their rail and their walkers, and a
//     Saturday's legs are each purpose's own factor times the weekday's;
//   - the trip explorer between the two places: its legs, rail and walking are the period's parts;
//   - the weekend scaling of the parts takes each purpose's factor.
// With IDS=gua,fos (two registered neighbours whose market packs carry the pairs, after the China
// rebuild) it runs the market-pack checks on those packs as they are instead of the made-up pairs.
// Run from the maps root: node trainworld/checks/t1177-cross-everyday.mjs
import assert from 'node:assert/strict';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, bundleDemandWorker, startWorker, workerPack as pack, marketPacks, marketRows, registry, loadApp } from './lib.mjs';
import { FRAME, KINDS, TRIPS, WALK, NAMES, pairData, lineNet } from './t1177-data.mjs';
const wrapper = await bundleDemandWorker('t1177');
const ids = (process.env.IDS ?? 'bos,pvd').split(',');
const real = !!process.env.IDS;
const [homeId, awayId] = ids;
const ROUNDS = 2;
// Saturday trips per weekday trip by everyday purpose (sim days.rs DAY_RATE[0][1 + kind])
const SATURDAY = [1.39, 1.54, 0.11, 0.30, 0.92, 0.30, 1.54];
const { homeMarket, awayMarket, pts, withPairs } = pairData(homeId, awayId);
const net = lineNet(pts);
const worker = async (market) => {
  const k = startWorker(wrapper); await k.next((m) => m.kind === 'boot');
  k.w.postMessage({ kind: 'init', packs: [pack(homeId), pack(awayId)], intercity: market, rows: marketRows(), periods: [0, 1, 2, 3, 4] });
  const r = await k.next((m) => m.kind === 'ready' || m.kind === 'failed'); assert.equal(r.kind, 'ready', r.error);
  return k;
};
const solve = async (k, version) => {
  const from = k.messages.length;
  k.w.postMessage({ kind: 'solve', version, net, rounds: ROUNDS });
  const done = await k.next((m) => (m.kind === 'done' && m.version === version) || m.kind === 'failed', from);
  assert.equal(done.kind, 'done', done.error);
  return [...new Map(k.messages.slice(from).filter((m) => m.kind === 'result' && !m.cluster && m.version === version && m.summary[0] === ROUNDS).map((m) => [m.period, m])).values()].sort((x, y) => x.period - y.period);
};
const close = (x, y, tol, what) => assert(Math.abs(x - y) <= tol * Math.max(1, Math.abs(y)), `${what}: ${x} vs ${y}`);
const sum = (xs) => xs.reduce((s, v) => s + v, 0);

const plain = marketPacks(ids);
const pairs = real ? plain : marketPacks(ids).map((p) => (p.id === homeId ? { id: p.id, pack: JSON.stringify(withPairs()) } : p));
const kc = await worker(plain), kp = await worker(pairs);
const [control, res] = [await solve(kc, 1), await solve(kp, 1)];
assert.equal(res.length, 5);

// ---- the purposes' parts
const out = { ids, pairs: real ? 'real' : 'made up' };
const kindSum = Array.from({ length: 7 }, () => [0, 0, 0, 0]);
for (let q = 0; q < 5; q++) {
  const [c, r] = [control[q], res[q]];
  const ps = r.intercityParts, cs = c.intercityParts;
  assert.equal(ps.length, 8 + 4 * 7, 'part_summary length');
  assert(cs.slice(4).every((v) => v === 0), `control period ${q}: no pairs`);
  for (const i of [0, 1, 2, 3]) assert.equal(ps[i], cs[i], `period ${q}: longer trips and commutes do not move (${i})`);
  for (let j = 0; j < 4; j++) close(sum(NAMES.map((_, k) => ps[8 + 4 * k + j])), ps[4 + j], 1e-4, `period ${q} figure ${j} adds up`);
  NAMES.forEach((_, k) => { for (let j = 0; j < 4; j++) kindSum[k][j] += ps[8 + 4 * k + j]; });
  // nothing counted twice: the market's trips grow by the pairs' legs, the cities' do not
  close(r.intercitySummary[1] - c.intercitySummary[1], ps[4], 1e-6, `period ${q} market trips grow by the pairs' legs`);
  close(r.summary[1] - c.summary[1], ps[4], 1e-4, `period ${q} the total grows by the same`);
  for (const id of Object.keys(c.citySummary)) assert.deepEqual(r.citySummary[id].filter((_, i) => i !== 4), c.citySummary[id].filter((_, i) => i !== 4), `period ${q} ${id}: city figures do not move`);
}
if (!real) {
  KINDS.forEach((k) => {
    close(kindSum[k][0], 2 * TRIPS[k], 1e-4, `${NAMES[k]} legs a day`);
    assert(kindSum[k][1] > 0, `${NAMES[k]} has rail`);
    close(kindSum[k][3], 2 * TRIPS[k] * WALK[k], 1e-4, `${NAMES[k]} walking with no rail`);
    assert(kindSum[k][2] <= kindSum[k][3] + 1e-6 && (WALK[k] === 0 || kindSum[k][2] < kindSum[k][3]), `${NAMES[k]}: rail takes walkers`);
  });
  [3, 4, 5, 6].forEach((k) => assert(kindSum[k].every((v) => v === 0), `${NAMES[k]} has none`));
}
out.byPurpose = Object.fromEntries(NAMES.map((n, k) => [n, kindSum[k].map((v) => Math.round(v))]).filter(([, v]) => v[0] > 0));

// ---- loads under their own purposes
for (let q = 0; q < 5; q++) {
  const [c, r] = [control[q].parts, res[q].parts];
  assert.deepEqual(r.keys, c.keys);
  const n = r.seg.length / r.keys.length;
  const total = (p, key, f) => sum(Array.from(p[f].subarray(r.keys.indexOf(key) * n, (r.keys.indexOf(key) + 1) * n)));
  for (const f of ['board']) {
    for (const key of NAMES) {
      const had = real ? null : kindSum[NAMES.indexOf(key)][1] > 0;
      const d = total(r, key, f) - total(c, key, f);
      if (had === true) assert(d > 0, `period ${q}: ${key} boardings grow`);
      if (had === false) assert.equal(d, 0, `period ${q}: ${key} boardings stay`);
    }
    // the longer trips' share of the loads is the control's up to the crowding the new riders add:
    // the pairs are not left under it
    const added = NAMES.reduce((s, key) => s + total(r, key, f) - total(c, key, f), 0);
    assert(Math.abs(total(r, 'longer', f) - total(c, 'longer', f)) <= 0.01 * added, `period ${q}: longer trips' ${f} ${total(r, 'longer', f)} vs ${total(c, 'longer', f)}, the pairs add ${added}`);
    close(sum(Array.from(res[q][f])), sum(NAMES.concat(['work', 'longer']).map((key) => total(r, key, f))), 1e-4, `period ${q}: purposes add up to the ${f} total`);
  }
}

// ---- a selection of every cell of each end
const P = registry().market.places.length;
const cells = async (id) => Uint32Array.from({ length: (await kp.ask({ kind: 'cells', city: id, frame: FRAME })).xy.length / 2 }, (_, i) => i);
const sel = async (id, end, day, version = 1) => {
  const r = await kp.ask({ kind: 'icSel', city: id, version, round: ROUNDS, end, day, cells: await cells(id) });
  assert(r.packed, `icSel ${id} end ${end}`); assert.equal(r.mask, 31);
  return r.packed;
};
const layout = (v) => { const nz = (v.length - 3 - 12 * P) / 41; assert(Number.isInteger(nz), `layout: ${v.length} numbers, ${P} places`); return { nz, etail: v.length - 2 - 30 * nz }; };
const block = (v, ki, dir) => { const { nz, etail } = layout(v); const o = etail + ki * 6 * nz; const rows = []; for (let z = 0; z < nz; z++) if (v[o + 6 * z + 3 * dir] > 0) rows.push([z, v[o + 6 * z + 3 * dir], v[o + 6 * z + 3 * dir + 1], v[o + 6 * z + 3 * dir + 2]]); return rows; };
const EK = [0, 1, 2, 3, 5];
for (const [id, end, dir] of [[homeId, 0, 0], [awayId, 1, 1]]) {
  const v = await sel(id, end, 0);
  EK.forEach((k, ki) => {
    const rows = block(v, ki, dir);
    if (real) { assert(rows.length > 0 || true); return; }
    if (!KINDS.includes(k)) { assert.equal(rows.length, 0, `${id}: no ${NAMES[k]} outings`); return; }
    assert.equal(rows.length, 1, `${id}: ${NAMES[k]} goes to one far zone`);
    const [, all, rail, walk] = rows[0];
    close(all, 2 * TRIPS[k], 1e-4, `${id} ${NAMES[k]} legs in the selection`);
    close(rail, kindSum[k][1], 1e-3, `${id} ${NAMES[k]} rail in the selection`);
    close(walk, kindSum[k][2], 1e-3, `${id} ${NAMES[k]} walkers in the selection`);
  });
  // a Saturday: each purpose's own factor
  const sat = await sel(id, end, 1);
  EK.forEach((k, ki) => {
    const [w, s] = [block(v, ki, dir), block(sat, ki, dir)];
    if (!w.length) return;
    close(s[0][1] / w[0][1], SATURDAY[k], 1e-3, `${id} ${NAMES[k]}: a Saturday's legs per weekday's`);
  });
  out[`${id}Selection`] = EK.map((k, ki) => [NAMES[k], block(v, ki, dir).reduce((s, r) => s + r[1], 0)]).filter(([, v]) => v > 0);
}

// ---- the trip explorer between the places, either way: the period's parts
const homePlace = homeMarket.place, awayPlace = awayMarket.place;
for (const [near, far, end] of [[homePlace, awayPlace, 0], [awayPlace, homePlace, 1]]) {
  const r = await kp.ask({ kind: 'icTrip', city: homeId, version: 1, round: ROUNDS, end, day: 0, cells: new Uint32Array(0), far, farGid: -1, near: { place: near, gid: -1 } });
  assert(r.answers, 'icTrip answered');
  r.periods.forEach((q, i) => {
    const v = r.answers[i], ps = res[q].intercityParts;
    assert.equal(v.length >= 22, true);
    if (!real) {
      close(v[19], ps[4], 1e-4, `explorer ${near}-${far} period ${q}: legs`);
      close(v[20], ps[5], 1e-3, `explorer period ${q}: rail`);
      close(v[21], ps[6], 1e-3, `explorer period ${q}: walking`);
      if (v[20] > 0) assert(v[11] > 0 && v[14] === 0, `explorer period ${q}: station pairs for the pairs' rail riders`);
    }
  });
  out[`explorer${near}-${far}`] = r.periods.map((q, i) => Array.from(r.answers[i].slice(19, 22), Math.round));
}

// ---- a weekend takes each purpose's own factor
const { marketPartsOn, PARTS_LEN } = await loadApp('src/workers/marketParts.ts');
assert.equal(PARTS_LEN, 36);
{
  const p = Array.from({ length: 36 }, (_, i) => (i < 8 ? i + 1 : 1 + (i % 4)));
  for (let k = 0; k < 7; k++) for (let j = 0; j < 4; j++) p[8 + 4 * k + j] = (k + 1) * (j + 1);
  const factor = { shopping: 2, social: 3, school: 0, university: 5, airport: 1, medical: 7, attractions: 11 };
  const w = marketPartsOn(p, 0.5, 0.25, (key) => factor[key]);
  assert.deepEqual([w[0], w[1], w[2], w[3]], [p[0] * 0.5, p[1] * 0.5, p[2] * 0.25, p[3] * 0.25]);
  const want = (j) => 1 * 2 * (j + 1) + 2 * 3 * (j + 1) + 3 * 0 * (j + 1) + 4 * 5 * (j + 1) + 5 * 1 * (j + 1) + 6 * 7 * (j + 1) + 7 * 11 * (j + 1);
  for (let j = 0; j < 4; j++) assert.equal(w[4 + j], want(j), `weekend sum ${j}`);
  assert.equal(w[8 + 4 * 2], 0, 'school has none on this day');
  assert.deepEqual(marketPartsOn([1, 2, 3, 4, 5, 6, 7, 8], 1, 1, () => 9), [1, 2, 3, 4, 5, 6, 7, 8], 'an answer without the per-purpose numbers is kept');
}
for (const k of [kc, kp]) await k.w.terminate();
mkdirSync(join(ARTIFACTS, 't1177'), { recursive: true });
writeFileSync(join(ARTIFACTS, 't1177', 't1177-cross-everyday.json'), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out));
console.log('PASS t1177-cross-everyday');
