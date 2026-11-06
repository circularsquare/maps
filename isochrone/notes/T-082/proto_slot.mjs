// T-082 prototype: canoe trips at L10, L3, L1 with the coarse levels as built (none), with the
// access time (acc), with the river graph contracted from L1 (slot), and both (slotacc).
//   ONLY=a,b VARS=none,acc,slot,slotacc node proto_slot.mjs tag  -> results/slot_<tag>.json
import { writeFileSync, readFileSync } from 'node:fs';
import * as H from './lib.mjs';
import * as S from './search_acc.mjs';
import { canoeBox, accessL1, coarseAcc, contract, accFromKm } from './slot.mjs';

const only = process.env.ONLY ? process.env.ONLY.split(',') : null;
const tag = process.argv[2] || 'all';
const pure = JSON.parse(readFileSync(H.SCR + 'results/pure_canoe.json', 'utf8'));
const VARIANTS = (process.env.VARS || 'none,acc,slot,slotacc').split(',');
H.cfg.hazard = process.env.HAZ !== 'off';
const out = [];
for (const reg of pure) {
    if (only && !only.includes(reg.name)) continue;
    const { lat, lng } = reg;
    const t0 = performance.now();
    const box = H.boxAround(lat, lng, 3.3, 3.3 / Math.cos(lat * Math.PI / 180));
    const w1 = H.win(1, box.west, box.south, box.east, box.north), w3 = H.win(3, box.west, box.south, box.east, box.north);
    const [b1, b3] = await Promise.all([H.loadBox(w1), H.loadBox(w3)]);
    const L1 = H.level(w1, b1.grid, b1.hz);
    const d1 = await canoeBox(L1);
    const t1 = await accessL1(L1, d1);
    const dests = reg.dests.map(d => ({ D: d.D, lat: d.lat, lng: d.lng }));
    const stats = {};
    let tE = null;
    for (const v of VARIANTS) {
        const L10 = await H.world10(false);
        L10.cn = null; await H.world10(true);
        const L3 = H.level(w3, b3.grid, b3.hz); await canoeBox(L3);
        if (v.startsWith('slot')) {
            stats.s10 = contract(L10, 0, 0, 10, true, L1, w1, d1);
            stats.s3 = contract(L3, w3.row0, w3.col0, 3, false, L1, w1, d1);
        }
        // access options: paint (the painted time of a canoe node on land), xfer share (getting in
        // or out within a cell), on which levels
        const O = { none: {}, slot: {}, acc: { paint: 1, xfer: 1 }, slotacc: { paint: 1, xfer: 1 },
            slotpaint: { paint: 1 }, slothalf: { paint: 1, xfer: 0.5 }, slotxh: { xfer: 0.5 },
            slotx1: { xfer: 1 }, slotxh10: { xfer: 0.5, only10: 1 }, slotx1_10: { xfer: 1, only10: 1 },
            slotxhE: { xfer: 0.5, e: 1 }, slotx1E: { xfer: 1, e: 1 }, slotxhE10: { xfer: 0.5, e: 1, only10: 1 },
            slotp1: { node: 1 }, slotp5: { node: 0.5 }, slotp75: { node: 0.75 } }[v];
        H.cfg.nodeShare = O.node || 0;
        if (O.paint || O.xfer) {
            if (O.e) {
                tE ||= await accessL1(L1, d1, true);
                L10.cn.acc = accFromKm(coarseAcc(w1, L1, tE, L10, 0, 0, 10, true), L10);
                if (!O.only10) L3.cn.acc = accFromKm(coarseAcc(w1, L1, tE, L3, w3.row0, w3.col0, 3, false), L3);
            } else {
                L10.cn.acc = coarseAcc(w1, L1, t1, L10, 0, 0, 10, true);
                if (!O.only10) L3.cn.acc = coarseAcc(w1, L1, t1, L3, w3.row0, w3.col0, 3, false);
            }
        }
        H.cfg.accPaint = !!O.paint;
        H.cfg.accXfer = !!O.xfer;
        H.cfg.accShare = O.xfer || 0;
        for (const [lv, L] of [[10, L10], [3, L3], [1, L1]]) {
            if (lv === 1 && v !== VARIANTS[0]) { for (const d of dests) d[`t1${v}`] = d[`t1${VARIANTS[0]}`]; continue; }
            const s = H.snap(L, lng, lat, L.win ? Math.ceil(10 / L.win.f) : 8);
            const res = await S.solve(L, { idx: [s], w: [0], t: [0], hz: [0] }, () => true, true);
            L.res = res; L.dist = S.cellTimes(L, res, true);
            for (const d of dests) d[`t${lv}${v}`] = H.timeAt(L, d.lat, d.lng).t;
            L.res = null;
        }
    }
    out.push({ name: reg.name, kind: reg.kind, lat, lng, stats, dests });
    console.error(`${reg.name}: ${JSON.stringify(stats)} ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
writeFileSync(H.SCR + `results/slot_${tag}.json`, JSON.stringify(out));
process.exit(0);
