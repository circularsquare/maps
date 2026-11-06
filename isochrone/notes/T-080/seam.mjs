// Window seams: an L1 window that does not hold the start, seeded on its ring from L10 (built now,
// or from crossing times), against the L1 solved whole. Hazard off.
//   inside: window time / whole-L1 time over the window's land cells
//   step:   window time / L10 time at points a quarter of the window in from its edge
//   node seam.mjs > results/seam.json
import { readFileSync, writeFileSync } from 'node:fs';
import * as H from './lib.mjs';
import { crossings, gridFrom } from './homog.mjs';
import { STARTS } from './regions.mjs';

H.cfg.hazard = false;
const only = process.env.ONLY ? process.env.ONLY.split(',') : null;
const geo = (a, b) => a < Infinity && b < Infinity ? Math.sqrt(a * b) : Math.min(a, b);
const out = [];
const P_SEED = 255;
for (const [name, kind, lat, lng] of STARTS) {
    if (only && !only.includes(name)) continue;
    const cos = Math.cos(lat * Math.PI / 180);
    const box = H.boxAround(lat, lng, 3.3, 3.3 / cos);
    const w1 = H.win(1, box.west, box.south, box.east, box.north), w10 = H.win(10, box.west, box.south, box.east, box.north);
    const g = (await H.loadBox(w1)).grid;
    const base = H.upscale(g, w1.cols, w1.rows, 10);
    const X = crossings(g, w1.cols, w1.rows, 10, 10, w1.north, w1.cell);
    const fix = gridFrom(base, X, (X, i) => geo(X.ewMin[i], X.nsMin[i]));
    const whole = H.level(w1, g, null);
    await H.solveFrom(whole, lat, lng, false);
    const C = {};
    for (const [k, grid] of [['cur', base], ['fix', fix]]) { C[k] = H.level(w10, grid, null); await H.solveFrom(C[k], lat, lng, false); }
    const res = { name, kind, wins: [] };
    for (const [dkm, bear] of [[120, 90], [120, 270], [200, 0], [200, 180]]) {
        const [cl, cg] = H.offset(lat, lng, dkm, bear);
        const ww = H.win(1, Math.floor((cg - 0.75 / cos) * 4) / 4, Math.floor((cl - 0.5) * 4) / 4, Math.ceil((cg + 0.75 / cos) * 4) / 4, Math.ceil((cl + 0.5) * 4) / 4);
        if (ww.west < box.west || ww.east > box.east || ww.south < box.south || ww.north > box.north) continue;
        const sub = new Uint8Array(ww.cols * ww.rows);
        const dc = ww.col0 - w1.col0, dr = ww.row0 - w1.row0;
        for (let r = 0; r < ww.rows; r++) for (let c = 0; c < ww.cols; c++) sub[r * ww.cols + c] = g[(r + dr) * w1.cols + c + dc];
        const one = { dkm, bear };
        for (const k of ['cur', 'fix']) {
            const L = H.level(ww, sub, null);
            const seeds = { idx: [], w: [], t: [], hz: [], pred: [] }, o = [0, 0, 0];
            const add = (r, c) => {
                const i = r * ww.cols + c; if (!sub[i]) return;
                const got = H.sampleNode(C[k], ww.west + (c + 0.5) * ww.cell, ww.north - (r + 0.5) * ww.cell, false, o);
                if (got) { seeds.idx.push(i); seeds.w.push(o[0]); seeds.t.push(o[1]); seeds.hz.push(o[2]); seeds.pred.push(P_SEED); }
            };
            for (let c = 0; c < ww.cols; c++) { add(0, c); add(ww.rows - 1, c); }
            for (let r = 1; r < ww.rows - 1; r++) { add(r, 0); add(r, ww.cols - 1); }
            const rr = await H.solve(L, seeds, () => true, false);
            const inside = [], step = [];
            for (let r = 0; r < ww.rows; r++) for (let c = 0; c < ww.cols; c++) {
                const i = r * ww.cols + c, t = rr.t[i];
                if (!(t < Infinity)) continue;
                const tw = whole.dist[(r + dr) * w1.cols + c + dc];
                if (tw < Infinity && tw > 0) inside.push(t / tw);
                // a ring a quarter of the way in
                const qr = Math.round(ww.rows / 4), qc = Math.round(ww.cols / 4);
                const onRing = ((r === qr || r === ww.rows - 1 - qr) && c >= qc && c < ww.cols - qc) || ((c === qc || c === ww.cols - 1 - qc) && r >= qr && r < ww.rows - qr);
                if (onRing && H.sampleNode(C[k], ww.west + (c + 0.5) * ww.cell, ww.north - (r + 0.5) * ww.cell, false, o)) step.push(t / o[1]);
            }
            one[k] = { inside: [H.quant(inside, 0.1), H.quant(inside, 0.5), H.quant(inside, 0.9)], step: [H.quant(step, 0.1), H.quant(step, 0.5), H.quant(step, 0.9)] };
        }
        res.wins.push(one);
    }
    out.push(res);
    console.error(name, JSON.stringify(res.wins.map(w => [w.cur.inside[1].toFixed(2), w.fix.inside[1].toFixed(2), w.cur.step[1].toFixed(2), w.fix.step[1].toFixed(2)])));
}
writeFileSync(H.SCR + 'results/seam.json', JSON.stringify(out));
process.exit(0);
