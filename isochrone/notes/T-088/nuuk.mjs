// Inner holes at L10 from Nuuk, listed with their nodes
import * as H from './lib.mjs';
const { eachCanoeNode } = await H.imp('canoe.js');
H.cfg.hazard = true;
const L = await H.world10(true);
await H.solveFrom(L, 64.18, -51.72, true);
const { W, H: Hh, grid, cn } = L, t = L.res.t, N = W * Hh;
const reached = (i) => { let a = grid[i] && t[i] <= 1024; eachCanoeNode(cn, i, (m) => { if (t[m] <= 1024) a = true; }); return a; };
for (let r = 1; r < Hh - 1; r++) for (let c = 0; c < W; c++) {
    const i = r * W + c;
    let n = grid[i] ? 1 : 0; eachCanoeNode(cn, i, () => { n++; });
    if (!n || reached(i)) continue;
    let k = 0, deep = true;
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        if (!dr && !dc) continue;
        const j = (r + dr) * W + (c + dc + W) % W;
        if (!reached(j)) continue;
        k++;
        let tb = grid[j] ? t[j] : Infinity, hb = grid[j] ? L.res.hz[j] : Infinity;
        eachCanoeNode(cn, j, (m) => { if (t[m] < tb) { tb = t[m]; hb = L.res.hz[m]; } });
        if (tb > 800 || hb > Math.log(100) - 0.7) deep = false;
    }
    if (k >= 6 && deep) {
        const nodes = []; eachCanoeNode(cn, i, (m) => nodes.push(m));
        console.log(`${(85 - (r + 0.5) / 12).toFixed(3)},${(-180 + (c + 0.5) / 12).toFixed(3)} grid ${grid[i]} kind ${cn.info[i] & 0xFF} info ${(cn.info[i] >>> 16).toString(2)} links ${((cn.info[i] >> 8) & 255).toString(2)} canoeOf ${cn.canoeOf[i]} extraN ${cn.extraN[i]} nodes ${nodes.join(',')} edges ${nodes.map(m => cn.rOff[m + 1] - cn.rOff[m]).join(',')} shore ${cn.shore[i]}`);
    }
}
process.exit(0);
