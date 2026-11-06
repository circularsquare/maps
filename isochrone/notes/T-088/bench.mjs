// buildCanoe time at L10, 5 runs: CANOE=... JSDIR=... node bench.mjs
import * as H from './lib.mjs';
const { loadWorld } = await H.imp('canoedata.js');
const { buildCanoe } = await H.imp('canoe.js');
const W = await H.world10(false);
const d = await loadWorld(W.W, W.H);
const ts = [];
for (let i = 0; i < 5; i++) { const t = performance.now(); buildCanoe(W, d, d.sea); ts.push(Math.round(performance.now() - t)); }
let pond = 0;
console.log('build ms', ts.join(' '), 'E', W.cn.E, 'nodes beyond cells; river/water edges', W.cn.rTo.length);
process.exit(0);
