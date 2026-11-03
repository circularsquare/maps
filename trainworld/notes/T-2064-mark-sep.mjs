// T-2064: a game save with every edge of its network marked grade-separated (`EdgeData.sep`, bit 7
// of an edge's flags, T-1751), everything else byte for byte as it was.
//   node trainworld/notes/T-2064-mark-sep.mjs <in.save> <out.save>
// For saves made from a real network before T-1751 (2026-10-28). Edges the player drew are marked
// too, so run it only on a save whose track is all imported.
import { readFileSync, writeFileSync } from 'node:fs';
import { gunzipSync, gzipSync } from 'node:zlib';

const [inPath, outPath] = process.argv.slice(2);
if (!inPath || !outPath) throw new Error('usage: T-2064-mark-sep.mjs <in.save> <out.save>');
const raw = gunzipSync(readFileSync(inPath));
if (raw.subarray(0, 4).toString('latin1') !== 'TWG1') throw new Error('not a game save');
const hn = raw.readUInt32LE(4);
const track = raw.subarray(8 + hn);
if (!['TWT4', 'TWT5', 'TWT6'].includes(track.subarray(0, 4).toString('latin1'))) throw new Error('not a TWT4-6 network');

let o = 20;
const u = () => { let v = 0n, sh = 0n; for (;;) { const b = track[o++]; if (b === undefined) throw new Error('truncated'); v |= BigInt(b & 0x7f) << sh; if (!(b & 0x80)) return Number(v); sh += 7n; } };
const skip = (n) => { for (let k = 0; k < n; k++) u(); };
const enc = (v) => { const out = []; for (;;) { const b = v & 0x7f; v = Math.floor(v / 128); if (v === 0) { out.push(b); return Buffer.from(out); } out.push(b | 0x80); } };

u(); // right hand
const nNodes = u();
for (let k = 0; k < nNodes; k++) {
  skip(3);
  if (u() & 2) { u(); const len = u(); o += len; }
}
const nEdges = u();
const parts = [];
let from = 0, marked = 0;
for (let k = 0; k < nEdges; k++) {
  skip(2);
  const at = o;
  const tb = u();
  if (!(tb & 128)) {
    parts.push(track.subarray(from, at), enc(tb | 128));
    from = o;
    marked++;
  }
  const n = u();
  skip(4 * n);
  if (tb & 8) skip(3 * u());
}
parts.push(track.subarray(from));
const out = Buffer.concat([raw.subarray(0, 8 + hn), ...parts]);
writeFileSync(outPath, gzipSync(out, { level: 9 }));
console.log(`${nEdges} edges, ${marked} marked separated, ${nEdges - marked} already were; wrote ${outPath}`);
