// T-2621: a capture of T-2606's form (plain arrays at the top) as t756-reach.mjs's <label>.net.json, so
// `node trainworld/checks/t756-reach.mjs compare <label> reach` gives every population's boardings by line.
//   node trainworld/notes/T-2621/to756.mjs <capture.json> <label>
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const [src, label] = process.argv.slice(2);
const c = JSON.parse(readFileSync(src, 'utf8'));
const types = { stXY: Float32Array, complex: Uint32Array, lineN: Uint32Array, stops: Uint32Array, times: Float32Array, tph: Float32Array, cars: Float32Array, hopKm: Float32Array };
const enc = (v) => ({ t: v.constructor.name, b: Buffer.from(v.buffer, v.byteOffset, v.byteLength).toString('base64') });
const net = { frame: c.frame, fare: c.fare };
for (const [k, T] of Object.entries(types)) net[k] = enc(T.from(c[k]));
const out = fileURLToPath(new URL(`../../../artifacts/trainworld/t756/${label}.net.json`, import.meta.url));
writeFileSync(out, JSON.stringify({ save: c.save, packs: c.packs, markets: c.markets, net }));
console.log(`${label}: ${c.stXY.length / 2} stations, ${c.lineN.length} lines -> ${out}`);
