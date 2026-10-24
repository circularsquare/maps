// T-761: a `t756-reach.mjs capture`'s network as plain JSON arrays, for the native market bench
// (`t761_market_bench` in sim/src/demand/intercity/tests.rs):
//   node trainworld/checks/t761-plain-net.mjs <label> <out.json>
import { readFileSync, writeFileSync } from 'node:fs';
const [label, out] = process.argv.slice(2);
const cap = JSON.parse(readFileSync(new URL(`../../artifacts/trainworld/t756/${label}.net.json`, import.meta.url), 'utf8'));
const dec = (o) => { const b = Buffer.from(o.b, 'base64'); const ab = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); return Array.from(new globalThis[o.t](ab)); };
const net = { frame: cap.net.frame, fare: cap.net.fare, markets: cap.markets };
for (const k of ['stXY', 'complex', 'lineN', 'stops', 'times', 'tph', 'cars', 'hopKm']) net[k] = dec(cap.net[k]);
writeFileSync(out, JSON.stringify(net));
console.log(net.stXY.length / 2, 'stations', net.lineN.length, 'lines', net.fare);
