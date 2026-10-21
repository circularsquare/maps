// T-854: a pack's .bin with its `nocar` array replaced. Modes:
//   zero  nobody without a car
//   flat  every cell at the pack's own people-weighted share
//   const:<s>  every cell at share s
//   dens  one curve on the 2 km zone's residential density: 0.75 / (1 + (11500 / d)^1.9)
//         (fitted on the US and English packs' census figures, fit.py)
// Also a CLI: node nocarpack.mjs <packs dir> <id> <mode> <out dir>  (writes <out>/<id>.bin + .json)
import { readFileSync, writeFileSync, copyFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const CELL_KM2 = 0.1053;
export const curve = (d) => 0.75 / (1 + Math.pow(11500 / Math.max(d, 1), 1.9));

export function modify(dir, id, mode) {
  const h = JSON.parse(readFileSync(join(dir, `${id}.json`), 'utf8'));
  const src = readFileSync(join(dir, `${id}.bin`));
  const buf = Buffer.from(src); // copy
  const A = Object.fromEntries(h.arrays.map((a) => [a.name, a]));
  if (!A.nocar || !A.pop) return { buf, note: 'no nocar' };
  const n = A.pop.count;
  const pop = (i) => buf.readFloatLE(A.pop.offset + 4 * i);
  let share;
  if (mode === 'zero') share = () => 0;
  else if (mode.startsWith('const:')) { const s = Number(mode.slice(6)); share = () => s; }
  else if (mode === 'flat') {
    let p = 0, c = 0;
    for (let i = 0; i < n; i++) { p += pop(i); c += buf.readFloatLE(A.nocar.offset + 4 * i); }
    const s = p > 0 ? c / p : 0;
    share = () => s;
  } else if (mode === 'dens') {
    const zone = (i) => buf.readUInt32LE(A.zone.offset + 4 * i);
    const zp = new Map(), zc = new Map();
    for (let i = 0; i < n; i++) { const z = zone(i); zp.set(z, (zp.get(z) ?? 0) + pop(i)); zc.set(z, (zc.get(z) ?? 0) + 1); }
    share = (i) => { const z = zone(i); return curve(zp.get(z) / (zc.get(z) * CELL_KM2)); };
  } else throw new Error('mode ' + mode);
  let p = 0, c = 0;
  for (let i = 0; i < n; i++) {
    const v = pop(i) * share(i);
    buf.writeFloatLE(v, A.nocar.offset + 4 * i);
    p += pop(i); c += v;
  }
  return { buf, note: `${id} ${mode}: car-free ${(100 * c / Math.max(p, 1)).toFixed(1)}%` };
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [dir, id, mode, out] = process.argv.slice(2);
  mkdirSync(out, { recursive: true });
  const { buf, note } = modify(dir, id, mode);
  writeFileSync(join(out, `${id}.bin`), buf);
  copyFileSync(join(dir, `${id}.json`), join(out, `${id}.json`));
  console.log(note);
}
