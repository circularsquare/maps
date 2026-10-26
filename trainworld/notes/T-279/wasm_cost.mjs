// T-279: what the routed drives cost a demand worker: decoding the table (time, WASM memory) and
// composing a market of a few zones with and without it.
//   TW_MARKET_DRIVES=<file> node trainworld/notes/T-279/wasm_cost.mjs [zone ids]
import { readFileSync } from 'node:fs';
import { loadApp, marketDrives, marketPacks, marketRows } from '../../checks/lib.mjs';
import init, { MarketPacks } from '../../app/src/wasm/demand/sim.js';

const wasm = await init({ module_or_path: readFileSync(new URL('../../app/src/wasm/demand/sim_bg.wasm', import.meta.url)) });
globalThis.location ??= { search: '', hash: '', href: 'http://localhost/' };
const app = await loadApp("export { marketWorld } from '@app/src/game/cities.ts';");
const ids = process.argv.slice(2).length ? process.argv.slice(2) : ['nyc', 'bos', 'phl'];
const world = app.marketWorld(marketRows());
const drives = marketDrives();
const mb = () => (wasm.memory.buffer.byteLength / 2 ** 20).toFixed(1);
const compose = (withDrives) => {
  const p = new MarketPacks(world);
  let t = performance.now(), n = 0;
  const m0 = mb();
  if (withDrives && drives) n = p.set_drives(drives);
  const decode = performance.now() - t;
  const m1 = mb();
  for (const m of marketPacks(ids)) p.add(m.pack);
  t = performance.now();
  const api = p.market();
  const ms = performance.now() - t;
  api.free();
  p.free();
  return { n, decode: decode.toFixed(1), compose: ms.toFixed(0), memBefore: m0, memAfter: m1 };
};
compose(false); // warm up
console.log('no table', JSON.stringify(compose(false)));
console.log('table', JSON.stringify(compose(true)), `${drives ? (drives.byteLength / 1e6).toFixed(2) : 0} MB file`);
console.log('no table again', JSON.stringify(compose(false)));
console.log('table again', JSON.stringify(compose(true)));
