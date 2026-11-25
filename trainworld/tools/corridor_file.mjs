// T-4199: the market's corridor file, app/src/game/market-corridors.bin: the corridors and the corridor book over
// the world's rows and routed drives, worked out once in the demand WASM (so the same bits as a load's), which a
// load reads instead of working them out in every demand worker (sim/src/demand/intercity/corridor_file.rs). The
// file is keyed by the world, the drives and the sim's sources, so a stale one is never used, only slower. Made
// after each demand WASM build (app/package.json, tools/snap_wasm.py), for a freeze's data by `freeze.py build`, and
// by a world run's join (`world_run.py step @corridors`); by hand after the market rows or the drives change.
// About 10 s on one core.
//   node trainworld/tools/corridor_file.mjs [--out <file>] [--strict]
// The rows are app/src/game/market-rows.json, the drives data/packs/market-drives.bin (or TW_MARKET_DRIVES). Without
// --strict a failure is a warning and exit 0, so a build goes on (its loads work the corridors out as before).
import { readFileSync, renameSync, writeFileSync, existsSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice(2);
const strict = args.includes('--strict');
const outAt = args.indexOf('--out');
const OUT = outAt >= 0 ? args[outAt + 1] : fileURLToPath(new URL('../app/src/game/market-corridors.bin', import.meta.url));

try {
  const t0 = performance.now();
  const { loadApp, marketRows, marketDrives } = await import('../checks/lib.mjs');
  const wasm = new URL('../app/src/wasm/demand/sim_bg.wasm', import.meta.url);
  if (!existsSync(wasm)) throw new Error('no demand WASM (npm run wasm:demand --prefix trainworld/app)');
  const { default: init, MarketPacks } = await import('../app/src/wasm/demand/sim.js');
  await init({ module_or_path: readFileSync(wasm) });
  // game/cities.ts reads the view from the URL as it loads
  globalThis.location ??= { search: '', hash: '', href: 'http://localhost/' };
  const { marketWorld, REGISTRY, fareBitsStale } = await loadApp("export { marketWorld, REGISTRY } from '@app/src/game/cities.ts'; export { fareBitsStale } from '@app/src/game/income.ts';");
  // the fare figures are fareBits.json's, the same in every engine, unless it is stale: then they are Node's, and
  // a browser whose last bits differ refuses the file
  const stale = fareBitsStale(REGISTRY.market.rule.eps);
  if (stale.length) console.warn(`corridor file: fareBits.json is stale for ${stale.length} countries (${stale.slice(0, 8).join(' ')}), so the file may be refused by browsers: node trainworld/tools/fare_bits.mjs makes it again`);
  // the world exactly as the demand workers give it (the attractions, which the key leaves out, aside)
  const packs = new MarketPacks(marketWorld(marketRows()));
  let drives = marketDrives();
  if (drives && drives[0] === 0x1f && drives[1] === 0x8b) drives = new Uint8Array(gunzipSync(drives));
  // as the worker: drives that do not read leave every drive the rule's
  if (drives) try { packs.set_drives(drives); } catch (err) { console.warn('corridor file: the drives would not read, so the rule\'s:', String(err)); }
  const bytes = packs.corridor_file();
  writeFileSync(OUT + '.part', bytes);
  renameSync(OUT + '.part', OUT);
  console.log(`corridor file: ${OUT}, ${(bytes.length / 1024).toFixed(0)} KB, ${((performance.now() - t0) / 1000).toFixed(1)} s`);
} catch (err) {
  console.warn(`corridor file not made (loads work the corridors out themselves): ${err?.stack ?? err}`);
  process.exit(strict ? 1 : 0);
}
