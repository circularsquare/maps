// T-1594: cold loads alternating demand WASM builds, back to back, each on a fresh Vite.
//   node runab.mjs <port> <save> <tag> <variant,variant,...>
// each variant is a folder scratch/wasm-<variant> copied over app/src/wasm/demand before its run
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { cpSync, rmSync, mkdirSync } from 'node:fs';
import { spawn } from 'node:child_process';

const TREE = 'C:/Users/anita/projects/maps/.tw-trees/t1594/trainworld';
const SCR = 'C:/Users/anita/AppData/Local/Temp/claude/c--Users-anita-projects-maps/7735d270-22c0-47bd-b8fa-dca6f93e8773/scratchpad/t1594';
const OUT = 'C:/Users/anita/projects/maps/artifacts/trainworld/t1594';
const [port, save, tag, variants, min = '60'] = process.argv.slice(2);
mkdirSync(OUT, { recursive: true });
const root = TREE + '/app';
const require = createRequire(root + '/package.json');
const { createServer } = await import(pathToFileURL(require.resolve('vite')).href);
const seen = {};
for (const v of variants.split(',')) {
  const n = (seen[v] = (seen[v] ?? 0) + 1);
  rmSync(root + '/src/wasm/demand', { recursive: true, force: true });
  cpSync(`${SCR}/wasm-${v}`, root + '/src/wasm/demand', { recursive: true });
  const server = await createServer({ root, configFile: root + '/vite.config.ts', cacheDir: SCR + '/vitecache', logLevel: 'warn', server: { port: Number(port), strictPort: true, watch: null, hmr: false } });
  await server.listen();
  const out = `${OUT}/${tag}-${v}-${n}.json`;
  const t = Date.now();
  console.log(`${new Date().toISOString()} ${v} #${n} -> ${out}`);
  // (async: the Vite server is in this process and must keep serving)
  const status = await new Promise((res) => spawn(process.execPath, [TREE + '/checks/t1594-measure.mjs', `http://localhost:${port}/`, save, out, min], { stdio: 'inherit' }).on('exit', res));
  console.log(`${new Date().toISOString()} ${v} #${n} exit ${status} in ${((Date.now() - t) / 1000).toFixed(0)} s`);
  await server.close();
}
console.log('ALL DONE');
process.exit(0);
