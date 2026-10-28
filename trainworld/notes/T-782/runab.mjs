// T-782 scratch: cold loads, each on a fresh Vite in this process (from T-1594's runab).
//   node runab.mjs <port> <save> <tag> <variant,variant,...> [min]
// variant "cur" runs the tree's own app/src/wasm/demand; any other copies scratch/wasm-<variant> over it first
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { cpSync, rmSync, mkdirSync } from 'node:fs';
import { spawn } from 'node:child_process';

const TREE = 'C:/Users/anita/projects/maps/.tw-trees/t782/trainworld';
const SCR = process.env.T782_SCRATCH ?? 'C:/Users/anita/AppData/Local/Temp/claude/c--Users-anita-projects-maps/7735d270-22c0-47bd-b8fa-dca6f93e8773/scratchpad/t782';
const OUT = 'C:/Users/anita/projects/maps/artifacts/trainworld/t782';
const [port, save, tag, variants, min = '60'] = process.argv.slice(2);
mkdirSync(OUT, { recursive: true });
const root = TREE + '/app';
const require = createRequire(root + '/package.json');
const { createServer } = await import(pathToFileURL(require.resolve('vite')).href);
const seen = {};
for (const v of variants.split(',')) {
  const n = (seen[v] = (seen[v] ?? 0) + 1);
  if (v !== 'cur') {
    rmSync(root + '/src/wasm/demand', { recursive: true, force: true });
    cpSync(`${SCR}/wasm-${v}`, root + '/src/wasm/demand', { recursive: true });
  }
  const server = await createServer({ root, configFile: root + '/vite.config.ts', cacheDir: SCR + '/vitecache', logLevel: 'warn', server: { port: Number(port), strictPort: true, watch: null, hmr: false } });
  await server.listen();
  const out = `${OUT}/${tag}-${v}-${n}.json`;
  const t = Date.now();
  console.log(`${new Date().toISOString()} ${v} #${n} -> ${out}`);
  const status = await new Promise((res) => spawn(process.execPath, [TREE + '/checks/t782-measure.mjs', `http://localhost:${port}/`, save, out, min], { stdio: 'inherit', env: { ...process.env } }).on('exit', res));
  console.log(`${new Date().toISOString()} ${v} #${n} exit ${status} in ${((Date.now() - t) / 1000).toFixed(0)} s`);
  await server.close();
}
console.log('ALL DONE');
process.exit(0);
