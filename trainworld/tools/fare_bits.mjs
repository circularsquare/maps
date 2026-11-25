// T-4199: app/src/game/fareBits.json, the market's fare figures (each country's minutes a dollar and its income
// groups, game/income.ts) as headless Chrome works them out. JS `**` gives other last bits in other engines, and
// the market's corridor file (tools/corridor_file.mjs) is keyed by these figures, so every engine takes these bits
// (income.ts `fareWeight`, `incomeGroups`). Make it again when countryIncome.json, the fare constants or the market
// rule's eps change; corridor_file.mjs warns when it is stale. A few seconds.
//   node trainworld/tools/fare_bits.mjs
import { readFileSync, renameSync, writeFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const app = fileURLToPath(new URL('../app', import.meta.url)).replace(/\\/g, '/');
const OUT = `${app}/src/game/fareBits.json`;
const { launchChrome } = await import('../checks/lib.mjs');
const { build } = await import(pathToFileURL(`${app}/node_modules/vite/dist/node/index.js`).href);
const eps = JSON.parse(readFileSync(`${app}/src/game/registry.json`, 'utf8')).market.rule.eps;
const entry = `export { fareBitsOwn } from '${app}/src/game/income.ts';`;
const bundle = await build({
  configFile: false, logLevel: 'silent',
  plugins: [{ name: 'entry', resolveId: (id) => (id.endsWith('fare-bits-entry') ? '\0fare-bits-entry' : undefined), load: (id) => (id === '\0fare-bits-entry' ? entry : undefined) }],
  build: { write: false, minify: false, lib: { entry: 'fare-bits-entry', formats: ['es'] } },
});
const code = (Array.isArray(bundle) ? bundle[0] : bundle).output.find((o) => o.type === 'chunk').code;
const { ctx, close } = await launchChrome({ prefix: 'tw-farebits-' });
let got;
try {
  const page = await ctx.newPage();
  got = await page.evaluate(async ([code, eps]) => {
    const url = URL.createObjectURL(new Blob([code], { type: 'text/javascript' }));
    const m = await import(url);
    return { by: navigator.userAgent.match(/Chrome\/[\d.]+/)?.[0] ?? navigator.userAgent, ...m.fareBitsOwn(eps) };
  }, [code, eps]);
} finally { await close(); }
// one country a line
const block = (o) => '{\n' + Object.entries(o).map(([k, v]) => `  ${JSON.stringify(k)}: ${JSON.stringify(v)}`).join(',\n') + '\n }';
writeFileSync(OUT + '.part', `{\n "by": ${JSON.stringify(got.by)},\n "eps": ${got.eps},\n "fw": ${block(got.fw)},\n "fg": ${block(got.fg)}\n}\n`);
renameSync(OUT + '.part', OUT);
console.log(`fare bits: ${OUT}, ${Object.keys(got.fw).length} countries' minutes a dollar, ${Object.keys(got.fg).length} countries' income groups, ${got.by}`);
