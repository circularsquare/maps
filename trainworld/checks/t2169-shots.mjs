// T-2169: pictures of t2169-water.mjs' render files (run it with --png first), painted by the diagram's own draw.ts in a
// static page, one per view and theme, into artifacts/trainworld/t2169/.
//   node trainworld/checks/t2169-shots.mjs <views.json>     views: [{ city, name, at: "Jamaica" | [x, y], px, w?, h?, dark? }]
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { ARTIFACTS, HERE, launchChrome } from './lib.mjs';

const views = JSON.parse(readFileSync(process.argv[2]));
const OUT = `${ARTIFACTS}t2169/`;
const { build } = await import(pathToFileURL(join(HERE, '../app/node_modules/vite/dist/node/index.js')).href);
const app = fileURLToPath(new URL('../app', import.meta.url)).replace(/\\/g, '/');
const entry = `export { paint } from '${app}/src/diagram/draw.ts'; export { SHADES } from '${app}/src/game/shadeTable.ts'; export { paintNames } from '${app}/src/diagram/names.ts';`;
const bundle = await build({
  configFile: false, logLevel: 'silent',
  plugins: [{ name: 'e', resolveId(id) { if (id.endsWith('shots-entry')) return '\0shots-entry'; }, load(id) { if (id === '\0shots-entry') return entry; } }],
  build: { write: false, minify: false, lib: { entry: 'shots-entry', formats: ['iife'], name: 'TW' } },
});
writeFileSync(`${OUT}paint.js`, (Array.isArray(bundle) ? bundle[0] : bundle).output.find((o) => o.type === 'chunk').code);
const cities = [...new Set(views.map((v) => v.city))];
writeFileSync(`${OUT}shots.html`, `<!doctype html><meta charset="utf-8"><body style="margin:0"><canvas id="c"></canvas>
<script src="paint.js"></script>${cities.map((c) => `<script src="render-${c}.js"></script>`).join('')}
<script>
window.render = (city, at, px, w, h, dark, dbg) => {
  const D = window.TW_DATA[city], c = document.getElementById('c');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  const d = { ...D, lines: D.lines.map((l) => ({ ...l, cmds: Float64Array.from(l.cmds), path: undefined })), small: Uint8Array.from(D.small),
    water: { rivers: Float64Array.from(D.water.rivers), sea: Float64Array.from(D.water.sea), band: D.water.band } };
  let cx, cy;
  if (typeof at === 'string') { const i = D.text.indexOf(at); if (i < 0) return 'no ' + at; cx = D.x[i]; cy = D.y[i]; } else [cx, cy] = at;
  const scale = px / d.pitch;
  const theme = dark
    ? { bg: '#22252b', ink: '#2b2b2b', pill: '#ffffff', selected: '#d9788e', border: '#73798b', edge: '#d8d8d8', water: TW.SHADES.dark.water }
    : { bg: '#fbf5ef', ink: '#2b2b2b', pill: '#ffffff', selected: '#d9788e', border: '#73798b', water: TW.SHADES.light.water };
  TW.paint(ctx, d, w, h, cx, cy, scale, theme);
  if (dbg) { ctx.save(); ctx.setTransform(scale, 0, 0, -scale, w / 2 - cx * scale, h / 2 + cy * scale); ctx.lineWidth = 1.5 / scale; for (const r of D.runs) { ctx.strokeStyle = '#000'; ctx.beginPath(); for (let k = 0; k < r.p.length; k += 2) k ? ctx.lineTo(r.p[k], r.p[k + 1]) : ctx.moveTo(r.p[k], r.p[k + 1]); ctx.stroke(); ctx.strokeStyle = '#f00'; ctx.lineWidth = 3 / scale; for (let k = 0; k < r.marks.length; k += 4) { ctx.beginPath(); ctx.moveTo(r.marks[k], r.marks[k + 1]); ctx.lineTo(r.marks[k + 2], r.marks[k + 3]); ctx.stroke(); ctx.fillStyle = '#f00'; ctx.fillRect(r.marks[k + 2] - 3 / scale, r.marks[k + 3] - 3 / scale, 6 / scale, 6 / scale); } ctx.lineWidth = 1.5 / scale; } for (let k = 0; k < D.failed.length; k += 4) { ctx.fillStyle = '#00f'; ctx.fillRect(D.failed[k] - 4 / scale, D.failed[k + 1] - 4 / scale, 8 / scale, 8 / scale); ctx.fillStyle = '#0a0'; ctx.fillRect(D.failed[k + 2] - 4 / scale, D.failed[k + 3] - 4 / scale, 8 / scale, 8 / scale); ctx.strokeStyle = '#00f'; ctx.beginPath(); ctx.moveTo(D.failed[k], D.failed[k + 1]); ctx.lineTo(D.failed[k + 2], D.failed[k + 3]); ctx.stroke(); } ctx.restore(); }
  TW.paintNames(ctx, D.plan, d, w, h, cx, cy, scale, 1, { ink: dark ? '#e8e2da' : '#2b2b2b', halo: theme.bg, font: 'system-ui, sans-serif' });
  return 'ok';
};
</script>`);
const { ctx, close } = await launchChrome({ prefix: 'tw-diagwater-', size: [1400, 900], args: [] });
const page = ctx.pages()[0] ?? (await ctx.newPage());
await page.goto(pathToFileURL(`${OUT}shots.html`).href);
for (const v of views) {
  const w = v.w ?? 1200, h = v.h ?? 800;
  const r = await page.evaluate(([c, at, px, w, h, dark, dbg]) => window.render(c, at, px, w, h, dark, dbg), [v.city, v.at, v.px, w, h, !!v.dark, !!v.debug]);
  if (r !== 'ok') { console.log(v.city, v.name, r); continue; }
  const file = `${OUT}${v.name}.png`;
  await page.locator('#c').screenshot({ path: file });
  console.log(file);
}
await close();
