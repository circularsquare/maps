// T-3192: every shader program's varyings fit in 8 vectors, packed as the GLSL ES 1.0.17 appendix A.7
// algorithm does it (ANGLE's VariablePacker, which Firefox runs at link time). 8 because Firefox with
// fingerprinting resistance (privacy.resistFingerprinting: LibreWolf, Mullvad, Tor; also
// webgl.min_capability_mode) reports MAX_VARYING_VECTORS 8 for WebGL2 too, and refuses a program that
// needs more: a player's game stopped at startup there. Prints one row per program; exits 1 if any is over.
//   node trainworld/checks/t3192-varyings.mjs [--budget N]
// The programs are the app's `compile(gl, VS, FS)` calls; their sources are the real strings, the
// shader modules bundled by Vite with their top-level shader consts exported.
import { readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';
import { pathToFileURL } from 'node:url';
import { HERE } from './lib.mjs';

const BUDGET = Number(process.argv.includes('--budget') ? process.argv[process.argv.indexOf('--budget') + 1] : 8);
const APP = join(HERE, '../app');
const SRC = join(APP, 'src');

function walk(dir, out = []) {
  for (const f of readdirSync(dir)) {
    const p = join(dir, f);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (/\.tsx?$/.test(f)) out.push(p);
  }
  return out;
}

// ---- the programs: compile(gl, A, B) calls, with where A and B are defined
const files = walk(SRC).map((p) => ({ p, text: readFileSync(p, 'utf8') }));
const shaderConsts = new Map(); // file -> names of top-level shader consts
for (const f of files) {
  const names = [...f.text.matchAll(/^(?:export )?const (\w+) = `#version/gm)].map((m) => m[1]);
  if (names.length) shaderConsts.set(f.p, names);
}
const programs = [];
for (const f of files) {
  if (/[\\/]map[\\/]gl\.ts$/.test(f.p)) continue; // compileOptional's own call
  for (const m of f.text.matchAll(/\bcompile(?:Optional)?\(gl, (\w+), (\w+)/g)) {
    const where = (name) => {
      if (shaderConsts.get(f.p)?.includes(name)) return f.p;
      const imp = [...f.text.matchAll(/import \{([^}]*)\} from "(\.[^"]+)"/g)].find((i) => i[1].split(',').map((s) => s.trim()).includes(name));
      if (!imp) throw new Error(`${relative(SRC, f.p)}: where is ${name}?`);
      return join(f.p, '..', imp[2] + '.ts');
    };
    programs.push({ file: relative(SRC, f.p).replace(/\\/g, '/'), vs: [where(m[1]), m[1]], fs: [where(m[2]), m[2]] });
  }
}
if (programs.length < 10) throw new Error(`only ${programs.length} programs found`);

// ---- their sources, bundled: each shader file also exports its shader consts as __glsl
const modules = [...shaderConsts.keys()];
const fwd = (p) => p.replace(/\\/g, '/');
const entry = modules.map((p, i) => `export { __glsl as m${i} } from ${JSON.stringify(fwd(p))};`).join('\n');
const { build } = await import(pathToFileURL(join(APP, 'node_modules/vite/dist/node/index.js')).href);
const bundle = await build({
  configFile: false, logLevel: 'silent', root: APP,
  plugins: [{
    name: 't3192',
    resolveId(id) { if (id.endsWith('glsl-entry')) return '\0glsl-entry'; },
    load(id) { if (id === '\0glsl-entry') return entry; },
    transform(code, id) {
      const names = shaderConsts.get(join(id.split('?')[0]));
      if (names) return code + `\nexport const __glsl = { ${names.join(', ')} };\n`;
    },
  }],
  // (no module's side effects: only what the shader strings are made of is kept, and nothing that wants a browser)
  build: { write: false, minify: false, lib: { entry: 'glsl-entry', formats: ['es'] }, rollupOptions: { treeshake: { moduleSideEffects: false } } },
});
const code = (Array.isArray(bundle) ? bundle[0] : bundle).output.find((o) => o.type === 'chunk').code;
// what the kept code's top level reads (the page's query flags)
const stub = { getItem: () => null, setItem() {}, removeItem() {} };
for (const [k, v] of Object.entries({ location: new URL('http://localhost/'), localStorage: stub, sessionStorage: stub })) if (!(k in globalThis)) globalThis[k] = v;
const tmp = join(tmpdir(), `tw-t3192-${process.pid}.mjs`);
writeFileSync(tmp, code);
let mod;
try { mod = await import(pathToFileURL(tmp).href); } finally { rmSync(tmp, { force: true }); }
const source = ([p, name]) => {
  const s = mod['m' + modules.indexOf(p)]?.[name];
  if (typeof s !== 'string') throw new Error(`no source for ${name} in ${relative(SRC, p)}`);
  return s;
};

// ---- varyings
const COMPONENTS = { float: [1, 1], int: [1, 1], uint: [1, 1], bool: [1, 1], vec2: [2, 1], ivec2: [2, 1], uvec2: [2, 1], bvec2: [2, 1], vec3: [3, 1], ivec3: [3, 1], uvec3: [3, 1], bvec3: [3, 1], vec4: [4, 1], ivec4: [4, 1], uvec4: [4, 1], bvec4: [4, 1], mat2: [2, 2], mat3: [3, 3], mat4: [4, 4], mat2x4: [4, 2], mat3x4: [4, 3], mat4x2: [2, 4], mat4x3: [3, 4], mat2x3: [3, 2], mat3x2: [2, 3] };
const stripComments = (s) => s.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/\/\/[^\n]*/g, ' ');
/** top-level `in`/`out` declarations of one storage: [{ type, name, size, decl }] */
function declared(src, storage) {
  const out = [];
  const re = new RegExp(`^\\s*(?:(?:flat|smooth|centroid)\\s+)*${storage}\\s+(?:(?:highp|mediump|lowp)\\s+)?(\\w+)\\s+([^;]+);`, 'gm');
  for (const m of src.matchAll(re)) {
    for (const part of m[2].split(',')) {
      const v = part.trim().match(/^(\w+)\s*(?:\[\s*(\d+)\s*\])?$/);
      if (v) out.push({ type: m[1], name: v[1], size: Number(v[2] ?? 1), decl: m[0] });
    }
  }
  return out;
}
const used = (src, decls, name) => {
  let body = src;
  for (const d of decls) body = body.replace(d.decl, ' ');
  return new RegExp(`\\b${name}\\b`).test(body);
};
/** the varyings Firefox's link check counts (WebGLShaderValidator CanLinkTo): the fragment shader's
 * statically used inputs that the vertex shader also uses, and its statically used gl_ built-ins */
function varyings(vsSrc, fsSrc) {
  const vs = stripComments(vsSrc), fs = stripComments(fsSrc);
  const vOut = declared(vs, 'out'), fIn = declared(fs, 'in');
  const list = [];
  for (const f of fIn) {
    const v = vOut.find((o) => o.name === f.name);
    if (!v) { if (used(fs, fIn, f.name)) throw new Error(`fragment input ${f.name} has no vertex output`); continue; }
    if (used(fs, fIn, f.name) && used(vs, vOut, v.name)) list.push(f);
  }
  for (const [name, type] of [['gl_FragCoord', 'vec4'], ['gl_PointCoord', 'vec2'], ['gl_FrontFacing', 'bool']]) {
    if (new RegExp(`\\b${name}\\b`).test(fs)) list.push({ type, name, size: 1 });
  }
  return list;
}

/** Whether `vars` pack into `maxRows` vectors (ANGLE VariablePacker::checkExpandedVariablesWithinPackingLimits). */
function fits(vars, maxRows) {
  const items = vars.map((v) => {
    const [cols, rows] = COMPONENTS[v.type] ?? (() => { throw new Error('type ' + v.type); })();
    return { cols, rows: rows * v.size, perElemRows: rows, size: v.size };
  });
  for (const it of items) if (it.size > Math.floor(maxRows / it.perElemRows)) return false;
  // by components per row, widest first, then larger first (the spec's type order for these types)
  items.sort((a, b) => b.cols - a.cols || b.rows - a.rows);
  const rowsUsed = new Array(maxRows).fill(0);
  let top = 0, bottom = maxRows - 1;
  const fill = (row, n, col, ncols) => {
    const mask = ((1 << ncols) - 1) << col;
    for (let r = row; r < row + n; r++) rowsUsed[r] |= mask;
    while (top < maxRows && rowsUsed[top] === 0xf) top++;
    while (bottom >= 0 && rowsUsed[bottom] === 0xf) bottom--;
  };
  let i = 0;
  for (; i < items.length && items[i].cols === 4; i++) top += items[i].rows;
  if (top > maxRows) return false;
  for (let r = 0; r < top; r++) rowsUsed[r] = 0xf;
  let three = 0;
  for (; i < items.length && items[i].cols === 3; i++) three += items[i].rows;
  if (top + three > maxRows) return false;
  const topBefore = top;
  fill(topBefore, three, 0, 3);
  const top2 = topBefore + three, avail = maxRows - top2;
  let in01 = avail, in23 = avail;
  for (; i < items.length && items[i].cols === 2; i++) {
    const n = items[i].rows;
    if (n <= in01) in01 -= n;
    else if (n <= in23) in23 -= n;
    else return false;
  }
  fill(top2, avail - in01, 0, 2);
  fill(maxRows - (avail - in23), avail - in23, 2, 2);
  for (; i < items.length; i++) {
    const n = items[i].rows;
    let best = -1, bestSize = maxRows + 1, bestRow = -1;
    for (let col = 0; col < 4; col++) {
      const bit = 1 << col;
      let start = -1;
      const consider = (end) => { const size = end - start; if (size >= n && size < bestSize) { bestSize = size; best = col; bestRow = start; } };
      for (let r = top; r <= bottom; r++) {
        if (!(rowsUsed[r] & bit)) { if (start < 0) start = r; }
        else if (start >= 0) { consider(r); start = -1; }
      }
      if (start >= 0) consider(bottom + 1);
    }
    if (best < 0) return false;
    fill(bestRow, n, best, 1);
  }
  return true;
}
const needed = (vars) => { for (let n = 0; n <= 64; n++) if (fits(vars, n)) return n; return Infinity; };

let bad = 0;
const rows = [];
for (const pr of programs) {
  const vars = varyings(source(pr.vs), source(pr.fs));
  const n = needed(vars);
  if (n > BUDGET) bad++;
  rows.push([`${pr.file} ${pr.vs[1]}+${pr.fs[1]}`, n, vars.length]);
}
const w = Math.max(...rows.map((r) => r[0].length));
for (const [name, n, count] of rows.sort((a, b) => b[1] - a[1])) console.log(`${n > BUDGET ? 'OVER' : '    '}  ${name.padEnd(w)}  ${String(n).padStart(2)} vectors  (${count} varyings)`);
console.log(`${bad ? 'FAIL' : 'PASS'}  t3192 varyings: ${programs.length} programs, ${bad} over ${BUDGET} vectors`);
process.exit(bad ? 1 : 0);
