// T-2583: the message files and the text still written inline (docs/localization.md).
//   node trainworld/checks/i18n.mjs                 messages, glossary, then the inline-literal worklist
//   node trainworld/checks/i18n.mjs --no-scan       messages and glossary only (smoke runs this)
//   node trainworld/checks/i18n.mjs --list <dir>    also each inline literal under app/src/<dir>, file:line
//   node trainworld/checks/i18n.mjs --stale         also each stale or unrecorded translation
// Exit 1 on a fault in the message files; the worklist and translation progress never fail it.
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, extname, join, relative } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const TW = join(dirname(fileURLToPath(import.meta.url)), "..");
const APP = join(TW, "app");
const SRC = join(APP, "src");
const I18N = join(SRC, "i18n");
const args = process.argv.slice(2);
const flag = (f) => args.includes(f);
const listDir = args.includes("--list") ? args[args.indexOf("--list") + 1] : null;

const req = createRequire(join(APP, "package.json"));
const { default: IMF } = await import(pathToFileURL(req.resolve("intl-messageformat")).href);
const { LANGS } = await import(pathToFileURL(join(I18N, "intl.ts")).href);

const faults = [];
const fault = (s) => faults.push(s);
const warnings = [];

// --- the files -------------------------------------------------------------------------------

/** A flat JSON object, with duplicate keys refused (JSON.parse keeps the last silently). */
function readFlat(path) {
  const raw = readFileSync(path, "utf8");
  let obj;
  try {
    obj = JSON.parse(raw);
  } catch (e) {
    fault(`${rel(path)}: not JSON: ${e.message}`);
    return {};
  }
  const seen = new Set();
  for (const m of raw.matchAll(/^\s*"((?:[^"\\]|\\.)*)"\s*:/gm)) {
    if (seen.has(m[1])) fault(`${rel(path)}: key ${m[1]} twice`);
    seen.add(m[1]);
  }
  for (const [k, v] of Object.entries(obj)) if (typeof v !== "string") fault(`${rel(path)}: ${k} is not a string`);
  return obj;
}
const rel = (p) => relative(TW, p).replace(/\\/g, "/");

const en = readFlat(join(I18N, "messages", "en.json"));
const context = readFlat(join(I18N, "context.json"));
const glossary = JSON.parse(readFileSync(join(I18N, "glossary.json"), "utf8"));
const others = LANGS.filter((l) => l !== "en");
const files = readdirSync(join(I18N, "messages")).filter((f) => f.endsWith(".json")).map((f) => f.slice(0, -5));
for (const f of files) if (!LANGS.includes(f)) fault(`messages/${f}.json: ${f} is not in LANGS (i18n/intl.ts)`);
for (const l of LANGS) if (!files.includes(l)) fault(`messages/${l}.json missing`);

// --- parsing and placeholders ----------------------------------------------------------------

// element types of @formatjs/icu-messageformat-parser
const LITERAL = 0, ARGUMENT = 1, NUMBER = 2, DATE = 3, TIME = 4, SELECT = 5, PLURAL = 6, POUND = 7, TAG = 8;
const intlLocale = (lang) => (lang === "en" ? "en-US" : lang);

function parse(lang, key, text) {
  try {
    return new IMF(text, intlLocale(lang), undefined, { ignoreTag: true }).getAst();
  } catch (e) {
    fault(`${lang} ${key}: does not parse (${e.message})`);
    return null;
  }
}

/** name -> kind ("text", "plural", "selectordinal", "select") and each select's and plural's option keys. */
function placeholders(lang, key, ast, out = new Map()) {
  for (const el of ast) {
    if (el.type === ARGUMENT) add(out, lang, key, el.value, "text");
    else if (el.type === NUMBER || el.type === DATE || el.type === TIME) fault(`${lang} ${key}: {${el.value}, ${["", "", "number", "date", "time"][el.type]}} formats in the message; format it in code and pass text (docs/localization.md)`);
    else if (el.type === SELECT || el.type === PLURAL) {
      const kind = el.type === SELECT ? "select" : el.pluralType === "ordinal" ? "selectordinal" : "plural";
      const p = add(out, lang, key, el.value, kind);
      if (el.offset) fault(`${lang} ${key}: plural offset is not used here`);
      for (const [opt, { value }] of Object.entries(el.options)) {
        p.options.add(opt);
        placeholders(lang, key, value, out);
      }
      if (el.type === PLURAL) {
        const cats = new Intl.PluralRules(intlLocale(lang), { type: el.pluralType }).resolvedOptions().pluralCategories;
        for (const opt of Object.keys(el.options)) if (!/^=\d+$/.test(opt) && !cats.includes(opt)) fault(`${lang} ${key}: plural option ${opt} is not a ${lang} category (${cats.join(", ")})`);
        for (const c of cats) if (c !== "many" && !(c in el.options)) warnings.push(`${lang} ${key}: plural {${el.value}} has no ${c}`);
      }
    } else if (el.type === TAG) fault(`${lang} ${key}: a tag; pass rich pieces as values instead`);
  }
  return out;
}
function add(out, lang, key, name, kind) {
  const had = out.get(name);
  if (had && had.kind !== kind) fault(`${lang} ${key}: {${name}} is both ${had.kind} and ${kind}`);
  if (!had) out.set(name, { kind, options: new Set() });
  return out.get(name);
}
/** The message's words, placeholders and ICU syntax left out, for the glossary. */
function literalText(ast) {
  let s = "";
  for (const el of ast) {
    if (el.type === LITERAL) s += el.value;
    else if (el.type === SELECT || el.type === PLURAL) for (const { value } of Object.values(el.options)) s += " " + literalText(value) + " ";
    else s += " ";
  }
  return s;
}

const enAst = new Map();
for (const [k, v] of Object.entries(en)) {
  if (!/^[a-z][A-Za-z0-9]*(\.[a-z][A-Za-z0-9]*)+$/.test(k)) fault(`en ${k}: key is not file.name in camelCase (docs/localization.md)`);
  if (!v.trim()) fault(`en ${k}: empty`);
  const ast = parse("en", k, v);
  if (ast) enAst.set(k, { ast, ph: placeholders("en", k, ast) });
  if (!(k in context)) fault(`en ${k}: no note in context.json`);
}
for (const k of Object.keys(context)) if (!(k in en)) fault(`context.json ${k}: no such key in en.json`);
const UPDATE_LINE_MAX = 80; // the start menu's update row; English only, translations may run longer (Anita, 2026-11-09)
for (const [k, v] of Object.entries(en)) if (/^updates\.v\d+Line$/.test(k) && v.length > UPDATE_LINE_MAX) fault(`${k}: ${v.length} characters, the update row's limit is ${UPDATE_LINE_MAX}`);

// --- glossary --------------------------------------------------------------------------------

const terms = glossary.terms ?? [];
for (const term of terms) {
  if (!term.en || !Array.isArray(term.forms) || !term.forms.length || !term.meaning) fault(`glossary: ${term.en ?? "?"} needs en, forms and meaning`);
  for (const l of others) if (!Array.isArray(term[l])) fault(`glossary: ${term.en} has no ${l} list`);
  for (const k of term.except ?? []) if (!(k in en)) fault(`glossary: ${term.en} excepts ${k}, not in en.json`);
}
const formRe = (term) => new RegExp(`(^|[^A-Za-z-])(${term.forms.map((f) => f.replace(/[-]/g, "\\-")).join("|")})(?![A-Za-z-])`, "i");
/** `text` with every longer term's forms that contain one of `term`'s blanked: "track" inside "passing track", "trip"
 * inside "round trip", is the longer term's word, not this one's. */
const outsideLonger = (text, term) => {
  const mine = term.forms.map((f) => f.toLowerCase());
  for (const o of terms) {
    if (o === term) continue;
    for (const f of o.forms) {
      const lf = f.toLowerCase();
      if (mine.some((g) => lf.length > g.length && lf.includes(g))) text = text.replace(new RegExp(f.replace(/[-]/g, "\\-"), "gi"), " ");
    }
  }
  return text;
};
const enTerms = new Map();   // key -> the checked terms its English uses
for (const [k, { ast }] of enAst) {
  const text = literalText(ast);
  enTerms.set(k, terms.filter((term) => term.check && !(term.except ?? []).includes(k) && formRe(term).test(outsideLonger(text, term))));
}

// --- each language ---------------------------------------------------------------------------

const progress = [];
for (const lang of others) {
  const path = join(I18N, "messages", `${lang}.json`);
  if (!existsSync(path)) continue;
  const msgs = readFlat(path);
  const srcPath = join(I18N, "sources", `${lang}.json`);
  const sources = existsSync(srcPath) ? readFlat(srcPath) : {};
  let done = 0, stale = 0, unrecorded = 0;
  for (const [k, v] of Object.entries(msgs)) {
    if (!(k in en)) { fault(`${lang} ${k}: no such key in en.json`); continue; }
    if (!v.trim()) { fault(`${lang} ${k}: empty; leave a key out to show the English`); continue; }
    done++;
    const ast = parse(lang, k, v);
    const src = enAst.get(k);
    if (!ast || !src) continue;
    const ph = placeholders(lang, k, ast);
    for (const [name, p] of src.ph) {
      const q = ph.get(name);
      if (!q) fault(`${lang} ${k}: {${name}} missing`);
      else if (q.kind !== p.kind) fault(`${lang} ${k}: {${name}} is ${q.kind}, English has ${p.kind}`);
      else if (p.kind === "select") for (const o of p.options) if (!q.options.has(o)) fault(`${lang} ${k}: select {${name}} has no ${o}`);
    }
    for (const name of ph.keys()) if (!src.ph.has(name)) fault(`${lang} ${k}: {${name}} is not in the English`);
    for (const term of enTerms.get(k) ?? []) {
      const want = term[lang] ?? [];
      if (want.length && !want.some((w) => v.toLowerCase().includes(w.toLowerCase()))) fault(`${lang} ${k}: English says "${term.en}", glossary wants ${want.join(" or ")}`);
    }
    if (!(k in sources)) { unrecorded++; if (flag("--stale")) console.log(`  ${lang} ${k}: no English recorded in sources/${lang}.json`); }
    else if (sources[k] !== en[k]) { stale++; if (flag("--stale")) console.log(`  ${lang} ${k}: stale; was "${sources[k]}", now "${en[k]}"`); }
  }
  for (const k of Object.keys(sources)) if (!(k in msgs)) fault(`sources/${lang}.json ${k}: no translation`);
  progress.push(`${lang} ${done}/${Object.keys(en).length}${stale ? `, ${stale} stale` : ""}${unrecorded ? `, ${unrecorded} without a source` : ""}`);
}

// --- self-test: the library does what the conventions rely on --------------------------------

const fmt = (s, v, l = "en") => new IMF(s, intlLocale(l), undefined, { ignoreTag: true }).format(v);
const expect = [
  [fmt("{n, plural, one {# car} other {# cars}}", { n: 1234 }), "1,234 cars"],
  [fmt("{n, plural, one {# car} other {# cars}}", { n: 1 }), "1 car"],
  [fmt("{b, select, true {built} other {blueprint}}", { b: String(true) }), "built"],
  [fmt("don't < 5, it''s '{x}'", {}), "don't < 5, it's {x}"],
  [new Intl.ListFormat("en-GB", { type: "conjunction" }).format(["a", "b", "c"]), "a, b and c"],
];
for (const [got, want] of expect) if (got !== want) fault(`self-test: got "${got}", want "${want}"`);

// --- report the files ------------------------------------------------------------------------

console.log(`i18n: ${Object.keys(en).length} English messages, ${terms.length} glossary terms; ${progress.join("; ")}`);
for (const w of warnings) console.log(`  warning: ${w}`);
for (const f of faults) console.log(`  FAULT: ${f}`);

// --- the inline literals still to move -------------------------------------------------------

if (!flag("--no-scan")) scan(await import(pathToFileURL(req.resolve("rolldown/parseAst")).href));

function scan({ parseAst }) {
  // whole files of debug or developer text, never shown to a player
  const SKIP = /^(i18n\/|wasm\/|perfFlags\.ts$|workers\/demandTest\.ts$|ui\/Debug\.tsx$|game\/fonts\.ts$)/;
  const CODE_CALLS = new Set(["t", "tRich", "addEventListener", "removeEventListener", "querySelector", "querySelectorAll", "getElementById", "closest",
    "createElement", "createElementNS", "setAttribute", "getAttribute", "removeAttribute", "toggleAttribute", "hasAttribute", "getLayer", "getSource", "setLayoutProperty", "setPaintProperty", "setFilter",
    "getContext", "postMessage", "fetch", "dataUrl", "import", "add", "remove", "toggle", "contains", "getItem", "setItem", "removeItem",
    "getUniformLocation", "getAttribLocation", "getExtension", "matchMedia", "setProperty", "getPropertyValue", "startsWith", "endsWith", "includes", "split", "replace", "replaceAll", "indexOf",
    "padStart", "join", "has", "get", "set", "delete", "Error", "TypeError", "RangeError", "log", "warn", "error", "debug", "info", "assert", "mark", "measure", "time", "timeEnd", "open", "transaction", "objectStore", "createObjectStore", "createIndex", "index",
    "URL", "Worker", "SharedWorker", "Blob", "Request", "Response", "Headers", "NumberFormat", "DisplayNames", "Collator", "ListFormat", "DateTimeFormat", "PluralRules", "toLocaleString", "toLocaleDateString", "toLocaleTimeString",
    "getComputedStyle", "parseFloat", "parseInt", "Number", "String", "RegExp", "on", "off", "once", "send", "ask", "query", "invoke", "call", "emit", "q", "post", "request", "queryClock", "glob"]);
  const CODE_ATTRS = new Set(["class", "className", "id", "key", "type", "name", "href", "src", "style", "role", "for", "htmlFor", "viewBox", "d", "fill", "stroke", "transform", "xmlns", "points", "rel", "target", "accept", "autocomplete", "inputMode", "inputmode", "pattern", "lang", "dir", "tabIndex", "method",
    "data-guide", "data-k", "data-tab", "data-id", "kind", "variant", "icon", "size", "mode", "align", "side", "width", "height", "x", "y", "cx", "cy", "r", "rx", "ry", "x1", "x2", "y1", "y2", "stroke-width", "stroke-linecap", "stroke-linejoin", "text-anchor", "dominant-baseline", "font-size", "font-weight",
    "opacity", "fill-opacity", "clip-path", "mask", "offset", "stop-color", "gradientUnits", "preserveAspectRatio", "draggable", "spellcheck", "enterKeyHint", "enterkeyhint", "colour", "color", "min", "max", "step", "value"]);
  const isProse = (s) => {
    const t = s.replace(/\$\{[^}]*\}/g, " ").trim();
    if (!/[A-Za-z]{2,}/.test(t)) return false;
    if (/^(https?:|\.?\.?\/|#[0-9a-f]{3,8}$|rgba?\(|var\(--|[a-z]+:\/\/)/i.test(t)) return false;
    if (/^[a-z0-9_.:\/-]+$/.test(t) && !/ /.test(t)) return false;   // ids, keys, paths, css tokens
    if (/^[a-z][a-zA-Z0-9]+$/.test(t) || /^[A-Z0-9_]+$/.test(t)) return false;   // camelCase, CONSTANTS
    if (/^[a-z][A-Za-z0-9]*(\.[a-z][A-Za-z0-9]*)+$/.test(t)) return false;   // message keys
    if (/[{};=]|=>|\bfunction\b|\bvoid\b|^\s*(precision|uniform|in |out |vec|float|#version)/.test(t)) return false;   // code, GLSL
    if (/^[a-z-]+( [a-z-]+)+$/.test(t) && t.split(" ").every((w) => /-/.test(w) || w.length < 3)) return false;   // class lists
    if (/^ *\w*([|:] *\w*)+$/.test(t) || /^[#.][a-z][\w .-]*$/.test(t) || /^([A-Z]{2} )+[A-Z]{2}$/.test(t) || /^(Feature|Point)$/.test(t) || /^(tool-tag|pi-mark|overlay|stn-label|chip|spin) [a-z-]+$/.test(t) || /^[a-z]+(,[a-z]+)+$/.test(t) || /^(color: |\d+ \d+px|\d+ \d+px )/.test(t) || /^\d+ \d+px\b/.test(t) || /^translate\(/.test(t) || /^[A-Z][a-z]+([A-Z][a-z0-9]+)+$/.test(t) || /^M[\d. -]+[A-Za-z][\dA-Za-z. -]*$/.test(t) || /^(Noto Sans|Zen Maru Gothic) Regular$/.test(t)) return false;   // CSS transforms, CamelCase codes and GeoJSON types, SVG paths, font stacks
    return true;
  };
  function* walk(d) {
    for (const n of readdirSync(d)) {
      const p = join(d, n);
      if (statSync(p).isDirectory()) yield* walk(p);
      else if ([".ts", ".tsx"].includes(extname(n)) && !n.endsWith(".d.ts")) yield p;
    }
  }
  const folderOf = (f) => (f.startsWith("ui/inspector/") ? "ui/inspector" : f.includes("/") ? f.split("/")[0] : "(root)");
  const rows = new Map();   // folder -> {strings: Set, sites, plurals}
  const listed = [];
  for (const path of walk(SRC)) {
    const file = relative(SRC, path).replace(/\\/g, "/");
    if (SKIP.test(file)) continue;
    const code = readFileSync(path, "utf8");
    const lines = code.split("\n");
    const starts = [0];
    for (let i = 0; i < code.length; i++) if (code[i] === "\n") starts.push(i + 1);
    const lineOf = (pos) => { let lo = 0, hi = starts.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (starts[m] <= pos) lo = m; else hi = m - 1; } return lo + 1; };
    const ignored = (ln) => /i18n-ignore/.test(lines[ln - 1] ?? "") || /i18n-ignore/.test(lines[ln - 2] ?? "");
    let ast;
    try { ast = parseAst(code, { lang: file.endsWith("x") ? "tsx" : "ts", sourceType: "module" }, path); }
    catch (e) { console.log(`  scan: ${file} does not parse: ${e.message}`); continue; }
    const row = rows.get(folderOf(file)) ?? { strings: new Set(), sites: 0, plurals: 0 };
    rows.set(folderOf(file), row);
    const found = (text, pos) => {
      const ln = lineOf(pos);
      if (ignored(ln)) return;
      row.strings.add(file + "\u0000" + text);
      row.sites++;
      if (listDir && file.startsWith(listDir)) listed.push(`${file}:${ln} ${text.replace(/\s+/g, " ").slice(0, 140)}`);
    };
    const pluralFound = (node) => {
      row.plurals++;
      if (listDir && file.startsWith(listDir)) listed.push(`${file}:${lineOf(node.start)} plural or list: ${code.slice(node.start, node.end).replace(/\s+/g, " ").slice(0, 120)}`);
    };
    // a literal inside a handler (onClick={() => ask({ text: "..." })}) is the function's, not the attribute's
    const attrOf = (anc) => {
      for (let i = anc.length - 1; i >= Math.max(0, anc.length - 6); i--) {
        if (/Function/.test(anc[i].type)) return null;
        if (anc[i].type === "JSXAttribute") return anc[i].name?.name?.name ?? anc[i].name?.name;
      }
      return null;
    };
    const visit = (node, parent, key, anc) => {
      if (!node || typeof node.type !== "string") return;
      const codeCtx = () => {
        if (!parent) return false;
        if (/^(Import|Export)/.test(parent.type) || parent.type === "ImportExpression") return true;
        if (["Property", "PropertyDefinition", "TSPropertySignature", "MethodDefinition"].includes(parent.type) && key === "key") return true;
        if (parent.type === "MemberExpression" && key === "property") return true;
        if (parent.type === "TSLiteralType" || parent.type === "TSEnumMember" || parent.type === "SwitchCase") return true;
        if (parent.type === "BinaryExpression" && ["===", "!==", "==", "!=", "in"].includes(parent.operator)) return true;
        if (parent.type === "CallExpression" || parent.type === "NewExpression") {
          const c = parent.callee, n = c?.type === "MemberExpression" ? c.property?.name : c?.name;
          if (CODE_CALLS.has(n)) return true;
        }
        const a = attrOf(anc);
        return !!a && (CODE_ATTRS.has(a) || /^on[A-Z]/.test(a));
      };
      if (node.type === "JSXText") {
        const t = node.value.replace(/\s+/g, " ").trim();
        if (/[A-Za-z]{2,}/.test(t)) found(t, node.start);
      } else if (node.type === "Literal" && typeof node.value === "string") {
        if (!codeCtx() && isProse(node.value)) found(node.value, node.start);
      } else if (node.type === "TemplateLiteral" && parent?.type !== "TaggedTemplateExpression") {
        const text = node.quasis.map((q, i) => q.value.cooked + (i < node.expressions.length ? "{" + i + "}" : "")).join("");
        // one word between two pieces ("{0} riders {1}") is a sentence, though alone it would pass for an id
        if (!codeCtx() && (isProse(text.replace(/\{\d+\}/g, " ")) || (/\}\s+[a-z]{3,}\s+\{/.test(text) && !/[;=#]/.test(text)))) found(text, node.start);
      }
      if (node.type === "CallExpression") {
        const c = node.callee, n = c?.type === "MemberExpression" ? c.property?.name : c?.name;
        if (["plural", "andList"].includes(n) && !ignored(lineOf(node.start))) pluralFound(node);
      } else if (node.type === "ConditionalExpression") {
        // a word in a branch ("stop" : "stops"); a message key chosen by the count ("topBar.inDays" against
        // "topBar.timingTomorrow"), a code value compared with and a bare "\n" are not
        const s = code.slice(node.test.start, node.test.end), whole = code.slice(node.start, node.end);
        const worded = (n) => {
          if (!n || typeof n.type !== "string") return false;
          if (n.type === "Literal") return typeof n.value === "string" && /[A-Za-z]{2,}/.test(n.value) && !/^[a-z][A-Za-z0-9]*(\.[a-z][A-Za-z0-9]*)+$/.test(n.value);
          if (n.type === "TemplateLiteral") return n.quasis.some((q) => /[A-Za-z]{2,}/.test(q.value.cooked));
          if (n.type === "BinaryExpression" && ["===", "!==", "==", "!="].includes(n.operator)) return false;
          if (n.type === "CallExpression" && ["t", "tr", "tRich"].includes(n.callee?.name)) return n.arguments.slice(1).some(worded);
          return Object.keys(n).some((k) => k !== "parent" && (Array.isArray(n[k]) ? n[k].some(worded) : n[k] && typeof n[k] === "object" && worded(n[k])));
        };
        if (/(===|!==|==)\s*1\b/.test(s) && (worded(node.consequent) || worded(node.alternate)) && whole.length < 200 && !ignored(lineOf(node.start))) pluralFound(node);
      }
      const next = [...anc, node];
      for (const k of Object.keys(node)) {
        if (k === "parent") continue;
        const v = node[k];
        if (Array.isArray(v)) v.forEach((x) => visit(x, node, k, next));
        else if (v && typeof v === "object" && typeof v.type === "string") visit(v, node, k, next);
      }
    };
    visit(ast, null, null, []);
  }
  const order = [...rows].sort((a, b) => b[1].strings.size - a[1].strings.size);
  const total = order.reduce((s, [, r]) => [s[0] + r.strings.size, s[1] + r.sites, s[2] + r.plurals], [0, 0, 0]);
  console.log("inline player text still to move to messages (stage 1's worklist):");
  console.log("  folder          strings  sites  plurals/lists");
  for (const [dir, r] of [...order, ["total", { strings: { size: total[0] }, sites: total[1], plurals: total[2] }]]) {
    console.log(`  ${dir.padEnd(14)} ${String(r.strings.size).padStart(8)} ${String(r.sites).padStart(6)} ${String(r.plurals).padStart(14)}`);
  }
  for (const l of listed) console.log("  " + l);
}

process.exit(faults.length ? 1 : 0);
