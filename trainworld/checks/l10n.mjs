// T-2636: the translate-and-review workflow's tools (the runbook is docs/localization.md "Translating").
//   node trainworld/checks/l10n.mjs status [<lang>]                         keys, translated and stale per folder
//   node trainworld/checks/l10n.mjs batch <lang> <scope> [--all] --out <f>  what to translate: key, English, note, glossary terms
//   node trainworld/checks/l10n.mjs merge <lang> <file>                     a {key: translation} file into messages/ and sources/, then checked
//   node trainworld/checks/l10n.mjs back <lang> <scope> --out <f>           the translations alone, for the back-translator
//   node trainworld/checks/l10n.mjs compare <lang> <back file> <scope> --out <f>   English, translation and back-translation side by side
//   node trainworld/checks/l10n.mjs lint <lang> <scope> [--opencc <dir>]    mechanical warnings (punctuation, spaces, English left, length, register)
//   node trainworld/checks/l10n.mjs opencc <scope> --opencc <dir>          zh-Hans through OpenCC s2twp against zh-TW, where they differ
// <scope> is --folder <dir under app/src> (its files' keys; "." for the root), --keys <prefix,prefix> or --all-keys.
// --opencc: the opencc-js package folder (npm install --prefix <scratch>/opencc opencc-js; then <scratch>/opencc/node_modules/opencc-js).
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmdirSync, statSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { spawnSync } from "node:child_process";
import { dirname, extname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const TW = join(dirname(fileURLToPath(import.meta.url)), "..");
const SRC = join(TW, "app", "src");
const I18N = join(SRC, "i18n");
const req = createRequire(join(TW, "app", "package.json"));
const { default: IMF } = await import(pathToFileURL(req.resolve("intl-messageformat")).href);
const { LANGS } = await import(pathToFileURL(join(I18N, "intl.ts")).href);

const [cmd, ...rest] = process.argv.slice(2);
const opt = (name) => (rest.includes(name) ? rest[rest.indexOf(name) + 1] : null);
const has = (name) => rest.includes(name);
const die = (msg) => { console.error(msg); process.exit(2); };

const readJson = (p, fallback = {}) => (existsSync(p) ? JSON.parse(readFileSync(p, "utf8")) : fallback);
const en = readJson(join(I18N, "messages", "en.json"));
const context = readJson(join(I18N, "context.json"));
const glossary = readJson(join(I18N, "glossary.json"));
const msgPath = (lang) => join(I18N, "messages", `${lang}.json`);
const srcPath = (lang) => join(I18N, "sources", `${lang}.json`);
const sorted = (obj) => Object.fromEntries(Object.keys(obj).sort().map((k) => [k, obj[k]]));
const langArg = (l) => (l && LANGS.includes(l) && l !== "en" ? l : die(`a language, one of ${LANGS.filter((x) => x !== "en").join(", ")}`));

// --- scope: which keys --------------------------------------------------------------------------

/** "LineInspector.tsx" -> "lineInspector", the key prefix its messages use (docs/localization.md "Keys"). */
const prefixOf = (file) => { const b = file.slice(0, -extname(file).length); return b[0].toLowerCase() + b.slice(1); };
function filesIn(dir) {
  const d = join(SRC, dir);
  if (!existsSync(d)) die(`no folder app/src/${dir}`);
  return readdirSync(d).filter((f) => /\.tsx?$/.test(f) && !f.endsWith(".d.ts") && statSync(join(d, f)).isFile());
}
/** Every folder under app/src with source files, "." for the root; i18n and wasm left out. */
function folders() {
  const out = [];
  const walk = (rel) => {
    const d = join(SRC, rel);
    if (readdirSync(d).some((f) => /\.tsx?$/.test(f))) out.push(rel || ".");
    for (const f of readdirSync(d)) if (statSync(join(d, f)).isDirectory() && !["i18n", "wasm"].includes(f)) walk(rel ? `${rel}/${f}` : f);
  };
  walk("");
  return out;
}
const prefixFolder = new Map();   // key prefix -> folder of the file it is named after
for (const dir of folders()) for (const f of filesIn(dir === "." ? "" : dir)) prefixFolder.set(prefixOf(f), dir);
const folderOfKey = (k) => prefixFolder.get(k.split(".")[0]) ?? "(no file: common and the like)";

function scopeKeys() {
  let keep;
  if (has("--all-keys")) keep = () => true;
  else if (opt("--folder")) {
    const dir = opt("--folder").replace(/\\/g, "/").replace(/\/$/, "");
    const prefixes = new Set(filesIn(dir === "." ? "" : dir).map(prefixOf));
    keep = (k) => prefixes.has(k.split(".")[0]);
  } else if (opt("--keys")) {
    const ps = opt("--keys").split(",").map((p) => p.replace(/\.$/, ""));
    keep = (k) => ps.some((p) => k === p || k.startsWith(p + "."));
  } else die("say which keys: --folder <dir>, --keys <prefix,...> or --all-keys");
  return Object.keys(en).filter(keep).sort();
}

// --- message text ---------------------------------------------------------------------------------

const intlLocale = (lang) => (lang === "en" ? "en-US" : lang);
/** The message's own words, each placeholder as \u0001, ICU syntax left out. */
function words(lang, text) {
  let ast;
  try { ast = new IMF(text, intlLocale(lang), undefined, { ignoreTag: true }).getAst(); } catch { return text; }
  const walk = (els) => els.map((el) => (el.type === 0 ? el.value : el.type === 5 || el.type === 6 ? Object.values(el.options).map((o) => walk(o.value)).join(" \u0002 ") : "\u0001")).join("");
  return walk(ast);
}
const termRe = (term) => new RegExp(`(^|[^A-Za-z-])(${term.forms.map((f) => f.replace(/-/g, "\\-")).join("|")})(?![A-Za-z-])`, "i");
/** The glossary terms an English message uses, worded for the translator. */
const termsFor = (lang, key) => (glossary.terms ?? []).filter((t) => termRe(t).test(words("en", en[key])))
  .map((t) => `${t.en} -> ${(t[lang] ?? []).join(" / ") || "(none yet)"}${t.notes ? ` (${t.notes})` : ""}${t.check && !(t.except ?? []).includes(key) ? "" : " [not checked]"}`);

function write(out, data) {
  if (!out) die("--out <file> is needed");
  writeFileSync(out, typeof data === "string" ? data : JSON.stringify(data, null, 1) + "\n", "utf8");
  console.log(`wrote ${out}`);
}

// --- commands -----------------------------------------------------------------------------------

if (cmd === "status") {
  const langs = rest[0] && !rest[0].startsWith("--") ? [langArg(rest[0])] : LANGS.filter((l) => l !== "en");
  const rows = new Map();
  for (const k of Object.keys(en)) {
    const f = folderOfKey(k);
    const r = rows.get(f) ?? { keys: 0, words: 0, by: {} };
    r.keys++;
    r.words += words("en", en[k]).split(/\s+/).filter((w) => /[A-Za-z]/.test(w)).length;
    rows.set(f, r);
  }
  for (const lang of langs) {
    const m = readJson(msgPath(lang)), s = readJson(srcPath(lang));
    for (const k of Object.keys(en)) {
      const r = rows.get(folderOfKey(k)), b = (r.by[lang] ??= { done: 0, stale: 0 });
      if (k in m) { b.done++; if (s[k] !== en[k]) b.stale++; }
    }
  }
  console.log(`folder${" ".repeat(34)}keys  words  ${langs.map((l) => l.padStart(14)).join("")}`);
  for (const [f, r] of [...rows].sort((a, b) => b[1].keys - a[1].keys)) {
    console.log(`${f.padEnd(40)}${String(r.keys).padStart(4)} ${String(r.words).padStart(6)}  ${langs.map((l) => { const b = r.by[l] ?? { done: 0, stale: 0 }; return `${b.done}/${r.keys}${b.stale ? ` ${b.stale}st` : ""}`.padStart(14); }).join("")}`);
  }
} else if (cmd === "batch") {
  const lang = langArg(rest[0]);
  const m = readJson(msgPath(lang)), s = readJson(srcPath(lang));
  const keys = scopeKeys().filter((k) => has("--all") || !(k in m) || s[k] !== en[k]);
  const items = keys.map((k) => ({
    key: k, en: en[k], note: context[k] ?? "",
    ...(termsFor(lang, k).length ? { terms: termsFor(lang, k) } : {}),
    ...(k in m ? { was: m[k], wasFrom: s[k] ?? "(not recorded)" } : {}),
  }));
  write(opt("--out"), { lang, made: new Date().toISOString(), count: items.length, items });
} else if (cmd === "merge") {
  const lang = langArg(rest[0]);
  const file = rest[1] ?? die("merge <lang> <file of {key: translation}>");
  const add = JSON.parse(readFileSync(file, "utf8"));
  const bad = Object.keys(add).filter((k) => !(k in en));
  if (bad.length) die(`not in en.json: ${bad.join(", ")}`);
  // several translator agents merge into one language's files: a lock folder, then temp + rename
  const lock = join(I18N, `.merge-${lang}.lock`);
  for (let i = 0; ; i++) {
    try { mkdirSync(lock); break; } catch { if (i > 300) die(`${lock} held for a minute; remove it if no merge is running`); Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 200); }
  }
  try {
    mkdirSync(join(I18N, "sources"), { recursive: true });
    const m = readJson(msgPath(lang)), s = readJson(srcPath(lang));
    for (const [k, v] of Object.entries(add)) {
      if (typeof v !== "string" || !v.trim()) { delete m[k]; delete s[k]; continue; }   // an empty value takes the key out (English shows)
      m[k] = v;
      s[k] = en[k];
    }
    for (const [p, obj] of [[msgPath(lang), m], [srcPath(lang), s]]) {
      writeFileSync(p + ".tmp", JSON.stringify(sorted(obj), null, 2) + "\n", "utf8");
      renameSync(p + ".tmp", p);
    }
  } finally {
    rmdirSync(lock);
  }
  console.log(`merged ${Object.keys(add).length} into messages/${lang}.json and sources/${lang}.json`);
  const r = spawnSync(process.execPath, [join(TW, "checks", "i18n.mjs"), "--no-scan"], { encoding: "utf8" });
  const mine = r.stdout.split("\n").filter((l) => l.includes("FAULT") && Object.keys(add).some((k) => l.includes(`${lang} ${k}:`)));
  for (const l of mine) console.log(l);
  console.log(mine.length ? `${mine.length} faults in this batch: fix them and merge again` : "checks/i18n.mjs: no fault in this batch");
  process.exit(mine.length ? 1 : 0);
} else if (cmd === "back") {
  const lang = langArg(rest[0]);
  const m = readJson(msgPath(lang));
  const items = scopeKeys().filter((k) => k in m).map((k) => ({ key: k, text: m[k] }));
  write(opt("--out"), { lang, count: items.length, items });
} else if (cmd === "compare") {
  const lang = langArg(rest[0]);
  const back = readJson(rest[1] ?? die("compare <lang> <back-translation file> <scope>"));
  const m = readJson(msgPath(lang));
  const items = scopeKeys().filter((k) => k in m).map((k) => ({ key: k, note: context[k] ?? "", en: en[k], text: m[k], back: back[k] ?? "(no back-translation)" }));
  write(opt("--out"), { lang, count: items.length, items });
} else if (cmd === "lint" || cmd === "opencc") {
  await lintOrDiff();
} else {
  console.log(readFileSync(fileURLToPath(import.meta.url), "utf8").split("\n").filter((l) => l.startsWith("//")).slice(0, 12).join("\n"));
  process.exit(cmd ? 2 : 0);
}

async function opencc() {
  const dir = opt("--opencc");
  if (!dir) return null;
  const OpenCC = await import(pathToFileURL(join(dir, "dist", "esm", "full.js")).href).catch(() => import(pathToFileURL(join(dir, "dist", "umd", "full.js")).href));
  return OpenCC.default ?? OpenCC;
}

async function lintOrDiff() {
  const CC = await opencc();
  if (cmd === "opencc") {
    if (!CC) die("--opencc <opencc-js folder> is needed");
    const s2twp = CC.Converter({ from: "cn", to: "twp" });
    const hans = readJson(msgPath("zh-Hans")), tw = readJson(msgPath("zh-TW"));
    let n = 0, same = 0;
    for (const k of scopeKeys()) {
      if (!(k in hans) || !(k in tw)) continue;
      n++;
      const conv = s2twp(hans[k]);
      if (conv === tw[k]) { same++; continue; }
      console.log(`${k}\n  en      ${en[k]}\n  s2twp   ${conv}\n  zh-TW   ${tw[k]}`);
    }
    console.log(`${n} keys in both; ${same} the same after s2twp, ${n - same} differ (read each: a Taiwan word, or a slip in either)`);
    return;
  }
  const lang = langArg(rest[0]);
  const m = readJson(msgPath(lang));
  const CJK = "぀-ヿ㐀-鿿豈-﫿";
  const cjk = new RegExp(`[${CJK}]`);
  const toTrad = CC && lang === "zh-TW" ? CC.Converter({ from: "cn", to: "tw" }) : null;
  const toSimp = CC && lang === "zh-Hans" ? CC.Converter({ from: "tw", to: "cn" }) : null;
  const KEEP = new Set(["km", "mi", "ft", "mph", "AI", "US", "MB", "GB", "CPU", "OK"]);
  // half-width CJK text measures about 0.55 of a full-width character
  const width = (s) => [...s].reduce((w, c) => w + (cjk.test(c) || /[　-〿＀-￯]/.test(c) ? 1 : c === " " ? 0.3 : c === "\u0001" ? 2.5 : c === "\u0002" ? 0 : 0.55), 0);
  const longest = (s) => s.split(" \u0002 ").reduce((a, b) => (width(b) > width(a) ? b : a), "");
  let count = 0;
  for (const k of scopeKeys()) {
    if (!(k in m)) continue;
    const text = m[k], w = words(lang, text), note = context[k] ?? "";
    const warn = (msg) => { count++; console.log(`${k}: ${msg}\n    ${text}`); };
    if (lang !== "es" && lang !== "pt" && lang !== "de" && lang !== "fr" && lang !== "tr" && lang !== "ru") {
      if (new RegExp(`[${CJK}][,.!?:;]|[,!?:;][${CJK}]`).test(w)) warn("half-width punctuation beside CJK; use 、。，！？：；");
      if (new RegExp(`\\([^)]*[${CJK}]`).test(w)) warn("half-width parentheses around CJK; use （）");
      if (new RegExp(`[${CJK}] +[A-Za-z0-9\\u0001]|[A-Za-z0-9\\u0001] +[${CJK}]`).test(w)) warn("a space between CJK and Latin, a digit or a placeholder; the page adds that gap itself (text-autospace)");
      if (/[０-９Ａ-Ｚａ-ｚ]/.test(w)) warn("full-width digits or letters; use ASCII");
      if (/[｡-ﾟ]/.test(w)) warn("half-width katakana");
      const latin = (w.match(/[A-Za-z]{2,}/g) ?? []).filter((x) => !KEEP.has(x));
      if (latin.length) warn(`English left: ${latin.join(", ")} (fine only for a name the data or the game gives)`);
    }
    if (lang === "ja") {
      if (/することができ|することが可能/.test(w)) warn("〜することができます: say できます or the verb itself");
      for (const s of w.split("。").slice(0, -1).map((x) => x.trim()).filter(Boolean)) {
        if (!/(です|ます|ません|ました|でした|ください|ましょう|か|ね|よ)[」）]?$/.test(s)) warn(`a sentence not in です・ます: "${s}。"`);
      }
    }
    if (lang.startsWith("zh")) {
      if (/您/.test(w)) warn("您: plain 你 in a game, or no pronoun");
      if (/赋能|賦能|无缝|無縫|沉浸式|一站式|旨在|助力/.test(w)) warn("a word that reads as machine-made (赋能, 无缝, 沉浸式, 一站式, 旨在, 助力)");
      // characters the converter changes, less the variants Taiwan writes as often as the standard ones (月台, 占用)
      const changed = (conv) => { const a = [...w], b = [...conv(w)]; return a.length !== b.length ? ["?"] : [...new Set(a.filter((c, i) => c !== b[i] && !"台占".includes(c)))]; };
      if (toTrad && changed(toTrad).length) warn(`simplified characters? ${changed(toTrad).join(" ")} (OpenCC cn->tw changes them)`);
      if (toSimp && changed(toSimp).length) warn(`traditional characters? ${changed(toSimp).join(" ")} (OpenCC tw->cn changes them)`);
    }
    if (lang === "fr") {
      if (w.includes("'")) warn("a straight apostrophe; French uses ’ (U+2019)");
      if (/[^\s  \d][:;?!](?=\s|$)/.test(w)) warn("no space before : ; ? or !; U+00A0 before :, U+202F before the others");
      if (/ [:;?!%](\s|$)/.test(w)) warn("a plain space before : ; ? ! or %; use U+00A0 before : and U+202F before the others");
      if (/«(?! )|(?<! )»/.test(w)) warn("guillemets need a no-break space inside");
      if (/\b(tu|ton|ta|tes|toi)\b/i.test(w)) warn("tu: the voice is vous");
      if (/(?<![\p{L}])(de|du|le|la|que|si|ce|ne|je|me|te|se) ?\u0001|(?<![\p{L}])(l|d|j|m|t|s|n|c|qu)’\u0001/iu.test(w)) warn("an article, preposition or elision directly before a placeholder: reword so the value stands alone (fine before a number or price)");
    }
    // a width check counts only a real difference: a full character or two more than the English
    const tw = width(longest(w)), ew = Math.max(1, width(longest(words("en", en[k]))));
    const narrow = /narrow|short|segmented|button/i.test(note);
    if (narrow ? tw > ew * 1.15 && tw > ew + 1.5 : tw > ew * 1.8 && tw > ew + 4) warn(`${(tw / ew).toFixed(2)}x the English's width${narrow ? " in a narrow place" : ""}`);
  }
  console.log(`${count} warnings for ${lang}${CC || !lang.startsWith("zh") ? "" : " (no --opencc: the script checks on Chinese were skipped)"}`);
}
