// T-2681: the registry's city names in Japanese and Chinese, from Wikidata labels, into
// app/src/i18n/place-names.json ({ metro id: { q, ja, "zh-Hans", "zh-TW" } }). Zones (`kind` "zone") keep
// their Latin names (docs/localization.md "City names").
//
//   node trainworld/tools/place_names.mjs --opencc <opencc-js folder>            add the cities not yet in the file
//   node trainworld/tools/place_names.mjs --opencc <dir> --relabel               also fetch every known item's labels again
//   node trainworld/tools/place_names.mjs --opencc <dir> --redo nyc,lon          match these ids again from scratch (`all`: every city)
//   --zones                                                                      the same for the zones, into place-names-zones.json
//
// Rerun after a join (registry.json rewritten); ids the registry no longer has are dropped. An entry whose item no longer
// carries the name the game shows (zone-names.json renamed the zone) is matched again. Every new match is vetted by
// place_names_rules.mjs vet() (T-3450): a town inside the unit named for the place gives way to the unit (Shiga's old
// town 志賀町 to Shiga Prefecture), and a museum, camp or former country is dropped. checks/place_names_match.mjs runs the
// same rules over the files. A wrong match the rules miss is fixed by adding the right item to PINNED
// (place_names_rules.mjs) and rerunning with --redo <id>. An entry with `q: null` is a city no item matched; it
// shows its Latin name and is not searched again unless named in --redo.
// --opencc: npm install --prefix <scratch>/opencc opencc-js, then <scratch>/opencc/node_modules/opencc-js. Without it,
// zh-TW falls back no further than Wikidata's zh-tw and zh-hant labels.
// Requests go to the public Wikidata SPARQL endpoint, one at a time and batched, with a generic User-Agent (the
// wbsearchentities API answered 429 to a search per city, 2026-11-07).

import { readFileSync, writeFileSync, renameSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { NEAR_KM, PLACE_CLASSES, NOT_PLACE_CLASSES, PINNED, HAND_LABELS, sparql, point, km, cleanName, fold, textsOf, labelFits, near, namesOf, factsOf, unitLabels, vet } from "./place_names_rules.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const REGISTRY = join(ROOT, "app/src/game/registry.json");
const ZONE_NAMES = join(ROOT, "app/src/game/zone-names.json");
const ZONES = process.argv.includes("--zones");
// zones are 3,500 rows, so they live in a file of their own that the app loads only for Japanese and Chinese
const OUT = join(ROOT, ZONES ? "app/src/i18n/place-names-zones.json" : "app/src/i18n/place-names.json");
/** A match by name with fewer sitelinks than this is checked against the places near the centre. */
const WEAK_LINKS = 20;

const args = process.argv.slice(2);
const opt = (name) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : undefined; };
const relabel = args.includes("--relabel");
const redo = new Set((opt("--redo") ?? "").split(",").filter(Boolean));

async function opencc() {
  const dir = opt("--opencc");
  if (!dir) return null;
  const m = await import(pathToFileURL(join(dir, "dist", "esm", "full.js")).href).catch(() => import(pathToFileURL(join(dir, "dist", "umd", "full.js")).href));
  const CC = m.default ?? m;
  return { toTW: CC.Converter({ from: "cn", to: "twp" }), toCN: CC.Converter({ from: "t", to: "cn" }) };
}

/** Items with a coordinate whose English label or alias is one of `texts`: { text: [{ id, coord, links }] }, 80 texts a query. */
async function byName(texts) {
  const out = new Map();
  for (let i = 0; i < texts.length; i += 80) {
    const batch = texts.slice(i, i + 80);
    const rows = await sparql(`SELECT ?item ?n ?coord ?links WHERE {
  VALUES ?n { ${batch.map((t) => `${JSON.stringify(t)}@en ${JSON.stringify(t)}@mul`).join(" ")} }
  ?item rdfs:label|skos:altLabel ?n ; wdt:P625 ?coord .
  OPTIONAL { ?item wikibase:sitelinks ?links }
  BIND(EXISTS { ?item wdt:P576 [] } AS ?gone)
  FILTER EXISTS { ?item wdt:P31/wdt:P279* ?cls VALUES ?cls { ${PLACE_CLASSES} } }
  FILTER NOT EXISTS { ?item wdt:P31/wdt:P279* ?no VALUES ?no { ${NOT_PLACE_CLASSES} } }
}`);
    for (const b of rows) {
      const p = point(b);
      if (!p) continue;
      const n = b.n.value;
      if (!out.has(n)) out.set(n, []);
      out.get(n).push({ id: b.item.value.split("/").pop(), coord: p, links: Number(b.links?.value ?? 0), gone: b.gone?.value === "true" });
    }
    console.log(`names ${Math.min(i + 80, texts.length)}/${texts.length}`);
  }
  return out;
}

const LABEL_LANGS = ["ja", "zh-hans", "zh-cn", "zh", "zh-tw", "zh-hant"];

/** Each item's labels in LABEL_LANGS, 200 items a query. */
async function labels(ids) {
  const out = new Map();
  for (let i = 0; i < ids.length; i += 200) {
    const batch = ids.slice(i, i + 200);
    const rows = await sparql(`SELECT ?item ?label WHERE {
  VALUES ?item { ${batch.map((id) => "wd:" + id).join(" ")} }
  ?item rdfs:label ?label FILTER(LANG(?label) IN (${LABEL_LANGS.map((l) => `"${l}"`).join(",")}))
}`);
    for (const b of rows) {
      const id = b.item.value.split("/").pop();
      if (!out.has(id)) out.set(id, {});
      out.get(id)[b.label["xml:lang"]] = b.label.value;
    }
  }
  return out;
}

/** "圣保罗 (巴西)" and "圣保罗（巴西）" as 圣保罗: a label's own disambiguation. */
const bare = (s) => s?.replace(/\s*[（(][^）)]*[）)]\s*$/, "").trim() || undefined;

/** A city as players say it: 上海 for 上海市, 東京 for 東京都, 釜山 for 釜山広域市, ソウル for ソウル特別市. The 市 stays
 * where the Latin name has its "City" too (吉林市 for Jilin City, クウェート市 for Kuwait City), and on a name of one
 * character before it. */
function short(s, latin) {
  if (!s) return s;
  const t = s.replace(/(広域市|广域市|廣域市|特別市|特别市)$/, "");
  if (t !== s) return t;
  if (/\bCity$/.test(latin)) return s;
  const m = /^(.{2,})[市都]$/u.exec(s);
  return m ? m[1] : s;
}

function names(L, CC, latin) {
  const ja = bare(L.ja);
  let hans = bare(L["zh-hans"] ?? L["zh-cn"]);
  if (!hans && L.zh) hans = CC ? CC.toCN(bare(L.zh)) : bare(L.zh);
  let tw = bare(L["zh-tw"] ?? L["zh-hant"]);
  if (!tw && hans && CC) tw = CC.toTW(hans);
  return { ja: short(ja, latin), "zh-Hans": short(hans, latin), "zh-TW": short(tw, latin) };
}

const registry = JSON.parse(readFileSync(REGISTRY, "utf8"));
const hand = JSON.parse(readFileSync(ZONE_NAMES, "utf8"));
const old = existsSync(OUT) ? JSON.parse(readFileSync(OUT, "utf8")) : {};
const CC = await opencc();
if (!CC) console.warn("no --opencc: zh-TW is only Wikidata's own, and a bare zh label is kept as written");

const cities = registry.cities.filter((c) => (c.kind === "zone") === ZONES).map((c) => ({ id: c.id, raw: c.name, name: hand[c.id]?.name ?? cleanName(c.name), center: c.center, box: c.box, country: c.country }));
let todo = cities.filter((c) => redo.has("all") || redo.has(c.id) || !(c.id in old));
const log = [];
// an item that no longer carries the name the game shows (the zone was renamed since it was matched) is matched again
const kept = cities.filter((c) => !todo.includes(c) && old[c.id]?.q && !PINNED[c.id] && !HAND_LABELS[c.id]);
if (kept.length) {
  const known = await namesOf([...new Set(kept.map((c) => old[c.id].q))], console.log);
  const stale = kept.filter((c) => !known.get(old[c.id].q).some((n) => labelFits(n, textsOf(c))));
  for (const c of stale) log.push(`${c.id}\t${c.name}\t${old[c.id].q}\tnamed otherwise, matched again`);
  todo = todo.concat(stale);
}
console.log(`${cities.length} cities in the registry, ${todo.length} to match`);

// match: the items whose English label or alias is the name (or its part before a comma, "and" or " - "), keep the
// best-linked one near the centre; one that still exists before one dissolved
const found = await byName([...new Set(todo.flatMap(textsOf))]);
const chosen = new Map();
const note = (c, best, how) => log.push(`${c.id}\t${c.name}\t${best?.id ?? "-"}\t${best ? `${Math.round(best.d)} km, ${best.links} links${how}` : "no item near"}`);
const weak = new Map();
for (const c of todo) {
  if (PINNED[c.id]) { chosen.set(c.id, PINNED[c.id]); log.push(`${c.id}\t${c.name}\t${PINNED[c.id]}\tpinned`); continue; }
  let best = null;
  for (const text of textsOf(c)) for (const e of found.get(text) ?? []) {
    const d = km(e.coord, c.center);
    if (!near(e.coord, c)) continue;
    if (!best || (best.gone && !e.gone) || (best.gone === e.gone && (e.links > best.links || (e.links === best.links && d < best.d)))) best = { ...e, d };
  }
  if (best && !best.gone && best.links >= WEAK_LINKS) { chosen.set(c.id, best.id); note(c, best, ""); }
  else weak.set(c, best);
}
// none, or only a little-known or dissolved item ("Bogota" for Bogotá, "New York" for New York City): the best-linked
// place near the centre whose English label or alias is the name or starts with it, ignoring case and accents; one that
// still exists first, then an exact one
for (const [c, first] of weak) {
  const rows = await sparql(`SELECT ?item ?label ?links ?coord ?gone WHERE {
  SERVICE wikibase:around { ?item wdt:P625 ?coord . bd:serviceParam wikibase:center "Point(${c.center[0]} ${c.center[1]})"^^geo:wktLiteral . bd:serviceParam wikibase:radius "${NEAR_KM / 2}" . }
  ?item wikibase:sitelinks ?links FILTER(?links >= 5)
  ?item rdfs:label|skos:altLabel ?label FILTER(LANG(?label) IN ("en", "mul"))
  BIND(EXISTS { ?item wdt:P576 [] } AS ?gone)
  FILTER EXISTS { ?item wdt:P31/wdt:P279* ?cls VALUES ?cls { ${PLACE_CLASSES} } }
  FILTER NOT EXISTS { ?item wdt:P31/wdt:P279* ?no VALUES ?no { ${NOT_PLACE_CLASSES} } }
}`);
  const texts = textsOf(c).map(fold);
  let best = null;
  for (const b of rows) {
    const label = fold(b.label.value), links = Number(b.links.value), live = b.gone?.value !== "true";
    const exact = texts.includes(label);
    if (!exact && !texts.some((t) => label.startsWith(t + " "))) continue;
    const cand = { id: b.item.value.split("/").pop(), links, exact, live, label: b.label.value, d: km(point(b), c.center) };
    if (!best || (cand.live - best.live || cand.exact - best.exact || cand.links - best.links) > 0) best = cand;
  }
  const firstWins = first && (!best || (first.gone !== best.live ? !first.gone : first.links >= best.links));
  if (!best || firstWins) { chosen.set(c.id, first?.id ?? null); note(c, first, ""); }
  else { chosen.set(c.id, best.id); note(c, best, `, by label "${best.label}"`); }
}
const qOf = (c) => (chosen.has(c.id) ? chosen.get(c.id) : old[c.id].q);
const fetchLabels = cities.filter((c) => qOf(c) && (chosen.has(c.id) || relabel)).map(qOf);
const L = await labels([...new Set(fetchLabels)]);

// vet each new match (place_names_rules.mjs vet): a town inside the unit named for the place gives way to the unit; a
// museum, camp, former country or far item is dropped
const vetting = todo.filter((c) => chosen.get(c.id) && !PINNED[c.id]);
if (vetting.length) {
  const ids = [...new Set(vetting.map((c) => chosen.get(c.id)))];
  const F = await factsOf(ids, console.log);
  const U = await unitLabels(F, vetting.map((c) => [chosen.get(c.id), c]), console.log);
  const swapped = [];
  for (const c of vetting) {
    const q = chosen.get(c.id);
    const faults = vet(q, c, names(L.get(q) ?? {}, CC, c.name), F.get(q), U);
    if (!faults.length) continue;
    const swap = faults.find((f) => f.better);
    chosen.set(c.id, swap ? swap.better : null);
    if (swap) swapped.push(swap.better);
    log.push(`${c.id}\t${c.name}\t${swap?.better ?? "-"}\t${q} refused: ${faults.map((f) => f.fault + (f.detail ? ` ${f.detail}` : "")).join("; ")}`);
  }
  for (const [q, v] of await labels(swapped.filter((q) => !L.has(q)))) L.set(q, v);
}

const out = {};
let matched = 0;
for (const c of cities) {
  const q = qOf(c);
  if (q && (chosen.has(c.id) || relabel)) {
    const n = names(L.get(q) ?? {}, CC, c.name);
    out[c.id] = { q, ...Object.fromEntries(Object.entries(n).filter(([, v]) => v)) };
  } else out[c.id] = q ? old[c.id] : { q: null };
  if (HAND_LABELS[c.id]) out[c.id] = { ...out[c.id], ...HAND_LABELS[c.id] };
  if (q) matched++;
}

const tmp = OUT + ".tmp";
writeFileSync(tmp, JSON.stringify(out, null, 0).replace(/\},"/g, '},\n"').replace(/^\{/, "{\n").replace(/\}$/, "\n}") + "\n");
renameSync(tmp, OUT);
const count = (k) => Object.values(out).filter((e) => e[k]).length;
console.log(`${cities.length} cities: ${matched} matched to an item; ja ${count("ja")}, zh-Hans ${count("zh-Hans")}, zh-TW ${count("zh-TW")}`);
if (log.length) {
  const logPath = opt("--log");
  if (logPath) writeFileSync(logPath, log.join("\n") + "\n");
  for (const l of log) if (l.split("\t")[2] === "-") console.log("no match: " + l);
}
