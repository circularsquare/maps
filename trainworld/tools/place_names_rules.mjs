// T-3450: what makes a Wikidata item the right one for a registry city or zone, shared by tools/place_names.mjs (which
// matches) and checks/place_names_match.mjs (which fails on a wrong match). docs/localization.md "City names".

/** A Wikidata item matches a registry city when one of its coordinates is this close to the city's centre. */
export const NEAR_KM = 60;
/** Only places count: human settlements and administrative units (not a stadium, an air base or a team). */
export const PLACE_CLASSES = "wd:Q486972 wd:Q56061";
/** Never a city's or zone's place, though Wikidata files some of them under settlements (the Auschwitz camp, alias
 * "Oświęcim", is one by subclass): museum, concentration camp. Not "building": Wikidata puts seaside resorts under it. */
export const NOT_PLACE_CLASSES = "wd:Q33506 wd:Q152081";
/** Places whose Japanese and Chinese names are written in hanzi or kanji, where two places of one Latin name can be
 * written differently (Shiga: 滋賀 the prefecture, 志賀 an old town inside it). */
const HAN_COUNTRIES = new Set(["JP", "CN", "TW", "HK", "MO", "KR"]);

/** Matches set by hand, where the rule picks a bigger unit of the same name over the city: Puebla's state (the city's
 * label is "Heroica Puebla de Zaragoza"), Puducherry's union territory, Siwan's district. */
export const PINNED = { pue: "Q125293", pud: "Q639421", maj: "Q2234098" };

/** Names set by hand where the metro's name is a region, not the city its centre lies in (the registry's "Osaka" is
 * Keihanshin, four prefectures, shown as Kansai; Hyogo's players read 大阪 as another prefecture). Wins over Wikidata. */
export const HAND_LABELS = {
  osa: { ja: "関西", "zh-Hans": "关西", "zh-TW": "關西" },
  dor: { ja: "ルール地方", "zh-Hans": "鲁尔区", "zh-TW": "魯爾區" },
  luo: { ja: "漯河", "zh-Hans": "漯河", "zh-TW": "漯河" }, // place_names_check.py cannot read 漯 (Luo), T-3324
  ram: { ja: "東エルサレム" }, // Wikidata's ja label came out as エルサレム, which reads as all of Jerusalem, T-3450
  // T-3756: zones whose item has no ja or zh label (katakana and Xinhua transliteration)
  ltx: { ja: "ラ・アスンタ", "zh-Hans": "拉阿松塔", "zh-TW": "拉阿松塔" },
  wae: { ja: "ワルネス" },
  vlt: { ja: "ビジャ・トゥナリ" },
  pca: { ja: "プカラニ" },
  anj: { ja: "ソコポ" },
  pcc: { ja: "ポコシ" },
};

const UA = "trainworld-l10n/1.0 (rail game city names; batch label fetch)";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function get(url, init = {}) {
  for (let attempt = 0; ; attempt++) {
    let r, json;
    try { // a dropped connection, or a query cut off by the endpoint's timeout (truncated JSON): ask again
      r = await fetch(url, { ...init, headers: { "User-Agent": UA, ...(init.headers ?? {}) } });
      if (r.ok) json = JSON.parse(await r.text());
    } catch (e) {
      if (attempt >= 5) throw e;
      await sleep(5000 * (attempt + 1));
      continue;
    }
    if (r.ok) return json;
    if ((r.status === 429 || r.status >= 500) && attempt < 5) {
      await sleep(Number(r.headers.get("retry-after")) * 1000 || 5000 * (attempt + 1));
      continue;
    }
    throw new Error(`${r.status} ${url.slice(0, 120)}`);
  }
}

export async function sparql(query) {
  const j = await get("https://query.wikidata.org/sparql", {
    method: "POST",
    headers: { Accept: "application/sparql-results+json", "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ query }).toString(),
  });
  await sleep(1500);
  return j.results.bindings;
}

export const qid = (b) => b.value.split("/").pop();

export const point = (b) => {
  const m = b.coord && /Point\(([-\d.eE]+) ([-\d.eE]+)\)/.exec(b.coord.value);
  return m ? [Number(m[1]), Number(m[2])] : null;
};

export function km([lon1, lat1], [lon2, lat2]) {
  const r = Math.PI / 180, a = Math.sin(((lat2 - lat1) * r) / 2) ** 2 + Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(((lon2 - lon1) * r) / 2) ** 2;
  return 12742 * Math.asin(Math.sqrt(a));
}

/** Whether a point is near enough to be the place: within NEAR_KM of its centre, or inside its box (a zone shown by a
 * town or state whose own point lies off the zone's centre: Vermont, Jefferson City). */
export const near = (p, c) => km(p, c.center) <= NEAR_KM || (c.box && p[0] >= c.box[0] && p[0] <= c.box[2] && p[1] >= c.box[1] && p[1] <= c.box[3]);

/** The same cleaning as app/src/game/zoneNames.ts cleanName, so the name searched is the name shown. */
export function cleanName(raw) {
  const s = raw.replace(/\s*\[[^\]]*\]/g, "").replace(/\s*\([^)]*\)\s*$/, "").replace(/ CC\b/g, "").replace(/\s+/g, " ").trim();
  return s || raw;
}

/** A name compared without case, accents or punctuation: "Đà Lạt" as "da lat", "Washington, D.C." as "washington dc". */
export const fold = (s) =>
  s.normalize("NFKD").replace(/\p{M}/gu, "").toLowerCase().replace(/đ/g, "d").replace(/ł/g, "l").replace(/ø/g, "o")
    .replace(/æ/g, "ae").replace(/ß/g, "ss").replace(/ı/g, "i").replace(/['’.]/g, "").replace(/[^\p{L}\p{N}]+/gu, " ").trim();

function editDistance(a, b) {
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    prev = cur;
  }
  return prev[b.length];
}

/** The texts a city or zone is matched by: the name the game shows, its part before a comma, "and" or " - ", and the
 * registry's own name only where it is a spelling of the shown one (Mahrajganj for Maharajganj). A zone renamed for its
 * main town (Sullivan County shown as Kingsport) is matched by the town, so Japanese and Chinese name what English does. */
export function textsOf(c) {
  const raw = cleanName(c.raw);
  const tight = (s) => fold(s).replace(/ /g, "");
  const texts = [c.name, c.name.split(/,| and | - /)[0].trim()];
  if (editDistance(tight(raw), tight(c.name)) <= 2) texts.push(raw);
  return [...new Set(texts)].filter(Boolean);
}

/** Whether a label is one of the texts or one followed by more words ("Shiga Prefecture" for Shiga), folded. */
export const labelFits = (label, texts) => {
  const l = fold(label);
  return texts.some((t) => { const f = fold(t); return l === f || l.startsWith(f + " "); });
};

/** Words a unit's English name carries beside the place it is named after ("Shiga Prefecture", "County of Kent"). */
const UNIT_WORDS = new Set("prefecture subprefecture county district province region state oblast krai governorate department regency municipality municipal city borough parish canton division of the metropolitan area autonomous special".split(" "));
const bareName = (s) => fold(s).split(" ").filter((w) => !UNIT_WORDS.has(w)).join(" ");
/** A Japanese or Chinese name without its unit word: 滋賀 for 滋賀県, 上川 for 上川総合振興局. */
const bareHan = (s) => {
  let t = (s ?? "").replace(/\s*[（(][^）)]*[）)]\s*$/, "").replace(/(\p{Script=Han}族)+自治(州|县|縣|区|區|旗)$/u, "").trim();
  for (let i = 0; i < 2; i++) t = t.replace(/(総合振興局|振興局|支庁|広域市|特別市|特别市|自治州|自治县|自治縣|自治区|地区|地區|[県郡市町村区區府道都州省县縣盟旗镇鎮乡鄉郷])$/u, "");
  return t;
};
const HAN = /^[\p{Script=Han}々ヶ]+$/u;
/** A town's or village's name: 岩手町, 志賀町, 某某镇. Smaller than any zone or registry city. */
const TOWN = /[町村镇鎮乡鄉郷]$/u;

const values = (ids) => ids.map((q) => "wd:" + q).join(" ");

/** Each item's English (and language-neutral) labels and aliases: { q: [name] }, 200 items a query. */
export async function namesOf(ids, log = () => {}) {
  const out = new Map(ids.map((q) => [q, []]));
  for (let i = 0; i < ids.length; i += 200) {
    const batch = ids.slice(i, i + 200);
    for (const b of await sparql(`SELECT ?item ?n WHERE { VALUES ?item { ${values(batch)} } ?item rdfs:label|skos:altLabel ?n FILTER(LANG(?n) IN ("en", "mul")) }`)) out.get(qid(b.item))?.push(b.n.value);
    log(`names ${Math.min(i + 200, ids.length)}/${ids.length}`);
  }
  return out;
}

/** What vet() needs about each item: its names, coordinates, kind, and the units it lies in (P131) whose name could be
 * the place's, with their Japanese and Chinese labels. 150 items a query. */
export async function factsOf(ids, log = () => {}) {
  const names = await namesOf(ids, log);
  const out = new Map(ids.map((q) => [q, { names: names.get(q) ?? [], coords: [], dissolved: false, historical: false, notPlace: false, units: {} }]));
  for (let i = 0; i < ids.length; i += 150) {
    const batch = ids.slice(i, i + 150);
    const rows = await sparql(`SELECT ?item ?coord ?gone ?hist ?bad WHERE { VALUES ?item { ${values(batch)} }
  OPTIONAL { ?item wdt:P625 ?coord }
  BIND(EXISTS { ?item wdt:P576 [] } AS ?gone)
  BIND(EXISTS { ?item wdt:P31/wdt:P279* wd:Q3024240 } AS ?hist)
  BIND(EXISTS { ?item wdt:P31/wdt:P279* ?no VALUES ?no { ${NOT_PLACE_CLASSES} } } AS ?bad) }`);
    for (const b of rows) {
      const f = out.get(qid(b.item));
      const p = point(b);
      if (p) f.coords.push(p);
      f.dissolved = b.gone.value === "true";
      f.historical = b.hist.value === "true";
      f.notPlace = b.bad.value === "true";
    }
    const units = await sparql(`SELECT ?item ?unit ?n ?cap WHERE { VALUES ?item { ${values(batch)} } ?item wdt:P131+ ?unit .
  ?unit rdfs:label ?n FILTER(LANG(?n) IN ("en", "mul")) OPTIONAL { ?unit wdt:P36 ?cap } }`);
    for (const b of units) {
      const u = (out.get(qid(b.item)).units[qid(b.unit)] ??= { names: [], caps: [] });
      if (!u.names.includes(b.n.value)) u.names.push(b.n.value);
      if (b.cap && !u.caps.includes(qid(b.cap))) u.caps.push(qid(b.cap));
    }
    log(`facts ${Math.min(i + 150, ids.length)}/${ids.length}`);
  }
  return out;
}

const namesake = (unit, c) => { const want = new Set(textsOf(c).map(bareName).filter(Boolean)); return unit.names.some((n) => want.has(bareName(n))); };

/** Japanese and Chinese labels and sitelinks of the units vet() compares with: those named for the place. places: [[q, c]]. */
export async function unitLabels(facts, places, log = () => {}) {
  const want = new Set();
  for (const [q, c] of places) for (const [u, unit] of Object.entries(facts.get(q)?.units ?? {})) if (namesake(unit, c)) want.add(u);
  const ids = [...want], out = new Map();
  for (let i = 0; i < ids.length; i += 200) {
    const batch = ids.slice(i, i + 200);
    const rows = await sparql(`SELECT ?item ?label ?links WHERE { VALUES ?item { ${values(batch)} } OPTIONAL { ?item wikibase:sitelinks ?links }
  OPTIONAL { ?item rdfs:label ?label FILTER(LANG(?label) IN ("ja", "zh-hans", "zh-cn", "zh", "zh-tw", "zh-hant")) } }`);
    for (const b of rows) {
      const e = out.get(qid(b.item)) ?? { links: Number(b.links?.value ?? 0) };
      if (b.label) e[b.label["xml:lang"]] = b.label.value;
      out.set(qid(b.item), e);
    }
    log(`units ${Math.min(i + 200, ids.length)}/${ids.length}`);
  }
  return out;
}

/**
 * Why an item is the wrong one for a city or zone, or [] when it is fine. c: { id, name (as shown), raw, center, country };
 * box?; shown: the entry's { ja, "zh-Hans", "zh-TW" }; f: factsOf()'s entry; U: unitLabels(). A namesake fault carries
 * `better`, the item to take instead.
 *  - far: no coordinate within NEAR_KM of the centre.
 *  - name: none of its English names is the shown name (a zone renamed since it was matched, Kingsport's Sullivan County).
 *  - not a place: a museum, building or camp.
 *  - historical: a former country (Kucha for Kuqa, the Kingdom of Montenegro for Crna Gora).
 *  - namesake: it lies inside a unit named for the place and is not that unit's seat, and its own name is a town's (岩手町
 *    for Iwate) or, where names are written in hanzi or kanji, spelt otherwise than the unit's (志賀 inside 滋賀県).
 */
export function vet(q, c, shown, f, U) {
  if (!f) return [{ fault: "no item" }];
  const out = [];
  if (!f.coords.some((p) => near(p, c))) out.push({ fault: "far" });
  if (!f.names.some((n) => labelFits(n, textsOf(c)))) out.push({ fault: "name" });
  if (f.notPlace) out.push({ fault: "not a place" });
  if (f.dissolved && f.historical) out.push({ fault: "historical" });
  const mine = [shown?.ja, shown?.["zh-Hans"], shown?.["zh-TW"]];
  const town = mine.some((l) => l && TOWN.test(l));
  let better = null;
  for (const [u, unit] of Object.entries(f.units)) {
    if (u === q || unit.caps.includes(q) || !namesake(unit, c)) continue;
    const L = U.get(u) ?? {};
    const theirs = [L.ja, L["zh-hans"] ?? L["zh-cn"] ?? L.zh, L["zh-tw"] ?? L["zh-hant"]];
    const spelt = HAN_COUNTRIES.has(c.country) && mine.some((m, i) => {
      const a = bareHan(m), b = bareHan(theirs[i]);
      return a && b && HAN.test(a) && HAN.test(b) && a !== b;
    });
    if ((town || spelt) && (!better || (L.links ?? 0) > better.links)) better = { id: u, links: L.links ?? 0, why: town ? "a town's name" : "spelt otherwise", unit: unit.names[0] };
  }
  if (better) out.push({ fault: "namesake", better: better.id, detail: `inside ${better.unit} (${better.id}), ${better.why}` });
  return out;
}
