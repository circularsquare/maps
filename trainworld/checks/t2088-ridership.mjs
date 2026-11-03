// T-2088: a standing check of demand on real networks against real ridership. Information, never a
// target (CLAUDE.md "Demand is simple on purpose"): a city outside 0.5-2x of the table is a NOTE, not a FAIL.
//   node trainworld/checks/t2088-ridership.mjs [url] [city ...]     (from the maps root)
// `jpn7` as a city name (or osa ngo fuk spk hij tyo7 sdj7) runs the jpn7 save instead, T-2093: see "the jpn7 run" below.
// It takes about 15-25 minutes and 2.5 GB a worker; the seven-city run is unchanged.
// Each `data/packs/real/<id>.save` (nyc lon par tyo seo tor sdj; name some to run fewer) is imported into
// a `?new=1&opened=all&demandWorkers=2&paused=1` game, waited to the weekday's demand final, and read:
// the rail trips a weekday (`view.railTripsPerDay`, a journey counted once), the stations, the network's
// km, and the metro lines' share. Set beside `research/ridership.csv` (T-2086) rows of the same city.
// With no url it starts its own Vite (port T2088_PORT or 6188). Headless Chrome, one throwaway profile a
// city. About 10 minutes for all seven (Tokyo, New York, London and Paris take most). Writes
// artifacts/trainworld/t2088/<id>.json and summary.json, and prints the table.
//
// How the two sides are counted (research/ridership.md "The table"):
//  - Game trips = railTripsPerDay: whole journeys, so it sets against the table's *entries* (and linked).
//  - Real total = the city's rows summed as listed ("listed"), and "entries-equivalent" ("eq"): unlinked
//    metro, light rail and tram rows divided by UNLINKED_PER_ENTRY, since unlinked runs 1.3-1.9x entries
//    (commuter rail rows are left as listed: a railway trip is one boarding). The ratio uses "eq".
//    Tokyo's rows are MLIT S12 operator entries, where a change between operators counts twice
//    (T-1742: 42.7M is about 1.25x the journeys): left as listed, so Tokyo's ratio is a floor.
//  - Population is the CSV's metro_pop_m (Demographia's built-up area) on both sides.
//  - Riders per km: the city's metro system only, because it is the one row the table gives a length
//    for in every city. Game: the metro lines' boardings (against an unlinked row) or those boardings
//    turned into journeys at the network's journeys-per-boarding (against an entries or linked row),
//    over the km of track those lines use (shared track once, parallel edges within ~30 m once).
//    Real: the metro row's weekday over its length_km.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { MAPS, ARTIFACTS, launchChrome, importSave, demandFinal, startVite } from './lib.mjs';

const UNLINKED_PER_ENTRY = 1.5;
const BAND = [0.5, 2];
const OUT = join(ARTIFACTS, 't2088');
const WORKERS = process.env.WORKERS ?? '2';

// ---- the table
function parseCsv(text) {
  const rows = [];
  let row = [], cell = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (c === '"') q = false; else cell += c;
    } else if (c === '"') q = true;
    else if (c === ',') { row.push(cell); cell = ''; }
    else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; row.push(cell); cell = ''; if (row.length > 1) rows.push(row); row = []; }
    else cell += c;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  const head = rows.shift();
  return rows.map((r) => Object.fromEntries(head.map((h, i) => [h, r[i] ?? ''])));
}
const num = (s) => (s === '' || s === undefined ? null : Number(s));

// ---- which real network, which csv rows, which game lines are the metro
const re = (r) => (n) => r.test(n);
const inSet = (...xs) => (n) => xs.includes(n);
const TYO_METRO = ['銀座線', '丸ノ内線（１）', '丸ノ内線（２）', '日比谷線', '東西線', '千代田線', '有楽町線', '半蔵門線', '南北線', '副都心線', '大江戸線', '浅草線', '三田線'];
const CITIES = {
  nyc: { csvCity: 'New York', metroRow: 'New York City Subway', metro: re(/^([1-7]|[A-Z]|[67]X|FX)$/) },
  lon: { csvCity: 'London', metroRow: 'London Underground', metro: inSet('Northern', 'Central', 'Victoria', 'Jubilee', 'Piccadilly', 'District', 'Metropolitan', 'Bakerloo', 'Circle', 'Waterloo & City', 'Hammersmith & City') },
  par: { csvCity: 'Paris', metroRow: 'Paris Metro', metro: re(/^\d+B?$/) },
  tyo: { csvCity: 'Tokyo', metroRow: 'Tokyo Metro and Toei subways within 50 km', metro: inSet(...TYO_METRO) },
  seo: { csvCity: 'Seoul', metroRow: 'Seoul Metro lines 1-8', metro: re(/^Line \d$/) },
  tor: { csvCity: 'Toronto', metroRow: 'TTC subway', metro: inSet('1', '2', '4') },
  sdj: { csvCity: 'Sendai', metroRow: null, metro: re(/南北線|東西線/) },
};

// ---- T-2093: the cities of the jpn7 save (all Japan's lines cut to the seven Japanese city packs)
// They have no save of their own, and the CSV's Japanese rows are MLIT S12 station sums inside a circle round the
// city's main station. So one jpn7 run is read station by station and each city's stations inside the same circle
// (same centre, same radius, haversine on the stations' lng/lat) are summed, then set against the CSV rows. The
// game's "entries" at a station are its boardings minus the riders changing line there (a journey once, as
// railTripsPerDay is); S12's are each operator's gate entries, so a change between operators counts twice in S12 and
// the ratio is a floor, as Tokyo's is. "boardings" counts every change too, the ceiling. S12 is an average day (all
// seven days): the table's main column is the weekday like the other cities, and the average day's is beside it.
// `tyo7` and `sdj7` are Tokyo and Sendai read out of the same run, to compare with their own saves' rows.
const JPN7_SAVE = 'trainworld/data/work/realnet/jpn7_real.save';
const JPN = {
  osa: { csvCity: 'Osaka', circle: { lng: 135.4959, lat: 34.7025, km: 40 } },
  ngo: { csvCity: 'Nagoya', circle: { lng: 136.8815, lat: 35.1709, km: 30 } },
  fuk: { csvCity: 'Fukuoka', circle: { lng: 130.4208, lat: 33.5902, km: 20 } },
  spk: { csvCity: 'Sapporo', circle: { lng: 141.3508, lat: 43.0687, km: 20 } },
  hij: { csvCity: 'Hiroshima', circle: { lng: 132.4753, lat: 34.3976, km: 20 } },
  tyo7: { csvCity: 'Tokyo', circle: { lng: 139.7671, lat: 35.6812, km: 50 } },
  sdj7: { csvCity: 'Sendai', circle: { lng: 140.8819, lat: 38.2601, km: 20 } },
};
const S12_FILE = 'riders/japanriders/data/S12-22/UTF-8/S12-22_NumberOfPassengers.geojson';
const S12_FIELD = 'S12_041'; // FY2019 boardings plus alightings, an average day; halved it is entries

const hav = (lng1, lat1, lng2, lat2) => {
  const r = Math.PI / 180, dp = (lat2 - lat1) * r, dl = (lng2 - lng1) * r;
  const a = Math.sin(dp / 2) ** 2 + Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(dl / 2) ** 2;
  return 2 * 6371.0088 * Math.asin(Math.sqrt(a));
};

/** [lng, lat, entries] of every S12 station line that has a FY2019 count (entries = the count halved) */
async function readS12() {
  const d = JSON.parse(await readFile(join(MAPS, S12_FILE), 'utf8'));
  const out = [];
  for (const f of d.features) {
    const v = f.properties[S12_FIELD];
    if (!(v > 0)) continue;
    const cs = f.geometry.coordinates;
    out.push([cs.reduce((s, c) => s + c[0], 0) / cs.length, cs.reduce((s, c) => s + c[1], 0) / cs.length, v / 2]);
  }
  return out;
}

function compareJapan(id, cfg, rows, g, s12) {
  const { lng, lat, km } = cfg.circle;
  const inside = (r) => hav(lng, lat, r[1], r[2]) <= km;
  const sum = (list) => {
    const mine = (list ?? []).filter(inside);
    return { stations: mine.length, entries: mine.reduce((s, r) => s + r[3] - r[4], 0), boardings: mine.reduce((s, r) => s + r[3], 0) };
  };
  const wd = sum(g.stationRows), av = sum(g.stationRowsAverage);
  // the S12 stations in the circle that have a game station within about 1 km: the rest are outside the pack's cut
  const cell = (x, y) => Math.round(x * 100) * 100003 + Math.round(y * 100);
  const have = new Set(g.stationRows.map((r) => cell(r[1], r[2])));
  let s12In = 0, s12Matched = 0, s12Lines = 0;
  for (const [x, y, e] of s12) {
    if (hav(lng, lat, x, y) > km) continue;
    s12In += e; s12Lines++;
    let hit = false;
    for (let dx = -1; dx <= 1 && !hit; dx++) for (let dy = -1; dy <= 1 && !hit; dy++) hit = have.has(cell(x + dx * 0.01, y + dy * 0.01));
    if (hit) s12Matched += e;
  }
  const cityRows = rows.filter((r) => r.city === cfg.csvCity);
  const pop = num(cityRows[0].metro_pop_m);
  const listed = cityRows.reduce((s, r) => s + (num(r.weekday_k) ?? 0), 0) / 1000;
  const out = {
    id, city: cfg.csvCity + (id.endsWith('7') ? ' (jpn7 run)' : ''), popM: pop, day: g.day, round: g.round, final: g.final, solveS: g.solveS,
    circle: cfg.circle,
    rows: cityRows.map((r) => ({ system: r.system, mode: r.mode, basis: r.count_basis || 'not stated', weekday_k: num(r.weekday_k), length_km: null, stations: null })),
    unlinkedShare: 0,
    game: { railTripsM: wd.entries / 1e6, perResident: wd.entries / 1e6 / pop, perResidentAverage: av.entries / 1e6 / pop, perResidentBoardings: wd.boardings / 1e6 / pop,
      stations: wd.stations, entriesAverageM: av.entries / 1e6, boardingsM: wd.boardings / 1e6, trackKm: null, metro: { perKm: null } },
    real: { listedM: listed, eqM: listed, perResidentListed: listed / pop, perResident: listed / pop, s12InCircleM: s12In / 1e6, s12Lines, metro: null },
    s12: { inCircleM: s12In / 1e6, matchedShare: s12In ? s12Matched / s12In : null },
  };
  out.ratio = { trips: out.game.perResident / out.real.perResident, tripsAverage: out.game.perResidentAverage / out.real.perResident,
    tripsBoardings: out.game.perResidentBoardings / out.real.perResident, perKm: null };
  out.notes = [];
  for (const [what, r] of [['station entries per resident, weekday', out.ratio.trips], ['station entries per resident, average day', out.ratio.tripsAverage]]) {
    if (r < BAND[0] || r > BAND[1]) out.notes.push(`NOTE ${id}: ${what} ${r.toFixed(2)}x the table (${r > 1 ? 'game high' : 'game low'}), outside ${BAND[0]}-${BAND[1]}x`);
  }
  return out;
}

// ---- the game side: one page a city
const pageUrl = (base) => `${base}${base.includes('?') ? '&' : '?'}debug=1&new=1&opened=all&demandWorkers=${WORKERS}&paused=1`;

/** `opt.save`: a save path from the maps root instead of data/packs/real/<id>.save; `opt.stations`: also return every
 * station's lng, lat and boardings (weekday and average day), for the circle sums (T-2093), and skip the edge list */
async function readGame(base, id, log, opt = {}) {
  const { ctx, close } = await launchChrome({ prefix: `tw-t2088-${id}-` });
  try {
    const page = ctx.pages()[0] ?? await ctx.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push('pageerror: ' + e.message.slice(0, 200)));
    page.on('console', (m) => { if (m.type() === 'error' && !/favicon|openfreemap|AJAXError|^\w{1,3}$/.test(m.text())) errors.push(m.text().slice(0, 200)); });
    // the weekday: the shown day is the average day unless told (smoke does the same)
    await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch { /* none */ } });
    await page.goto(pageUrl(base));
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
    await importSave(page, await readFile(join(MAPS, opt.save ?? `trainworld/data/packs/real/${id}.save`)));
    await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 900000 });
    const t = Date.now();
    log(`${id}: save in, waiting for demand`);
    await demandFinal(page, { timeout: 3 * 3600000, polling: 2000, lines: 3 });
    const solveS = (Date.now() - t) / 1000;
    const g = await page.evaluate((withStations) => {
      const w = window.tw.world.value, v = window.twDemand.view(), s = window.twDemand.log.at(-1);
      // [id, lng, lat, boardings, transfers] a station, for the weekday view and the average day's (day 6)
      const stationsOf = (view) => view ? w.save.stations.map((st) => { const d = view.stations.get(st.id); return [st.id, st.lng, st.lat, d?.boardings ?? 0, d?.transfers ?? 0]; }) : null;
      const avg = withStations ? window.twDemand.days()[6]?.view : null;
      const lines = w.save.lines.map((l) => {
        const d = v.lines.get(l.id), st = w.lineStats[l.id];
        const n = l.stops.length;
        return { id: l.id, name: l.name, stops: n, km: st?.stopS?.[0] ? (st.stopS[0][n - 1] - st.stopS[0][0]) / 1000 : 0, riders: d?.ridersPerDay ?? 0 };
      });
      const node = (i) => w.nodes.get(i);
      const edges = withStations ? [] : w.edges.filter((e) => e.built && !e.off).map((e) => {
        const a = node(e.a), b = node(e.b);
        return a && b ? [a.x, a.y, b.x, b.y, e.lengthM, a.lat, e.lines ?? []] : null;
      }).filter(Boolean);
      return {
        day: v.day, version: v.version, final: v.final, round: v.round, railTripsPerDay: v.railTripsPerDay, tripsPerDay: v.tripsPerDay, split: v.split,
        status: String(window.twDemand.client.status), cities: s?.cities, stations: w.save.stations.length, lines, edges,
        ...(withStations ? { stationRows: stationsOf(v), stationRowsAverage: stationsOf(avg), averageRailTrips: avg?.railTripsPerDay ?? null } : {}),
      };
    }, !!opt.stations);
    return { ...g, solveS, errors };
  } finally {
    await close();
  }
}

/** km of track in `edges` ([ax, ay, bx, by, lengthM, lat, lines]) with edges lying within ~30 m of an earlier edge counted once */
function uniqueKm(edges) {
  const claim = new Map();
  let km = 0;
  const key = (ix, iy) => ix * 4194304 + iy;
  edges.forEach((e, idx) => {
    const [ax, ay, bx, by, L, lat] = e;
    const k = 1 / Math.cos(lat * Math.PI / 180); // web mercator metres per ground metre
    const cell = 20 * k, end = 15;
    const steps = Math.max(2, Math.ceil(L / 10));
    let counted = 0, dup = 0;
    const mine = [];
    for (let s = 0; s <= steps; s++) {
      const f = s / steps, x = ax + (bx - ax) * f, y = ay + (by - ay) * f;
      const ix = Math.floor(x / cell), iy = Math.floor(y / cell);
      mine.push([ix, iy]);
      if (f * L < end || (1 - f) * L < end) continue;
      counted++;
      let hit = false;
      for (let dx = -1; dx <= 1 && !hit; dx++) for (let dy = -1; dy <= 1 && !hit; dy++) { const c = claim.get(key(ix + dx, iy + dy)); if (c !== undefined && c !== idx) hit = true; }
      if (hit) dup++;
    }
    for (const [ix, iy] of mine) if (!claim.has(key(ix, iy))) claim.set(key(ix, iy), idx);
    km += (L / 1000) * (counted ? 1 - dup / counted : 1);
  });
  return km;
}

// ---- both sides together
function compare(id, cfg, rows, g) {
  const cityRows = rows.filter((r) => r.city === cfg.csvCity);
  const isUnlinkedMetro = (r) => r.count_basis === 'unlinked' && ['metro', 'light rail', 'tram'].includes(r.mode);
  let listed = 0, eq = 0, unlinkedK = 0, km = 0, kmRiders = 0, stations = 0;
  for (const r of cityRows) {
    const w = num(r.weekday_k) ?? 0;
    listed += w;
    eq += isUnlinkedMetro(r) ? w / UNLINKED_PER_ENTRY : w;
    if (r.count_basis === 'unlinked') unlinkedK += w;
    if (num(r.length_km)) { km += num(r.length_km); kmRiders += w; }
    stations += num(r.stations) ?? 0;
  }
  const pop = num(cityRows[0].metro_pop_m);
  const realTrips = { listedM: listed / 1000, eqM: eq / 1000 };
  const gameTripsM = g.railTripsPerDay / 1e6;
  const out = {
    id, city: cfg.csvCity, popM: pop, day: g.day, round: g.round, final: g.final, solveS: g.solveS,
    rows: cityRows.map((r) => ({ system: r.system, mode: r.mode, basis: r.count_basis || 'not stated', weekday_k: num(r.weekday_k), length_km: num(r.length_km), stations: num(r.stations) })),
    unlinkedShare: listed ? unlinkedK / listed : 0,
    game: { railTripsM: gameTripsM, perResident: gameTripsM / pop, stations: g.stations, lines: g.lines.length, boardingsM: g.lines.reduce((s, l) => s + l.riders, 0) / 1e6 },
    real: { listedM: realTrips.listedM, eqM: realTrips.eqM, perResidentListed: realTrips.listedM / pop, perResident: realTrips.eqM / pop, stations },
  };
  out.ratio = { trips: out.game.perResident / out.real.perResident, tripsListed: out.game.perResident / out.real.perResidentListed };
  // riders per km on the metro system
  const boardAll = out.game.boardingsM * 1e6;
  const journeysPerBoarding = boardAll ? g.railTripsPerDay / boardAll : 0;
  out.game.journeysPerBoarding = journeysPerBoarding;
  const metroIds = new Set(g.lines.filter((l) => cfg.metro(l.name ?? '')).map((l) => l.id));
  const metroEdges = g.edges.filter((e) => e[6].some((x) => metroIds.has(String(x))));
  const metroBoard = g.lines.filter((l) => metroIds.has(l.id)).reduce((s, l) => s + l.riders, 0);
  out.game.metro = { lines: metroIds.size, boardings: metroBoard, journeys: metroBoard * journeysPerBoarding, km: uniqueKm(metroEdges), sumLineKm: g.lines.filter((l) => metroIds.has(l.id)).reduce((s, l) => s + l.km, 0) };
  // an unlinked real row counts every boarding, so it sets against the game's line boardings; an entries or linked row against journeys
  const mr = cfg.metroRow && cityRows.find((r) => r.system === cfg.metroRow);
  out.game.metro.basis = mr?.count_basis === 'unlinked' ? 'boardings' : 'journeys';
  out.game.metro.perKm = out.game.metro.km ? out.game.metro[out.game.metro.basis] / out.game.metro.km : null;
  out.game.trackKm = uniqueKm(g.edges);
  out.game.edgeKm = g.edges.reduce((s, e) => s + e[4], 0) / 1000;
  out.game.perKm = out.game.trackKm ? g.railTripsPerDay / out.game.trackKm : null;
  out.real.metro = mr ? { system: mr.system, weekday_k: num(mr.weekday_k), length_km: num(mr.length_km), basis: mr.count_basis, perKm: num(mr.length_km) ? num(mr.weekday_k) * 1000 / num(mr.length_km) : null } : null;
  out.real.perKmAll = km ? kmRiders * 1000 / km : null;
  out.ratio.perKm = out.real.metro?.perKm ? out.game.metro.perKm / out.real.metro.perKm : null;
  out.notes = [];
  for (const [what, r] of [['rail trips per resident', out.ratio.trips], ['riders per km', out.ratio.perKm]]) {
    if (r !== null && (r < BAND[0] || r > BAND[1])) out.notes.push(`NOTE ${id}: ${what} ${r.toFixed(2)}x the table (${r > 1 ? 'game high' : 'game low'}), outside ${BAND[0]}-${BAND[1]}x`);
  }
  return out;
}

const f2 = (x) => (x === null || x === undefined ? '-' : x.toFixed(2));
const f0 = (x) => (x === null || x === undefined ? '-' : Math.round(x).toLocaleString('en-US'));
function table(results) {
  const head = ['city', 'game trips/res', 'real', 'ratio', 'game riders/km', 'real', 'ratio', 'stations', 'track km', 'unlinked'];
  const lines = [head, head.map(() => '---')];
  for (const r of results) {
    lines.push([r.city, f2(r.game.perResident), f2(r.real.perResident) + (r.unlinkedShare > 0.5 ? '*' : ''), f2(r.ratio.trips), f0(r.game.metro.perKm), f0(r.real.metro?.perKm), f2(r.ratio.perKm), f0(r.game.stations), f0(r.game.trackKm), `${(100 * r.unlinkedShare).toFixed(0)}%`]);
  }
  return lines.map((l) => '| ' + l.join(' | ') + ' |').join('\n');
}

async function main() {
  const args = process.argv.slice(2);
  const url = args.find((a) => /^https?:/.test(a));
  const ids = args.filter((a) => !/^https?:/.test(a));
  // `jpn7` stands for the jpn7 run's cities (T-2093); with no names the seven cities with saves run, as before
  const named = ids.flatMap((a) => (a === 'jpn7' ? Object.keys(JPN) : [a]));
  const todo = (named.length ? named : Object.keys(CITIES)).filter((id) => !JPN[id]);
  const jpn = named.filter((id) => JPN[id]);
  for (const id of [...todo, ...jpn]) if (!CITIES[id] && !JPN[id]) throw new Error(`unknown city ${id}; known: ${Object.keys(CITIES).join(' ')} jpn7 (= ${Object.keys(JPN).join(' ')})`);
  const rows = parseCsv(await readFile(join(MAPS, 'trainworld/research/ridership.csv'), 'utf8'));
  await mkdir(OUT, { recursive: true });
  const t0 = Date.now();
  const log = (s) => console.log(`${((Date.now() - t0) / 1000).toFixed(0).padStart(5)} s  ${s}`);
  let vite = null;
  let base = url;
  if (!base) { vite = await startVite(Number(process.env.T2088_PORT ?? 6188)); base = vite.url; }
  const results = [], jpnResults = [];
  try {
    for (const id of todo) {
      try {
        const g = await readGame(base, id, log);
        if (g.status.startsWith('failed')) throw new Error(`demand ${g.status}`);
        const r = compare(id, CITIES[id], rows, g);
        r.errors = g.errors;
        r.at = new Date().toISOString();
        await writeFile(join(OUT, `${id}.json`), JSON.stringify(r, null, 1) + '\n');
        results.push(r);
        log(`${id}: ${f0(g.railTripsPerDay)} rail trips, final ${g.final} round ${g.round} in ${g.solveS.toFixed(0)} s${g.errors.length ? `, ${g.errors.length} console errors` : ''}`);
      } catch (e) {
        log(`${id}: FAILED ${e.message.split('\n')[0]}`);
      }
    }
    if (jpn.length) {
      try {
        const [g, s12] = [await readGame(base, 'jpn7', log, { save: JPN7_SAVE, stations: true }), await readS12()];
        if (g.status.startsWith('failed')) throw new Error(`demand ${g.status}`);
        await writeFile(join(OUT, 'jpn7-stations.json'), JSON.stringify({ at: new Date().toISOString(), day: g.day, railTripsPerDay: g.railTripsPerDay, averageRailTrips: g.averageRailTrips, columns: ['id', 'lng', 'lat', 'boardings', 'transfers'], weekday: g.stationRows, average: g.stationRowsAverage }) + '\n');
        log(`jpn7: ${f0(g.railTripsPerDay)} rail trips across ${g.stations} stations, final ${g.final} round ${g.round} in ${g.solveS.toFixed(0)} s${g.errors.length ? `, ${g.errors.length} console errors` : ''}`);
        for (const id of jpn) {
          const r = compareJapan(id, JPN[id], rows, g, s12);
          r.errors = g.errors;
          r.at = new Date().toISOString();
          await writeFile(join(OUT, `${id}.json`), JSON.stringify(r, null, 1) + '\n');
          jpnResults.push(r);
        }
      } catch (e) {
        log(`jpn7: FAILED ${e.message.split('\n')[0]}`);
      }
    }
  } finally {
    vite?.child.kill();
  }
  if (jpnResults.length) {
    await writeFile(join(OUT, 'summary-jpn7.json'), JSON.stringify({ at: new Date().toISOString(), band: BAND, cities: jpnResults.map(({ id, city, popM, circle, game, real, ratio, s12 }) => ({ id, city, popM, circle, game, real, ratio, s12 })) }, null, 1) + '\n');
    console.log('\njpn7 run: station entries inside the S12 circle (weekday; a journey once) against the CSV\'s S12 sums (an average day, a change between operators counted twice)');
    console.log(['city', 'game stations', 'game entries M', 'S12 M', 'ratio weekday', 'ratio average day', 'ratio boardings', 'S12 near a game station'].join(' | '));
    for (const r of jpnResults) console.log([r.city, r.game.stations, r.game.railTripsM.toFixed(2), r.real.listedM.toFixed(2), f2(r.ratio.trips), f2(r.ratio.tripsAverage), f2(r.ratio.tripsBoardings), `${(100 * r.s12.matchedShare).toFixed(0)}%`].join(' | '));
    console.log(table(jpnResults));
    for (const r of jpnResults) for (const n of r.notes) console.log(n);
  }
  if (!results.length) { if (!jpnResults.length) { console.log('no city finished'); process.exitCode = 1; } return; }
  await writeFile(join(OUT, 'summary.json'), JSON.stringify({ at: new Date().toISOString(), unlinkedPerEntry: UNLINKED_PER_ENTRY, band: BAND, cities: results.map(({ id, city, popM, game, real, ratio, unlinkedShare }) => ({ id, city, popM, game, real, ratio, unlinkedShare })) }, null, 1) + '\n');
  console.log('\nGame (weekday rail trips, a journey once) against research/ridership.csv; "real" on entries-equivalent counts, * = over half the real rows are unlinked');
  console.log(table(results));
  for (const r of results) for (const n of r.notes) console.log(n);
  if (!results.some((r) => r.notes.length)) console.log('every ratio inside the band');
  console.log(`wrote ${OUT}`);
}
main().catch((e) => { console.error(e); process.exit(1); });
