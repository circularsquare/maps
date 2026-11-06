// T-2668: the options on the measured test lines. Riders depend only on the perceived fare (fare $ x
// minutes a $), so one sweep per line gives riders at any perceived multiple k of today's default;
// fare income = riders(k) x today's mean fare x the option's fare multiple.
import fs from "fs";
const dir = "C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/b35f881f-a7d6-4f9f-9c17-4b4b2ffbdf21/scratchpad/farescale/";
const T = JSON.parse(fs.readFileSync("C:/Users/anita/projects/maps/trainworld/app/src/game/countryIncome.json", "utf8"));
const curve = (tag, base, km) => {
  const pts = [];
  for (const f of fs.readdirSync(dir + "out")) {
    const m = f.match(new RegExp(`^${tag}_([0-9.]+)\\.json$`));
    if (!m) continue;
    const j = JSON.parse(fs.readFileSync(dir + "out/" + f, "utf8"));
    const rail = j.day.commute[1] + j.day.everyday[1], pkm = j.lines.pkm.reduce((a, b) => a + b, 0);
    pts.push({ k: +m[1], rail, pkm });
  }
  pts.sort((a, b) => a.k - b.k);
  const at = (k, key) => { // log-log interpolation
    let i = 0; while (i < pts.length - 2 && pts[i + 1].k < k) i++;
    const a = pts[i], b = pts[i + 1], t = (Math.log(k) - Math.log(a.k)) / (Math.log(b.k) - Math.log(a.k));
    return Math.exp(Math.log(a[key]) + t * (Math.log(b[key]) - Math.log(a[key])));
  };
  const inc = (k, mult) => mult * (base * at(k, "rail") + km * at(k, "pkm"));
  // income-max multiple at today's weights (k = mult)
  let best = 0, bestK = 1;
  for (let k = 0.3; k <= 6; k *= 1.01) { const v = inc(k, k); if (v > best) { best = v; bestK = k; } }
  return { tag, at, inc, bestK, best, pts };
};
const tij = curve("tij", 0.65, 0.055), tijbig = curve("tijbig", 0.65, 0.055), snd = curve("snd", 2.5, 0.2);
for (const c of [tij, tijbig, snd]) {
  const d = c.inc(1, 1);
  const e = Math.log(c.at(1.4, "rail") / c.at(1, "rail")) / Math.log(1.4);
  console.log(`${c.tag}: default riders ${c.at(1, "rail").toFixed(0)}, income ${d.toFixed(0)}; income-max at ${c.bestK.toFixed(2)}x, ${c.best.toFixed(0)} (default earns ${(100 * d / c.best).toFixed(0)}%, carries ${(c.at(1, "rail") / c.at(c.bestK, "rail")).toFixed(2)}x the riders); elasticity 1->1.4x ${e.toFixed(2)}`);
}

// costs, per real day (the game's day is 100x both sides)
const trainKm = (tph, km) => (tph[0] * 14 + tph[1] * 4) * 2 * km;
const LKM = 16.6, STATIONS = 15;
const running = (tph, cars, km) => trainKm(tph, km) * (4 + 2 * cars);
const upkeep = (km, st, perKm = 250) => perKm * km + 500 * st; // viaduct
const light = { tph: [12, 6, 0], cars: 4 }, heavy = { tph: [24, 12, 0], cars: 6 };
const costTij = (s) => running(s.tph, s.cars, LKM) + upkeep(LKM, STATIONS);
const costSnd = running(light.tph, light.cars, 15.7) + upkeep(15.7, 14);
console.log(`costs a real day: Tijuana light ${costTij(light).toFixed(0)} (running ${running(light.tph, light.cars, LKM).toFixed(0)}), heavy ${costTij(heavy).toFixed(0)}; San Diego light ${costSnd.toFixed(0)}`);

const G = T.usd, r = (c) => G[c] / G.US;
// options: fare exponent ef, value-of-time exponent ev, cost exponent ec (running + upkeep x r^ec; 0 = world price)
const options = [
  { id: "A", name: "today", ef: 0.7, ev: 0.7, ec: 0 },
  { id: "B", name: "fares ^0.5", ef: 0.5, ev: 0.7, ec: 0 },
  { id: "C", name: "value of time ^0.5", ef: 0.7, ev: 0.5, ec: 0 },
  { id: "D", name: "both ^0.5", ef: 0.5, ev: 0.5, ec: 0 },
  { id: "E", name: "costs ^0.7 (running + upkeep)", ef: 0.7, ev: 0.7, ec: 0.7 },
  { id: "F", name: "value of time ^0.5 + costs ^0.5", ef: 0.7, ev: 0.5, ec: 0.5 },
  { id: "G", name: "value of time ^0.6 + costs ^0.5", ef: 0.7, ev: 0.6, ec: 0.5 },
  { id: "H", name: "both ^0.5 + costs ^0.5", ef: 0.5, ev: 0.5, ec: 0.5 },
  { id: "I", name: "value of time ^0.6 + costs ^0.7", ef: 0.7, ev: 0.6, ec: 0.7 },
  { id: "J", name: "value of time ^0.5 + costs ^0.7", ef: 0.7, ev: 0.5, ec: 0.7 },
];
const rows = [];
const fmt = (v) => (v < 0 ? "-" : "") + "$" + Math.round(Math.abs(v) / 1000) + "k";
for (const o of options) {
  const rr = r("MX"), k = rr ** (o.ef - o.ev), fm = rr ** (o.ef - 0.7), cf = rr ** o.ec;
  const out = { ...o, k: +k.toFixed(3), fareMult: +fm.toFixed(3), costFactor: +cf.toFixed(3) };
  for (const [nm, c, s] of [["light", tij, light], ["heavy", tijbig, heavy]]) {
    const riders = c.at(k, "rail"), inc = c.inc(k, fm), cost = costTij(s) * cf;
    out[nm] = { riders: Math.round(riders), income: Math.round(inc), cost: Math.round(cost), profit: Math.round(inc - cost) };
  }
  // the default's place against the income max in Tijuana (light): income-max fare multiple of the option's default
  let best = 0, bm = 1;
  for (let m = 0.2; m <= 10; m *= 1.01) { const v = tij.inc(k * m, fm * m); if (v > best) { best = v; bm = m; } }
  out.tijMaxMult = +bm.toFixed(2); out.tijDefaultShare = +(out.light.income / best).toFixed(2);
  // per city: base fare, minutes of the base fare, break-even riders per car (4 cars, a 5.5 km trip)
  out.cities = {};
  for (const [city, cc] of [["Tijuana", "MX"], ["Mumbai", "IN"], ["Tokyo", "JP"], ["New York", "US"]]) {
    const x = r(cc), base = 2.5 * x ** o.ef, rate = 0.2 * x ** o.ef, w = 4.4 * x ** -o.ev;
    const perPkm = (base + rate * 5.5) / 5.5, carKm = 3 * x ** o.ec;
    const kk = x ** (o.ef - o.ev), curveC = cc === "US" || cc === "JP" ? snd : tij;
    const rid = curveC.at(kk, "rail") / curveC.at(1, "rail");
    const inc = (curveC.inc(kk, 1) / curveC.inc(1, 1)) * x ** (o.ef - 0.7);
    out.cities[city] = { base: +base.toFixed(2), baseMin: +(base * w).toFixed(1), breakEvenPerCar: +(carKm / perPkm).toFixed(1), riders: +rid.toFixed(2), income: +inc.toFixed(2) };
  }
  rows.push(out);
  console.log(`\n${o.id} ${o.name}: MX perceived x${k.toFixed(2)}, fare x${fm.toFixed(2)}, costs x${cf.toFixed(2)}`);
  console.log(`  Tijuana light: riders ${out.light.riders}, income ${fmt(out.light.income)}, costs ${fmt(out.light.cost)}, profit ${fmt(out.light.profit)}; heavy: riders ${out.heavy.riders}, income ${fmt(out.heavy.income)}, costs ${fmt(out.heavy.cost)}, profit ${fmt(out.heavy.profit)}; income-max at ${out.tijMaxMult}x the default (default earns ${Math.round(100 * out.tijDefaultShare)}%)`);
  for (const [c, v] of Object.entries(out.cities)) console.log(`  ${c}: base $${v.base}, ${v.baseMin} min, break-even ${v.breakEvenPerCar} riders a car, riders x${v.riders}, fare income x${v.income}`);
}
fs.writeFileSync(dir + "options.json", JSON.stringify(rows, null, 1));
