// Log-log fit of base short urban rail fare (US$) on GDP per head at market rates (countryIncome.json usd).
import fs from "fs";
const dir = "C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/b35f881f-a7d6-4f9f-9c17-4b4b2ffbdf21/scratchpad/farescale/";
const T = JSON.parse(fs.readFileSync("C:/Users/anita/projects/maps/trainworld/app/src/game/countryIncome.json", "utf8"));
const lines = fs.readFileSync(dir + "fares.csv", "utf8").trim().split(/\r?\n/).slice(1);
function parse(l) {
  const out = []; let cur = "", q = false;
  for (const ch of l) { if (ch === '"') q = !q; else if (ch === "," && !q) { out.push(cur); cur = ""; } else cur += ch; }
  out.push(cur); return out;
}
// Corrections from known figures (see notes): New York $3.00 from Jan 2026 (T-1521's table); Paris EUR 2.55 2026.
const fix = { "New York": 3.0, "Paris": 2.88 };
const rows = [];
for (const l of lines) {
  const f = parse(l);
  const [city, iso, mode, , , , , usd, , , conf] = f;
  if (!usd) continue;
  const g = T.usd[iso], p = T.gdp[iso];
  if (!g) { console.log("no gdp", iso); continue; }
  rows.push({ city, iso, mode, usd: fix[city] ?? +usd, g, p, weak: /weak/.test(conf), bus: /^(Tijuana|Johannesburg)$/.test(city), sub: /suburban/.test(mode) });
}
function ols(rs, key = "g") {
  const xs = rs.map((r) => Math.log(r[key])), ys = rs.map((r) => Math.log(r.usd));
  const n = xs.length, mx = xs.reduce((a, b) => a + b) / n, my = ys.reduce((a, b) => a + b) / n;
  let sxx = 0, sxy = 0; for (let i = 0; i < n; i++) { sxx += (xs[i] - mx) ** 2; sxy += (xs[i] - mx) * (ys[i] - my); }
  const b = sxy / sxx, a = my - b * mx;
  let sse = 0; for (let i = 0; i < n; i++) sse += (ys[i] - a - b * xs[i]) ** 2;
  const se = Math.sqrt(sse / (n - 2) / sxx), r2 = 1 - sse / ys.reduce((s, y) => s + (y - my) ** 2, 0);
  return { n, b, se, a, r2, rmse: Math.sqrt(sse / n) };
}
function boot(rs, key = "g", k = 4000) {
  let seed = 7; const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
  const bs = [];
  for (let i = 0; i < k; i++) { const s = rs.map(() => rs[Math.floor(rnd() * rs.length)]); bs.push(ols(s, key).b); }
  bs.sort((a, b) => a - b); return [bs[Math.floor(k * 0.05)], bs[Math.floor(k * 0.95)]];
}
// Theil-Sen (median of pairwise slopes), robust to outliers
function theil(rs, key = "g") {
  const sl = [];
  for (let i = 0; i < rs.length; i++) for (let j = i + 1; j < rs.length; j++) {
    const dx = Math.log(rs[j][key]) - Math.log(rs[i][key]); if (Math.abs(dx) < 0.05) continue;
    sl.push((Math.log(rs[j].usd) - Math.log(rs[i].usd)) / dx);
  }
  sl.sort((a, b) => a - b); return sl[Math.floor(sl.length / 2)];
}
const sets = {
  "all rows": rows,
  "rail only (no buses, no Mumbai suburban)": rows.filter((r) => !r.bus && !r.sub),
  "rail only, no Iran/Uzbekistan (exchange-rate outliers)": rows.filter((r) => !r.bus && !r.sub && !["IR", "UZ"].includes(r.iso)),
  "rail only, not weak": rows.filter((r) => !r.bus && !r.sub && !r.weak),
  "one per country (median), rail, no IR/UZ": (() => {
    const by = {}; for (const r of rows.filter((r) => !r.bus && !r.sub && !["IR", "UZ"].includes(r.iso))) (by[r.iso] ??= []).push(r);
    return Object.values(by).map((rs) => { const u = rs.map((r) => r.usd).sort((a, b) => a - b); return { ...rs[0], usd: u[Math.floor((u.length - 1) / 2)] }; });
  })(),
};
const res = {};
for (const [name, rs] of Object.entries(sets)) {
  const o = ols(rs), ci = boot(rs), ts = theil(rs), op = ols(rs, "p");
  res[name] = { ...o, ci, theil: ts, pppB: op.b, pppSe: op.se };
  console.log(`${name}: n=${o.n} b=${o.b.toFixed(3)} se=${o.se.toFixed(3)} 90%boot=[${ci[0].toFixed(2)},${ci[1].toFixed(2)}] theil=${ts.toFixed(3)} r2=${o.r2.toFixed(2)} rmse(log)=${o.rmse.toFixed(2)} | vs PPP gdp b=${op.b.toFixed(3)}`);
}
// Game's default against real, per row
const fitR = ols(sets["rail only, no Iran/Uzbekistan (exchange-rate outliers)"]);
const tenth = (v) => Math.min(19.9, Math.max(0.1, Math.round(v * 10) / 10));
const s = (c) => tenth((T.usd[c] / T.usd.IN) ** 0.7);
const out = [];
for (const r of rows) {
  const game = 2.5 * s(r.iso) / s("US");
  const fitv = Math.exp(fitR.a + fitR.b * Math.log(r.g));
  out.push({ city: r.city, iso: r.iso, gdp: r.g, real: r.usd, game: +game.toFixed(2), ratio: +(r.usd / game).toFixed(2), fit: +fitv.toFixed(2) });
}
out.sort((a, b) => a.gdp - b.gdp);
for (const o of out) console.log(`${o.city.padEnd(40)} ${o.iso} gdp ${o.gdp} real ${o.real} game ${o.game} real/game ${o.ratio} fit ${o.fit}`);
fs.writeFileSync(dir + "fit.json", JSON.stringify({ res, out, fit: fitR }, null, 1));
