// T-1521 round 3: market rows for the corridor runs, with what the app adds to each row (`fw`, `fg`,
// `cu`, game/cities.ts `marketWorld`) under each variant, plus `af`, the plane fare's factor (the
// run's crate copy only). Writes ../artifacts/trainworld/T-1521/corridor/rows_<variant>.json.
//   node trainworld/notes/T-1521/corridor_rows.mjs   (from the tree's maps root)
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { loadApp } from '../../checks/lib.mjs';

const app = await loadApp("export * from '@app/src/game/income.ts';");
const { fareWeight, incomeGroups, carUsdPerKm, LONG_TRIP_MONEY } = app;
const TW = new URL('../../', import.meta.url);
const rows = JSON.parse(readFileSync(new URL('app/src/game/market-rows.json', TW), 'utf8'));
const inc = JSON.parse(readFileSync(new URL('app/src/game/countryIncome.json', TW), 'utf8'));
const OUT = new URL('../../../artifacts/trainworld/T-1521/corridor/', TW);
mkdirSync(OUT, { recursive: true });

// price level: PPP conversion factor over the market rate (World Bank PA.NUS.PPP and PA.NUS.FCRF,
// 2025), approximated from the income table's two GDP figures where no factor is at hand
const PLR = { US: 1, JP: 97.08 / 149.66, CN: 3.398 / 7.19, KR: 816.33 / 1422.44, FR: 0.6776 / 0.88, DE: 0.71 / 0.88, ES: 0.571 / 0.88, GB: 0.6771 / 0.76, IN: 20.089 / 87.16, NG: 195.02 / 1518.38 };
const plr = (c) => PLR[c] ?? Math.min(1.5, (inc.usd[c] ?? inc.usd.IN) / (inc.gdp[c] ?? inc.gdp.IN));
const UPKEEP = 0.073; // AAA 2026: maintenance, repair and tyres 11.7 cents a mile
const eps = JSON.parse(readFileSync(new URL('app/src/game/registry.json', TW), 'utf8')).market.rule.eps;
const builtGroups = (c) => incomeGroups(c, eps).flatMap((g) => [g.share, g.factor]);
const oldW = (c) => 5 * ((inc.gdp[c] ?? inc.gdp.IN) / 60000) ** -0.5;

const VARIANTS = {
  before: (c) => ({ fw: oldW(c), fg: builtGroups(c).map((v, i) => (i % 2 ? v / LONG_TRIP_MONEY : v)), cu: 0.15, af: 1 }),
  built: (c) => ({ fw: fareWeight(c), fg: builtGroups(c), cu: carUsdPerKm(c), af: 1 }),
  plr: (c) => ({ fw: fareWeight(c), fg: builtGroups(c), cu: carUsdPerKm(c), af: plr(c) }),
  upkeep: (c) => ({ fw: fareWeight(c), fg: builtGroups(c), cu: carUsdPerKm(c) + UPKEEP, af: 1 }),
};
for (const [name, f] of Object.entries(VARIANTS)) {
  const out = {};
  for (const [id, r] of Object.entries(rows)) out[id] = { ...r, ...f(r.country) };
  writeFileSync(new URL(`rows_${name}.json`, OUT), JSON.stringify(out));
}
writeFileSync(new URL('af_plr.json', OUT), JSON.stringify(Object.fromEntries(Object.keys(inc.usd).map((c) => [c, +plr(c).toFixed(4)]))));
writeFileSync(new URL('fare_w.json', OUT), JSON.stringify({ built: fareWeight('IN'), before: oldW('IN') }));
console.log('weights', ['US', 'JP', 'CN', 'FR', 'ES', 'GB'].map((c) => `${c} ${fareWeight(c).toFixed(2)} old ${oldW(c).toFixed(2)} plr ${plr(c).toFixed(3)} cu ${carUsdPerKm(c).toFixed(3)}`).join('; '));
console.log('wrote', Object.keys(VARIANTS).join(', '), 'to', OUT.pathname);
