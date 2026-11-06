// How GDP per head at PPP scales with GDP per head at market rates across countries (countryIncome.json),
// so a value-of-time elasticity on real (PPP) income can be read as one on market dollars.
import fs from "fs";
const T = JSON.parse(fs.readFileSync("C:/Users/anita/projects/maps/trainworld/app/src/game/countryIncome.json", "utf8"));
const xs = [], ys = [];
for (const c of Object.keys(T.usd)) if (T.gdp[c] > 0 && T.usd[c] > 0) { xs.push(Math.log(T.usd[c])); ys.push(Math.log(T.gdp[c])); }
const n = xs.length, mx = xs.reduce((a, b) => a + b) / n, my = ys.reduce((a, b) => a + b) / n;
let sxx = 0, sxy = 0; for (let i = 0; i < n; i++) { sxx += (xs[i] - mx) ** 2; sxy += (xs[i] - mx) * (ys[i] - my); }
const b = sxy / sxx;
console.log(`n ${n}: ln PPP = a + ${b.toFixed(3)} ln market`);
// value of time in market $ ~ PPP income^e x price level; price level = PPP factor = market/PPP ~ market^(1-b)
for (const e of [0, 0.14, 0.3, 0.5, 0.7, 1]) console.log(`real-income elasticity ${e} -> market exponent ${(e * b + (1 - b)).toFixed(2)}`);
for (const m of [0.5, 0.6, 0.7]) console.log(`market exponent ${m} -> real-income elasticity ${((m - (1 - b)) / b).toFixed(2)}`);
