// A one-line network for calib (RealNet JSON): stations about every 1.2 km along waypoints.
// node gen_line.mjs <pack json> <out json> <name> lon,lat lon,lat ...
import fs from "fs";
const [packPath, out, name, ...pts] = process.argv.slice(2);
const pack = JSON.parse(fs.readFileSync(packPath, "utf8"));
const { lon: lon0, lat: lat0 } = pack.origin;
const R = 6371008.8, k = Math.PI / 180;
const xy = ([lon, lat]) => [(lon - lon0) * k * Math.cos(lat0 * k) * R, (lat - lat0) * k * R];
const way = pts.map((p) => xy(p.split(",").map(Number)));
// resample at ~1.2 km
const seg = [];
let total = 0;
for (let i = 1; i < way.length; i++) { const d = Math.hypot(way[i][0] - way[i - 1][0], way[i][1] - way[i - 1][1]); seg.push(d); total += d; }
const n = Math.max(2, Math.round(total / 1200) + 1);
const st = [];
for (let s = 0; s < n; s++) {
  let t = (total * s) / (n - 1), i = 0;
  while (i < seg.length - 1 && t > seg[i]) { t -= seg[i]; i++; }
  const f = Math.min(1, t / seg[i]);
  st.push([way[i][0] + f * (way[i + 1][0] - way[i][0]), way[i][1] + f * (way[i + 1][1] - way[i][1])]);
}
const hop = st.slice(1).map((p, i) => Math.hypot(p[0] - st[i][0], p[1] - st[i][1]));
const run = hop.map((d) => Math.round(d / (40 / 3.6) + 30)); // 40 km/h between stops plus a 30 s dwell
const d0 = [...run, 0], d1 = [0, ...run];
const tph = (process.env.TPH ?? "12,6,0").split(",").map(Number); // high (6-20), medium (20-24), low (0-6)
const CARS = Number(process.env.CARS ?? 4);
const stops = st.map((_, i) => i);
const net = {
  what: `farescale test line ${name}`, city: pack.city, pack: pack.city,
  stations: st.map((p, i) => ({ key: `s${i}`, name: `s${i}`, x: p[0], y: p[1] })),
  lines: [{ stops, tph, cars: CARS, operator: "test", name }],
  demand: { st_xy: st.flat(), line_n: [n], stops, times: [...d0, ...d0, ...d0, ...d1, ...d1, ...d1], tph, cars: [CARS] },
};
fs.writeFileSync(out, JSON.stringify(net));
console.log(`${name}: ${n} stations, ${(total / 1000).toFixed(1)} km of stops ${(hop.reduce((a, b) => a + b, 0) / 1000).toFixed(1)} km, run ${(run.reduce((a, b) => a + b, 0) / 60).toFixed(1)} min`);
