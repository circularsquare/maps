// T-2193: a ferry line made through the track model (WASM) in Node, as the Ferry tool (T-2194) will
// make one: two ports and the Staten Island Ferry's crossing, Whitehall to St George, routed as the
// water worker (T-2192) gives a route (notes/T-2177.md "F1 -> F2 route format": x, y, kind, m/s a
// point; 25 km/h 200 m or more from land, 10 km/h at 20 m, straight between).
//   node trainworld/checks/t2193-ferry.mjs
// Constructs it, checks the price, the boats and the run time (notes/T-2177.md 3.4: about 22 min),
// saves, loads into a fresh model, and follows its boats over an hour. Exits 1 on any failure.
import { readFileSync } from 'node:fs';
import init, { TrackApi } from '../app/src/wasm/track/sim.js';

await init({ module_or_path: readFileSync(new URL('../app/src/wasm/track/sim_bg.wasm', import.meta.url)) });
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}`); if (!ok) failed++; };

// the legacy frame round the origin: x east, y north, ground metres
const O = [-74.04, 40.67], R = 6371008.8, rad = Math.PI / 180;
const xy = (lng, lat) => [(lng - O[0]) * rad * R * Math.cos(O[1] * rad), (lat - O[1]) * rad * R];
const K_ROUTE = 0, K_PORT = 1;
const kmh = (v) => v / 3.6;

const api = new TrackApi(O[0], O[1], true);
api.set_cash(1000);
const port = (landing, berth, name) => {
  const [x, y] = xy(...landing), [bx, by] = xy(...berth);
  if (api.add_port(x, y, bx, by, name) !== 0) throw new Error(`port ${name}: ${api.issue_kinds()}`);
  return api.last_port();
};
const whitehall = port([-74.0131, 40.7015], [-74.0136, 40.7008], 'Whitehall');
const george = port([-74.0721, 40.6441], [-74.0712, 40.6447], 'St George');
const [a, b] = [api.port_berth(whitehall), api.port_berth(george)];

// the crossing: points every 10 m within 250 m of each berth, every 500 m between; speed from the room
// to land, taken as 5 m at the berth plus the distance out from it
const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
const at = [];
for (let d = 0; d < 250; d += 10) at.push(d);
for (let d = 250; d < len - 250; d += 500) at.push(d);
for (let d = len - 250; d < len; d += 10) at.push(d);
at.push(len);
const room = (d) => 5 + Math.min(d, len - d);
const speed = (m) => kmh(m >= 200 ? 25 : m <= 20 ? 10 : 10 + 15 * (m - 20) / 180);
const route = [];
at.forEach((d, i) => {
  const t = d / len;
  route.push(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), i === 0 || i === at.length - 1 ? K_PORT : K_ROUTE, speed(room(d)));
});
const points = new Float64Array(route), ports = new Uint32Array([whitehall, george]);

const line = api.new_line_id();
const quote = api.ferry_quote(line, ports, points, 60);
check(quote[3] === 1 && quote[2] === 20, `quote: ${(quote[0] / 1000).toFixed(2)} km, ${(quote[1] / 60).toFixed(1)} min a crossing, $${quote[2]}M of ports`);
check(api.set_ferry_line(line, ports, points, 900, 1200, 1800, 60, 300, 0x1f77b4, 'Ferry 1') === 0, `a ferry line over two ports (${api.issue_kinds() || 'ok'})`);
check(api.is_ferry(line) && !api.is_bus(line) && api.is_port(whitehall), 'it is a ferry line between ports');

const cash = api.cash();
check(api.construct_line(line) === 0, `constructed (${api.issue_kinds() || 'ok'})`);
const boats = api.boats_needed();
const paid = cash - api.cash();
check(Math.abs(paid - (20 + 8 * boats)) < 1e-6 && api.fleet_ferry() === boats && api.fleet() === 0, `paid $${paid.toFixed(1)}M: two ports at $10M and ${boats} boats at $8M`);
const run = api.stop_times(line, 0, 0).at(-1);
check(run > 19 * 60 && run < 24 * 60, `Whitehall to St George ${(run / 60).toFixed(1)} min over ${(api.ferry_m(line) / 1000).toFixed(2)} km (notes: about 22, real 25)`);
check(Math.abs(api.stop_times(line, 1, 0).at(-1) - run) < 1e-6, 'as long the other way');
const upkeep = api.upkeep();
check(Math.abs(upkeep[7] - 0.02) < 1e-12 && upkeep[8] === 2, `ports' upkeep $${(upkeep[7] * 1000).toFixed(0)}k a day`);

// save and load into a fresh model; the fleet travels in the game's own header, as for buses
const saved = api.save();
check(Buffer.from(saved).includes('LFR1') && TrackApi.save_unread(saved) === 0, `the save carries an LFR1 block (${saved.length} bytes)`);
const back = new TrackApi(O[0], O[1], true);
check(back.load(saved), 'it loads');
back.set_fleet_ferry(api.fleet_ferry());
const same = (x, y) => x.length === y.length && x.every((v, i) => Object.is(v, y[i]));
check(back.is_ferry(line) && back.is_port(george) && same(back.ferry_route(line), api.ferry_route(line)), 'the route and the ports come back');
check(same(back.stop_times(line, 0, 0), api.stop_times(line, 0, 0)) && back.boats_needed() === boats, 'the same run times and boats');
check(same(back.save(), saved), 'saved again, to the bit');
const info = back.lines_info();
check(info[0] === line && info[3] === 1, 'running after the load');

// its boats over an hour of a weekday morning: where each is from its trip's phases
const epoch = 86400 + 8 * 3600;
const trips = back.trips(epoch), meta = back.profile_meta(), phases = back.line_phases(line);
const first = meta[line * 6 * 4];
const stateAt = (p, tau) => {
  let k = 0;
  for (let i = 0; i < p.length; i += 4) if (p[i] <= tau) k = i;
  const [t, s, v, acc] = p.slice(k, k + 4);
  let d = tau - t;
  if (acc < 0) d = Math.min(d, v / -acc);
  return s + v * d + 0.5 * acc * d * d;
};
let mine = 0, seen = 0, worst = 0;
const poly = [back.run_polyline(line, 0, 10), back.run_polyline(line, 1, 10)];
for (let i = 0; i < trips.length; i += 4) {
  const prof = trips[i], r = Math.floor(prof / 3) % 2;
  if (Math.floor(prof / 6) !== line) continue;
  mine++;
  const m = meta.slice(prof * 4, prof * 4 + 4);
  const p = phases.slice((m[0] - first) * 4, (m[0] - first + m[1]) * 4);
  // half way through its run: the boat is on the line between the berths, about half way over
  const s = stateAt(p, run / 2);
  const k = Math.min(Math.round(s / 10), poly[r].length / 2 - 1);
  const [x, y] = [poly[r][2 * k], poly[r][2 * k + 1]];
  const from = r === 0 ? a : b;
  worst = Math.max(worst, Math.abs(Math.hypot(x - from[0], y - from[1]) - len / 2));
  seen++;
}
check(mine >= 8, `${mine} boat trips drawn in the hour from 8:00 (a boat every 15 minutes each way)`);
check(seen > 0 && worst < 100, `half way through its run every boat is half way over (worst ${worst.toFixed(0)} m off)`);
process.exit(failed ? 1 : 0);
