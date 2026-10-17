// How much a history adds to the real save. usage: node realsize.mjs [round]
import { readFileSync } from 'node:fs';
import { gunzipSync, gzipSync } from 'node:zlib';

const file = readFileSync('C:/Users/anita/projects/maps/trainworld/T-007-real-nyc.save');
const raw = gunzipSync(file);
const n = raw.readUInt32LE(4);
const hdr = JSON.parse(raw.subarray(8, 8 + n).toString('utf8'));
const track = raw.subarray(8 + n);
console.log('file gz', file.length, 'raw', raw.length, 'header raw', n, 'track raw', track.length);
console.log('header keys', Object.keys(hdr).join(','), 'ledger', hdr.ledger?.length, 'hours', hdr.hours?.length, 'regions', Object.keys(hdr.regions ?? {}));
console.log('last day', JSON.stringify(hdr.ledger?.at(-1)));

const build = (h) => {
  const json = Buffer.from(JSON.stringify(h), 'utf8');
  const head = Buffer.alloc(8);
  head.write('TWG1', 0);
  head.writeUInt32LE(json.length, 4);
  return gzipSync(Buffer.concat([head, json, track])).length;
};
const rnd = (a, b) => a + Math.random() * (b - a);
const round = process.argv[2] ? Number(process.argv[2]) : null;
const num = (v, d) => (round === null ? v : Math.round(v * 10 ** d) / 10 ** d);
// a day of a region with real-looking magnitudes: fares and running in US$M, rides a count
const day = (d, scale) => ({ day: d, fares: num(rnd(80, 130) * scale, 6), running: num(rnd(10, 14) * scale, 6), trains: 0, build: 0, rides: num(rnd(4e5, 6e5) * scale, 2) });
const hour = (h, scale) => ({ hour: h, fares: num(rnd(2, 7) * scale, 6), running: num(rnd(0.3, 0.6) * scale, 6), trains: 0, build: 0, rides: num(rnd(1e4, 3e4) * scale, 2) });
const history = (days, regions) => {
  const reg = {};
  for (let r = 0; r < regions; r++) {
    const s = r === 0 ? 1 : 0.25;
    reg[r === 0 ? '' : 'z' + r] = { ledger: Array.from({ length: days }, (_, i) => day(100 + i, s)), hours: Array.from({ length: 48 }, (_, i) => hour(2400 + i, s)) };
  }
  return { ledger: Array.from({ length: days }, (_, i) => day(100 + i, 1)), hours: Array.from({ length: 48 }, (_, i) => hour(2400 + i, 1)), regions: reg };
};
const base = { ...hdr };
delete base.ledger; delete base.hours; delete base.regions;
const none = build({ ...base, ledger: [], hours: [] });
console.log('no history at all', none);
for (const regions of [1, 3, 6]) {
  for (const days of [30, 60]) {
    const total = build({ ...base, ...history(days, regions) });
    console.log(`${days} days + 48 hours, network plus ${regions - 1} extra regions (+ one for the network in 'regions'): file ${total}, adds ${total - none}`);
  }
}
const old = { ...base, ledger: history(30, 1).ledger.map(({ rides, ...d }) => d), hours: history(30, 1).hours.map(({ rides, ...d }) => d) };
console.log('old-style 30 days, 48 hours, no regions', build(old) - none);
