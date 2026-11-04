// T-279: named corridors' car before and after the routed drives (game/marketRule.ts, the sim's
// mirror): the car's perceived minutes, its drive, any ferry, and the car's share of the trips not
// by rail against the plane. TW_MARKET_DRIVES=<file> node trainworld/notes/T-279/corridors.mjs
import { loadApp, marketDrives, marketRows } from '../../checks/lib.mjs';

globalThis.location ??= { search: '', hash: '', href: 'http://localhost/' };
const rule = await loadApp("export * from '@app/src/game/marketRule.ts';");
const rows = JSON.parse(marketRows());
const P = [['ber', 'ham', 'Berlin - Hamburg'], ['tai', 'zht', 'Taipei - Hualien (zht)'], ['nyc', 'bos', 'New York - Boston'], ['hel', 'pos', 'Helsinki - Tallinn (pos)'], ['war', 'poz', 'Warsaw - Poznan'], ['seo', 'shy', 'Seoul - Shenyang'], ['seo', 'dal', 'Seoul - Dalian']];
const fig = () => Object.fromEntries(P.filter(([a, b]) => rows[a] && rows[b]).map(([a, b, n]) => {
  const c = rule.corridorOf(rows[a].place, rows[b].place);
  return [n, c && { km: Math.round(c.km), car: c.car && Math.round(c.car.total), drive: c.car && Math.round(c.car.driveMin), road: c.car && Math.round(c.car.roadKm), ferry: c.car?.crossings.map((x) => `${x.kind} ${Math.round(x.km ?? 0)} km`).join(' '), carShare: c.car && +c.carShare.toFixed(3) }];
}));
rule.setMarketRows(rows);
const before = fig();
rule.setMarketDrives(marketDrives());
rule.setMarketRows(rows);
const after = fig();
for (const n of Object.keys(before)) console.log(n, JSON.stringify(before[n]), '->', JSON.stringify(after[n]));
