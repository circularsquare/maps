// T-1504 probe command: installs window.__t (the explorer between two places, summed over the
// day's periods, with the biggest period's station pairs and routes), then answers the named pairs.
window.__t = {
  async ask(near, far) {
    const ps = await window.twDemand.client.marketTrip('home', new Uint32Array(0), far, -1, { place: near, gid: -1 });
    if (!ps) return null;
    const day = { trips: 0, rail: 0, air: 0, car: 0, everydayLegs: 0, everydayRail: 0, commuteLegs: 0, commuteRail: 0 };
    let big = null;
    for (const p of ps) {
      if (!p) continue;
      day.trips += p.trips; day.rail += p.rail; day.air += p.air; day.car += p.car;
      day.everydayLegs += p.everyday.legs; day.everydayRail += p.everyday.rail; day.commuteLegs += p.commute.legs; day.commuteRail += p.commute.rail;
      if (!big || p.trips > big.trips) big = p;
    }
    const b = big && {
      railP: big.railPerceived, carP: big.carPerceived, refP: big.refPerceived, km: big.km, served: big.served, baseFareP: big.baseFarePerceived,
      minPerUsd: big.minPerUsd, why: big.why, zonePairs: big.zonePairs, cut: big.cut,
      pairs: big.pairs.slice(0, 6).map((x) => ({ a: x.access, b: x.egress, aP: x.accessPerceived, bP: x.egressPerceived, strat: x.strategy, share: x.share })),
      routes: big.routes.slice(0, 3).map((r) => ({ share: r.share, pair: r.pair, legs: r.legs.map((l) => ({ kind: l.kind, line: l.line, from: l.from, to: l.to, stops: l.stops, min: l.min, perc: l.perceived, fare: l.farePerceived, waitMin: l.waitMin, waitP: l.waitPerceived, change: l.changePerceived })) })),
    };
    return { day, big: b };
  },
};
const P = { bei: 2688, sha: 17, gua: 2594, shz: 2598, wuh: 2624, chs: 2675, tia: 2701, nan: 2694, hag: 2632, zhe: 2710, xia: 2639, chg: 2630 };
const out = {};
for (const [a, b] of [['bei', 'sha'], ['gua', 'shz'], ['wuh', 'chs'], ['bei', 'tia'], ['sha', 'nan'], ['sha', 'hag'], ['bei', 'gua'], ['wuh', 'gua'], ['bei', 'wuh'], ['zhe', 'wuh'], ['bei', 'zhe']]) {
  out[`${a}-${b}`] = await window.__t.ask(P[a], P[b]);
}
return out;
