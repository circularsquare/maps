// T-3887: the page side of tools/frontier.py, run through `play.mjs do --file`. frontier.py appends one call:
//   return await FRONTIER.build(spec)     a line of one mode along the corridor, built and constructed
//   return await FRONTIER.service(args)   that line's headways set, its trips calculated, its day's money read
// Everything is read the way the game shows it: fares from the weekday's income (Money tab), running cost and
// upkeep as the Lines tab works them out (game/profitEstimate.ts, game/today.ts), building cost from the cash.
const FRONTIER = {
  async build(spec) {
    return page.evaluate(async (spec) => {
      const tw = window.tw, c = tw.client;
      const send = async (op) => {
        const w0 = tw.world.value, t = performance.now();
        const a = await c.edit(op);
        while (tw.world.value === w0 && performance.now() - t < 3000) await new Promise((r) => setTimeout(r, 30));
        return a;
      };
      const nearNode = ([lng, lat], skip) => {
        const p = tw.lngLatToPack(lng, lat);
        let best = null, bd = Infinity;
        for (const n of tw.world.value.nodes.values()) {
          if (skip?.has(n.id)) continue;
          const d = Math.hypot(n.x - p.x, n.y - p.y);
          if (d < bd) { bd = d; best = n.id; }
        }
        return best;
      };
      const free = ([lng, lat], level) => { const p = tw.lngLatToPack(lng, lat); return { kind: 'free', x: p.x, y: p.y, level }; };
      const issues = [];
      let dropped = 0;
      const step = async (what, op) => { const a = await send(op); if (a?.ok === false) issues.push(`${what}: ${(a.issues ?? []).join(', ')}`); return a; };
      const cash0 = tw.moneyState.value.cash;
      let line = null;
      if (spec.mode === 'bus') {
        // points every ~100 m along the corridor, each stretch on the road (K_ROAD 2), the ends clicked (K_CORNER 1):
        // a straight arterial of the city's grid, at its drawn length
        const route = [];
        spec.path.forEach(([lng, lat], k) => {
          const p = tw.lngLatToPack(lng, lat);
          const kind = (k === 0 || k === spec.path.length - 1 ? 1 : 0) | (k > 0 ? 2 : 0);
          route.push(p.x, p.y, kind);
        });
        const a = await step('bus line', { op: 'addBusLine', route, name: spec.name, colour: '#c0392b' });
        line = a?.line ?? null;
        if (line !== null && spec.busSpacing) await step('spacing', { op: 'busSpacing', line, spacing: spec.busSpacing });
        if (line !== null) await step('construct', { op: 'construct', edges: [], stations: [], buses: [line] });
      } else {
        const old = new Set([...tw.world.value.nodes.values()].map((n) => n.id));
        const mine = (p) => nearNode(p, old);
        const pts = spec.stations;
        const flags = { single: false, hsr: !!spec.hsr, tram: !!spec.tram };
        for (let k = 1; k < pts.length; k++) {
          const from = k === 1 ? free(pts[0], spec.level) : { kind: 'node', node: mine(pts[k - 1]) };
          const ask = { op: 'route', from, to: free(pts[k], spec.level), pis: [], ...flags };
          if (spec.street) { ask.street = true; ask.streetLine = [...pts[k - 1], ...pts[k]]; }
          await step(`track ${k}`, ask);
        }
        const nodes = pts.map(mine);
        // a stop the game refuses over water (a river the line runs along) is left out, the ends excepted
        const kept = [];
        for (const [k, n] of nodes.entries()) {
          const a = await send({ op: 'addStation', at: { kind: 'node', node: n }, platform: spec.platform, name: '' });
          const why = (a?.issues ?? []).join(', ');
          if (a?.ok !== false) kept.push(n);
          else if (k > 0 && k < nodes.length - 1 && /OverWater/.test(why)) dropped++;
          else issues.push(`station ${k + 1}: ${why}`);
        }
        await step('construct', { op: 'constructAll' });
        const a = await step('line', { op: 'addLine', stops: kept, name: spec.name, colour: '#0a7d3b' });
        line = a?.line ?? null;
        if (line !== null && spec.cars) await step('cars', { op: 'lineCars', line, cars: spec.cars });
        if (line !== null && spec.fast) await step('fast', { op: 'lineFast', line, fast: true });
      }
      const w = tw.world.value;
      const l = w.save.lines.find((x) => x.num === line);
      return { line, id: l?.id ?? null, issues, built: cash0 - tw.moneyState.value.cash, stops: l?.stops.length ?? 0, dropped, stations: w.save.stations.length, edges: w.edges.length, km: l ? w.lineStats[l.id]?.km : null };
    }, spec);
  },

  // the trip explorer's own answer for each pair of points (lng, lat): the shares walking, by train (bus included) and
  // by car of the first cell's homes going to the second's, as a commute in the morning peak (period 0)
  async trips({ pairs, purpose = -1, period = 0 }) {
    return page.evaluate(async ({ pairs, purpose, period }) => {
      const trip = await import('/src/game/trip.ts');
      const tw = window.tw, out = [];
      for (const [a, b] of pairs) {
        const pa = tw.lngLatToPack(a[0], a[1]), pb = tw.lngLatToPack(b[0], b[1]);
        const t = await trip.askTrip([pa.x, pa.y], [pb.x, pb.y], purpose);
        const p = t?.periods?.[period];
        out.push(p ? { pWalk: p.pWalk, pTrain: p.pTrain, pDrive: p.pDrive, distM: p.distM, rail: p.railOffered, why: p.why,
          railPerceived: p.railPerceived, roadPerceived: p.roadPerceived, walkMin: p.walkMin } : null);
      }
      return out;
    }, { pairs, purpose, period });
  },

  async service({ line, seconds }) {
    const before = await page.evaluate(async ({ line, seconds }) => {
      const dem = await import('/src/game/demand.ts');
      const tw = window.tw, cash0 = tw.moneyState.value.cash;
      const version = dem.demandView.value?.version ?? 0;
      const a = await tw.client.edit({ op: 'setHeadway', line, seconds });
      return { cash0, version, ok: a?.ok !== false, issues: a?.issues ?? [] };
    }, { line, seconds });
    if (!before.ok) return { ok: false, issues: before.issues };
    await wait.final({ after: before.version, timeout: 165000 });
    return page.evaluate(async ({ line, cash0 }) => {
      const money = await import('/src/game/money.ts');
      const pe = await import('/src/game/profitEstimate.ts');
      const per = await import('/src/game/periods.ts');
      const dem = await import('/src/game/demand.ts');
      const tw = window.tw, w = tw.world.value, m = tw.moneyState.value;
      const hours = per.PERIODS.map((p) => p.to - p.from);
      const l = w.save.lines.find((x) => x.num === line), st = w.lineStats[l.id];
      const lm = money.incomeByDay.value[0]?.lines.get(l.id);
      const ld = dem.demandView.value?.lines.get(l.id);
      // the busiest stretch's riders an hour, one direction, in each period (loads are a period's riders)
      const peak = ld ? Math.max(0, ...ld.loads.map((s, p) => Math.max(0, ...s) / hours[p])) : 0;
      const board = ld?.board ? ld.board.reduce((a, b) => a + b.reduce((x, y) => x + y, 0), 0) : null;
      const sum = (arr) => (arr ?? []).reduce((a, v, p) => a + v * hours[p], 0);
      return {
        ok: true, version: dem.demandView.value?.version, final: !!dem.demandView.value?.final,
        km: st.km, cars: st.cars, tram: st.tram, fast: st.fast, bus: !!st.bus, need: st.trainsNeeded, roundTripS: st.roundTripS,
        tph: l.tphP, riders: ld?.ridersPerDay ?? 0, boardSum: board, peakPerHour: peak, fullestPct: ld?.fullestPct ?? null,
        fares: lm ? sum(lm.perHour) * 1e6 : 0, rides: lm ? sum(lm.rides) : 0,
        running: pe.lineRunningADay(l, st, m), upkeep: st.upkeep, bought: cash0 - m.cash, cash: m.cash,
        netAll: pe.dayProfit(money.incomeByDay.value[0], m),
      };
    }, { line, cash0: before.cash0 });
  },
};
