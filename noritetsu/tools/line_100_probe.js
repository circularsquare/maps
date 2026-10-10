/* THE "100%-ABLE" CHECKER (2026-10-07; handoff_notes/full_line_not_100.md). Evaluated in the
   page (dist/index.html at localhost:8800/noritetsu/), e.g. with a copy of
   neighborhoods/tools/screenshot.js that prepends `window.CK = {...};`. window.CK: {ccs:
   ['us', ...], kind: 'listed' | 'service', detail: failing lines listed per country, only:
   [line ids], noAlong: true to measure without along.json}. Nothing is saved: rides are made
   with newRide and credited with rideCredit, never put in RIDES.

   1. Every line: emulates what a rider can pick in its strip diagram (stops only; junctions
      are never ride ends): every stop pair by the drawn way and the shortest way, then where
      the line is not yet whole the other way round, the "other track" chain ways from the
      stops nearest each chain, every stop to every row past a junction end and row to row.
      It keeps the rides that add a section and checks lineKm(line, credit) === line.km.
   2. ONE PICK END TO END on every unbranched line (its stops form a path: each joined to at
      most two others, through junctions, two ends; and no junction end of its track other
      than a run on past an end stop): the default way between the two end stops, or the
      first row past a junction where the line runs on beyond its end stop. `unbranched`,
      `e2ePass`, `e2eFail` (with what is left); `unbranchedLoose` / `e2ePassLoose` count the
      lines whose stops form a path whatever their junction ends.
   3. COUNTRIES AND OPERATORS (2026-10-08): every ride of 1 together, then each country's,
      operator's and operator-in-a-country's total (index.html unionTotals) must read exactly
      100% wherever all its lines did (`groups`), and everywhere once the lines 1 could not
      finish are ridden whole too (`groupsWhole`).
   Also `pairs` / `offDiagram`: stop pairs whose shortest way leaves the drawn track.
   In the US, Canada and Australia the listed lines are the routes (OSM lines, named trains
   included), the corridors and the register track the routes leave uncovered (index.html
   ROUTE_CCS, 2026-10-09); `routes` says how many of each, the corridors by name and how much
   track was left out of the totals as slivers under the gap tolerance. */
(async () => {
  const CK = window.CK || {};
  const ccs = CK.ccs || ['us'];
  const kind = CK.kind || 'listed';
  const detail = CK.detail != null ? CK.detail : 40;
  const ONLY = CK.only ? new Set(CK.only) : null;
  const t0 = Date.now();
  const tick = () => new Promise(r => setTimeout(r, 0));
  for (const cc of ccs) await loadRegion(cc);
  await tick();
  // CK.noAlong: as before along.json (the footprint siblings only).
  if (CK.noAlong && typeof ALONG !== 'undefined') { ALONG.clear(); ALONG_INTO.clear(); }

  // Rides from the picks, as commitRide would make them.
  const viewWas = VIEW, pickWas = PICK;
  const ridesOf = (line, pick) => {
    VIEW = { kind: 'line', id: line.id };
    PICK = { around: false, ...pick };
    let run = null;
    try { run = pickedRun(); } catch (e) { run = null; }
    if (!run || !run.p || PICK.from === PICK.to) return [];
    if (run.legs) return run.legs.map(l => newRide(l.line.id, l.from, l.to, '', { cuts: l.cuts }));
    return [newRide(line.id, PICK.from, PICK.to, '', { around: run.around,
      via: run.via ? run.via.key : null,
      drawn: !!run.drawn && !run.around && !run.direct && !run.via })];
  };
  const restore = () => { VIEW = viewWas; PICK = pickWas; };

  // Credit of a set of rides along one line's sections.
  const fracOf = (line, rides) => {
    const raw = new Map(), direct = new Set();
    for (const r of rides) {
      const c = rideCredit(r);
      for (const g of c.direct) direct.add(g);
      for (const [t, sp] of c.R) {
        if (!raw.has(t)) raw.set(t, []);
        raw.get(t).push(...sp);
      }
    }
    const R = new Map();
    for (const [t, iv] of raw) R.set(t, mergeSpans(iv, secKm(t)));
    const frac = new Map();
    for (const [a, b, km, gid] of line.sections) {
      if (direct.has(gid)) { frac.set(gid, 1); continue; }
      const sp = mapBack(gid, R);
      if (sp.length) frac.set(gid, spansLength(sp));
    }
    return frac;
  };

  // pathBetween's Dijkstra run to the end: the tree its paths from `from` come from.
  const tree = (g, from) => {
    const dist = new Map([[from, 0]]), prev = new Map(), seen = new Set();
    for (;;) {
      let u = null, best = Infinity;
      for (const [k, d] of dist) if (!seen.has(k) && d < best) { best = d; u = k; }
      if (u === null) break;
      seen.add(u);
      for (const [v, km, gid] of g.get(u) || []) {
        const nd = best + km;
        if (nd < (dist.has(v) ? dist.get(v) : Infinity)) { dist.set(v, nd); prev.set(v, [u, gid]); }
      }
    }
    return prev;
  };

  // Stopless trees (pruning non-stop leaves): gid -> true.
  const stoplessOf = line => {
    const adj = new Map(), deg = new Map(), gone = new Set();
    for (const [a, b, km, gid] of line.sections) {
      if (a === b) continue;
      for (const [u, v] of [[a, b], [b, a]]) {
        if (!adj.has(u)) adj.set(u, []);
        adj.get(u).push([v, gid]);
        deg.set(u, (deg.get(u) || 0) + 1);
      }
    }
    const todo = [...deg].filter(([n, d]) => d === 1 && !isStop(n)).map(([n]) => n);
    while (todo.length) {
      const v = todo.pop();
      if (deg.get(v) !== 1) continue;
      const e = adj.get(v).find(x => !gone.has(x[1]));
      if (!e) continue;
      gone.add(e[1]); deg.set(v, 0); deg.set(e[0], deg.get(e[0]) - 1);
      if (deg.get(e[0]) === 1 && !isStop(e[0])) todo.push(e[0]);
    }
    return gone;
  };

  const waitGeo = async ids => {
    const ps = [];
    for (const id of ids) { geoNow(id); ps.push(geomFor(id)); }
    await Promise.all(ps);
    await tick(); await tick();
  };

  const out = {};
  const allRides = [], tested = new Set(), failed = new Set(), missOf = new Map();
  for (const cc of ccs) {
    /* Listed lines; in the US, Canada and Australia that takes in the routes, named trains
       included (index.html ROUTE_CCS, 2026-10-09). Elsewhere listed() never holds for one. */
    let lines = LINES.filter(l => (l.region === cc || (l.regions || []).includes(cc))
      && (kind === 'service' ? l.service : listed(l)));
    if (ONLY) lines = lines.filter(l => ONLY.has(l.id));
    // Geometry first where it changes routing (straight gaps).
    await waitGeo(lines.filter(l => l.straight_sections).map(l => l.id));
    for (const l of lines) if (l._gapsPending) { l._g = null; }
    const res = { lines: lines.length, pass: 0, fail: 0, nostops: 0, failKm: 0, causes: {}, failing: [],
                  pairs: 0, offDiagram: 0, offLines: 0, unbranched: 0, e2ePass: 0, e2eFail: [] };
    for (const line of lines) {
      tested.add(line.id);
      const running = line.sections.filter(s => !CLOSED.has(s[3]) && s[0] !== s[1]);
      if (!running.length || !(line.km > 0)) { res.pass++; continue; }
      const stops = [...lineStations(line)].filter(isStop);
      const g = lineGraph(line);
      const covered = new Set(), rides = [];
      const take = rs => {
        const gs = rs.flatMap(r => rideGids(r));
        if (gs.some(x => !covered.has(x))) { gs.forEach(x => covered.add(x)); rides.push(...rs); return true; }
        return false;
      };
      const NEW = typeof drawnGraph === 'function';
      let offLine = false;
      const gD = NEW ? drawnGraph(line) : null;
      const pathOf = (prev, s, t) => {
        if (!prev.has(t)) return null;
        const gids = [];
        for (let c = t; c !== s;) { const p = prev.get(c); gids.push(p[1]); c = p[0]; }
        return gids.reverse();
      };
      const pairs = (from, to) => {
        for (const s of from) {
          const pF = tree(g, s), pD = gD ? tree(gD, s) : null;
          for (const t of to) {
            if (t === s) continue;
            const sh = pathOf(pF, s, t);
            if (!sh) continue;
            const dr = pD ? pathOf(pD, s, t) : null;
            const differs = dr && (dr.length !== sh.length || dr.some((x, i) => x !== sh[i]));
            res.pairs++;
            if (differs) { res.offDiagram++; offLine = true; }
            if (differs && dr.some(x => !covered.has(x))) take([newRide(line.id, s, t, '', { drawn: true })]);
            if (sh.some(x => !covered.has(x))) take([newRide(line.id, s, t, '')]);
          }
        }
      };
      pairs(stops, stops);
      if (offLine) res.offLines++;
      let frac = fracOf(line, rides);
      const full = () => lineKm(line, frac) === line.km;
      /* ONE PICK, END TO END, on a line whose stops form a path (each stop joined to at most
         two others, through junctions; two ends): its default way must make it whole. */
      {
        const adjS = new Map(stops.map(s => [s, new Set()]));
        for (const s of stops) {
          const seen = new Set([s]), todo = [s];
          while (todo.length) {
            const u = todo.pop();
            for (const [v] of g.get(u) || []) {
              if (seen.has(v)) continue;
              seen.add(v);
              if (isStop(v)) adjS.get(s).add(v); else todo.push(v);
            }
          }
        }
        const deg = [...adjS.values()].map(x => x.size);
        const nEdges = deg.reduce((a, b) => a + b, 0) / 2;
        const endsS = stops.filter(s => adjS.get(s).size === 1);
        // The junction end past an end stop: a chain of junctions with nothing branching.
        const beyond = e => {
          for (const [first] of (g.get(e) || []).filter(([v]) => !isStop(v))) {
            let prev = e, cur = first;
            for (let k = 0; k < 500; k++) {
              const nb = (g.get(cur) || []).filter(([v]) => v !== prev);
              if (!nb.length) return cur;
              if (nb.length !== 1 || isStop(nb[0][0])) break;
              prev = cur; cur = nb[0][0];
            }
          }
          return null;
        };
        // ... and no other junction end anywhere (a branch to a junction is a branch).
        const leaves = [...g.keys()].filter(n => (g.get(n) || []).length === 1);
        const okEnds = new Set([...endsS, ...endsS.map(beyond).filter(Boolean)]);
        const loose = stops.length >= 2 && deg.every(d => d <= 2) && endsS.length === 2
          && nEdges === stops.length - 1;
        const strict = loose && leaves.every(n => okEnds.has(n));
        if (loose) {
          res.unbranchedLoose = (res.unbranchedLoose || 0) + 1;
          if (strict) res.unbranched++;
          /* Where the line runs on past an end stop to a junction, the rider picks the first
             stop listed past it (a continuation row) instead. */
          for (let k = 0; k < 20 && throughEnds(line).pending; k++) await new Promise(r => setTimeout(r, 150));
          const endRow = e => {
            const j = beyond(e);
            if (!j) return null;
            let rows = [];
            try { rows = contRows(line, j); } catch (err) {}
            const c = rows.find(x => x.kind === 'thru') || rows[0];
            return c ? { j, c } : null;
          };
          const A = endRow(endsS[0]), B = endRow(endsS[1]);
          const ref = x => ({ j: x.j, st: x.c.st, key: x.c.key });
          let pick = { from: endsS[0], to: endsS[1] };
          if (A && B && A.c.kind === 'thru' && B.c.kind === 'thru')
            pick = { from: A.c.st, to: B.c.st, through: { from: ref(A), to: ref(B) } };
          else if (B && B.c.kind === 'thru') pick = { from: endsS[0], to: B.c.st, through: { from: null, to: ref(B) } };
          else if (A && A.c.kind === 'thru') pick = { from: A.c.st, to: endsS[1], through: { from: ref(A), to: null } };
          else if (B) pick = { from: endsS[0], to: B.c.st, past: { line: B.c.line.id, board: B.c.board, j: B.j } };
          else if (A) pick = { from: A.c.st, to: endsS[1], past: { line: A.c.line.id, board: A.c.board, j: A.j } };
          const r1 = ridesOf(line, pick);
          restore();
          const f1 = fracOf(line, r1);
          const ok = r1.length && lineKm(line, f1) === line.km;
          if (ok) res.e2ePassLoose = (res.e2ePassLoose || 0) + 1;
          if (ok && strict) res.e2ePass++;
          else if (strict) res.e2eFail.push({ id: line.id, name: lineName(line), pct: +(100 * lineKm(line, f1) / line.km).toFixed(2),
            left: running.filter(s => (f1.get(s[3]) || 0) < 1).slice(0, 4)
              .map(s => `${s[3]} ${stName(s[0])} - ${stName(s[1])} ${s[2]} f=${(f1.get(s[3]) || 0).toFixed(3)}`) });
        }
      }
      if (!full()) {
        // The other way round, every pair.
        if (stops.length <= 120) {
          for (let i = 0; i < stops.length; i++) for (let j = i + 1; j < stops.length; j++) {
            const alt = pathAround(line, stops[i], stops[j]);
            if (alt && alt.gids.some(x => !covered.has(x)))
              take(ridesOf(line, { from: stops[i], to: stops[j], around: true }));
          }
        }
        // Ways over track no other pick rides (lineOrphans), from the stops nearest each chain.
        if (NEW) {
          const pickSet = new Set(stops);
          const frontier = (from, block) => {
            const seen = new Set([from, ...block]), out = [], todo = [from];
            while (todo.length) {
              const u = todo.pop();
              for (const [v] of g.get(u) || []) {
                if (seen.has(v)) continue;
                seen.add(v);
                if (pickSet.has(v)) out.push(v); else todo.push(v);
              }
            }
            if (pickSet.has(from)) out.push(from);
            return out;
          };
          for (const ch of lineOrphans(line)) {
            const inner = ch.nodes.slice(1, -1);
            const near = [...new Set([...frontier(ch.u, inner), ...frontier(ch.v, inner)])];
            for (const s of near) for (const t of near) {
              if (s === t) continue;
              VIEW = { kind: 'line', id: line.id };
              PICK = { from: s, to: t, around: false, via: `${ch.u}|${ch.v}` };
              const run = pickedRun();
              if (run && run.via) take([newRide(line.id, s, t, '', { via: run.via.key })]);
            }
          }
          restore();
        }
        // Continuation rows past junction ends.
        const tp = throughEnds(line);
        if (tp.pending) {
          for (let k = 0; k < 20 && throughEnds(line).pending; k++) await new Promise(r => setTimeout(r, 150));
        }
        let ends = new Map();
        try { ends = contEnds(line); } catch (e) {}
        for (const [j, rows] of ends) for (const c of rows) for (const s of stops) {
          const pick = c.kind === 'past'
            ? { from: s, to: c.st, past: { line: c.line.id, board: c.board, j } }
            : { from: s, to: c.st, through: { from: null, to: { j, st: c.st, key: c.key } } };
          take(ridesOf(line, pick));
        }
        // Between two continuation rows at different junction ends.
        const flat = [...ends].flatMap(([j, rows]) => rows.filter(c => c.kind === 'thru').map(c => ({ j, c })));
        for (const A of flat) for (const B of flat) {
          if (A.j === B.j) continue;
          take(ridesOf(line, { from: A.c.st, to: B.c.st,
            through: { from: { j: A.j, st: A.c.st, key: A.c.key }, to: { j: B.j, st: B.c.st, key: B.c.key } } }));
        }
        restore();
        frac = fracOf(line, rides);
      }
      allRides.push(...rides);
      if (full()) { res.pass++; continue; }
      failed.add(line.id);
      res.fail++;
      if (stops.length < 2) res.nostops++;
      const gaps = straightGaps(line), stopless = stoplessOf(line);
      const left = [];
      for (const [a, b, km, gid] of running) {
        const f = frac.get(gid) || 0;
        if (f >= 1) continue;
        const why = stops.length < 2 ? 'no stops'
          : gaps.has(gid) ? 'straight gap'
          : stopless.has(gid) ? 'stopless tail'
          : covered.has(gid) ? 'on a path, partly credited'
          : f > 0 ? 'partly credited, never on a path'
          : 'never on a path (alternative)';
        res.causes[why] = (res.causes[why] || 0) + 1;
        left.push({ gid, km: +(km * (1 - f)).toFixed(3), f: +f.toFixed(3), why,
                    a: stName(a), b: stName(b) });
      }
      const miss = line.km - lineKm(line, frac);
      missOf.set(line.id, miss);
      res.failKm += miss;
      res.failing.push({ id: line.id, name: lineName(line), km: +line.km.toFixed(2),
                         missKm: +miss.toFixed(3), stops: stops.length, rides: rides.length,
                         left: left.sort((x, y) => y.km - x.km).slice(0, 6) });
    }
    restore();
    // What the routes did to the country's lists (ROUTE_CCS): rows, corridors, track kept.
    if (typeof routesOf === 'function' && ROUTE_CCS.has(cc)) {
      const R = routesOf(cc), inf = [...R.info.entries()];
      const nm = id => lineName(LINE_BY_ID.get(id));
      res.routes = {
        rows: R.rows.size, tracks: inf.length,
        trackKept: inf.filter(([, i]) => !i.covered).length,
        trackHidden: inf.filter(([, i]) => i.covered && !i.corridor).length,
        single: inf.filter(([, i]) => i.single && !i.corridor).length,
        sliverKm: +inf.filter(([, i]) => i.covered && !i.corridor).reduce((s, [, i]) => s + i.gap, 0).toFixed(3),
        corridors: inf.filter(([, i]) => i.corridor).map(([id, i]) => `${nm(id)} (${i.services.length})`),
      };
    }
    res.failKm = +res.failKm.toFixed(1);
    res.failing.sort((x, y) => y.missKm - x.missKm);
    res.failingIds = res.failing.map(f => f.id);
    if (detail >= 0) res.failing = res.failing.slice(0, detail);
    out[cc] = res;
  }

  /* 3. COUNTRIES AND OPERATORS (Anita, 2026-10-08: "100%ing a country should be possible. it
     should be the union of 100%ing all the individual lines. same for an agency."). Every
     ride above together, credited at once (creditFor, nothing saved), and each group of
     unionTotals checked: a country (c:<cc>), an operator (o:<key>), an operator in one country
     (r:<cc>|<key>). A group whose lines all reached 100% here must read 100% (`pctLabel`) and
     its km ridden equal its total exactly. Done twice: `groups`, the picks alone, where a
     group with a line that failed above is counted as blocked, not failed; and
     `groupsWhole`, with each failed line ridden whole as well (wholeRide, its "rode all of
     it"), so every group is checked as if every line could be finished. An operator with
     lines in a country not probed is left out. CK.groups = false skips this. */
  if (kind === 'listed' && CK.groups !== false && typeof unionTotals === 'function') {
    out.groups = checkGroups(allRides, failed);
    out.groupsWhole = checkGroups([...allRides, ...[...failed].map(id => wholeRide(LINE_BY_ID.get(id)))], new Set());
  }
  out.secs = (Date.now() - t0) / 1000;
  return out;

  function checkGroups(allRides, failed) {
    const C = creditFor(allRides);
    const U = unionTotals(), D = unionDone(C);
    const keyOf = new Map([...U.groups].map(([k, g]) => [g, k]));
    const members = new Map();
    for (const { line, gs } of U.members) for (const g of gs) {
      const k = keyOf.get(g);
      if (!members.has(k)) members.set(k, new Set());
      members.get(k).add(line.id);
    }
    const inCcs = k => k.startsWith('o:') || ccs.includes(k.slice(2).split('|')[0]);
    const G = { checked: 0, pass: 0, exact: 0, blocked: 0, notProbed: 0, fail: [], linesNotWholeTogether: 0 };
    for (const [k, ids] of members) {
      if (!inCcs(k)) continue;
      if ([...ids].some(id => !tested.has(id))) { G.notProbed++; continue; }
      const total = U.groups.get(k).total, done = D.get(k) || 0;
      /* Blocked by a line that failed above: what is left of the group should be no more than
         what those lines miss (a little over where owner and own lengths differ). */
      const miss = [...ids].reduce((s, id) => s + (missOf.get(id) || 0), 0);
      if ([...ids].some(id => failed.has(id))) {
        G.blocked++;
        if (total - done > 1.05 * miss + 0.05) {
          G.blockedOver = G.blockedOver || [];
          G.blockedOver.push({ key: k, shortKm: +(total - done).toFixed(3), linesMissKm: +miss.toFixed(3) });
        }
        continue;
      }
      G.checked++;
      if (done === total) G.exact++;
      if (total <= 0 || pctLabel(done / total) === '100') G.pass++;
      else G.fail.push({ key: k, total: +total.toFixed(3), done: +done.toFixed(3),
                         shortKm: +(total - done).toFixed(4), lines: ids.size });
    }
    // A line whole on its own rides is whole with every ride together (credit only adds).
    for (const id of tested) {
      if (failed.has(id)) continue;
      const l = LINE_BY_ID.get(id);
      if (l && l.km > 0 && l.sections.some(s => !CLOSED.has(s[3]) && s[0] !== s[1])
          && lineKm(l, C.frac) !== l.km) G.linesNotWholeTogether++;
    }
    G.fail.sort((a, b) => b.shortKm - a.shortKm);
    G.failCount = G.fail.length;
    if (detail >= 0) G.fail = G.fail.slice(0, detail);
    G.rides = allRides.length;
    G.countries = Object.fromEntries([...members.keys()].filter(k => k.startsWith('c:') && inCcs(k))
      .map(k => [k.slice(2), { total: +U.groups.get(k).total.toFixed(2), done: +(D.get(k) || 0).toFixed(2),
                               shown: pctLabel((D.get(k) || 0) / Math.max(U.groups.get(k).total, 0.001)) }]));
    return G;
  }
})()
