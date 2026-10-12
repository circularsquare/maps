// Shared-world demand worker (T-116/T-117). Owns New York local commutes,
// Philadelphia local commutes and the separate regional market. Each assigned period
// advances all populations from one frozen total load vector at every crowding round.
// Network messages supersede work between populations; queries require the same
// version AND refinement round across all this worker's periods.
//
// One worker of the pool has no periods (`aux`, T-119): it only sets the network up. It is never
// busy, so the questions that need no period's result (the cells, a bubble's commuters by mode
// from the sums it is handed) are answered at once, not after a busy worker's current pass.

import init, { DemandApi, EverydayApi, IntercityApi, demand_periods, release_skims, wasm_memory_bytes } from "../wasm/demand/sim"; // demand only (T-051)
import { CITIES, cityToMarket, cityToWorld, HOME_CITY, setWorldFrame, worldToCity, worldToMarket, type CityId } from "../game/cities";
import type { DemandNetwork, FromDemand, ToDemand } from "./demandProtocol";

let api: DemandApi | null = null;
const cities = new Map<CityId, DemandApi>();
const everyday = new Map<CityId, EverydayApi>();
const EVERYDAY_PURPOSES = ["Shopping and errands", "Social and leisure", "School (metro public)", "University (metro IPEDS)"];
let intercity: IntercityApi | null = null;
let mine: number[] = [];
let pending: { version: number; net: DemandNetwork; rounds: number } | null = null;
let busy = false;
/** the version whose network `api` holds (queries about any other get nulls) */
let current = -1;
/** the round each of this worker's periods has reached; -1 before a result, -2 while a stage runs */
const roundsHeld: number[] = [];
/**
 * What `subSums` answers, taken the moment all this worker's periods have a round, per city and
 * round (T-119). The bubbles ask for it right after a round is published, when this worker is
 * already part way into the next one; the live state would be a mixture of two rounds by then, so
 * the answer is kept from when it was whole.
 */
const sumsByRound = new Map<string, { mask: number; sums: Float32Array; everyday: Float32Array | null }>();

function sendMessage(msg: FromDemand, transfer: Transferable[] = []) {
  postMessage(msg, { transfer });
}

/** Yield to the event loop so a waiting message (a newer network) is delivered. */
function yieldNow(): Promise<void> {
  return new Promise((resolve) => {
    const ch = new MessageChannel();
    ch.port1.onmessage = () => resolve();
    ch.port2.postMessage(0);
  });
}

onmessage = async (e: MessageEvent<ToDemand>) => {
  const msg = e.data;
  if (msg.kind === "init") {
    try {
      const t0 = performance.now();
      await init();
      const t1 = performance.now();
      for (const pack of msg.packs) {
        cities.set(pack.id, new DemandApi(pack.header, new Uint8Array(pack.bin)));
        if (!msg.aux && pack.id === "nyc" && JSON.parse(pack.header).arrays.some((a: {name: string}) => a.name === "errand_jobs")) everyday.set(pack.id, new EverydayApi(pack.header, new Uint8Array(pack.bin)));
      }
      api = cities.get(HOME_CITY)!;
      if (!msg.aux) intercity = new IntercityApi(msg.intercity);
      const t2 = performance.now();
      mine = msg.periods;
      const warmMs = [...cities.values()].reduce((sum, a) => sum + a.warm_up(), 0);
      const cityInfo = Object.fromEntries([...cities].map(([id, a]) => {
        const info = Array.from(a.info());
        // Auxiliary worker only serves commute allocation; compute workers report all local demand.
        const extra = everyday.get(id)?.info();
        if (extra) {
          let outings = 0, walking = 0;
          for (let k=0;k<extra.length;k+=3) { outings += extra[k]; walking += extra[k]*extra[k+1]; }
          info[3] = (info[2]*info[3]+walking)/(info[2]+outings);
          info[2] += outings;
        }
        return [id, info];
      })) as Record<CityId, number[]>;
      sendMessage({ kind: "ready", cityInfo, wasmMs: t1 - t0, openMs: t2 - t1, warmMs, info: cityInfo[HOME_CITY], periods: Array.from(demand_periods()), memBytes: wasm_memory_bytes() });
      if (pending && !busy) run();
    } catch (err) {
      sendMessage({ kind: "failed", error: String(err) });
    }
  } else if (msg.kind === "solve") {
    pending = msg;
    if (api && !busy) run();
  } else {
    query(msg);
  }
};

/**
 * Each local cell's 600 m square (the longer-trip market's endpoint sites are built from the same
 * cells, pipeline/intercity.py), and the commuters living and working in each square, so a
 * selection of cells can be turned into the share of every square it holds (T-118). Built once
 * per city on the first longer-trip query.
 */
interface Squares {
  key: Int32Array;
  total: Map<number, [number, number]>;
}
const SQUARE_M = 600, KEY_SPAN = 4096, KEY_OFFSET = 2048;
const squares = new Map<CityId, Squares>();
function squaresOf(id: CityId, a: DemandApi): Squares {
  let s = squares.get(id);
  if (s) return s;
  const xy = a.cell_xy(), w = a.cell_commuters(), n = xy.length / 2;
  const key = new Int32Array(n), total = new Map<number, [number, number]>();
  for (let c = 0; c < n; c++) {
    const p = cityToMarket(xy[2 * c], xy[2 * c + 1], id);
    const k = (Math.floor(p.x / SQUARE_M) + KEY_OFFSET) * KEY_SPAN + Math.floor(p.y / SQUARE_M) + KEY_OFFSET;
    key[c] = k;
    const t = total.get(k) ?? [0, 0];
    t[0] += w[c]; t[1] += w[n + c];
    total.set(k, t);
  }
  squares.set(id, (s = { key, total }));
  return s;
}
/** A selection's squares: their centres (world metres) and the share of each that is selected. */
function selectionSquares(id: CityId, a: DemandApi, end: number, cells: Uint32Array) {
  const s = squaresOf(id, a), n = s.key.length;
  const w = a.cell_commuters();
  const picked = new Map<number, number>();
  for (const c of cells) if (c < n) picked.set(s.key[c], (picked.get(s.key[c]) ?? 0) + w[end * n + c]);
  const xy = new Float32Array(2 * picked.size), share = new Float32Array(picked.size);
  let i = 0;
  for (const [k, sel] of picked) {
    const tot = s.total.get(k)![end];
    xy[2 * i] = (Math.floor(k / KEY_SPAN) - KEY_OFFSET + 0.5) * SQUARE_M;
    xy[2 * i + 1] = ((k % KEY_SPAN) - KEY_OFFSET + 0.5) * SQUARE_M;
    share[i] = tot > 0 ? Math.min(1, sel / tot) : 1;
    i++;
  }
  return { xy, share };
}
/** The market (IntercityApi's) of a local city. */
const MARKET_OF: Record<CityId, number> = Object.fromEntries(CITIES.map(c => [c.id, c.market]));

/** The demand views' queries (T-078). They run between periods, never inside a solve. */
function query(msg: Exclude<ToDemand, { kind: "init" | "solve" }>) {
  const a = cities.get(msg.city);
  type WithoutCity<T> = T extends {city: CityId} ? Omit<T, "city"> : T;
  const sendMessage = (reply: WithoutCity<FromDemand>, transfer: Transferable[] = []) => {
    postMessage({ ...reply, city: msg.city }, { transfer });
  };
  const live = !!a && msg.kind !== "cells" && msg.version === current && mine.every(q => roundsHeld[q] === msg.round);
  try {
    if (msg.kind === "cells") {
      if (!a) return sendMessage({ kind: "failed", error: "cells asked before the city opened" });
      setWorldFrame(msg.frame);
      const xy = a.cell_xy(), h3 = a.cell_h3(), zone = a.cell_zone(), zoneXY = a.zone_xy();
      for (let k = 0; k < xy.length; k += 2) {
        const p = cityToWorld(xy[k], xy[k + 1], msg.city); xy[k] = p.x; xy[k + 1] = p.y;
      }
      for (let k = 1; k < zoneXY.length; k += 2) {
        const p = cityToWorld(zoneXY[k], zoneXY[k + 1], msg.city); zoneXY[k] = p.x; zoneXY[k + 1] = p.y;
      }
      sendMessage({ kind: "cells", id: msg.id, xy, h3, zone, zoneXY }, [xy.buffer, h3.buffer, zone.buffer, zoneXY.buffer]);
    } else if (msg.kind === "subSums") {
      // from the copy taken when the round was whole, not from the state now (see sumsByRound)
      const kept = a && msg.version === current ? sumsByRound.get(`${msg.city}:${msg.round}`) : undefined;
      const sums = kept ? kept.sums.slice() : null;
      const extra = kept?.everyday?.slice() ?? null;
      sendMessage({ kind: "subSums", id: msg.id, version: msg.version, mask: kept ? kept.mask : 0, sums, everyday: extra }, [...(sums ? [sums.buffer] : []), ...(extra ? [extra.buffer] : [])]);
    } else if (msg.kind === "cellModes") {
      // reads the network's subzones and the sums it is given, not any period's state, so it holds
      // for the version whatever round this worker is in
      const modes = a && msg.version === current ? a.cell_modes(msg.sums) : null;
      sendMessage({ kind: "cellModes", id: msg.id, version: msg.version, modes: modes && modes.length ? modes : null }, modes ? [modes.buffer] : []);
    } else if (msg.kind === "station") {
      const packed = live ? a!.station_riders(msg.station) : null;
      sendMessage({ kind: "station", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, packed: packed && packed.length ? packed : null }, packed ? [packed.buffer] : []);
    } else if (msg.kind === "flowRail") {
      const sums = live ? a!.flow_rail(msg.end, msg.cells) : null;
      // empty is a real answer on a network with no running line (no subzones, T-099)
      sendMessage({ kind: "flowRail", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, sums }, sums ? [sums.buffer] : []);
    } else if (msg.kind === "purposes") {
      const work = live ? a!.purpose_timing(msg.end, msg.cells) : null;
      const layer = live ? everyday.get(msg.city) : undefined;
      const extra = layer?.selection(msg.end, msg.cells) ?? null;
      const names = EVERYDAY_PURPOSES.slice(0, layer?.purpose_count() ?? 0);
      sendMessage({ kind: "purposes", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, work, everyday: extra, names }, [...(work ? [work.buffer] : []), ...(extra ? [extra.buffer] : [])]);
    } else if (msg.kind === "flows") {
      const packed = live ? a!.flows(msg.end, msg.cells, msg.sums) : null;
      sendMessage({ kind: "flows", id: msg.id, version: msg.version, packed: packed && packed.length ? packed : null }, packed ? [packed.buffer] : []);
    } else if (msg.kind === "icSel") {
      const ok = live && !!intercity;
      const sq = ok ? selectionSquares(msg.city, a!, msg.end, msg.cells) : null;
      const packed = sq ? Float64Array.from(intercity!.selection_flows(MARKET_OF[msg.city], sq.xy, sq.share)) : null;
      sendMessage({ kind: "icSel", id: msg.id, version: msg.version, mask: ok ? intercity!.solved_mask() : 0, packed }, packed ? [packed.buffer] : []);
    } else if (msg.kind === "icStation") {
      const ok = live && !!intercity;
      const packed = ok ? Float64Array.from(intercity!.station_flows(msg.station)) : null;
      sendMessage({ kind: "icStation", id: msg.id, version: msg.version, mask: ok ? intercity!.solved_mask() : 0, packed: packed && packed.length ? packed : null }, packed ? [packed.buffer] : []);
    }
  } catch (err) {
    // a failed query answers with nothing; the pool itself carries on
    console.error("demand query:", err);
    if (msg.kind === "subSums" || msg.kind === "station") sendMessage({ kind: msg.kind, id: msg.id, version: msg.version, mask: 0, ...(msg.kind === "subSums" ? { sums: null, everyday: null } : { packed: null }) } as FromDemand);
    else if (msg.kind === "purposes") sendMessage({kind: "purposes", id: msg.id, version: msg.version, mask: 0, work: null, everyday: null, names: []});
    else if (msg.kind === "cellModes") sendMessage({ kind: "cellModes", id: msg.id, version: msg.version, modes: null });
    else if (msg.kind === "flowRail") sendMessage({ kind: "flowRail", id: msg.id, version: msg.version, mask: 0, sums: null });
    else if (msg.kind === "flows") sendMessage({ kind: "flows", id: msg.id, version: msg.version, packed: null });
    else if (msg.kind === "icSel" || msg.kind === "icStation") sendMessage({ kind: msg.kind, id: msg.id, version: msg.version, mask: 0, packed: null });
  }
}

async function run() {
  busy = true;
  try {
    while (pending && api) {
      const job = pending;
      pending = null;
      const n = job.net;
      setWorldFrame(n.frame);
      let setupMs = 0;
      for (const [id, a] of cities) {
        const xy = n.stXY.slice();
        for (let k = 0; k < xy.length; k += 2) {
          const p = worldToCity(xy[k], xy[k + 1], id); xy[k] = p.x; xy[k + 1] = p.y;
        }
        setupMs += a.set_network(xy, n.lineN, n.stops, n.times, n.tph, n.cars);
        a.set_fares(n.fare.base, n.fare.perKm, n.hopKm);
        const extra = everyday.get(id);
        if (extra) {
          setupMs += extra.set_network(xy, n.lineN, n.stops, n.times, n.tph, n.cars);
          extra.set_fares(n.fare.base, n.fare.perKm, n.hopKm);
        }
      }
      const regional = intercity;
      if (regional) {
        const mxy = n.stXY.slice();
        for (let k = 0; k < mxy.length; k += 2) {
          const p = worldToMarket(mxy[k], mxy[k + 1]); mxy[k] = p.x; mxy[k + 1] = p.y;
        }
        setupMs += regional.set_network(mxy, n.lineN, n.stops, n.times, n.tph, n.cars);
        regional.set_fares(n.fare.base, n.fare.perKm, n.hopKm);
      }
      current = job.version;
      roundsHeld.fill(-1);
      sumsByRound.clear();
      let stale = false;
      // Periods are independent. Every population in a period advances from one frozen
      // combined load vector, never from whichever population happened to run first.
      const populations: (DemandApi | IntercityApi | EverydayApi)[] = [...cities.values(), ...everyday.values()];
      if (regional) populations.push(regional);
      for (let round = 0; round <= job.rounds && !stale; round++) {
        for (const q of mine) {
          await yieldNow();
          if (pending) { stale = true; break; }
          const t0 = performance.now();
          const sum = (read: (a: DemandApi | IntercityApi | EverydayApi) => Float32Array): Float32Array => {
            const out = new Float32Array(2 * n.stops.length);
            for (const a of populations) { const v = read(a); for (let k = 0; k < out.length; k++) out[k] += v[k] ?? 0; }
            return out;
          };
          // The period is part way between rounds until every population has had it; a query in
          // between (the yields below) must not read it.
          roundsHeld[q] = -2;
          const total = round === 0 ? null : sum(a => a.seg(q));
          for (const a of populations) {
            if (total === null) a.solve(q);
            else if (!a.crowd_with_external(q, total)) throw new Error("shared crowding layout disagrees");
            // a population's pass is up to a second of one call: let a waiting query or a newer
            // network in between
            await yieldNow();
            if (pending) { stale = true; break; }
          }
          release_skims();
          if (stale) break;
          roundsHeld[q] = round;
          if (q === mine[mine.length - 1]) {
            for (const [id, a] of cities) {
              const kept = { mask: a.solved_mask(), sums: a.sub_sums(), everyday: everyday.get(id)?.cell_modes() ?? null };
              sumsByRound.set(`${id}:${round}`, kept);
              // and sent unasked, ahead of this round's last result: the client holds every
              // worker's by the time it publishes the round, and the bubbles need no query to a
              // worker that is busy with the next round
              const copy = kept.sums.slice();
              const extra = kept.everyday?.slice() ?? null;
              sendMessage({ kind: "sums", version: job.version, round, city: id, mask: kept.mask, sums: copy, everyday: extra }, [copy.buffer, ...(extra ? [extra.buffer] : [])]);
            }
          }
          const ms = performance.now() - t0;
          const seg = sum(a => a.seg(q)), board = sum(a => a.board(q)), alight = sum(a => a.alight(q)), loadOfCrush = sum(a => a.load_of_crush(q));
          const citySummary = Object.fromEntries([...cities].map(([id, a]) => [id, Array.from(a.summary(q))])) as Record<CityId, number[]>;
          for (const [id, a] of everyday) { const v = a.summary(q); for (const k of [1, 2, 3, 5]) citySummary[id][k] += v[k] ?? 0; }
          const intercitySummary = Array.from(regional!.summary(q));
          const intercityBoard = regional!.board(q);
          const summary = [round, 0, 0, 0, ms, 0, 0, 0, 0, 0];
          for (const v of [...Object.values(citySummary), intercitySummary]) for (const k of [1, 2, 3, 5]) summary[k] += v[k] ?? 0;
          sendMessage(
            { kind: "result", version: job.version, period: q, summary, citySummary, intercitySummary, corridors: Array.from(regional!.corridor_summary(q)), intercityBoard, seg, board, alight, loadOfCrush, ms, setupMs, memBytes: wasm_memory_bytes() },
            [seg.buffer, board.buffer, intercityBoard.buffer, alight.buffer, loadOfCrush.buffer],
          );
          setupMs = 0;
        }
      }
      if (!stale) sendMessage({ kind: "done", version: job.version, memBytes: wasm_memory_bytes() });
    }
  } catch (err) {
    sendMessage({ kind: "failed", error: String(err) });
  }
  busy = false;
}
