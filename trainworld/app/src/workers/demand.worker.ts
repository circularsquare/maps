// A demand worker (T-026, T-021; SPEC 4.4, 6.5, 7). Owns one DemandApi (its own copy of the
// city) and solves a fixed set of the day's periods on each network it is sent: the free-flow
// first estimate for every period first, then the crowding rounds. Between periods it lets
// messages in, so a newer network supersedes the running solve within one period's work.

import init, { DemandApi, demand_periods, wasm_memory_bytes } from "../wasm/demand/sim"; // demand only (T-051)
import type { DemandNetwork, FromDemand, ToDemand } from "./demandProtocol";

let api: DemandApi | null = null;
let mine: number[] = [];
let pending: { version: number; net: DemandNetwork; rounds: number } | null = null;
let busy = false;
/** the version whose network `api` holds (queries about any other get nulls) */
let current = -1;

function send(msg: FromDemand, transfer: Transferable[] = []) {
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
      const header = msg.header;
      api = new DemandApi(header, new Uint8Array(msg.bin));
      const t2 = performance.now();
      mine = msg.periods;
      const warmMs = api.warm_up();
      send({ kind: "ready", wasmMs: t1 - t0, openMs: t2 - t1, warmMs, info: Array.from(api.info()), periods: Array.from(demand_periods()), memBytes: wasm_memory_bytes() });
      if (pending && !busy) run();
    } catch (err) {
      send({ kind: "failed", error: String(err) });
    }
  } else if (msg.kind === "solve") {
    pending = msg;
    if (api && !busy) run();
  } else {
    query(msg);
  }
};

/** The demand views' queries (T-078). They run between periods, never inside a solve. */
function query(msg: Exclude<ToDemand, { kind: "init" | "solve" }>) {
  const a = api;
  const live = !!a && msg.kind !== "cells" && msg.version === current;
  try {
    if (msg.kind === "cells") {
      if (!a) return send({ kind: "failed", error: "cells asked before the city opened" });
      const xy = a.cell_xy(), h3 = a.cell_h3(), zone = a.cell_zone(), zoneXY = a.zone_xy();
      send({ kind: "cells", id: msg.id, xy, h3, zone, zoneXY }, [xy.buffer, h3.buffer, zone.buffer, zoneXY.buffer]);
    } else if (msg.kind === "subSums") {
      const sums = live ? a!.sub_sums() : null;
      send({ kind: "subSums", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, sums }, sums ? [sums.buffer] : []);
    } else if (msg.kind === "cellModes") {
      const modes = live ? a!.cell_modes(msg.sums) : null;
      send({ kind: "cellModes", id: msg.id, version: msg.version, modes: modes && modes.length ? modes : null }, modes ? [modes.buffer] : []);
    } else if (msg.kind === "station") {
      const packed = live ? a!.station_riders(msg.station) : null;
      send({ kind: "station", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, packed: packed && packed.length ? packed : null }, packed ? [packed.buffer] : []);
    } else if (msg.kind === "flowRail") {
      const sums = live ? a!.flow_rail(msg.end, msg.cells) : null;
      // empty is a real answer on a network with no running line (no subzones, T-099)
      send({ kind: "flowRail", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, sums }, sums ? [sums.buffer] : []);
    } else if (msg.kind === "flows") {
      const packed = live ? a!.flows(msg.end, msg.cells, msg.sums) : null;
      send({ kind: "flows", id: msg.id, version: msg.version, packed: packed && packed.length ? packed : null }, packed ? [packed.buffer] : []);
    }
  } catch (err) {
    // a failed query answers with nothing; the pool itself carries on
    console.error("demand query:", err);
    if (msg.kind === "subSums" || msg.kind === "station") send({ kind: msg.kind, id: msg.id, version: msg.version, mask: 0, ...(msg.kind === "subSums" ? { sums: null } : { packed: null }) } as FromDemand);
    else if (msg.kind === "cellModes") send({ kind: "cellModes", id: msg.id, version: msg.version, modes: null });
    else if (msg.kind === "flowRail") send({ kind: "flowRail", id: msg.id, version: msg.version, mask: 0, sums: null });
    else if (msg.kind === "flows") send({ kind: "flows", id: msg.id, version: msg.version, packed: null });
  }
}

async function run() {
  busy = true;
  try {
    while (pending && api) {
      const job = pending;
      pending = null;
      const n = job.net;
      let setupMs = api.set_network(n.stXY, n.lineN, n.stops, n.times, n.tph, n.cars);
      current = job.version;
      let stale = false;
      for (let round = 0; round <= job.rounds && !stale; round++) {
        for (const q of mine) {
          await yieldNow();
          if (pending) {
            stale = true;
            break;
          }
          const t0 = performance.now();
          if (round === 0) api.solve(q);
          else api.crowd(q);
          const ms = performance.now() - t0;
          const seg = api.seg(q), board = api.board(q), alight = api.alight(q), loadOfCrush = api.load_of_crush(q);
          send(
            { kind: "result", version: job.version, period: q, summary: Array.from(api.summary(q)), seg, board, alight, loadOfCrush, ms, setupMs, memBytes: wasm_memory_bytes() },
            [seg.buffer, board.buffer, alight.buffer, loadOfCrush.buffer],
          );
          setupMs = 0;
        }
      }
      if (!stale) send({ kind: "done", version: job.version, memBytes: wasm_memory_bytes() });
    }
  } catch (err) {
    send({ kind: "failed", error: String(err) });
  }
  busy = false;
}
