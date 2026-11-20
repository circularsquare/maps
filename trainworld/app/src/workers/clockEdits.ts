// Edits: `apply` and its helpers (T-2374).

import { TrackApi } from "../wasm/track/sim"; // the track model only (T-051)
import { isCity } from "../game/cities";
import { headwaysFor, NEW_LINE } from "../game/newLine";
import { zonesOf } from "../game/fareZones";
import { letterOf } from "../game/lineLetter";
import { cloneName, cloneSymbol } from "../game/cloneNames";
import { capTransfer, faresOf, tidyHsr, tidyTransfer } from "../game/fareCurve";
import { FARE_BASELINE, FARE_E, keepOwn } from "../game/income";
import { type SaveHeader } from "./saveFile";
import type { EditOp, PaidFor, PiInput, RouteAsk, RouteEnd, StationAt } from "./protocol";
import { saves } from "./slots";
import { addPorts, newLineRows, newLineWeekends, nextBusName, renameComplex, renameStation } from "./clockDemand";
import { cityFares, setLineFare } from "./clockFares";
import { stateRows } from "./clockMain";
import { gameNow } from "./clockMoney";
import { api, fares, freezeCheckpoint, hiddenLines, lineShapes, loanOp, newLineCars, newLineHeadway, newLineWeekendShare, putFares, putNewLineCars, putNewLineWeekendShare, send, TRAM_MAX_CARS, TRAM_NEW_CARS } from "./clockState";

// ---------------------------------------------------------------- edits

function endArr(e: RouteEnd): number[] {
  if (e.kind === "free") return [0, 0, e.x, e.y, e.level];
  if (e.kind === "node") return [1, e.node, 0, 0, 0];
  return [2, e.edge, e.x, e.y, 0];
}
export function freeEnds(at: Extract<StationAt, { kind: "free" }>): Float64Array {
  return new Float64Array([0, 0, at.x0, at.y0, at.level, 0, 0, at.x1, at.y1, at.level]);
}
export function pisArr(p: PiInput[]): Float64Array {
  return new Float64Array(p.flatMap((v) => [v.x, v.y, v.radius ?? 0, v.level]));
}
function colourInt(c: string) {
  return parseInt(c.slice(1), 16);
}

/** T-1899: the share of a line's route km on high-speed track, 0 with no route. */
function hsrShare(a: TrackApi, line: number): number {
  const path = a.line_path(line);
  if (!path.length) return 0;
  const hsr = new Set(a.hsr_edges()), len = new Map<number, number>(), info = a.edges_info();
  for (let i = 0; i < info.length; i += 10) len.set(info[i], info[i + 5]);
  let all = 0, fast = 0;
  for (const p of path) {
    const m = len.get(p >> 1) ?? 0;
    all += m;
    if (hsr.has(p >> 1)) fast += m;
  }
  return all > 0 ? fast / all : 0;
}

type Applied = { r: number; fault?: number[]; line?: number; node?: number; skipped?: number; charge?: number; trainCharge?: number; why?: string[] };

/** T-1899: a new line's stop edit (`edit`), its trains then high-speed when more than half its
 * route's km is high-speed track and ordinary otherwise. The type is set before the stops in the undo
 * history (an undo takes the stop back first), and nothing the first try bought is kept. */
function withAutoFast(a: TrackApi, line: number, edit: () => number): Applied {
  const was = [a.cash(), a.fleet(), a.fleet_fast()] as const;
  const back = () => (a.undo(), a.set_cash(was[0]), a.set_fleet(was[1]), a.set_fleet_fast(was[2]));
  const r = edit();
  if (r !== 0 || !a.line_path(line).length) return { r };
  const fast = hsrShare(a, line) > 0.5;
  if (fast === a.line_fast(line)) return { r };
  back();
  if (a.set_line_fast(line, fast) !== 0) return { r: edit() };
  const charge = a.last_charge(), trainCharge = a.last_train_charge();
  if (edit() !== 0) {
    back();
    return { r: edit() };
  }
  return { r: 0, charge: charge + a.last_charge(), trainCharge: trainCharge + a.last_train_charge() };
}

/** T-3319: the header's shapes, of the lines there are (a deleted line's stays in hand, for an undo) */
export function savedShapes(): Pick<SaveHeader, "shapes"> {
  if (!lineShapes.size) return {};
  const info = api!.lines_info(), alive = new Set<string>();
  for (let i = 0; i < info.length; i += 22) alive.add(String(info[i]));
  const shapes = Object.fromEntries([...lineShapes].filter(([id]) => alive.has(id)));
  return Object.keys(shapes).length ? { shapes } : {};
}

/** T-3420: the header's hidden lines, of the lines there are */
export function savedHidden(): Pick<SaveHeader, "hidden"> {
  if (!hiddenLines.size) return {};
  const info = api!.lines_info(), alive = new Set<string>();
  for (let i = 0; i < info.length; i += 22) alive.add(String(info[i]));
  const hidden = [...hiddenLines].filter((id) => alive.has(id));
  return hidden.length ? { hidden } : {};
}

/** T-3319: a new line takes no shape left by a deleted line whose id it reuses; a copy takes its source's. */
export function newLineShape(op: EditOp, line: number) {
  if (op.op !== "addLine" && op.op !== "addBusLine" && op.op !== "addFerryLine" && op.op !== "copyLine") return;
  const id = String(line), from = op.op === "copyLine" ? lineShapes.get(String(op.line)) : undefined;
  // T-3420: a new line or a copy is shown, whatever a deleted line of its id was
  if (hiddenLines.delete(id)) send({ kind: "hiddenLines", lines: [...hiddenLines] });
  if (from === lineShapes.get(id)) return;
  if (from) lineShapes.set(id, from);
  else lineShapes.delete(id);
  send({ kind: "lineShapes", shapes: Object.fromEntries(lineShapes) });
}

export function apply(op: EditOp): Applied {
  freezeCheckpoint("apply-start", { op: op.op });
  const start = performance.now();
  try { return applyInner(op); }
  finally { freezeCheckpoint("apply-end", { op: op.op, ms: performance.now() - start }); }
}
function applyInner(op: EditOp): Applied {
  const a = api!;
  a.set_game_day(Math.floor(gameNow() / 86400)); // T-2871: what this edit constructs is dated today
  switch (op.op) {
    case "route": {
      const r = a.add_route(...routeArgs(a, op));
      return { r, node: r === 0 ? a.route_end() : undefined };
    }
    case "edgePis":
      a.set_design_speed(designOf(op));
      return { r: a.set_edge_pis(op.edge, pisArr(op.pis)) };
    case "moveNode":
      a.set_design_speed(designOf(op));
      return { r: a.move_node(op.node, op.x, op.y) };
    case "addPoint":
      return { r: op.x === undefined || op.y === undefined ? a.add_point_middle(op.edge) : a.add_point(op.edge, op.x, op.y) };
    case "moveNodes":
      a.set_design_speed(designOf(op));
      return { r: a.move_nodes(new Uint32Array(op.nodes), op.dx, op.dy) };
    case "edgeLevel":
      return { r: a.set_edge_level(op.edge, op.level) };
    case "edgesLevel": {
      const r = a.set_edges_level(new Uint32Array(op.edges), op.level);
      return { r, skipped: a.last_skipped() };
    }
    case "edgesKind": {
      const r = a.set_edges_kind(new Uint32Array(op.edges), op.kind);
      return { r, skipped: a.last_skipped() };
    }
    case "edgesTracks": {
      const r = a.set_edges_tracks(new Uint32Array(op.edges), op.tracks);
      const skipped = a.last_skipped();
      // T-3321: why some stayed, for the notice
      return { r, skipped, why: r === 0 && skipped ? issues() : undefined };
    }
    case "rebuild": {
      const r = a.rebuild_pieces(new Uint32Array(op.edges), new Uint32Array(op.stations), op.level ?? 99, op.kind, op.tracks);
      return { r, skipped: a.last_skipped() };
    }
    case "removeEdge":
      return { r: a.delete_edge(op.edge) };
    case "removePieces":
      return { r: a.delete_pieces(new Uint32Array(op.edges), new Uint32Array(op.stations)) };
    case "flying":
      return { r: a.set_flying(op.node, op.flying) };
    case "stationUpgrade":
      return { r: a.set_station_upgrade(new Uint32Array(op.nodes), op.bit, op.on) };
    case "addStation":
      a.set_station_one(op.at.one ? new Float64Array([op.at.one.x, op.at.one.y]) : new Float64Array(0));
      return op.at.kind === "node"
        ? { r: a.add_station(1, op.at.node, 0, 0, op.platform, op.name) }
        : op.at.kind === "edge"
          ? { r: a.add_station(2, op.at.edge, op.at.x, op.at.y, op.platform, op.name) }
          : (a.set_route_kind(op.at.tram ? 3 : 0), a.set_route_hsr(!!op.at.hsr), { r: a.add_free_station(freeEnds(op.at), op.at.quad ? 4 : 2, op.platform, op.name) });
    case "crossover":
      return { r: a.add_crossover(op.edge, op.x, op.y) };
    case "doubleUp":
      return { r: a.double_up(op.edge) };
    case "oneWay":
      return { r: a.set_edges_oneway(new Uint32Array([op.edge]), new Uint8Array([op.dir])) };
    case "oneWays":
      return { r: a.set_edges_oneway(new Uint32Array(op.edges), new Uint8Array(op.dirs)) };
    case "pieceLimit":
      return { r: a.set_edges_limit(new Uint32Array(op.edges), op.kmh) };
    case "stationTracks":
      return { r: a.set_station_tracks(op.node, op.one) };
    case "stationWay":
      return { r: a.set_station_way(op.node, op.heading) };
    case "reverseStation":
      return { r: a.reverse_station(op.node) };
    case "stationProps": {
      // the name a station shows is not always its stored one (T-3288, T-3759): sent back as shown, it is
      // no rename; another name with the platform as it is renames as the station panel does
      const stored = a.node_name(op.node), shown = stateRows.shownName(op.node) ?? stored;
      if (op.name === shown) return { r: a.set_station_props(op.node, op.platform, stored) };
      if (op.platform === stateRows.nodeRow(op.node)?.platform) return { r: renameStation(a, op.node, op.name) };
      return { r: a.set_station_props(op.node, op.platform, op.name) };
    }
    case "renameStation":
      return { r: renameStation(a, op.node, op.name) };
    case "renameComplex":
      return { r: renameComplex(a, op.node, op.name) };
    case "bindStation":
      return { r: a.set_station_bound(op.node, op.bound) };
    case "bindAll":
      return { r: a.bind_stations(a.unbound_stations()) };
    case "unbindAll":
      return { r: a.unbind_stations(new Uint32Array(op.nodes), op.names) };
    case "removeStation":
      return { r: a.remove_station(op.node) };
    case "construct":
      return { r: a.construct_with_buses(new Uint32Array(op.edges), new Uint32Array(op.stations), new Uint32Array(op.buses ?? [])) };
    case "constructAll":
      return { r: a.construct_all() };
    case "blueprintOff":
      return { r: a.set_blueprint_off(new Uint32Array(op.edges), new Uint32Array(op.stations), op.off) };
    case "constructLine":
      return { r: a.construct_line(op.line) };
    case "addBusLine": {
      const id = a.new_line_id();
      const { dwellS, turnaroundS } = NEW_LINE.bus;
      const [h, m, l] = headwaysFor("bus", newLineHeadway)!;
      return newLineWeekends(a, newLineRows(a, a.set_bus_line(id, Float64Array.from(op.route), h, m, l, dwellS, turnaroundS, colourInt(op.colour), op.name || nextBusName(a)), id, "bus"));
    }
    case "busRoute":
      return { r: a.set_bus_route(op.line, Float64Array.from(op.route)) };
    case "busSpacing":
      return { r: a.set_bus_spacing(op.line, op.spacing) };
    case "busHand":
      return { r: a.set_bus_hand(op.line) };
    case "busHandStop":
      return { r: a.set_bus_stop(op.line, op.x, op.y, op.on) };
    case "addFerryLine": {
      const ports = addPorts(a, op.ports);
      if (!ports) return { r: 1 };
      const id = a.new_line_id();
      const { dwellS, turnaroundS } = NEW_LINE.ferry;
      const [h, m, l] = headwaysFor("ferry", newLineHeadway)!;
      return newLineWeekends(a, newLineRows(a, a.set_ferry_line(id, ports, Float64Array.from(op.route), h, m, l, dwellS, turnaroundS, colourInt(op.colour), op.name || nextBusName(a, "Ferry ")), id, "ferry")); // i18n-ignore: a line's saved name
    }
    case "ferryRoute": {
      const ports = addPorts(a, op.ports);
      return ports ? { r: a.set_ferry_route(op.line, ports, Float64Array.from(op.route)) } : { r: 1 };
    }
    case "addLine": {
      const id = a.new_line_id();
      const { trains: [h, m, l], dwellS, turnaroundS } = NEW_LINE.train;
      // T-2926: a line on tram track starts with the tram's dwell and turnaround and 2 cars
      const tram = a.stops_tram(new Uint32Array(op.stops)); // T-3889: any tram stop
      const [tramDwell, tramTurn] = tram ? TrackApi.kind_defaults(3) : [dwellS, turnaroundS];
      const r = a.set_line_with_trains(id, new Uint32Array(op.stops), h, m, l, tramDwell, tramTurn, tram ? TRAM_NEW_CARS : newLineCars, colourInt(op.colour), op.name || a.next_line_name());
      if (r !== 0) return { r, line: id };
      // T-1899: a line started on high-speed track (a track's New line) runs high-speed trains
      let charge = a.last_charge(), trainCharge = a.last_train_charge();
      let fast = false;
      if (op.autoFast && hsrShare(a, id) > 0.5 && a.set_line_fast(id, true) === 0) {
        fast = true;
        charge += a.last_charge();
        trainCharge += a.last_train_charge();
      }
      // T-3543, T-3588: the headways the player made the default for this kind of line; a refused edit (cash) leaves it on its counts
      // T-3767: one for each of the line's rows
      const rows = a.line_period_count(id), wanted = headwaysFor(tram ? "tram" : fast ? "fast" : "rail", newLineHeadway, rows);
      if (wanted && (rows === 3 ? a.set_auto_headway(id, wanted[0], wanted[1], wanted[2]) : a.set_line_row_headways(id, Uint32Array.from(wanted))) === 0) {
        charge += a.last_charge();
        trainCharge += a.last_train_charge();
      }
      // T-3588: the weekend percent "Apply to all lines" set for new lines
      if (newLineWeekendShare !== null && newLineWeekendShare !== 100 && a.set_line_weekend_share(id, newLineWeekendShare) === 0) {
        charge += a.last_charge();
        trainCharge += a.last_train_charge();
      }
      return { r, line: id, charge, trainCharge };
    }
    case "copyLine": {
      const id = a.new_line_id();
      const existing = new Set<string>(), symbols = new Set<string>();
      const info = a.lines_info();
      let colour = 0;
      for (let i = 0; i < info.length; i += 22) {
        const nm = a.line_name(info[i]);
        existing.add(nm);
        symbols.add((a.line_symbol(info[i]) || letterOf(nm)).toLowerCase());
        if (info[i] === op.line) colour = info[i + 1];
      }
      const from = a.line_name(op.line);
      let name = cloneName(from, existing);
      if (op.name?.trim()) {
        const base = op.name.trim();
        name = base;
        for (let n = 2; existing.has(name); n++) name = `${base} ${n}`;
      }
      const r = a.copy_line(op.line, id, name);
      // T-1872: the copy always has a symbol of its own (the source's, or the one it shows, stepped on)
      // T-2791: a lettered name ("Line 1d") gives the symbol too ("1d") when that is free, so bullet and name agree
      const tail = /\d+[A-Za-z]$/.exec(name)?.[0];
      const symbol = tail && !op.name?.trim() && tail.length <= 5 && !symbols.has(tail.toLowerCase()) ? tail : cloneSymbol(a.line_symbol(op.line) || letterOf(from), symbols);
      if (r !== 0) return { r, line: id };
      a.set_line_badge(id, name, colour, symbol);
      // T-3834: a copy starts on the new-line default of its kind in every period the line uses, as a new line does
      let charge = 0, trainCharge = 0;
      const kind = a.is_ferry(op.line) ? "ferry" : a.is_bus(op.line) ? "bus" : a.line_tram(op.line) ? "tram" : a.line_fast(op.line) ? "fast" : "rail";
      const rows = a.line_period_count(id), wanted = headwaysFor(kind, newLineHeadway, rows);
      if (wanted && (rows === 3 ? a.set_auto_headway(id, wanted[0], wanted[1], wanted[2]) : a.set_line_row_headways(id, Uint32Array.from(wanted))) === 0) {
        charge += a.last_charge();
        trainCharge += a.last_train_charge();
      }
      if (newLineWeekendShare !== null && newLineWeekendShare !== 100 && a.set_line_weekend_share(id, newLineWeekendShare) === 0) {
        charge += a.last_charge();
        trainCharge += a.last_train_charge();
      }
      return { r, line: id, charge, trainCharge };
    }
    case "setStops": {
      const set = () => a.set_stops(op.line, new Uint32Array(op.stops));
      return op.autoFast ? withAutoFast(a, op.line, set) : { r: set() };
    }
    case "insertStop": {
      // one call: the stop put in, and an end it leaves on a loop of track turned round it (T-3122)
      const set = () => a.add_stop(op.line, op.station);
      const out = op.autoFast ? withAutoFast(a, op.line, set) : { r: set() };
      return out.r === 2 ? { ...out, r: INSERT_REFUSED } : out;
    }
    case "oneWayStop": {
      const set = () => a.one_way_stop(op.line, op.station);
      const out = op.autoFast ? withAutoFast(a, op.line, set) : { r: set() };
      return out.r === 2 ? { ...out, r: INSERT_REFUSED } : out;
    }
    case "dropStop": {
      const r = a.drop_stop(op.line, op.index);
      return { r: r === 0 ? 0 : DROP_REFUSED };
    }
    case "endLineAt": {
      const r = a.end_line_at(op.line, op.index);
      return { r: r === 0 ? 0 : DROP_REFUSED };
    }
    case "orderStops": {
      const fault = [...a.order_fault(new Uint32Array(op.stops))];
      if (fault.length === 3) return { r: ORDER_REFUSED, fault };
      return { r: a.order_stops(op.line, new Uint32Array(op.stops)) };
    }
    case "sortStops":
      return { r: a.sort_stops(op.line) };
    case "turnEnd":
      return { r: a.turn_end(op.line, op.station) };
    case "openLoop":
      return { r: a.open_loop(op.line) };
    case "stopPassed": {
      const r = a.stop_passed(op.line, op.station);
      return { r: r === 0 ? 0 : INSERT_REFUSED };
    }
    case "setTrainCount":
      return { r: op.unrecorded ? a.set_train_counts_unrecorded(op.line, ...op.trains) : a.set_train_counts(op.line, ...op.trains) };
    case "setHeadway":
      return { r: op.seconds ? a.set_auto_headway(op.line, ...op.seconds) : a.clear_auto_headway(op.line) };
    case "weekendShare":
      return { r: a.set_line_weekend_share(op.line, op.share) };
    case "weekendCounts":
      return { r: op.headways ? a.set_line_weekend_headways(op.line, ...op.counts, ...op.headways) : a.set_line_weekend_counts(op.line, ...op.counts) };
    case "weekendShareAll":
      putNewLineWeekendShare(op.share);
      saves.dirty = true;
      return { r: a.set_weekend_share_all(op.share) };
    case "setPeriods":
      return { r: op.line === undefined ? a.set_game_periods(op.rows) : a.set_line_periods(op.line, op.rows) };
    case "setHours":
      return { r: op.line === undefined ? a.set_game_hours(new Uint32Array(op.cuts)) : a.set_line_hours(op.line, new Uint32Array(op.cuts)) };
    case "setRowTrains":
      return { r: a.set_line_row_trains(op.line, Uint32Array.from(op.trains)) };
    case "setRowHeadways":
      return { r: a.set_line_row_headways(op.line, Uint32Array.from(op.seconds)) };
    case "rowWeekend":
      return { r: a.set_line_row_weekend(op.line, Uint32Array.from(op.counts), Uint32Array.from(op.headways)) };
    case "lineLook": {
      let colour = 0;
      if (op.colour === undefined) {
        const info = a.lines_info();
        for (let i = 0; i < info.length; i += 22) if (info[i] === op.line) colour = info[i + 1];
      }
      const name = op.name ?? a.line_name(op.line), col = op.colour === undefined ? colour : colourInt(op.colour);
      return { r: op.letter === undefined ? a.set_line_look(op.line, name, col) : a.set_line_badge(op.line, name, col, op.letter.trim()) };
    }
    case "removeLine":
      return { r: a.remove_line(op.line) };
    case "lineCars": {
      const tram = a.line_tram(op.line);
      const cars = Math.max(0, Math.min(tram ? TRAM_MAX_CARS : 255, Math.round(op.cars)));
      const r = a.set_line_cars(op.line, cars);
      if (r === 0 && !tram) putNewLineCars(cars);
      return { r };
    }
    case "lineFast":
      return { r: a.set_line_fast(op.line, op.fast) };
    case "lineOneWay":
      return { r: a.set_line_one_way(op.line, op.oneWay) };
    case "linePairs":
      return { r: a.set_line_pairs(op.line, op.on) };
    case "lineLimit":
      return { r: a.set_line_limit(op.line, op.kmh) };
    case "lineFareDist":
      return { r: a.set_line_fare_dist(op.line, op.dist) };
    case "lineGroup":
      return { r: a.set_line_group(new Uint32Array(op.lines), op.group === "new" ? a.new_line_group() : op.group) };
    case "groupMembers":
      return { r: a.set_group_members(new Uint32Array(op.lines), new Uint32Array(op.groups), op.group === "new" ? a.new_line_group() : op.group, op.parent) };
    case "groupName":
      return { r: a.set_line_group_name(op.group, op.name) };
    case "groupLook":
      return { r: a.set_line_group_look(op.group, op.symbol.trim(), op.colour === null ? -1 : colourInt(op.colour)) };
    case "fares":
      putFares(capTransfer({ ...(op.nodes ? faresOf({ nodes: op.nodes }, fares) : faresOf({ base: op.base, perKm: op.perKm }, fares)), scales: fares.scales, baseline: FARE_BASELINE, basis: "usd", income: FARE_E, ...(fares.hsr !== undefined ? { hsr: fares.hsr } : {}), ...(fares.ferry !== undefined ? { ferry: fares.ferry } : {}), ...(fares.bus !== undefined ? { bus: fares.bus } : {}), ...(fares.transfer !== undefined ? { transfer: fares.transfer } : {}), ...(fares.ref ? { ref: fares.ref } : {}), ...(fares.lines ? { lines: fares.lines } : {}), ...zonesOf(fares), ...cityFares(fares) }));
      return { r: 0 };
    case "fareZones": {
      // T-3740: the fare areas as a whole; an id is never used twice
      const { zones: _, zoneNext: __, ...rest } = fares;
      putFares({ ...rest, ...zonesOf({ zones: op.zones, zoneNext: Math.max(fares.zoneNext ?? 1, op.zoneNext ?? 1) }) });
      return { r: 0 };
    }
    case "stationFares":
      return { r: a.set_station_fares(new Uint32Array(op.nodes), new Uint32Array(op.codes)) };
    case "fareLine":
      return { r: Number.isFinite(op.mul) && setLineFare(a, op.line, op.mul) ? 0 : 1 };
    case "fareHsr":
      if (!Number.isFinite(op.mul)) return { r: 1 };
      putFares({ ...fares, hsr: tidyHsr(op.mul) });
      return { r: 0 };
    case "fareFerry":
      if (!Number.isFinite(op.mul)) return { r: 1 };
      putFares({ ...fares, ferry: tidyHsr(op.mul) });
      return { r: 0 };
    case "fareBus":
      if (!Number.isFinite(op.mul)) return { r: 1 };
      putFares({ ...fares, bus: tidyHsr(op.mul) });
      return { r: 0 };
    case "fareTransfer": {
      if (!Number.isFinite(op.usd)) return { r: 1 };
      const { transfer: _, ...rest } = fares, v = tidyTransfer(op.usd);
      putFares(capTransfer(v > 0 ? { ...rest, transfer: v } : rest));
      return { r: 0 };
    }
    case "fareScale": {
      const scales = { ...fares.scales };
      if (op.scale === null || !Number.isFinite(op.scale)) delete scales[op.country];
      else scales[op.country] = keepOwn(op.scale);
      putFares({ ...fares, scales });
      return { r: 0 };
    }
    case "fareCityScale": {
      if (typeof op.city !== "string" || !isCity(op.city)) return { r: 1 };
      const { cityScales: was, ...rest } = fares, own = { ...was };
      if (op.scale === null || !Number.isFinite(op.scale)) delete own[op.city];
      else own[op.city] = keepOwn(op.scale);
      putFares(Object.keys(own).length ? { ...rest, cityScales: own } : rest);
      return { r: 0 };
    }
    case "fareByCity": {
      const { byCity: _, ...rest } = fares;
      putFares(op.on ? { ...rest, byCity: true } : rest);
      return { r: 0 };
    }
    case "fareRef":
      if (typeof op.country !== "string" || !op.country) return { r: 1 };
      putFares({ ...fares, ref: op.country });
      return { r: 0 };
    case "borrow":
    case "repay":
      return { r: loanOp(op) };
    case "link":
      return { r: op.on ? a.link_stations(op.a, op.b) : a.unlink_stations(op.a, op.b) };
    case "addBuilding":
      return { r: a.add_building(new Float64Array(op.corners), op.name) };
    case "reshapeBuilding":
      return { r: a.reshape_building(op.id, new Float64Array(op.corners)) };
    case "renameBuilding":
      return { r: a.rename_building(op.id, op.name) };
    case "removeBuilding":
      return { r: a.remove_building(op.id) };
    case "undo":
      return { r: a.undo() };
    case "redo":
      return { r: a.redo() };
  }
}

/** T-3844: what an edit paid for, named in its answer: the line a line edit set (its trains), the stations and
 * pieces a station or piece edit changed. `chargeShares` places the charge by it when the world shows no new
 * track or station. Not named: undo and redo (they pay only for trains, of whichever line the undone edit set),
 * and the edits that set every line at once (`setPeriods`, `setHours` and `weekendShareAll` without a line). */
export function paidFor(op: EditOp): PaidFor | undefined {
  const o = op as Record<string, unknown>, paid: PaidFor = {};
  if (typeof o.line === "number" && op.op !== "copyLine") paid.line = o.line;
  const nodes: number[] = [], edges: number[] = [];
  switch (op.op) {
    case "stationUpgrade": case "moveNodes": nodes.push(...op.nodes); break;
    case "flying": case "stationTracks": case "stationWay": case "reverseStation": case "stationProps": case "moveNode": nodes.push(op.node); break;
    case "edgeLevel": case "doubleUp": case "edgePis": edges.push(op.edge); break;
    case "edgesLevel": case "edgesKind": case "edgesTracks": edges.push(...op.edges); break;
    case "rebuild": edges.push(...op.edges); nodes.push(...op.stations); break;
  }
  if (nodes.length) paid.nodes = nodes;
  if (edges.length) paid.edges = edges;
  return paid.line !== undefined || nodes.length || edges.length ? paid : undefined;
}

/** `apply`'s answer for an insertStop with nowhere to go (the track model's own codes are >= 0) */
export const INSERT_REFUSED = -1;
/** the same for a dropStop or endLineAt the line cannot take (T-1791) */
export const DROP_REFUSED = -2;
/** the same for an orderStops with no route between two stops in a row (T-2895) */
export const ORDER_REFUSED = -3;

export function issues(): string[] {
  return api!.issue_kinds().split("\n").filter(Boolean);
}

/** T-2218: the "buildingQuote" query's answer: [m², US$M, stations touching, built ones], then the
 * refusal's kinds as character codes, joined by newlines (none: it can be built). */
export function buildingQuoteAnswer(id: number | null, corners: number[]): Float64Array {
  const q = api!.building_quote(id ?? 0xffffffff, new Float64Array(corners));
  const kinds = api!.issue_kinds();
  return new Float64Array([...q, ...Array.from(kinds, (c) => c.charCodeAt(0))]);
}

/** T-3321: the "rebuildQuote" query's answer: the quote's length, `TrackApi::rebuild_quote`'s quote, then the
 * kinds of why pieces are refused as character codes, joined by newlines. */
export function rebuildQuoteAnswer(edges: number[], stations: number[], level: number | null, kind: number, tracks: number): Float64Array {
  const q = api!.rebuild_quote(new Uint32Array(edges), new Uint32Array(stations), level ?? 99, kind, tracks);
  const kinds = api!.issue_kinds();
  const out = new Float64Array(1 + q.length + kinds.length);
  out[0] = q.length;
  out.set(q, 1);
  for (let i = 0; i < kinds.length; i++) out[1 + q.length + i] = kinds.charCodeAt(i);
  return out;
}

export function doubleUpAnswer(edge: number): Float64Array {
  const price = api!.preview_double_up(edge);
  if (price >= 0) return new Float64Array([price]);
  const kinds = api!.issue_kinds();
  return new Float64Array([-1, ...Array.from(kinds, (c) => c.charCodeAt(0))]);
}

/** T-1685: the pieces a route beside track runs along, for `set_route_beside` (none: u32::MAX) */
const besideOf = (r: { beside?: [number, number] }): [number, number] => r.beside ?? [0xffffffff, 0xffffffff];
/** The design speed new track is laid for, km/h, where the request names none */
export const designOf = (r: { design?: number }) => r.design ?? 160;
/** `r`'s settings set on the track model, and its ends, points and tracks as `TrackApi.add_route`
 * and `preview_route` take them. */
export function routeArgs(a: TrackApi, r: RouteAsk): [Float64Array, Float64Array, number, boolean] {
  a.set_route_tangent(!!r.through);
  a.set_design_speed(designOf(r));
  a.set_flyover_speed(r.flyover ?? 80);
  a.set_route_kind(r.tram ? (r.street ? 2 : 3) : 0);
  a.set_route_street(new Float64Array(r.tram && r.street ? r.streetLine ?? [] : []));
  a.set_route_hsr(!!r.hsr);
  a.set_route_ramp(!!r.ramp);
  a.set_route_beside(...besideOf(r));
  a.set_route_join(new Float64Array((r.join ?? [null, null, null, null]).map((v) => v ?? NaN)));
  // T-3092: with ramps on, new junctions on double track are laid as flyovers
  return [new Float64Array([...endArr(r.from), ...endArr(r.to)]), pisArr(r.pis), r.single ? 1 : r.quad ? 4 : 2, !!r.ramp];
}
