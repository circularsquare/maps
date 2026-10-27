// T-1464: a bus line's panel, the rail line panel cut down (notes/T-1445.md): its schedule in buses,
// its stops with the street stops folded into a row, and the past 24 hours with the riders changing to
// trains. No train kind, length, stop editor or related-lines diagram. LineInspector.tsx hands a bus
// line over to it.

import type { ComponentChildren } from "preact";
import { useEffect, useState } from "preact/hooks";
import { demandAt, lineSeconds } from "../../game/clock";
import { busesText, busRunRule, count, distance, headway, money, plural, speed } from "../../game/format";
import { LINE_PALETTE } from "../../game/palette";
import { PERIODS } from "../../game/periods";
import { runningToday, today, useToday } from "../../game/today";
import { accountVersion } from "../../game/boarding";
import { kindIn, lineCountry } from "../../game/holidays";
import { DEMAND_INDEX, DEMANDS, type Demand, type LineInput, type LineStats } from "../../game/types";
import { ask, renameLinePrompt, selection, tool } from "../../game/ui";
import { liveMoney, moneyState } from "../../game/money";
import { editBusRoute } from "../../game/lineActions";
import { busDraft } from "../../map/busDraw";
import { ColourPicker } from "../ColourPicker";
import { autoTrainsFor, edit, headwayFromMinutes, headwayShown, linesById, nextHeadway, refusedForCash, renameLine, say, setAutoTrains, setHeadway, stationNames, stepHeadway, stepTrainCount, trainsShown, world } from "../../game/world";
import { Chip, Icon, Row, Segmented, Stepper } from "../widgets";
import { NameEdit } from "./NameEdit";
import { locateLine } from "../locate";
import { demandPending, Spinner } from "../Spinner";
import { demandView, complexTransfers } from "../../game/demand";
import { useClockQuery } from "../useClockQuery";
import { placesNear, hoodsNear, townsNear } from "../../game/placeList";
import { endsInStation, placeName } from "../../map/stationNames";
import type { ClockBus } from "../../workers/protocol";
import { DEMAND_LABEL, levelTimes, StopTransfers, useLiveColour } from "./LineInspector";

/** A bus holds this many riders at crush (`BUS_CRUSH`, sim/src/track/params.rs). */
const BUS_CRUSH = 80;
/** How long a refused edit's control shakes, ms (style.css `.shake`). */
const SHAKE_MS = 300;
/** Numbers per period in `TrackApi.line_periods` (`PERIOD_STRIDE` in boarding.rs). */
const PERIOD_STRIDE = 11;

const headwayText = (s: number) => (s > 0 ? `${+(s / 60).toFixed(2)} min` : "none");
const minutesText = (s: number) => `${Math.round(s / 60)}m`;

// ---------------------------------------------------------------- the stops

/** Where (x, y) falls along the route, in the route's own units; only used to put points in order. */
function along(route: number[], x: number, y: number): number {
  let best = Infinity, at = 0, run = 0;
  for (let i = 0; i + 3 < route.length; i += 2) {
    const [ax, ay, bx, by] = [route[i], route[i + 1], route[i + 2], route[i + 3]];
    const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy, len = Math.sqrt(len2);
    const t = len2 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / len2)) : 0;
    const d = Math.hypot(ax + t * dx - x, ay + t * dy - y);
    if (d < best) { best = d; at = run + t * len; }
    run += len;
  }
  return at;
}

/** One row of the stops list: an end or a station by its stop index, or a run of street stops. */
type StopRow = { kind: "end" | "station"; i: number; station: string } | { kind: "street"; n: number; from: number };

/** The ends and stations met, in order, each gap's street stops folded into one row: the stops
 * demand sees between them (kinds 2 and 3) and the drawn street stops. */
function stopRows(bus: ClockBus): StopRow[] {
  const anchors = bus.stops.map((s, i) => ({ s, i })).filter((a) => a.s.kind < 2);
  const where = anchors.map((a) => along(bus.route, a.s.x, a.s.y));
  const between = anchors.map(() => 0);
  const gapAt = (pos: number) => {
    let k = 0;
    while (k + 1 < anchors.length && where[k + 1] <= pos) k++;
    return k;
  };
  let k = -1;
  bus.stops.forEach((s) => {
    if (s.kind < 2) k++;
    else if (anchors.length) between[Math.max(0, k)]++;
  });
  for (let j = 0; j + 1 < bus.street.length; j += 2) if (anchors.length) between[gapAt(along(bus.route, bus.street[j], bus.street[j + 1]))]++;
  const rows: StopRow[] = [];
  anchors.forEach((a, k) => {
    rows.push({ kind: a.s.kind === 0 ? "end" : "station", i: a.i, station: a.s.station >= 0 ? String(a.s.station) : "" });
    if (k + 1 < anchors.length && between[k] > 0) rows.push({ kind: "street", n: between[k], from: k });
  });
  return rows;
}

/** Every stop of a route, the street stops included. */
const stopCount = (bus: ClockBus) => bus.stops.length + (bus.street.length >> 1);

const endNames = new Map<string, string>();
const endKey = (lng: number, lat: number) => `${lng.toFixed(4)},${lat.toFixed(4)}`;

/** An end's name as a new station is named from its place (the list's neighbourhood, else the nearest
 * place, else the town), minus the street the basemap would give at street zoom: a name that stays
 * the same at any zoom. "End" where the places list holds nothing near. */
async function nameEnd(lng: number, lat: number, taken: Set<string>): Promise<string> {
  const [listed, towns] = await Promise.all([placesNear(lng, lat).catch(() => []), townsNear(lng, lat).catch(() => [])]);
  const near = listed.filter((n) => !endsInStation(n.place.name, n.place.kind));
  return placeName(hoodsNear(near), taken) ?? placeName(near, taken) ?? placeName(towns.filter((n) => !endsInStation(n.place.name, n.place.kind)), taken) ?? "End";
}

/** The names of a route's two ends, by stop index; empty until the places list has answered. */
function useEndNames(bus: ClockBus | undefined, stationsTaken: Set<string>): Map<number, string> {
  const [, bump] = useState(0);
  const ends = (bus?.stops ?? []).map((s, i) => ({ s, i })).filter((a) => a.s.kind === 0);
  const key = ends.map((a) => endKey(a.s.lng, a.s.lat)).join("|");
  useEffect(() => {
    let live = true;
    void (async () => {
      const taken = new Set(stationsTaken);
      for (const { s } of ends) {
        const k = endKey(s.lng, s.lat);
        if (!endNames.has(k)) endNames.set(k, await nameEnd(s.lng, s.lat, taken));
        taken.add(endNames.get(k)!);
      }
      if (live) bump((n) => n + 1);
    })();
    return () => { live = false; };
  }, [key]);
  return new Map(ends.flatMap((a) => (endNames.has(endKey(a.s.lng, a.s.lat)) ? [[a.i, endNames.get(endKey(a.s.lng, a.s.lat))!] as const] : [])));
}

// ---------------------------------------------------------------- the schedule

/** What a stepper's button leads to: the bus count, and for a running line the buses it needs beyond
 * its busiest level, bought unless the depot has spare ones, or the buses it frees. */
function changeTip(counts: [number, number, number], lev: number, to: number, st: LineStats): string {
  const head = busesText(to);
  const m = moneyState.value;
  if (!m || st.status !== "running") return head;
  const next = [...counts];
  next[lev] = to;
  const more = Math.max(...next) - Math.max(...counts);
  if (more > 0) {
    const bought = Math.max(0, more - Math.max(0, m.fleetBus - m.busesNeeded));
    return head + "\n" + (bought > 0 ? `Buys ${busesText(bought)} for ${money(bought * m.busPrice * 1e6)}` : `Uses ${busesText(more)} from the depot`);
  }
  return more < 0 ? head + `\nFrees ${busesText(-more)}, which stay in the depot` : head;
}

/** One demand level's bus count and headway cells: a count stepper and the headway it gives, or the
 * headway stepper and the count it gives. A count refused for cash is red and shakes once. */
function LevelCells({ l, st, d, auto }: { l: LineInput; st: LineStats; d: Demand; auto: [number, number, number] | null }) {
  const lev = DEMAND_INDEX[d];
  const trip = st.roundTripByLevel[lev];
  const autoPending = auto ? auto.map((s, i) => s !== st.autoS?.[i]) : null;
  const counts = (auto
    ? st.trainsNeeded.map((n, i) => (autoPending![i] ? autoTrainsFor(st.roundTripByLevel[i], auto[i], 0, false) : n))
    : (trainsShown(l.id) ?? st.trainsNeeded)) as [number, number, number];
  const r = refusedForCash.value;
  const refused = r && r.id === l.id && r.levs[lev] ? r : null;
  const broke = refused !== null && liveMoney.value.cash < refused.need;
  const shake = refused !== null && performance.now() - refused.at < SHAKE_MS;
  const wrap = (child: ComponentChildren) => <span key={refused?.at ?? 0} class={shake ? "shake" : undefined}>{child}</span>;
  const n = <span class={broke ? "bad" : undefined}>{counts[lev]}</span>;
  if (!auto) {
    const countPending = counts[lev] !== st.trainsNeeded[lev];
    const tph = countPending ? (trip > 0 ? (counts[lev] * 3600) / trip : 0) : l.tph[d];
    return (
      <>
        <td>
          {wrap(<Stepper value={n} label="buses on the line" jump onStep={(s) => stepTrainCount(l.id, d, s)}
            tipDown={changeTip(counts, lev, Math.max(0, counts[lev] - 1), st)} tipUp={changeTip(counts, lev, counts[lev] + 1, st)} />)}
        </td>
        <td>{headway(tph)}</td>
      </>
    );
  }
  const typeTip = "Click to type a headway in minutes. 0 is no service.";
  return (
    <>
      <td>
        <span class={"num" + (broke ? " bad" : "")} data-tip={`${busesText(counts[lev])} on the line keep the headway over its round trip`}>{counts[lev]}</span>
      </td>
      <td>
        {wrap(<Stepper wide label="headway" value={headwayText(auto[lev])} onStep={(s) => stepHeadway(l.id, d, s)}
          tipDown={changeTip(counts, lev, autoTrainsFor(trip, nextHeadway(auto[lev], -1), 0, false), st)}
          tipUp={changeTip(counts, lev, autoTrainsFor(trip, nextHeadway(auto[lev], 1), 0, false), st)}
          type={{ text: auto[lev] ? String(+(auto[lev] / 60).toFixed(2)) : "0", set: (v) => setHeadway(l.id, d, headwayFromMinutes(v)), tip: typeTip }} />)}
      </td>
    </>
  );
}

/** The line's average speed end to end, there and back over the round trip less the turning at the ends. */
function AverageSpeed({ st, lev }: { st: LineStats; lev: number }) {
  const rt = st.roundTripByLevel[lev] ?? st.roundTripS;
  const s = rt - 2 * st.turnaroundS;
  if (!(s > 0) || !(st.km > 0)) return null;
  const kmh = (2 * st.km) / (s / 3600);
  return (
    <li data-tip={`${distance(st.km * 2000)} there and back in ${Math.round(s / 60)} min, the round trip less the turning at the ends. Dwell at stations is in it.`}>
      <span class="grow">Average speed</span>
      <span class="val">{speed(kmh)}</span>
    </li>
  );
}

// ---------------------------------------------------------------- the past 24 hours

/** The riders a day changing between this bus and a rail line, from demand's transfer pairs at the
 * stations it meets (T-176); null until demand has answered for the solve shown. */
function useChangesToTrains(l: LineInput, bus: ClockBus): number | null {
  const [n, setN] = useState<number | null>(null);
  const dv = demandView.value;
  const members = bus.stops.filter((s) => s.station >= 0).map((s) => String(s.station));
  const key = l.id + "|" + members.join(",");
  useEffect(() => {
    if (!dv) return;
    if (!members.length) { setN(0); return; }
    let live = true;
    void complexTransfers(members).then((r) => {
      if (!live || !r) return;
      const stats = world.peek()?.lineStats ?? {};
      const rail = (id: string) => !!stats[id] && !stats[id].bus;
      let sum = 0;
      for (const p of r.pairs) if ((p.fromLine === l.id && rail(p.toLine)) || (p.toLine === l.id && rail(p.fromLine))) sum += p.riders;
      setN(sum);
    });
    return () => { live = false; };
  }, [dv, key]);
  return n;
}

/** The day's rider-seconds aboard over what the trips could hold at crush; null with no trips. */
function averageLoad(v: Float64Array, st: LineStats): number | null {
  let aboard = 0, room = 0;
  for (let i = 0; i < PERIODS.length; i++) {
    const o = i * PERIOD_STRIDE;
    if (!(v[o] > 0)) continue;
    const times = st.fromStartByLevel[DEMAND_INDEX[PERIODS[i].demand]] ?? st.fromStartS;
    aboard += v[o + 5];
    room += v[o] * (times[times.length - 1] ?? 0) * BUS_CRUSH;
  }
  return room > 0 ? Math.min(1, aboard / room) : null;
}

function PastDay({ l, st, bus, periods, kind }: { l: LineInput; st: LineStats; bus: ClockBus; periods: Float64Array | null; kind: number }) {
  const d = today.value.get(l.id);
  const changing = useChangesToTrains(l, bus);
  if (!d) return null;
  const load = periods ? averageLoad(periods, st) : null;
  const run = runningToday(l, st);
  const m = moneyState.value;
  const rule = m ? `\nRunning costs ${busRunRule(m)}` : "";
  return (
    <>
      <div class="h divided">
        <span data-tip="The past 24 hours">Past 24 hours</span>
      </div>
      <ul class="rows">
        <li data-tip={`${d.riders.toLocaleString("en-US")} riders boarded in the past 24 hours`}>
          <span class="grow">Riders</span>
          <span class="val">{count(d.riders)}</span>
        </li>
        {changing !== null && (
          <li data-tip="Riders a day who change between this bus and a train">
            <span class="grow">Changing to trains</span>
            <span class="val">{count(changing)}</span>
          </li>
        )}
        <li data-tip={`Fares ${money(d.fares)} minus running cost ${money(run)} for the buses and upkeep of its route ${money(st.upkeep)}: ${money(d.fares - run - st.upkeep)} net${rule}`}>
          <span class="grow">Profit</span>
          <span class="val">{money(d.fares)} −<span class="crowd-bad">{money(run + st.upkeep)}</span></span>
        </li>
        {load !== null && (
          <li data-tip={`Riders aboard on average, as a share of the ${BUS_CRUSH} a bus holds at crush, over every trip on a ${["weekday", "Saturday", "Sunday", "holiday"][kind] ?? "day"}`}>
            <span class="grow">Average load</span>
            <span class="val">{Math.round(load * 100)}%</span>
          </li>
        )}
        {d.fullTrains > 0 && <Row label="Buses leaving full"><span data-tip={`${count(d.fullTrains)}/${count(d.trains)} buses left full`}>{count(d.fullTrains)}/{count(d.trains)}</span></Row>}
      </ul>
    </>
  );
}

// ---------------------------------------------------------------- the panel

export function BusInspector({ id }: { id: string }) {
  const w = world.value;
  const [editFor, setEditFor] = useState<string | null>(null);
  const [previewOf, setPreviewOf] = useState<{ id: string; hex: string } | null>(null);
  const editing = editFor === id;
  const setEditing = (open: boolean) => setEditFor(open ? id : null);
  const preview = previewOf?.id === id ? previewOf.hex : null;
  const setPreview = (hex: string | null) => setPreviewOf(hex === null ? null : { id, hex });
  const [copying, setCopying] = useState(false);
  const l = w ? linesById(w).get(id) : undefined;
  const { dragColour, commitColour } = useLiveColour(l, setPreview);
  const [symbol, setSymbol] = useState("");
  useEffect(() => { setSymbol(l?.letter ?? ""); }, [l?.letter]);
  useEffect(() => { if (preview !== null && l && l.colour.toLowerCase() === preview.toLowerCase()) setPreview(null); }, [l?.colour, preview]);
  useEffect(() => {
    if (!editing) return;
    const away = (e: PointerEvent) => { if (!(e.target as Element | null)?.closest?.("[data-line-edit]")) setEditing(false); };
    const esc = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || (e.target as Element | null)?.closest?.(".name-edit")) return;
      e.preventDefault();
      e.stopPropagation();
      setEditing(false);
    };
    window.addEventListener("pointerdown", away, true);
    window.addEventListener("keydown", esc, true);
    return () => {
      window.removeEventListener("pointerdown", away, true);
      window.removeEventListener("keydown", esc, true);
    };
  }, [editing]);
  const now = demandAt(lineSeconds(w?.lineStats[id]));
  const lev = DEMAND_INDEX[now];
  const kind = kindIn(lineCountry(l?.stops ?? []), Math.floor(lineSeconds(w?.lineStats[id]) / 86400));
  const acct = accountVersion.value;
  const periods = useClockQuery(l ? { what: "linePeriods", line: l.num, day: kind } : null, [l?.num, w?.version, acct, kind], (d) => (d.length ? d : null));
  useToday();
  const st = w && l ? w.lineStats[l.id] : undefined;
  const bus = st?.bus;
  const endName = useEndNames(bus, new Set((w?.save.stations ?? []).map((s) => s.name)));
  if (!w || !l || !st || !bus) return null;
  const names = stationNames(w);

  const times = st.fromStartByLevel[lev] ?? st.fromStartS;
  const auto = headwayShown(l.id);
  const rows = stopRows(bus);
  const drawing = tool.value === "bus" && busDraft.value?.line === l.num;

  const build = async () => {
    const a = await edit({ op: "constructLine", line: l.num });
    if (a.ok) say(`${l.name} constructed for ${money(a.charge * 1e6)}. Its buses start with the next departure.`);
  };
  const remove = () => {
    ask.value = { text: bus.built ? `Delete ${l.name}? Nothing is refunded for the route.` : `Delete ${l.name}?`, yes: "Delete line", no: "Keep line", run: () => {
      ask.value = null;
      void edit({ op: "removeLine", line: l.num }).then((a) => {
        if (!a.ok) return;
        const selected = selection.peek();
        if (selected?.kind === "line" && selected.line === l.id) selection.value = null;
      });
    }};
  };
  const copy = async () => {
    if (copying) return;
    setCopying(true);
    try {
      const a = await edit({ op: "copyLine", line: l.num });
      if (!a.ok || a.line === undefined) return;
      renameLinePrompt.value = String(a.line);
      selection.value = { kind: "line", line: String(a.line) };
    } finally { setCopying(false); }
  };

  const shown = preview ?? l.colour;
  const stopName = (r: StopRow & { kind: "end" | "station" }) => (r.kind === "end" ? endName.get(r.i) ?? "End" : names.get(r.station) ?? "Station");
  const stopTime = (i: number) => times[i] !== undefined && (
    <span class="n num muted" data-tip={i === 0 ? "The first stop" : `Reached ${Math.round(times[i] / 60)} min after the first stop`}>{minutesText(times[i])}</span>
  );
  const between = (r: StopRow & { kind: "street" }) => {
    const from = rows.filter((x) => x.kind !== "street")[r.from] as StopRow & { kind: "end" | "station" } | undefined;
    const to = rows.filter((x) => x.kind !== "street")[r.from + 1] as StopRow & { kind: "end" | "station" } | undefined;
    return from && to ? `${plural(r.n, "street stop")} between ${stopName(from)} and ${stopName(to)}` : plural(r.n, "street stop");
  };

  return (
    <>
      <div class="title" data-line-edit>
        <button class={"badge-edit" + (editing ? " on" : "")} data-tip="Edit the name, symbol and colour (double click also opens it)" aria-label="Edit the name, symbol and colour"
          onClick={() => setEditing(true)} onDblClick={() => setEditing(true)}><Chip line={{ ...l, colour: shown }} size="lg" plain /></button>
        <NameEdit key={l.id} name={l.name} onRename={(n) => renameLine(l.id, n)} prompt={renameLinePrompt.value === id} onPrompted={() => { renameLinePrompt.value = null; }}
          control={{ open: editing, set: setEditing }}>
          <button class="btn icon" data-tip="Show on the map" aria-label="Show on the map" onClick={() => locateLine(l.stops)}>
            <Icon.locate />
          </button>
        </NameEdit>
      </div>
      {editing && (
        <div class="line-identity" data-line-edit>
          <label>Symbol <input aria-label="Line symbol" value={symbol} onInput={(e) => setSymbol(e.currentTarget.value)} onChange={(e) => void edit({ op: "lineLook", line: l.num, name: l.name, colour: l.colour, letter: e.currentTarget.value.trim() })} /></label>
          <ColourPicker value={l.colour} presets={LINE_PALETTE} onPreview={dragColour} onCommit={commitColour} />
        </div>
      )}
      <div class="sub muted">
        {plural(stopCount(bus), "stop")}, {distance(bus.roadM)}
      </div>
      {st.status === "planned" && (
        <div class="callout">
          <p>Planned</p>
          <button class="btn text" data-tip={`Route ${money(st.buildCost)}${st.trainCost > 0 ? `, buses ${money(st.trainCost)}` : ""}. Buses run once it is constructed.`} onClick={build}>
            Construct for {money(st.buildCost + st.trainCost)}
          </button>
        </div>
      )}

      <div class="line-stop-actions">
        <button class={"btn text" + (drawing ? " on" : "")} data-tip={drawing ? "Save the route (Enter)" : bus.built ? "Change the route. The old route is not refunded" : "Change the route"} onClick={() => editBusRoute(l)}><Icon.pencil />{drawing ? "Save route" : "Edit route"}</button>
        <button class="btn text" onClick={() => void copy()} disabled={copying}>Copy line</button>
        <button class="btn text danger" onClick={remove}>Delete line</button>
      </div>

      <div class="h">
        Schedule
        <span class="right">
          <Segmented<boolean>
            options={[
              [false, <span data-tip="Set the number of buses on the line for each demand level">Buses</span>],
              [true, <span data-tip="Set the headway for each demand level. The line runs the fewest buses that keep it over the round trip.">Headway</span>],
            ]}
            value={auto !== null}
            onChange={(v) => setAutoTrains(l.id, v)}
          />
        </span>
      </div>
      <table class="sched">
        <tr>
          <th>Demand</th>
          <th>Buses</th>
          <th>Headway</th>
        </tr>
        {DEMANDS.map((d) => (
          <tr class={d === now ? "now" : ""}>
            <td>{st.status === "running" && demandPending.value && <Spinner class="row-dot" tip="Figures appear when demand has loaded" />}<span data-tip={levelTimes(d) + "\nHow many buses run in this time interval."}>{DEMAND_LABEL[d]}</span></td>
            <LevelCells l={l} st={st} d={d} auto={auto} />
          </tr>
        ))}
      </table>
      <ul class="rows">
        <AverageSpeed st={st} lev={lev} />
      </ul>

      <div class="h divided">Stops</div>
      <ul class="bus-stops" style={{ "--c": shown }}>
        {rows.map((r) => r.kind === "street" ? (
          <li class="street" data-tip={between(r)}>
            <i class="bs-mk" />
            <span class="grow muted">{plural(r.n, "street stop")}</span>
          </li>
        ) : (
          <li class={r.kind}>
            <i class={"bs-mk " + r.kind} />
            <span class="grow">
              {r.kind === "station"
                ? <span class="stop-name" onClick={() => (selection.value = { kind: "station", station: r.station })}>{stopName(r)}</span>
                : <span data-tip="An end of the route">{stopName(r)}</span>}
            </span>
            {r.kind === "station" && <StopTransfers l={l} i={r.i} />}
            {stopTime(r.i)}
          </li>
        ))}
      </ul>

      {st.status === "running" && <PastDay l={l} st={st} bus={bus} periods={periods} kind={kind} />}
    </>
  );
}
