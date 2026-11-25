// T-1464: a bus line's panel, the rail line panel cut down (notes/T-1445.md): its schedule in buses,
// its stops with the street stops folded into a row, and the past 24 hours with the riders changing to
// trains. No train kind, length, stop editor or related-lines diagram. LineInspector.tsx hands a bus
// line over to it. T-2194: a ferry line too (its row is a bus row with `ferry` set), in boats, its
// stops its ports. The schedule with its crowding dots and schedule panel button (T-3703), the title, the past 24 hours,
// the stop times, the north-up order and the round trip are the rail panel's (schedule.tsx,
// LineTitle.tsx, LineInspector.tsx; T-2373, T-2919).

import { ConstructButton } from "./ConstructButton";
import type { ComponentChildren } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { shown } from "../../game/pending";
import { demandAt, lineSeconds } from "../../game/clock";
import { periodAt } from "../../game/periods";
import { count, distance, distUnit, duration, exact, metresFromShown, metresToShown, millions, money, perDistance, shortDistance, speed } from "../../game/format";
import { moneyState } from "../../game/money";
import { BOATS, boatOf, boatStock, type Boat } from "../../game/boats";
import { t } from "../../i18n";
import { tipMutedLine } from "../tip";
import { useToday } from "../../game/today";
import { DEMAND_INDEX, type LineInput, type LineStats } from "../../game/types";
import { ask, display, lineDraft, selection, tool } from "../../game/ui";
import { live } from "../../map/live";
import { editBusRoute, editFerryRoute, toggleEditStops } from "../../game/lineActions";
import { busDraft } from "../../map/busDraw";
import { ferryDraft } from "../../map/ferryDraw";
import { edit, fleetOf, linesById, namedBusStops, say, stationNames, world } from "../../game/world";
import { Alert, Icon, Row, Segmented, Stepper, SwapLabel } from "../widgets";
import { useLineQuote } from "../../game/quotes";
import { nearestStation } from "../../game/places";
import { fullTitle, heldTrains, LANE_KIND } from "../../game/capacityKinds";
import { useClockQuery } from "../useClockQuery";
import { demandView, complexTransfers } from "../../game/demand";
import { placesNear, hoodsNear, townsNear } from "../../game/placeList";
import { endsInStation, placeName } from "../../map/stationNames";
import type { ClockBus } from "../../workers/protocol";
import { limitsFrom, RoundTrip, ScheduleButtons, StopToday, StopTransfers, StretchHalf, stretchLooks } from "./LineInspector";
import { LineFare, LineMoney } from "./LineMoney";
import { PurposeSplit } from "./PurposeSplit";
import { drawnReversed } from "../../game/northUp";
import { AverageSpeed, boatsBought, NO_LIMITS, PastDay, rowLabel, Schedule, useLinePeriods } from "./schedule";
import { LineTitle, useLineTitle } from "./LineTitle";
import { afterCopy } from "../../game/lineGroups";

/** What a bus stands at a stop that is no train station, s (`bus::STREET_DWELL_S`). */
const STREET_STAND_S = 20;

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
type StopRow = { kind: "end" | "station"; i: number; station: string } | { kind: "street"; n: number; from: number } | { kind: "named"; i: number };

/** The ends and stations met, in order, each gap's street stops folded into one row: the stops
 * demand sees between them (kinds 2 and 3) and the drawn street stops. */
function stopRows(bus: ClockBus, named: ReadonlyMap<string, string>): StopRow[] {
  const anchors = bus.stops.map((s, i) => ({ s, i })).filter((a) => a.s.kind < 2);
  const where = anchors.map((a) => along(bus.route, a.s.x, a.s.y));
  // T-4080: each gap's stops in order along the route: the street stops, and the ones the player named by their index
  const gaps = anchors.map(() => [] as { pos: number; named: number }[]);
  const gapAt = (pos: number) => {
    let k = 0;
    while (k + 1 < anchors.length && where[k + 1] <= pos) k++;
    return k;
  };
  let k = -1;
  bus.stops.forEach((s, i) => {
    if (s.kind < 2) k++;
    else if (anchors.length) gaps[Math.max(0, k)].push({ pos: along(bus.route, s.x, s.y), named: s.station < 0 && named.has(String(s.id)) ? i : -1 });
  });
  for (let j = 0; j + 1 < bus.street.length; j += 2) if (anchors.length) {
    const pos = along(bus.route, bus.street[j], bus.street[j + 1]);
    gaps[gapAt(pos)].push({ pos, named: -1 });
  }
  const rows: StopRow[] = [];
  anchors.forEach((a, k) => {
    rows.push({ kind: a.s.kind === 0 ? "end" : "station", i: a.i, station: a.s.station >= 0 ? String(a.s.station) : "" });
    if (k + 1 >= anchors.length) return;
    let run = 0;
    for (const g of gaps[k].sort((x, y) => x.pos - y.pos)) {
      if (g.named < 0) { run++; continue; }
      if (run) rows.push({ kind: "street", n: run, from: k });
      run = 0;
      rows.push({ kind: "named", i: g.named });
    }
    if (run) rows.push({ kind: "street", n: run, from: k });
  });
  return rows;
}

/** Every stop of a route, the street stops included; a loop's first stop once. */
const stopCount = (bus: ClockBus, loop: boolean) => bus.stops.length + (bus.street.length >> 1) - (loop ? 1 : 0);

const endNames = new Map<string, string>();
const endKey = (lng: number, lat: number) => `${lng.toFixed(4)},${lat.toFixed(4)}`;

/** An end's name as a new station is named from its place (the list's neighbourhood, else the nearest
 * place, else the town), minus the street the basemap would give at street zoom: a name that stays
 * the same at any zoom. "End" where the places list holds nothing near. */
async function nameEnd(lng: number, lat: number, taken: Set<string>): Promise<string> {
  const [listed, towns] = await Promise.all([placesNear(lng, lat).catch(() => []), townsNear(lng, lat).catch(() => [])]);
  const near = listed.filter((n) => !endsInStation(n.place.name, n.place.kind));
  return placeName(hoodsNear(near), taken) ?? placeName(near, taken) ?? placeName(towns.filter((n) => !endsInStation(n.place.name, n.place.kind)), taken) ?? t("busInspector.end");
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

/** how far a row is dragged before it counts as a drag, not a click (px) */
const DRAG_PX = 4;

/** T-2275: a ferry line's ports drawn as a rail line's stops strip, the Traffic view's stretch widths
 * included (T-2274). A click on a port's dot takes it out of the line and a drag of its row moves it,
 * the route searched again through the ports (`FerryDraw.setPorts`); not while the ferry tool has the
 * line, nor with only two ports. A loop (T-2899) keeps its order: taking out its first or last port opens
 * it, and a ring of three with one taken out is a line of two. */
function FerryPorts({ l, st, bus, colour, editable, flip, time }: { l: LineInput; st: LineStats; bus: ClockBus; colour: string; editable: boolean; flip: boolean; time: (i: number) => ComponentChildren }) {
  const [drag, setDrag] = useState<{ from: number; gap: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const dragged = useRef(false);
  const ports = bus.ferry?.ports ?? [];
  const n = ports.length;
  const names = new Map((world.value?.save.stations ?? []).map((s) => [s.num, s.name]));
  const nameOf = (i: number) => names.get(ports[i]?.node ?? -1) ?? t("busInspector.port");
  const looks = display.value.trackColour === "traffic" && st.status === "running" && l.stops.length === n ? stretchLooks(l, ports.map((_, i) => nameOf(i)), []) : null;
  const loop = n >= 4 && ports[0].node === ports[n - 1].node;
  const canEdit = editable && !busy && n > 2;
  const canMove = canEdit && !loop;
  const apply = async (order: number[]) => {
    const f = live.tools?.ferry;
    if (!f || busy) return;
    setBusy(true);
    try { await f.setPorts(l.num, order); } finally { setBusy(false); }
  };
  const all = ports.map((_, i) => i);
  // T-2919: the rows north end on top as a rail line's; a row's place in the list to its port
  const shown = flip ? all.slice().reverse() : all;
  const remove = (i: number) => void apply(loop ? (i === 0 || i === n - 1 ? all.slice(0, -1) : all.filter((k) => k !== i && !(n <= 4 && k === n - 1))) : all.filter((k) => k !== i));
  const pointerDown = (e: PointerEvent, from: number) => {
    if (!canMove || e.button !== 0 || e.pointerType === "touch" || (e.target as Element | null)?.closest?.(".stop-hit, button")) return;
    const x0 = e.clientX, y0 = e.clientY;
    let on = false, gap = from;
    const move = (m: PointerEvent) => {
      if (!on && Math.hypot(m.clientX - x0, m.clientY - y0) < DRAG_PX) return;
      on = true;
      const row = document.elementFromPoint(m.clientX, m.clientY)?.closest("[data-port]") as HTMLElement | null;
      if (row) {
        const b = row.getBoundingClientRect(), k = Number(row.dataset.port);
        gap = m.clientY - b.top < b.height / 2 ? k : k + 1;
      }
      setDrag({ from, gap });
    };
    const end = (u: PointerEvent | KeyboardEvent) => {
      removeEventListener("pointermove", move);
      removeEventListener("pointerup", end);
      removeEventListener("keydown", key);
      setDrag(null);
      if (!on) return;
      dragged.current = true;
      setTimeout(() => { dragged.current = false; }, 0);
      if (u.type !== "pointerup" || gap === from || gap === from + 1) return;
      const order = shown.filter((_, k) => k !== from);
      order.splice(gap > from ? gap - 1 : gap, 0, shown[from]);
      void apply(flip ? order.reverse() : order);
    };
    const key = (k: KeyboardEvent) => { if (k.key === "Escape") end(k); };
    addEventListener("pointermove", move);
    addEventListener("pointerup", end);
    addEventListener("keydown", key);
  };
  const moving = drag && drag.gap !== drag.from && drag.gap !== drag.from + 1 ? drag : null;
  return (
    <ul class={"stops ferry-ports" + (looks ? " traffic" : "") + (canMove ? " movable" : "") + (drag ? " dragging" : "")} style={{ "--c": colour }}>
      {shown.map((i, at) => {
        const p = ports[i];
        // the stretches above and below the row
        const above = flip ? (i < n - 1 ? i : -1) : i - 1, below = flip ? i - 1 : i < n - 1 ? i : -1;
        return (
          <li data-port={at} class={"xfer" + (st.oneWayAt?.[String(p.node)] !== undefined ? " oneway" : "") + (drag?.from === at ? " dragged" : "") + (moving?.gap === at ? " drop-before" : moving && at === n - 1 && moving.gap === n ? " drop-after" : "")}
            onPointerDown={(e) => pointerDown(e, at)}>
            {looks && above >= 0 && looks[above] && <StretchHalf s={looks[above]} down={false} />}
            {looks && below >= 0 && looks[below] && <StretchHalf s={looks[below]} down />}
            {canEdit && <span class="stop-hit" data-tip={t("busInspector.removeStop", { name: nameOf(i) })} onClick={() => remove(i)} />}
            <span class="grow">
              <span class="stop-name" onClick={() => { if (!dragged.current) selection.value = { kind: "station", station: String(p.node) }; }}>{nameOf(i)}</span>
              {time(i)}
            </span>
            <StopTransfers l={l} i={i} />
            <StopToday line={l.id} i={i} />
          </li>
        );
      })}
    </ul>
  );
}

/** T-2885: the steps of a bus line's stop spacing under Auto, m. The first, 400, is the stops placing
 * themselves (the sim stores it as 0, T-2947). A typed distance may fall between. */
const SPACING_STEPS = [400, 600, 800, 1000, 1500, 2000, 3000, 5000, 8000, 10_000, 15_000, 20_000, 30_000, 50_000, 100_000];

/** T-2937: the Stops heading with its Auto and Manual buttons, and under Auto the Stop spacing row.
 * Manual (T-2891) opens the stop editing; any spacing, 0 included, leaves it. */
function StopsHead({ l, bus }: { l: LineInput; bus: ClockBus }) {
  // what was asked for shows at once, until the rows that hold it (game/busAhead.ts)
  const hand = shown("line:" + l.num, "hand", bus.hand);
  const spacing = Math.max(SPACING_STEPS[0], shown("line:" + l.num, "spacing", bus.spacing));
  const send = (to: { hand: boolean; spacing: number }) =>
    edit(to.hand ? { op: "busHand", line: l.num } : { op: "busSpacing", line: l.num, spacing: to.spacing }).then((a) => {
      if (a.ok && to.hand) toggleEditStops(l);
      return a.ok;
    });
  const choose = (manual: boolean) => { if (manual !== hand) void send({ hand: manual, spacing: 0 }); };
  const step = (d: number) => {
    const at = SPACING_STEPS.reduce((best, s, i) => (s <= spacing ? i : best), 0);
    const to = Math.max(0, Math.min(SPACING_STEPS.length - 1, d > 0 || SPACING_STEPS[at] === spacing ? at + d : at + d + 1));
    return SPACING_STEPS[to] === spacing ? false : send({ hand: false, spacing: SPACING_STEPS[to] });
  };
  const typed = (v: number) => void send({ hand: false, spacing: Math.max(SPACING_STEPS[0], metresFromShown(v)) });
  return (
    <>
      <div class="h divided">
        {t("busInspector.stops")}
        <span class="right" data-tip={hand ? t("busInspector.spacingHandTip") : undefined}>
          <Segmented<boolean> options={[[false, t("busInspector.spacingAuto")], [true, t("busInspector.spacingHand")]]} value={hand} onChange={choose} />
        </span>
      </div>
      {!hand && (
        <ul class="rows">
          <Row label={<span data-tip={t("busInspector.spacingTip", { auto: spacing === SPACING_STEPS[0] })}>{t("busInspector.spacing")}</span>}>
            <Stepper wide label={t("busInspector.spacing")} value={shortDistance(spacing)} onStep={step}
              type={{ text: String(+metresToShown(spacing).toFixed(2)), set: typed }} />
          </Row>
        </ul>
      )}
    </>
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

function ChangingToTrains({ l, bus }: { l: LineInput; bus: ClockBus }) {
  const n = useChangesToTrains(l, bus);
  if (n === null) return null;
  return (
    <li data-tip={t("busInspector.changingTip", { vehicle: bus.ferry ? "boat" : "bus" })}>
      <span class="grow">{t("busInspector.changing")}</span>
      <span class="val">{count(n)}</span>
    </li>
  );
}

// ---------------------------------------------------------------- the panel

/** T-3892: a ferry line's boat size, where a train line has its train type: each hover its riders and speed, and on
 * a running line what the switch buys beyond that size's spare boats (the old ones stay in the depot). */
function BoatSize({ l, st, bus }: { l: LineInput; st: LineStats; bus: ClockBus }) {
  const m = moneyState.value, now = shown("line:" + l.num, "boat", boatOf(bus));
  const tip = (b: Boat) => {
    let s = t("busInspector.boatTip", { riders: exact(BOATS[b].riders), speed: speed(BOATS[b].kmh) });
    const need = fleetOf(st);
    if (m && b !== now && st.status === "running" && need > 0) {
      const k = boatStock(m, b);
      s += "\n" + boatsBought({ price: k.price, spare: Math.max(0, k.owned - k.needed) }, need);
    }
    return s;
  };
  const sizes: [Boat, string][] = [[0, t("busInspector.boatSmall")], [1, t("busInspector.boatMedium")], [2, t("busInspector.boatLarge")]];
  return (
    <li>
      <span class="grow">{t("busInspector.boat")}</span>
      <Segmented<Boat>
        options={sizes.map(([b, name]) => [b, <span data-tip={tip(b)}>{name}</span>])}
        value={now}
        onChange={(boat) => { if (boat !== now) void edit({ op: "ferryBoat", line: l.num, boat }); }}
      />
    </li>
  );
}

/** T-3907: a bus line on BRT or off it: bus lanes along the whole route, its buses at no less than the BRT floor.
 * T-3944: an upgrade as a station's are (StationUpgrades.tsx): outlined when off, dashed while its lanes are in the
 * blueprint, filled once built. A click puts it in the blueprint or takes it out, or takes built lanes away (nothing
 * refunded); a constructed line's lanes in the blueprint are built by Construct beside it or the Build tab's. */
function BrtRow({ l, bus }: { l: LineInput; bus: ClockBus }) {
  const m = moneyState.value;
  const asked = shown<boolean | null>("line:" + l.num, "brt", null);
  const on = asked ?? (bus.brt || bus.brtBp);
  const built = on && asked === null && bus.brt && bus.built;
  // the Construct beside it: a constructed line whose lanes are in the blueprint
  const pending = asked === null && bus.built && bus.brtBp;
  const q = useLineQuote(pending ? l.num : null);
  // T-3907: paid and upkept over the road in town only; T-3378: lanes another BRT line has are not paid again
  const shared = bus.townM - bus.newM;
  const price = m ? (bus.newM / 1000) * m.brtRouteKm * 1e6 : 0;
  const act = t(on ? (built ? "stationUpgrades.actBuilt" : "stationUpgrades.actPlanned") : "stationUpgrades.actAdd");
  const tip = m ? [t("busInspector.brt"), t("busInspector.brtTip", { speed: speed(m.brtKmh), price: millions(perDistance(m.brtRouteKm)), upkeep: money(perDistance(m.upkeepBrtKm * m.economy)), unit: distUnit(), town: distance(bus.townM) }),
    ...(shared >= 50 ? [t("busInspector.brtShared", { shared: distance(shared) })] : []), ...(built ? [] : [t("stationUpgrades.costs", { price: money(price) })]), act].join("\n") : undefined;
  const set = (to: boolean) => {
    void edit({ op: "busBrt", line: l.num, on: to }).then((a) => {
      if (!a.ok) return;
      if (to) say(t("stationUpgrades.added", { name: t("busInspector.brt") }));
      else if (built) say(t("stationUpgrades.removed", { name: t("busInspector.brt"), station: l.name }));
    });
  };
  const build = async () => {
    const a = await edit({ op: "constructLine", line: l.num });
    if (a.ok) say(t("busInspector.brtBuilt", { name: l.name, price: money(a.charge * 1e6) }));
  };
  return (
    <li class="brt-up">
      <button class={"btn icon upgrade" + (!on ? "" : built ? " on" : " planned")} aria-label={t("busInspector.brt")} aria-pressed={on} data-tip={tip} onClick={() => set(!on)}>
        <Icon.brt />
      </button>
      <span class="grow" data-tip={tip}>{t("busInspector.brt")}</span>
      {pending && (
        <button class="btn text" onClick={() => void build()}>
          {q ? t("constructButton.priced", { price: money(q.build + q.trains) }) : t("constructButton.plain")}
        </button>
      )}
    </li>
  );
}

/** T-3378: a constructed BRT line's fullest stretch of lane now, buses an hour one way against the most it takes;
 * red once full, when the lines on it run fewer buses than asked. */
function LaneRow({ bus, period }: { bus: ClockBus; period: number }) {
  const most = bus.laneLoad[5] ?? 0, n = bus.laneLoad[period] ?? 0;
  if (!bus.brt || !bus.built || !(n > 0) || !(most > 0)) return null;
  const full = n >= most - 0.05;
  return (
    <li>
      <span class="grow">{t("busInspector.lane")}</span>
      <span class={"val" + (full ? " bad" : "")} data-tip={full ? t("busInspector.laneFullTip", { most: Math.round(most) }) : undefined}>{t("busInspector.laneLoad", { n: Math.round(n), most: Math.round(most) })}</span>
    </li>
  );
}

interface LaneHold { row: number; rows: number; wait: number; place: string }
/** T-3378: a BRT line's `line_held` (`line_row_held`): where a full lane keeps some of its buses from running, by
 * row, and the headways the lanes limit it to (`limitsFrom`). */
function useLaneHeld(l: LineInput | undefined, bus: ClockBus | undefined) {
  const w = world.value;
  return useClockQuery(l && bus?.brt && bus.built ? { what: "lineHeld", line: l.num, rows: true } : null, [l?.num, w?.version, bus?.brt], (d) => {
    const rows = Math.round(d.length / 13), held: LaneHold[] = [];
    for (let r = 0; r < rows; r++) {
      const [wait, kind, , x, y] = d.subarray(7 * r, 7 * r + 5);
      if (kind === LANE_KIND && heldTrains(wait) > 0) held.push({ row: r, rows, wait, place: fullTitle(kind, nearestStation(x, y)?.name ?? "", "bus") });
    }
    return { held, limits: limitsFrom(d, "bus") };
  });
}

/** T-3378: the alert of a full BRT lane, as a rail line's of full track (`lineInspector.heldLevel`). */
function LaneHeld({ st, lanes }: { st: LineStats; lanes: LaneHold[] }) {
  if (st.status !== "running" || lanes.length === 0) return null;
  return (
    <ul class="alerts">
      {[...new Set(lanes.map((h) => h.place))].map((place) => (
        <Alert tip={t("busInspector.laneFullTip", { most: Math.round(st.bus?.laneLoad[5] ?? 0) })}>
          {place}
          {lanes.filter((h) => h.place === place).map((h) => <div class="detail">{t("lineInspector.heldLevel", { level: rowLabel(h.rows, h.row), text: t("busInspector.laneHeld", { n: heldTrains(h.wait) }) })}</div>)}
        </Alert>
      ))}
    </ul>
  );
}

export function BusInspector({ id }: { id: string }) {
  const w = world.value;
  const [copying, setCopying] = useState(false);
  const l = w ? linesById(w).get(id) : undefined;
  const title = useLineTitle(id, l);
  const now = demandAt(lineSeconds(w?.lineStats[id]));
  const lev = DEMAND_INDEX[now];
  const { kind, periods } = useLinePeriods(id, l);
  useToday();
  const st = w && l ? w.lineStats[l.id] : undefined;
  const bus = st?.bus;
  const endName = useEndNames(bus, new Set((w?.save.stations ?? []).map((s) => s.name)));
  const lanes = useLaneHeld(l, bus);
  if (!w || !l || !st || !bus) return null;
  const names = stationNames(w);

  const times = st.fromStartByLevel[lev] ?? st.fromStartS;
  const rows = stopRows(bus, namedBusStops(w));
  const ferry = !!bus.ferry;
  const drawing = ferry ? tool.value === "ferry" && ferryDraft.value?.line === l.num : tool.value === "bus" && busDraft.value?.line === l.num;
  const editingStops = tool.value === "line" && lineDraft.value.line === l.num;

  const build = async () => {
    const a = await edit({ op: "constructLine", line: l.num });
    if (a.ok) say(t("busInspector.constructed", { name: l.name, price: money(a.charge * 1e6), ferry }));
  };
  const removeNow = () => void edit({ op: "removeLine", line: l.num }).then((a) => {
    if (!a.ok) return;
    const selected = selection.peek();
    if (selected?.kind === "line" && selected.line === l.id) selection.value = null;
  });
  const remove = (e: MouseEvent) => {
    // T-2919: asked only where it cannot be undone, a constructed bus route (final, nothing refunded);
    // a planned bus line or a ferry line (T-2194: its ports stay, its boats go to the depot) goes at once
    if (ferry || !bus.built || e.shiftKey) return removeNow();
    ask.value = { text: t("busInspector.deleteQuestion", { name: l.name, built: true }), yes: t("busInspector.deleteYes"), no: t("busInspector.deleteNo"), run: () => {
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
      selection.value = afterCopy(world.peek()?.save.lines ?? [], a.line);
    } finally { setCopying(false); }
  };

  const shown = title.shown;
  const stopName = (r: StopRow & { kind: "end" | "station" }) => (r.kind === "end" ? endName.get(r.i) ?? t("busInspector.end") : names.get(r.station) ?? t("busInspector.station"));
  // T-2919: the rail panel's order and stop times: the north (or west) end on top, each stop's time
  // from the one on top after its name
  const first = bus.stops[0], last = bus.stops[bus.stops.length - 1];
  const flip = drawnReversed(first, last, !!st.loop);
  const total = times[times.length - 1] ?? 0;
  const stopTime = (i: number) => {
    const top = flip ? i === bus.stops.length - 1 : i === 0;
    return !top && times[i] !== undefined && <span class="t muted num">{duration(flip ? total - times[i] : times[i])}</span>;
  };
  // a bus stands the line's dwell at train stations and a short stand at its other stops; a boat the dwell at each port
  const mid = bus.stops.slice(1, -1);
  const lap = mid.filter((s) => s.station >= 0).length * st.dwellS + (mid.filter((s) => s.station < 0).length + (bus.street.length >> 1)) * STREET_STAND_S;
  const standing = ferry ? undefined : st.loop ? lap + st.dwellS : 2 * lap;
  const between = (r: StopRow & { kind: "street" }) => {
    const anchored = rows.filter((x) => x.kind === "end" || x.kind === "station");
    const from = anchored[r.from] as StopRow & { kind: "end" | "station" } | undefined;
    const to = anchored[r.from + 1] as StopRow & { kind: "end" | "station" } | undefined;
    return from && to ? t("busInspector.streetBetween", { count: r.n, from: stopName(from), to: stopName(to) }) : t("busInspector.streetStops", { count: r.n });
  };

  return (
    <>
      <LineTitle l={l} t={title} />
      <div class="sub muted">
        {t("busInspector.sub", { count: stopCount(bus, !!st.loop), ferry, distance: distance(bus.roadM) })}
      </div>
      {st.status === "planned" && (
        <div class="callout">
          <p>{t("busInspector.planned")}</p>
          {ferry
            ? <ConstructButton num={l.num} build={build} kind="boat" />
            : <ConstructButton num={l.num} build={build} kind="bus" />}
        </div>
      )}

      <div class="line-stop-actions">
        <button class={"btn text" + (drawing ? " on" : "")} data-tip={drawing ? t("busInspector.saveTip") : bus.built && !ferry ? t("busInspector.changeTipBuilt") : t("busInspector.changeTip")} onClick={() => (ferry ? editFerryRoute(l) : editBusRoute(l))}><Icon.pencil />{drawing ? t("busInspector.save") : t("busInspector.edit")}</button>
        {bus.hand && <button class={"btn text" + (editingStops ? " on" : "")} data-tip={editingStops ? t("lineInspector.finishTip") : t("busInspector.editStopsTip")} onClick={() => toggleEditStops(l)}><Icon.pencil /><SwapLabel on={editingStops} off={t("lineInspector.editStops")} onText={t("lineInspector.stopEditing")} /></button>}
        <button class="btn text" onClick={() => void copy()} disabled={copying}>{t("busInspector.copy")}</button>
        <button class="btn text danger" data-tip={tipMutedLine(t("common.shiftSkip"))} onClick={remove}>{t("busInspector.deleteYes")}</button>
      </div>

      <Schedule l={l} st={st} period={periodAt(lineSeconds(st))} limits={lanes?.limits ?? NO_LIMITS} crowd={periods} copy={<ScheduleButtons l={l} st={st} />} />
      <ul class="rows">
        {ferry && <BoatSize l={l} st={st} bus={bus} />}
        {!ferry && <BrtRow l={l} bus={bus} />}
        {!ferry && <LaneRow bus={bus} period={periodAt(lineSeconds(st))} />}
        <AverageSpeed st={st} lev={lev} />
      </ul>
      <LaneHeld st={st} lanes={lanes?.held ?? []} />

      {st.status === "running" && <div class="line-day"><PastDay l={l} st={st} periods={periods} kind={kind} extra={<><PurposeSplit scope={[{ line: l.id }]} unit={t("lineInspector.ridersUnit")} /><ChangingToTrains l={l} bus={bus} /></>} /></div>}

      {ferry ? (
        <div class="h divided">
          {t("busInspector.ports")}
          <span class="right muted small">{t("cityDetails.boardingsToday")}</span>
        </div>
      ) : <StopsHead l={l} bus={bus} />}
      {ferry ? <FerryPorts l={l} st={st} bus={bus} colour={shown} editable={!drawing} flip={flip} time={stopTime} /> : <ul class="bus-stops" style={{ "--c": shown }}>
        {(flip ? rows.slice().reverse() : rows).map((r) => r.kind === "street" ? (
          <li class="street" data-tip={between(r)}>
            <i class="bs-mk" />
            <span class="grow muted">{t("busInspector.streetStops", { count: r.n })}</span>
          </li>
        ) : r.kind === "named" ? (
          <li class="named">
            <i class="bs-mk" />
            <span class="grow">
              <span class="stop-name" onClick={() => { const s = bus.stops[r.i]; selection.value = { kind: "busStop", stop: String(s.id), at: [s.lng, s.lat] }; }}>{names.get(String(bus.stops[r.i].id))}</span>
              {stopTime(r.i)}
            </span>
          </li>
        ) : (
          <li class={r.kind}>
            <i class={"bs-mk " + r.kind} />
            <span class="grow">
              {r.kind === "station"
                ? <span class="stop-name" onClick={() => (selection.value = { kind: "station", station: r.station })}>{stopName(r)}</span>
                : <span data-tip={t("busInspector.endTip")}>{stopName(r)}</span>}
              {stopTime(r.i)}
            </span>
            {r.kind === "station" && <StopTransfers l={l} i={r.i} />}
          </li>
        ))}
      </ul>}

      {l.stops.length >= 2 && <RoundTrip num={l.num} st={st} lev={lev} stops={l.stops.length} standing={standing} />}
      <LineFare l={l} />
      {st.status === "running" && <LineMoney num={l.num} />}
    </>
  );
}
