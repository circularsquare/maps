// T-1464: a bus line's panel, the rail line panel cut down (notes/T-1445.md): its schedule in buses,
// its stops with the street stops folded into a row, and the past 24 hours with the riders changing to
// trains. No train kind, length, stop editor or related-lines diagram. LineInspector.tsx hands a bus
// line over to it. T-2194: a ferry line too (its row is a bus row with `ferry` set), in boats, its
// stops its ports. The schedule, title and past 24 hours are the rail panel's (schedule.tsx,
// LineTitle.tsx, T-2373).

import { ConstructButton } from "./ConstructButton";
import type { ComponentChildren } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { demandAt, lineSeconds } from "../../game/clock";
import { count, distance, money } from "../../game/format";
import { t } from "../../i18n";
import { useToday } from "../../game/today";
import { DEMAND_INDEX, type LineInput, type LineStats } from "../../game/types";
import { ask, display, selection, tool } from "../../game/ui";
import { live } from "../../map/live";
import { editBusRoute, editFerryRoute } from "../../game/lineActions";
import { busDraft } from "../../map/busDraw";
import { ferryDraft } from "../../map/ferryDraw";
import { edit, linesById, say, stationNames, world } from "../../game/world";
import { Icon } from "../widgets";
import { demandView, complexTransfers } from "../../game/demand";
import { placesNear, hoodsNear, townsNear } from "../../game/placeList";
import { endsInStation, placeName } from "../../map/stationNames";
import type { ClockBus } from "../../workers/protocol";
import { StopTransfers, StretchHalf, stretchLooks } from "./LineInspector";
import { AverageSpeed, PastDay, Schedule, useLinePeriods } from "./schedule";
import { LineTitle, useLineTitle } from "./LineTitle";
import { afterCopy } from "../../game/lineGroups";

const minutesText = (s: number) => t("inspector.minShort", { n: Math.round(s / 60) });

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
 * line, nor with only two ports. */
function FerryPorts({ l, st, bus, colour, editable, time }: { l: LineInput; st: LineStats; bus: ClockBus; colour: string; editable: boolean; time: (i: number) => ComponentChildren }) {
  const [drag, setDrag] = useState<{ from: number; gap: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const dragged = useRef(false);
  const ports = bus.ferry?.ports ?? [];
  const n = ports.length;
  const names = new Map((world.value?.save.stations ?? []).map((s) => [s.num, s.name]));
  const nameOf = (i: number) => names.get(ports[i]?.node ?? -1) ?? t("busInspector.port");
  const looks = display.value.trackColour === "traffic" && st.status === "running" && l.stops.length === n ? stretchLooks(l, ports.map((_, i) => nameOf(i)), []) : null;
  const canEdit = editable && !busy && n > 2;
  const apply = async (order: number[]) => {
    const f = live.tools?.ferry;
    if (!f || busy) return;
    setBusy(true);
    try { await f.setPorts(l.num, order); } finally { setBusy(false); }
  };
  const all = ports.map((_, i) => i);
  const remove = (i: number) => void apply(all.filter((k) => k !== i));
  const pointerDown = (e: PointerEvent, from: number) => {
    if (!canEdit || e.button !== 0 || e.pointerType === "touch" || (e.target as Element | null)?.closest?.(".stop-hit, button")) return;
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
      const order = all.filter((k) => k !== from);
      order.splice(gap > from ? gap - 1 : gap, 0, from);
      void apply(order);
    };
    const key = (k: KeyboardEvent) => { if (k.key === "Escape") end(k); };
    addEventListener("pointermove", move);
    addEventListener("pointerup", end);
    addEventListener("keydown", key);
  };
  const moving = drag && drag.gap !== drag.from && drag.gap !== drag.from + 1 ? drag : null;
  return (
    <ul class={"stops ferry-ports" + (looks ? " traffic" : "") + (canEdit ? " movable" : "") + (drag ? " dragging" : "")} style={{ "--c": colour }}>
      {ports.map((p, i) => (
        <li data-port={i} class={"xfer" + (drag?.from === i ? " dragged" : "") + (moving?.gap === i ? " drop-before" : moving && i === n - 1 && moving.gap === n ? " drop-after" : "")}
          onPointerDown={(e) => pointerDown(e, i)}>
          {looks && i > 0 && looks[i - 1] && <StretchHalf s={looks[i - 1]} down={false} />}
          {looks && i < n - 1 && looks[i] && <StretchHalf s={looks[i]} down />}
          {canEdit && <span class="stop-hit" data-tip={t("busInspector.removeStop", { name: nameOf(i) })} onClick={() => remove(i)} />}
          <span class="grow">
            <span class="stop-name" onClick={() => { if (!dragged.current) selection.value = { kind: "station", station: String(p.node) }; }}>{nameOf(i)}</span>
          </span>
          <StopTransfers l={l} i={i} />
          {time(i)}
        </li>
      ))}
    </ul>
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
  if (!w || !l || !st || !bus) return null;
  const names = stationNames(w);

  const times = st.fromStartByLevel[lev] ?? st.fromStartS;
  const rows = stopRows(bus);
  const ferry = !!bus.ferry;
  const drawing = ferry ? tool.value === "ferry" && ferryDraft.value?.line === l.num : tool.value === "bus" && busDraft.value?.line === l.num;

  const build = async () => {
    const a = await edit({ op: "constructLine", line: l.num });
    if (a.ok) say(t("busInspector.constructed", { name: l.name, price: money(a.charge * 1e6), ferry }));
  };
  const removeNow = () => void edit({ op: "removeLine", line: l.num }).then((a) => {
    if (!a.ok) return;
    const selected = selection.peek();
    if (selected?.kind === "line" && selected.line === l.id) selection.value = null;
  });
  const remove = () => {
    // T-2194: a ferry line leaves its ports and its boats go to the depot, so it goes at once (undoable)
    if (ferry || stopCount(bus) === 0) return removeNow();
    ask.value = { text: t("busInspector.deleteQuestion", { name: l.name, built: bus.built }), yes: t("busInspector.deleteYes"), no: t("busInspector.deleteNo"), run: () => {
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
  const stopTime = (i: number) => times[i] !== undefined && (
    <span class="n num muted" data-tip={i === 0 ? t("busInspector.firstStop") : t("busInspector.reached", { n: Math.round(times[i] / 60) })}>{minutesText(times[i])}</span>
  );
  const between = (r: StopRow & { kind: "street" }) => {
    const from = rows.filter((x) => x.kind !== "street")[r.from] as StopRow & { kind: "end" | "station" } | undefined;
    const to = rows.filter((x) => x.kind !== "street")[r.from + 1] as StopRow & { kind: "end" | "station" } | undefined;
    return from && to ? t("busInspector.streetBetween", { count: r.n, from: stopName(from), to: stopName(to) }) : t("busInspector.streetStops", { count: r.n });
  };

  return (
    <>
      <LineTitle l={l} t={title} />
      <div class="sub muted">
        {t("busInspector.sub", { count: stopCount(bus), ferry, distance: distance(bus.roadM) })}
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
        <button class="btn text" onClick={() => void copy()} disabled={copying}>{t("busInspector.copy")}</button>
        <button class="btn text danger" onClick={remove}>{t("busInspector.deleteYes")}</button>
      </div>

      <Schedule l={l} st={st} now={now} />
      <ul class="rows">
        <AverageSpeed st={st} lev={lev} />
      </ul>

      <div class="h divided">{ferry ? t("busInspector.ports") : t("busInspector.stops")}</div>
      {ferry ? <FerryPorts l={l} st={st} bus={bus} colour={shown} editable={!drawing} time={stopTime} /> : <ul class="bus-stops" style={{ "--c": shown }}>
        {rows.map((r) => r.kind === "street" ? (
          <li class="street" data-tip={between(r)}>
            <i class="bs-mk" />
            <span class="grow muted">{t("busInspector.streetStops", { count: r.n })}</span>
          </li>
        ) : (
          <li class={r.kind}>
            <i class={"bs-mk " + r.kind} />
            <span class="grow">
              {r.kind === "station"
                ? <span class="stop-name" onClick={() => (selection.value = { kind: "station", station: r.station })}>{stopName(r)}</span>
                : <span data-tip={t("busInspector.endTip")}>{stopName(r)}</span>}
            </span>
            {r.kind === "station" && <StopTransfers l={l} i={r.i} />}
            {stopTime(r.i)}
          </li>
        ))}
      </ul>}

      {st.status === "running" && <PastDay l={l} st={st} periods={periods} kind={kind} extra={<ChangingToTrains l={l} bus={bus} />} />}
    </>
  );
}
