// T-2949, T-3890: a bus stop's panel, made like a station's: its name, the lines stopping at its place
// (the stops and train stations demand counts as one place, game/busLinks.ts `busPlaces`) with their
// boardings today, the riders waiting and the places a short walk away with other lines. A train
// station's stop opens the station's panel instead (map/pick.ts).

import { effect } from "@preact/signals";
import { t } from "../../i18n";
import { busPlaces, groundM, type BusPlace, type BusPlaces } from "../../game/busLinks";
import { count, length } from "../../game/format";
import type { LineInput, WorldView } from "../../game/types";
import { selection } from "../../game/ui";
import { today, useToday } from "../../game/today";
import { busStopsById, edit, linesById, stationNames, stationsById, world } from "../../game/world";
import { live } from "../../map/live";
import { locateStation } from "../locate";
import { Built, Chip, Icon } from "../widgets";
import { BoardingsToday, Waiting } from "./StationInspector";
import { NameEdit } from "./NameEdit";

/** A stop placed again by an edit is found again this near where it was, ground m. */
const REFIND_M = 60;
/** Places a walk away listed at most. */
const WALKS_SHOWN = 8;

/** The stop again after an edit placed the stops anew: the nearest within `REFIND_M` of where it was
 * (ids are places in a line's list, so after an undo one can name another stop), the same id winning a
 * tie; null when none is. */
function refind(p: BusPlaces, id: number, at: [number, number]) {
  let best: BusPlace["stops"][number] | null = null, d = REFIND_M;
  for (const pl of p.places) for (const s of pl.stops) {
    const m = groundM([s.lng, s.lat], at);
    if (m < d || (m === d && s.id === id)) (d = m), (best = s);
  }
  return best;
}

// the selection follows its stop: to the stop placed nearest where it was, to the train station's
// panel when that is now a station's stop, or away when there is none
effect(() => {
  const s = selection.value, w = world.value;
  if (s?.kind !== "busStop" || !w) return;
  const p = busPlaces(w), found = refind(p, Number(s.stop), s.at);
  if (!found) selection.value = null;
  else if (found.station >= 0) selection.value = stationsById(w).has(String(found.station)) ? { kind: "station", station: String(found.station) } : null;
  else if (String(found.id) !== s.stop || found.lng !== s.at[0] || found.lat !== s.at[1]) selection.value = { kind: "busStop", stop: String(found.id), at: [found.lng, found.lat] };
});

/** The lines at a place in line order, a hidden twin as the line it runs for (T-3333). */
function placeLines(w: WorldView, pl: BusPlace): LineInput[] {
  const byId = linesById(w), out: LineInput[] = [];
  for (const id of pl.lines) {
    const l = byId.get(id), shown = l?.twinOf ? byId.get(l.twinOf) : l;
    if (shown && !out.includes(shown)) out.push(shown);
  }
  return out;
}

/** A place's name: its train station's, else its first stop's. */
function placeName(w: WorldView, pl: BusPlace): string {
  const st = pl.stations.length ? stationsById(w).get(pl.stations[0]) : undefined;
  return st?.name ?? stationNames(w).get(String(pl.stops[0]?.id)) ?? t("world.busStop");
}

/** Open a place a walk away: its train station's panel, else its first plain stop's. */
function openPlace(pl: BusPlace | null, stations: string[]) {
  const plain = pl?.stops.find((s) => s.station < 0);
  if (stations.length) selection.value = { kind: "station", station: stations[0] };
  else if (plain) selection.value = { kind: "busStop", stop: String(plain.id), at: [plain.lng, plain.lat] };
  live.overlay?.invalidate();
}

export function BusStopInspector({ stop, at }: { stop: string; at: [number, number] }) {
  const w = world.value;
  useToday(); // boardings in the past 24 hours, kept fresh while the panel is open
  if (!w) return null;
  const p = busPlaces(w), k = p.placeOf.get(Number(stop));
  const pl = k === undefined ? null : p.places[k];
  const me = pl?.stops.find((s) => String(s.id) === stop);
  if (!pl || !me) return null;
  const lines = placeLines(w, pl);
  // a line's visits to this place: its stops here, a bus line's and a train line's at its stations
  const ids = new Set([...pl.stops.map((s) => String(s.id)), ...pl.stations]);
  const visits = (l: LineInput) => l.stops.flatMap((x, i) => (ids.has(x) ? [i] : []));
  const day = today.value;
  const known = lines.some((l) => day.has(l.id));
  const stations = stationsById(w);
  const name = stationNames(w).get(stop) ?? t("world.busStop");
  const walks = pl.walks.slice(0, WALKS_SHOWN);
  // T-4080: the name goes to the kept stop this place has, else a new one at this stop (its own place: the named stop stays, its neighbours on an automatic line can be laid out again)
  const here = busStopsById(w).get(stop);
  const rename = (n: string) => edit({ op: "renameBusStop", stops: [me.id, ...pl.stops.filter((s) => s.station < 0 && s.id !== me.id).map((s) => s.id)], x: here?.x ?? 0, y: here?.y ?? 0, name: n }).then((a) => a.ok);
  return (
    <>
      <div class="title">
        <NameEdit key={stop} name={name} onRename={rename} clearable={!!here}>
          <button class="btn icon" data-tip={t("lineTitle.show")} aria-label={t("lineTitle.show")} onClick={() => locateStation(at[0], at[1])}>
            <Icon.locate />
          </button>
        </NameEdit>
      </div>
      {(lines.length > 1 || pl.stations.length > 0) && (
        <div class="sub muted">
          {lines.length > 1 && t("busStopInspector.lines", { count: lines.length })}
          {pl.stations.slice(0, 1).map((id) => (
            <>
              {lines.length > 1 && " · "}
              <button class="btn text inline" onClick={() => openPlace(null, [id])}>{t("busStopInspector.atStation", { name: stations.get(id)?.name ?? "" })}</button>
            </>
          ))}
        </div>
      )}
      <Built built={me.built} />
      <div class="h">{t("stationInspector.lines")}{known && <span class="right muted small">{t("cityDetails.boardingsToday")}</span>}</div>
      <ul class="rows pick">
        {lines.map((l) => {
          const b = visits(l).reduce((n, i) => n + (day.get(l.id)?.stops[i] ?? 0), 0);
          return (
            <li onClick={() => (selection.value = { kind: "line", line: l.id })}>
              <Chip line={l} />
              <span class="grow">{l.name}</span>
              <span class="val">{day.has(l.id) ? count(b) : ""}</span>
            </li>
          );
        })}
      </ul>
      {known && <BoardingsToday lines={lines} at={visits} />}
      <Waiting node={Number(stop)} lines={lines} />
      {walks.length > 0 && (
        <>
          <div class="h">{t("busStopInspector.onFoot")}</div>
          <ul class="rows pick">
            {walks.map((x) => {
              const other = x.place >= 0 ? p.places[x.place] : null;
              const chips = other ? placeLines(w, other) : x.stations.flatMap((id) => placeLines(w, { stops: [], stations: [id], lines: linesAtStation(w, id), walks: [] }));
              const label = other ? placeName(w, other) : stations.get(x.stations[0])?.name ?? "";
              const seen = new Set<string>();
              const shown = chips.filter((l) => !seen.has(l.name) && !!seen.add(l.name));
              return (
                <li onClick={() => openPlace(other, other ? other.stations : x.stations)}>
                  <span class="grow chip-line">
                    <span>{label}</span>
                    <span class="bullets">{shown.slice(0, 6).map((l) => <Chip line={l} size="sm" />)}</span>
                  </span>
                  <span class="val">{length(Math.max(10, Math.round(x.m / 10) * 10))}</span>
                </li>
              );
            })}
          </ul>
        </>
      )}
    </>
  );
}

/** The lines stopping at a train station, by id. */
function linesAtStation(w: WorldView, id: string): string[] {
  return w.save.lines.filter((l) => l.stops.includes(id)).map((l) => l.id);
}

