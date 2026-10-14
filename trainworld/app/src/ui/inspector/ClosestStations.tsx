// "Closest stations" (T-312, T-323): the 3 stations nearest on foot to a place, built or blueprint,
// each with the chips of the lines stopping there and its walking minutes (game/walks.ts). Shared
// by the station inspector (the station's own complex and name left out) and the inspector of a
// selected demand bubble (its centre). A click on a row selects that station and moves the map to
// it if it is off screen.

import { count } from "../../game/format";
import { selection } from "../../game/ui";
import type { LineInput, StationInput } from "../../game/types";
import { closestStations, minLabel } from "../../game/walks";
import { linesAt, world } from "../../game/world";
import { live } from "../../map/live";
import { Chip } from "../widgets";

/** Chips shown a row; the rest are one "+n". */
const CHIPS = 6;

/** The lines stopping at a station, a chip for each line name once (branches of one line are one). */
function chipsAt(id: string): LineInput[] {
  const seen = new Set<string>();
  return linesAt(id).filter((l) => !seen.has(l.name) && !!seen.add(l.name));
}

function open(t: StationInput) {
  selection.value = { kind: "station", station: t.id };
  const map = live.map;
  if (map && !map.getBounds().contains([t.lng, t.lat])) map.jumpTo({ center: [t.lng, t.lat] });
  live.overlay?.invalidate();
}

export function ClosestStations({ lng, lat, exclude, own }: { lng: number; lat: number; exclude?: ReadonlySet<string>; own?: string }) {
  const w = world.value;
  if (!w) return null;
  const rows = closestStations(lng, lat, w.save.stations, exclude, 3, own ?? null);
  if (!rows.length) return null;
  return (
    <>
      <div class="h">Closest stations</div>
      <ul class="rows pick">
        {rows.map((r) => {
          const chips = chipsAt(r.station.id);
          return (
            <li onClick={() => open(r.station)}>
              <span class={"grow" + (r.station.built ? "" : " muted")} style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "4px" }}>
                <span>{r.station.name}{r.station.built ? "" : " (blueprint)"}</span>
                {chips.slice(0, CHIPS).map((l) => <Chip line={l} size="sm" />)}
                {chips.length > CHIPS && <span class="muted small">+{chips.length - CHIPS}</span>}
              </span>
              <span class="val" data-tip={r.metres === null ? "Water in the way" : `${count(Math.round(r.metres))} m on foot`}>
                {r.metres === null ? "no walk" : minLabel(r.metres)}
              </span>
            </li>
          );
        })}
      </ul>
    </>
  );
}
