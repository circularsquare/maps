// A station's commuters (T-084, T-097, SPEC 8): how many live or work within its reach and ride
// from it, and where they go. The chosen end's cells are opaque discs on the map and the places at
// the far end partly see-through ones below them (map/demandViews.ts); the swatches here are the
// map's discs.

import { demandView } from "../../game/demand";
import { count } from "../../game/format";
import { CATCHMENT_COLOUR, DESTINATION_ALPHA, DESTINATION_COLOUR } from "../../game/palette";
import { stationSide, stationView } from "../../map/demandViews";
import { Row } from "../widgets";

/** Places listed at the far end. */
const PLACES = 8;

export function StationRiders({ id }: { id: string }) {
  const v = stationView.value;
  const side = stationSide.value;
  if (!demandView.value || !v || v.stationId !== id) return null;
  const disc = <i class="mk" style={{ background: CATCHMENT_COLOUR }} />;
  const blank = <i class="mk" />;
  return (
    <>
      <div class="h">
        Commuters<span class="right muted small">a day</span>
      </div>
      <ul class="rows pick">
        <li class={side === "home" ? "on" : ""} onClick={() => (stationSide.value = "home")}>
          {side === "home" ? disc : blank}
          <span class="grow">Live nearby</span>
          <span class="val">{count(v.home)}</span>
        </li>
        <li class={side === "work" ? "on" : ""} onClick={() => (stationSide.value = "work")}>
          {side === "work" ? disc : blank}
          <span class="grow">Work nearby</span>
          <span class="val">{count(v.work)}</span>
        </li>
      </ul>
      {v.places.length > 0 && (
        <>
          <div class="h">
            <i class="mk far" style={{ background: DESTINATION_COLOUR, opacity: DESTINATION_ALPHA }} />
            {side === "home" ? "Where they work" : "Where they live"}
          </div>
          <ul class="rows">
            {v.places.slice(0, PLACES).map((p) => (
              <Row label={p.name}>{count(p.riders)}</Row>
            ))}
          </ul>
        </>
      )}
    </>
  );
}
