// A station's commuters (T-084, T-097, SPEC 8): how many live or work within its reach and ride
// from it, and where they go. The chosen end's cells are opaque discs on the map and the places at
// the far end partly see-through ones below them (map/demandViews.ts); the swatches here are the
// map's discs.

import { demandView } from "../../game/demand";
import { count } from "../../game/format";
import { CATCHMENT_COLOUR, DESTINATION_ALPHA, DESTINATION_COLOUR, MODE_COLOURS } from "../../game/palette";
import { purposeWords } from "../../game/purposes";
import { purposeFilter, shownDayPhrase } from "../../game/ui";
import { stationSide, stationView, type StationView } from "../../map/demandViews";
import { Icon, Row } from "../widgets";

/** Places listed at the far end. */
const PLACES = 8;

const pct = (n: number, of: number) => {
  const p = of > 0 ? (100 * n) / of : 0;
  return n >= 0.5 && p < 0.5 ? "<1%" : Math.round(p) + "%";
};

/** How the commuters who live nearby reach the station (T-374): a bar and a key in the purpose
 * split's style (PurposeSplit.tsx), walked in the walk colour and rode in the drive colour. */
function AccessSplit({ v }: { v: StationView }) {
  const all = v.walked + v.rode;
  if (!(all > 0)) return null;
  const parts = [
    { n: v.walked, icon: Icon.walk, colour: MODE_COLOURS.walk, cls: "seg-walk", what: "Walked" },
    { n: v.rode, icon: Icon.ride, colour: MODE_COLOURS.drive, cls: "seg-drive", what: "Drove, took a bus or cycled" },
  ];
  const exact = (n: number) => Math.round(n).toLocaleString("en");
  const tip = (p: (typeof parts)[number]) => `${p.what}: ${pct(p.n, all)}, ${exact(p.n)} of the ${exact(all)} commuters ${shownDayPhrase.value} who live nearby`;
  return (
    <ul class="rows">
      <li class="purpose-split">
        <div class="bar split-bar" role="img" aria-label={parts.map(tip).join("; ")}>
          {parts.map((p) => (p.n > 0 ? <i class={"seg " + p.cls} style={{ width: (100 * p.n) / all + "%" }} data-tip={tip(p)} /> : null))}
        </div>
        <ul class="split-key">
          {parts.map((p) => (
            <li class={p.n > 0 ? "" : "muted"} data-tip={tip(p)}>
              <span class="mode-icon" style={{ color: p.colour }}>{p.icon()}</span>
              <span class="num">{pct(p.n, all)}</span>
            </li>
          ))}
        </ul>
      </li>
    </ul>
  );
}

/** `scope`: the stations counted, ids joined by commas (the complex's in complex view, T-146). */
export function StationRiders({ id, scope = id }: { id: string; scope?: string }) {
  const v = stationView.value;
  const side = stationSide.value;
  if (!demandView.value || !v || v.stationId !== id || v.scope !== scope) return null;
  const disc = <i class="mk" style={{ background: CATCHMENT_COLOUR }} />;
  const blank = <i class="mk" />;
  return (
    <>
      <div class="h">
        Commuters<span class="right muted small">{shownDayPhrase.value}</span>
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
      <AccessSplit v={v} />
      {purposeFilter.value !== "all" && purposeFilter.value !== "work" && (
        <p class="muted small" data-tip={`The map shows ${purposeWords(purposeFilter.value)} trips only`}>Not drawn under this purpose filter.</p>
      )}
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
