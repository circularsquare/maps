// A station's commuters (T-084, T-097, SPEC 8): how many live or work within its reach and ride
// from it, and where they go. Listed here only, not drawn on the map.

import { demandView } from "../../game/demand";
import { count, exact } from "../../game/format";
import { modeColours } from "../../game/palette";
import { shownDayPhrase } from "../../game/ui";
import { stationSide, stationView, type StationView } from "../../map/demandViews";
import { Icon, Row, SegBar } from "../widgets";
import { pctOf } from "./tripText";

/** Places listed at the far end. */
const PLACES = 8;

/** How the commuters who live nearby reach the station (T-374): a bar and a key in the purpose
 * split's style (PurposeSplit.tsx), walked in the walk colour and rode in the drive colour. */
function AccessSplit({ v }: { v: StationView }) {
  const all = v.walked + v.rode;
  if (!(all > 0)) return null;
  const c = modeColours();
  const parts = [
    { n: v.walked, icon: Icon.walk, colour: c.walk, cls: "seg-walk", what: "Walked" },
    { n: v.rode, icon: Icon.ride, colour: c.drive, cls: "seg-drive", what: "Drove, took a bus or cycled" },
  ];
  const tip = (p: (typeof parts)[number]) => `${p.what}: ${pctOf(p.n, all)}, ${exact(p.n)} of the ${exact(all)} commuters ${shownDayPhrase.value} who live nearby`;
  return (
    <ul class="rows">
      <li class="purpose-split">
        <SegBar kind="split-bar" label={parts.map(tip).join("; ")} parts={parts.map((p) => ({ share: p.n / all, cls: p.cls, tip: tip(p) }))} />
        <ul class="split-key">
          {parts.map((p) => (
            <li class={p.n > 0 ? "" : "muted"} data-tip={tip(p)}>
              <span class="mode-icon" style={{ color: p.colour }}>{p.icon()}</span>
              <span class="num">{pctOf(p.n, all)}</span>
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
  return (
    <>
      <div class="h">
        Commuters<span class="right muted small">{shownDayPhrase.value}</span>
      </div>
      <ul class="rows pick">
        <li class={side === "home" ? "on" : ""} onClick={() => (stationSide.value = "home")}>
          <span class="grow">Live nearby</span>
          <span class="val">{count(v.home)}</span>
        </li>
        <li class={side === "work" ? "on" : ""} onClick={() => (stationSide.value = "work")}>
          <span class="grow">Work nearby</span>
          <span class="val">{count(v.work)}</span>
        </li>
      </ul>
      <AccessSplit v={v} />
      {v.places.length > 0 && (
        <>
          <div class="h">{side === "home" ? "Where they work" : "Where they live"}</div>
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
