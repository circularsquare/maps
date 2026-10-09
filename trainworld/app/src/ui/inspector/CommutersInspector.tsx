// Commuter bubbles picked on the map (T-097, SPEC 8): how many commuters live (or work) there and
// how they travel, and where they work (or live), the far end drawn on the map as bubbles too.

import { count } from "../../game/format";
import { BUBBLE_ALPHA, modeMix } from "../../game/palette";
import { commuterPick } from "../../map/commuterView";
import { Row } from "../widgets";

const PLACES = 8;

const swatch = (t: number, w: number, d: number) => {
  const c = modeMix(t, w, d);
  return <i class="mk" style={{ background: `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${BUBBLE_ALPHA})` }} />;
};

function Modes({ totals }: { totals: [number, number, number] }) {
  const all = totals[0] + totals[1] + totals[2];
  const pct = (v: number) => (all > 0 ? Math.round((100 * v) / all) + "%" : "");
  const rows: [string, number, [number, number, number]][] = [
    ["Train", totals[0], [1, 0, 0]],
    ["Walking", totals[1], [0, 1, 0]],
    ["Driving", totals[2], [0, 0, 1]],
  ];
  return (
    <ul class="rows">
      <Row label="Commuters a day" bold>
        {count(all)}
      </Row>
      {rows.map(([name, v, m]) => (
        <li>
          {swatch(...m)}
          <span class="grow">{name}</span>
          <span class="val">{count(v)}</span>
          <span class="val w-num">{pct(v)}</span>
        </li>
      ))}
    </ul>
  );
}

export function CommutersInspector() {
  const p = commuterPick.value;
  if (!p || p.sel?.kind !== "commuters") return null;
  const home = p.sel.end === "home";
  return (
    <>
      <div class="title">
        {swatch(...p.totals)}
        <span class="name">{home ? "Commuters who live here" : "Commuters who work here"}</span>
      </div>
      <div class="sub muted">{p.sel.areas === 1 ? "One area" : `${p.sel.areas} areas`}</div>
      <Modes totals={p.totals} />
      <div class="h">
        {home ? "Where they work" : "Where they live"}
        <span class="right muted small">{p.far ? "commuters a day" : ""}</span>
      </div>
      {p.far ? (
        <ul class="rows">
          {p.far.places.slice(0, PLACES).map((pl) => (
            <Row label={pl.name}>{count(pl.riders)}</Row>
          ))}
        </ul>
      ) : (
        <p class="muted">Working it out.</p>
      )}
    </>
  );
}
