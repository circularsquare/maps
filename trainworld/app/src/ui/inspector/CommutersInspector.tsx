// Commuter bubbles picked on the map (T-097, SPEC 8): how many commuters live (or work) there and
// how they travel, and where they work (or live), the far end drawn on the map as bubbles too.

import { capitalise, count } from "../../game/format";
import { shownDayPhrase } from "../../game/ui";
import { BUBBLE_ALPHA, modeMix, towardNone } from "../../game/palette";
import { purposeWords } from "../../game/purposes";
import { t } from "../../i18n";
import { commuterPick } from "../../map/commuterView";
import { Row } from "../widgets";
import { ClosestStations } from "./ClosestStations";
import { PurposeBreakdown } from "./PurposeBreakdown";
import { pctWhole } from "./tripText";

const swatch = (t: number, w: number, d: number) => {
  const c = modeMix(t, w, d);
  return <i class="mk" style={{ background: `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${BUBBLE_ALPHA})` }} />;
};

/** The selection's mark: the mode mix, pulled to the No route grey as its bubbles are (T-2215). */
function TitleMark({ totals }: { totals: [number, number, number, number] }) {
  const c = towardNone(modeMix(totals[0], totals[1], totals[2]), totals[0] + totals[1] + totals[2], totals[3]);
  return <i class="mk" style={{ background: `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${BUBBLE_ALPHA})` }} />;
}

function Modes({ totals }: { totals: [number, number, number, number] }) {
  const all = totals[0] + totals[1] + totals[2] + totals[3];
  const pct = (v: number) => (all > 0 ? pctWhole(v / all) : "");
  const rows: [string, number, [number, number, number] | null][] = [
    [t("commutersInspector.train"), totals[0], [1, 0, 0]],
    [t("commutersInspector.walking"), totals[1], [0, 1, 0]],
    [t("commutersInspector.driving"), totals[2], [0, 0, 1]],
    // T-2215: trips between two land parts no road joins, not by rail
    ...(totals[3] > 0 ? [[t("commutersInspector.noRoute"), totals[3], null] as [string, number, null]] : []),
  ];
  return (
    <ul class="rows">
      <Row label={t("commutersInspector.legs", { phrase: shownDayPhrase.value })} bold>
        {count(all)}
      </Row>
      {rows.map(([name, v, m]) => (
        <li data-tip={m ? undefined : t("inspector.noRoadEnds")}>
          {m ? swatch(...m) : <i class="mk" style={{ background: "var(--mode-none)" }} />}
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
  const end = p.sel.end;
  return (
    <>
      <div class="title">
        <TitleMark totals={p.totals} />
        <span class="name">{t("commutersInspector.title", { end })}</span>
      </div>
      <div class="sub muted" data-tip={t("commutersInspector.legsTip")}>{t("commutersInspector.areas", { count: p.sel.areas })}</div>
      <Modes totals={p.totals} />
      {p.at && p.sel.areas === 1 && <ClosestStations lng={p.at[0]} lat={p.at[1]} />}
      {p.purpose && p.purpose !== "all" && <p class="muted small">
        {t("commutersInspector.only", { purpose: capitalise(purposeWords(p.purpose)) })}
      </p>}
      <PurposeBreakdown purposes={p.purposes} />
    </>
  );
}
