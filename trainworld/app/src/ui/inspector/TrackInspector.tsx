// The track inspector (SPEC 8): what a piece of track is. T-3746: information only; a click on track makes it
// the pick too, and the selection bar at the top of the map holds every control (ui/Blueprint.tsx).

import { distance, levelLabel, money, speed, upkeepText } from "../../game/format";
import { selection } from "../../game/ui";
import { t, tSeries } from "../../i18n";
import { world } from "../../game/world";
import { createLine } from "../../game/lineActions";
import { moneyState } from "../../game/money";
import { edgeUpkeep } from "../../game/regionMoney";
import { PieceLimit } from "./PieceLimit";
import { Built, Chip, Icon, Row } from "../widgets";
// Shift+click has no visible control, so its instruction is a tag (T-3746)
import { touchInput } from "../../map/touch";

export function TrackInspector({ edge }: { edge: number }) {
  const w = world.value;
  const e = w?.edges.find((e) => e.id === edge);
  if (!w || !e) return null;
  const lines = w.save.lines.filter((l) => e.lines.includes(l.id));
  const lv = e.levelMin === e.levelMax ? t("trackInspector.level", { level: levelLabel(e.levelMin) }) : t("trackInspector.levels", { from: levelLabel(e.levelMin), to: levelLabel(e.levelMax) });
  // a crossover is one thing: both diagonals are priced, built and removed together (T-391)
  const other = e.crossover ? w.edges.find((x) => x.id === e.partner) : undefined;
  const cost = e.cost + (other?.cost ?? 0);
  const m = moneyState.value;
  return (
    <>
      <div class="title">
        <span class="name" data-tip={touchInput.value ? undefined : t("trackInspector.shiftTip")}>{e.kind && !e.crossover ? t(e.kind === 4 ? "trackInspector.nameBusway" : e.kind === 2 ? "trackInspector.nameStreetTram" : "trackInspector.nameTram") : t("trackInspector.name", { kind: e.crossover ? "crossover" : e.hsr ? "hsr" : e.tracks === 2 ? "double" : "single" })}</span>
      </div>
      <div class="sub muted">
        {tSeries([distance(e.lengthM), e.kind === 4 && !e.crossover ? t("trackInspector.kindBusway") : t("trackInspector.kind", { kind: e.crossover ? "crossover" : e.kind ? (e.kind === 2 ? "streetTram" : "tram") : e.hsr ? "hsr" : e.tracks === 2 ? "double" : "single" }), lv])}
      </div>
      <Built built={e.built} />
      {!e.built && e.off && <p class="muted" data-tip={t("trackInspector.disabledTip")}>{t("trackInspector.disabled")}</p>}
      <ul class="rows top-speed">
        {!e.crossover && e.kind !== 4 && <PieceLimit e={e} />}
        <Row label={t("trackInspector.slowest")}>{speed(e.vminKmh)}</Row>
        <Row label={t("trackInspector.cost")}>{money(cost)}</Row>
        {!e.crossover && m && (
          <Row label={t("trackInspector.upkeep")}>
            <span class="money-neg" data-tip={t("trackInspector.upkeepTip", { text: upkeepText(m, e.built ? t("trackInspector.upkeepLead") : t("trackInspector.upkeepLeadPlanned")) })}>
              {t("inspector.perDay", { value: money(edgeUpkeep({ ...e, built: true }, m)) })}
            </span>
          </Row>
        )}
      </ul>
      {lines.length > 0 && <div class="h">{t("trackInspector.linesHere")}</div>}
      {lines.length === 0 && (
        <div class="h">
          {t("trackInspector.linesHere")}
          <span class="right">
            <button class="btn icon" data-tip={t("trackInspector.newLineTip")} aria-label={t("trackInspector.newLine")} onClick={() => void createLine(e)}><Icon.plus /></button>
          </span>
        </div>
      )}
      <ul class="rows pick">
        {lines.map((l) => (
          <li onClick={() => (selection.value = { kind: "line", line: l.id })}>
            <Chip line={l} />
            <span class="grow">{l.name}</span>
          </li>
        ))}
      </ul>
    </>
  );
}
