// The track inspector (SPEC 8): a stretch of track and double up; a blueprint piece's level is set in the
// selection bar, which a click on it opens too (T-2283).

import { distance, levelLabel, money, speed, upkeepText } from "../../game/format";
import { selection } from "../../game/ui";
import { t, tSeries } from "../../i18n";
import { edit, say, world } from "../../game/world";
import { createLine } from "../../game/lineActions";
import { removeThing } from "../../game/remove";
import { describeIssues } from "../../game/issues";
import { moneyState } from "../../game/money";
import { edgeUpkeep } from "../../game/regionMoney";
import { TRACK_SPECS } from "../../game/trackKinds";
import { Actions, Built, Check, Chip, ConstructRemove, Icon, Row } from "../widgets";
import { setOneWay } from "../../game/oneWay";
import { useClockQuery } from "../useClockQuery";

/** Double up: an express pair beside this double track as far as its run goes, as
 * blueprint. The hover gives the price or why not. */
function DoubleUp({ edge, version }: { edge: number; version: number }) {
  const answer = useClockQuery<{ price: number } | { why: string }>({ what: "doubleUp", edge }, [edge, version],
    (d) => (d[0] >= 0 ? { price: d[0] } : { why: describeIssues(String.fromCharCode(...d.subarray(1)).split("\n").filter(Boolean)) }), true);
  if (!answer) return null;
  const tip = "price" in answer ? t("trackInspector.doubleUpTip", { price: money(answer.price * 1e6) }) : answer.why;
  return (
    <button class="btn icon" disabled={"why" in answer} data-tip={tip} aria-label={t("trackInspector.doubleUp")} onClick={() => void edit({ op: "doubleUp", edge })}>
      <Icon.doubleUp />
    </button>
  );
}

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
  const construct = async () => {
    const a = await edit({ op: "construct", edges: [e.id], stations: [] });
    if (a.ok) say(t("trackInspector.constructed", { crossover: !!e.crossover, price: money(a.charge * 1e6) }));
  };
  // constructed: asks first above the bottom bar (T-096)
  const remove = () => void removeThing({ kind: "track", edge: e.id });
  return (
    <>
      <div class="title">
        <span class="name">{e.kind && !e.crossover ? t(e.kind === 2 ? "trackInspector.nameStreetTram" : "trackInspector.nameTram") : t("trackInspector.name", { kind: e.crossover ? "crossover" : e.hsr ? "hsr" : e.tracks === 2 ? "double" : "single" })}</span>
        {!e.built && !e.crossover && (
          <button class="btn icon" data-tip={t("trackInspector.addPointTip")} aria-label={t("trackInspector.addPoint")} onClick={() => void edit({ op: "addPoint", edge: e.id })}>
            <Icon.addPoint />
          </button>
        )}
        {e.tracks === 2 && !e.crossover && <DoubleUp edge={e.id} version={w.version} />}
      </div>
      <div class="sub muted">
        {tSeries([distance(e.lengthM), lv])}
      </div>
      <ConstructRemove built={e.built} construct={construct} constructLabel={t("trackInspector.construct", { price: money(cost) })} remove={remove}
        removeLabel={t("trackInspector.remove", { kind: e.crossover ? "crossover" : e.built ? "track" : "blueprint" })} removeTip="" />
      <Built built={e.built} />
      {!e.built && e.off && <p class="muted" data-tip={t("trackInspector.disabledTip")}>{t("trackInspector.disabled")}</p>}
      {e.tracks === 1 && !e.crossover && !e.hsr && !e.kind && (
        <Actions>
          {/* T-3137: the switch shows the way it runs, taken by itself included; off where the run gives a way pins both ways */}
          <span data-tip={e.oneWay && !e.setWay ? t("trackInspector.oneWayAutoTip") : undefined}>
            <Check checked={!!e.oneWay} onChange={(on) => void setOneWay(e.id, on ? e.autoWay ?? 1 : e.autoWay ? 3 : 0)}>{t("trackInspector.oneWay")}</Check>
          </span>
          {e.oneWay && <button class="btn text" onClick={() => void setOneWay(e.id, e.oneWay === 1 ? 2 : 1, true)}>{t("trackInspector.reverse")}</button>}
          {!!e.setWay && (e.setWay === 3 ? 0 : e.setWay) !== (e.autoWay ?? 0) && (
            <button class="btn text" onClick={() => void setOneWay(e.id, 0)}>{t("trackInspector.autoWay")}</button>
          )}
        </Actions>
      )}
      <ul class="rows">
        {!e.crossover && (
          <li data-tip={e.kind ? undefined : t(e.hsr ? "trackInspector.speedHsr" : "trackInspector.speedNormal")}>
            <span class="grow">{t("trackInspector.speedLimit")}</span>
            <span class="val">{speed(e.kind ? TRACK_SPECS[e.kind === 2 ? "streetTram" : "tram"].vCapKmh : e.hsr ? 320 : 160)}</span>
          </li>
        )}
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
