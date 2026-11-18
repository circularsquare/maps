// The track inspector (SPEC 8): a stretch of track and double up; a blueprint piece's level is set in the
// selection bar, which a click on it opens too (T-2283).

import { distance, levelLabel, money, speed, upkeepText } from "../../game/format";
import { selection } from "../../game/ui";
import { t, tSeries } from "../../i18n";
import { edit, say, world } from "../../game/world";
import { createLine } from "../../game/lineActions";
import { removeThing, removeTracks } from "../../game/remove";
import { describeIssues } from "../../game/issues";
import { moneyState } from "../../game/money";
import { edgeUpkeep } from "../../game/regionMoney";
import { PieceLimit } from "./PieceLimit";
import { PieceRebuild } from "./PieceRebuild";
import { Actions, Built, Check, Chip, ConstructRemove, Icon, Row } from "../widgets";
import { setOneWay } from "../../game/oneWay";
import { useClockQuery } from "../useClockQuery";
import { settable } from "../../game/trackSet";
import { touchInput } from "../../map/touch";
import { mapBearing } from "../../map/live";
import type { TrackEdge } from "../../game/types";

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

/** The way a one-way piece runs, as an arrow turned to where its direction points on screen (its a-to-b bearing minus the map's). */
function WayArrow({ e }: { e: TrackEdge }) {
  const nodes = world.value?.nodes, a = nodes?.get(e.a), b = nodes?.get(e.b);
  if (!a || !b) return null;
  const [from, to] = e.oneWay === 2 ? [b, a] : [a, b];
  const north = (to.lat - from.lat), east = (to.lng - from.lng) * Math.cos((from.lat * Math.PI) / 180);
  const deg = (Math.atan2(east, north) * 180) / Math.PI - mapBearing.value;
  return <span class="way-arrow" aria-hidden="true" style={{ transform: `rotate(${deg}deg)` }}><Icon.arrowUp /></span>;
}

const kindOf =(e: TrackEdge) => (e.kind ? (e.kind === 2 ? "streetTram" : "tram") : e.hsr ? "hsr" : e.tracks === 2 ? "double" : "single");

/** T-3634: built pieces Shift+clicked together (game/trackSet.ts): their totals, one Rebuild for all, and the
 * lines on any of them. */
export function TracksInspector({ edges: ids }: { edges: number[] }) {
  const w = world.value;
  if (!w) return null;
  const set = new Set(ids);
  const edges = w.edges.filter((e) => set.has(e.id));
  if (!edges.length) return null;
  const m = moneyState.value;
  const lines = w.save.lines.filter((l) => edges.some((e) => e.lines.includes(l.id)));
  const lo = Math.min(...edges.map((e) => e.levelMin)), hi = Math.max(...edges.map((e) => e.levelMax));
  const lv = lo === hi ? t("trackInspector.level", { level: levelLabel(lo as TrackEdge["levelMin"]) }) : t("trackInspector.levels", { from: levelLabel(lo as TrackEdge["levelMin"]), to: levelLabel(hi as TrackEdge["levelMin"]) });
  const kind = edges.every((e) => kindOf(e) === kindOf(edges[0])) ? t("trackInspector.kind", { kind: kindOf(edges[0]) }) : "";
  const sum = (f: (e: TrackEdge) => number) => edges.reduce((s, e) => s + f(e), 0);
  return (
    <>
      <div class="title">
        <span class="name" data-tip={touchInput.value ? undefined : t("trackInspector.shiftSetTip")}>{t("trackInspector.pieces", { count: edges.length })}</span>
      </div>
      <div class="sub muted">{tSeries([distance(sum((e) => e.lengthM)), kind, lv].filter(Boolean))}</div>
      <PieceRebuild edges={edges}><button class="btn text" onClick={() => void removeTracks(ids)}>{t("trackInspector.remove", { kind: "track" })}</button></PieceRebuild>
      <ul class="rows top-speed">
        <Row label={t("trackInspector.cost")}>{money(sum((e) => e.cost))}</Row>
        {m && (
          <Row label={t("trackInspector.upkeep")}>
            <span class="money-neg" data-tip={t("trackInspector.upkeepTip", { text: upkeepText(m, t("trackInspector.upkeepLead")) })}>
              {t("inspector.perDay", { value: money(sum((e) => edgeUpkeep(e, m))) })}
            </span>
          </Row>
        )}
      </ul>
      {lines.length > 0 && <div class="h">{t("trackInspector.linesHere")}</div>}
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
  const shiftTip = !touchInput.value; // T-3634: Shift+click has no visible control, so its instruction is a tag
  const construct = async () => {
    const a = await edit({ op: "construct", edges: [e.id], stations: [] });
    if (a.ok) say(t("trackInspector.constructed", { crossover: !!e.crossover, price: money(a.charge * 1e6) }));
  };
  // constructed: asks first above the bottom bar (T-096)
  const remove = () => void removeThing({ kind: "track", edge: e.id });
  return (
    <>
      <div class="title">
        <span class="name" data-tip={shiftTip && settable(e) ? t("trackInspector.shiftAddTip") : undefined}>{e.kind && !e.crossover ? t(e.kind === 2 ? "trackInspector.nameStreetTram" : "trackInspector.nameTram") : t("trackInspector.name", { kind: e.crossover ? "crossover" : e.hsr ? "hsr" : e.tracks === 2 ? "double" : "single" })}</span>
        {!e.built && !e.crossover && (
          <button class="btn icon" data-tip={t("trackInspector.addPointTip")} aria-label={t("trackInspector.addPoint")} onClick={() => void edit({ op: "addPoint", edge: e.id })}>
            <Icon.addPoint />
          </button>
        )}
        {e.tracks === 2 && !e.crossover && <DoubleUp edge={e.id} version={w.version} />}
      </div>
      <div class="sub muted">
        {tSeries([distance(e.lengthM), t("trackInspector.kind", { kind: e.crossover ? "crossover" : e.kind ? (e.kind === 2 ? "streetTram" : "tram") : e.hsr ? "hsr" : e.tracks === 2 ? "double" : "single" }), lv])}
      </div>
      {/* T-3408: built track rebuilds here, Rebuild beside Remove */}
      {e.built && !e.crossover ? (
        <PieceRebuild key={e.id} edges={[e]}><button class="btn text" onClick={remove}>{t("trackInspector.remove", { kind: "track" })}</button></PieceRebuild>
      ) : (
        <ConstructRemove built={e.built} construct={construct} constructLabel={t("trackInspector.construct", { price: money(cost) })} remove={remove}
          removeLabel={t("trackInspector.remove", { kind: e.crossover ? "crossover" : e.built ? "track" : "blueprint" })} removeTip="" />
      )}
      <Built built={e.built} />
      {!e.built && e.off && <p class="muted" data-tip={t("trackInspector.disabledTip")}>{t("trackInspector.disabled")}</p>}
      {e.tracks === 1 && !e.crossover && (
        <Actions>
          {/* T-3137: the switch shows the way it runs, taken by itself included; off where the run gives a way pins both ways */}
          <span data-tip={e.oneWay && !e.setWay ? t("trackInspector.oneWayAutoTip") : undefined}>
            <Check checked={!!e.oneWay} onChange={(on) => void setOneWay(e.id, on ? e.autoWay ?? 1 : e.autoWay ? 3 : 0)}>{t("trackInspector.oneWay")}</Check>
          </span>
          {e.oneWay && <WayArrow e={e} />}
          {e.oneWay && <button class="btn text" onClick={() => void setOneWay(e.id, e.oneWay === 1 ? 2 : 1, true)}>{t("trackInspector.reverse")}</button>}
          {!!e.setWay && (e.setWay === 3 ? 0 : e.setWay) !== (e.autoWay ?? 0) && (
            <button class="btn text" onClick={() => void setOneWay(e.id, 0)}>{t("trackInspector.autoWay")}</button>
          )}
        </Actions>
      )}
      <ul class="rows top-speed">
        {!e.crossover && <PieceLimit key={e.id} e={e} />}
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
