// The track inspector (SPEC 8): a stretch of track, its elevation while a blueprint, and double up.

import { distance, levelLabel, money, speed, upkeepText } from "../../game/format";
import { LEVELS, type Level } from "../../game/types";
import { parseEdgePis, type EdgePis } from "../../map/edgeEdit";
import { selection } from "../../game/ui";
import { edit, say, world } from "../../game/world";
import { createLine } from "../../game/lineActions";
import { removeThing } from "../../game/remove";
import { describeIssues } from "../../game/issues";
import { moneyState } from "../../game/money";
import { edgeUpkeep } from "../../game/regionMoney";
import { Built, Chip, ConstructRemove, Icon, Row } from "../widgets";
import { useClockQuery } from "../useClockQuery";

/** A blueprint stretch's elevation. Corners are dragged on the map (map/edgeEdit.ts). */
function Shape({ edge, version }: { edge: number; version: number }) {
  const shape = useClockQuery<EdgePis>({ what: "edgePis", edge }, [edge, version], parseEdgePis, true);
  if (!shape) return null;
  const pis = shape.pis;
  const track = world.value?.edges.find(e => e.id === edge);
  // a straight piece has no corners, so its elevation is its ends', moved by edgesLevel as in the selection bar
  const level = pis.length ? (pis.every((p) => p.level === pis[0].level) ? pis[0].level : null) : track && track.levelMin === track.levelMax ? track.levelMin : null;
  const setLevel = async (l: Level) => {
    if (pis.length) return void edit({ op: "edgeLevel", edge, level: l });
    const a = await edit({ op: "edgesLevel", edges: [edge], level: l }, true);
    if (!a.ok) say(`This track cannot go to elevation ${levelLabel(l)}.`, true);
  };
  return (
    <>
      <div class="h">Elevation</div>
      <div class="group levels">
        {LEVELS.map((l) => (
          <button class={"btn" + (l === level ? " on" : "")} onClick={() => void setLevel(l)}>
            {levelLabel(l)}
          </button>
        ))}
      </div>
    </>
  );
}

/** Double up: an express pair beside this double track as far as its run goes, as
 * blueprint. The hover gives the price or why not. */
function DoubleUp({ edge, version }: { edge: number; version: number }) {
  const answer = useClockQuery<{ price: number } | { why: string }>({ what: "doubleUp", edge }, [edge, version],
    (d) => (d[0] >= 0 ? { price: d[0] } : { why: describeIssues(String.fromCharCode(...d.subarray(1)).split("\n").filter(Boolean)) }), true);
  if (!answer) return null;
  const tip = "price" in answer ? `Double up to quad track for ${money(answer.price * 1e6)} (blueprint)\nThe express pair runs beside it to the next junction or track end each way` : answer.why;
  return (
    <button class="btn icon" disabled={"why" in answer} data-tip={tip} aria-label="Double up" onClick={() => void edit({ op: "doubleUp", edge })}>
      <Icon.doubleUp />
    </button>
  );
}

export function TrackInspector({ edge }: { edge: number }) {
  const w = world.value;
  const e = w?.edges.find((e) => e.id === edge);
  if (!w || !e) return null;
  const lines = w.save.lines.filter((l) => e.lines.includes(l.id));
  const lv = e.levelMin === e.levelMax ? `Elevation ${levelLabel(e.levelMin)}` : `Elevation ${levelLabel(e.levelMin)} to ${levelLabel(e.levelMax)}`;
  // a crossover is one thing: both diagonals are priced, built and removed together (T-391)
  const other = e.crossover ? w.edges.find((x) => x.id === e.partner) : undefined;
  const cost = e.cost + (other?.cost ?? 0);
  const m = moneyState.value;
  const construct = async () => {
    const a = await edit({ op: "construct", edges: [e.id], stations: [] });
    if (a.ok) say(`${e.crossover ? "Crossover" : "Track"} constructed for ${money(a.charge * 1e6)}.`);
  };
  // constructed: asks first above the bottom bar (T-096)
  const remove = () => void removeThing({ kind: "track", edge: e.id });
  return (
    <>
      <div class="title">
        <span class="name">{e.crossover ? "Crossover" : e.hsr ? "High-speed track" : e.tracks === 2 ? "Double track" : "Single track"}</span>
        {e.tracks === 2 && !e.crossover && !e.hsr && <DoubleUp edge={e.id} version={w.version} />}
      </div>
      <div class="sub muted">
        {distance(e.lengthM)}, {lv}
      </div>
      <ConstructRemove built={e.built} construct={construct} constructLabel={`Construct for ${money(cost)}`} remove={remove}
        removeLabel={e.crossover ? "Remove crossover" : e.built ? "Remove track" : "Remove blueprint"} removeTip="" />
      <Built built={e.built} />
      {!e.built && e.off && <p class="muted" data-tip="Will not be constructed unless enabled">Disabled</p>}
      <ul class="rows">
        {!e.crossover && (
          <li data-tip={e.hsr ? "High-speed trains run up to 320 km/h here and normal trains up to 160 km/h" : "Trains run up to 160 km/h here. Over 160 needs high-speed track."}>
            <span class="grow">Speed limit</span>
            <span class="val">{speed(e.hsr ? 320 : 160)}</span>
          </li>
        )}
        <Row label="Slowest curve">{speed(e.vminKmh)}</Row>
        <Row label="Cost to construct">{money(cost)}</Row>
        {!e.crossover && m && (
          <Row label="Upkeep">
            <span data-tip={`${upkeepText(m, e.built ? "Upkeep" : "Once constructed, upkeep")} The lines here share it by their trains per day.`}>
              {money(edgeUpkeep({ ...e, built: true }, m))}/day
            </span>
          </Row>
        )}
      </ul>
      {!e.built && !e.crossover && <Shape edge={e.id} version={w.version} />}
      {lines.length > 0 && <div class="h">Lines here</div>}
      {lines.length === 0 && (
        <div class="h">
          Lines here
          <span class="right">
            <button class="btn icon" data-tip="New line, starting at the stations on this track (N)" aria-label="New line" onClick={() => void createLine(e)}><Icon.plus /></button>
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
