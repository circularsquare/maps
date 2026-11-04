// The track inspector (SPEC 8): a stretch of track and double up; a blueprint piece's level is set in the
// selection bar, which a click on it opens too (T-2283).

import { distance, levelLabel, money, speed, upkeepText } from "../../game/format";
import { selection } from "../../game/ui";
import { edit, say, world } from "../../game/world";
import { createLine } from "../../game/lineActions";
import { removeThing } from "../../game/remove";
import { describeIssues } from "../../game/issues";
import { moneyState } from "../../game/money";
import { edgeUpkeep } from "../../game/regionMoney";
import { Built, Chip, ConstructRemove, Icon, Row } from "../widgets";
import { useClockQuery } from "../useClockQuery";

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
  const lv = e.levelMin === e.levelMax ? `Level ${levelLabel(e.levelMin)}` : `Level ${levelLabel(e.levelMin)} to ${levelLabel(e.levelMax)}`;
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
            <span class="money-neg" data-tip={`${upkeepText(m, e.built ? "Upkeep" : "Once constructed, upkeep")} The lines here share it by their trains per day.`}>
              {money(edgeUpkeep({ ...e, built: true }, m))}/day
            </span>
          </Row>
        )}
      </ul>
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
