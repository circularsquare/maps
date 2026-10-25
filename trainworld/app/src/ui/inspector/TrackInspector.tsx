// The track inspector (SPEC 8): a stretch of track, its corners and elevation while a blueprint, and double up.

import { useState } from "preact/hooks";
import { distance, length, levelLabel, money, speed, upkeepText } from "../../game/format";
import { LEVELS, type Level } from "../../game/types";
import { parseEdgePis, type EdgePi, type EdgePis } from "../../map/edgeEdit";
import { selection } from "../../game/ui";
import { edit, say, world } from "../../game/world";
import { createLine } from "../../game/lineActions";
import { live } from "../../map/live";
import { removeThing } from "../../game/remove";
import { describeIssues } from "../../game/issues";
import { moneyState } from "../../game/money";
import { edgeUpkeep } from "../../game/regionMoney";
import { Built, Chip, ConstructRemove, Icon, Row, Stepper } from "../widgets";
import { useClockQuery } from "../useClockQuery";

/** Radii the curve stepper walks through, m; past the last is auto (SPEC 6.1). */
const RADII = [100, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500, 1796];

/** A blueprint stretch's shape (T-064, T-092): a row per corner (PI) with its curve's speed and a
 * radius stepper (auto, or set), and the stretch's level. Dragging the corners is on the map
 * (map/edgeEdit.ts). */
function Shape({ edge, version }: { edge: number; version: number }) {
  const shape = useClockQuery<EdgePis>({ what: "edgePis", edge }, [edge, version], parseEdgePis, true);
  const [busy, setBusy] = useState(false);
  if (!shape) return null;
  const pis = shape.pis;
  const w = world.value;
  const track = w?.edges.find(e => e.id === edge);
  const from = track ? w?.nodes.get(track.a) : undefined, to = track ? w?.nodes.get(track.b) : undefined;
  if (!from || !to) return null;
  const put = async (next: EdgePi[]) => {
    if (busy) return;
    setBusy(true);
    try { await live.edgeEditor?.setPis(edge, next); }
    finally { setBusy(false); }
  };
  const add = (i: number) => {
    const a = i === 0 ? from : pis[i - 1], b = i === pis.length ? to : pis[i];
    const x = (a.x + b.x) / 2, y = (a.y + b.y) / 2;
    const p: EdgePi = { x, y, level: Math.round((a.level + b.level) / 2), radius: 0, used: 0, kmh: 0, mx: x, my: y };
    void put([...pis.slice(0, i), p, ...pis.slice(i)]);
  };
  const step = (i: number, d: number) => {
    const p = pis[i];
    const cur = p.radius > 0 ? p.radius : Infinity;
    let r: number;
    if (d < 0) r = [...RADII].reverse().find((x) => x < Math.min(cur, p.used > 0 ? p.used + 1 : cur)) ?? RADII[0];
    else r = RADII.find((x) => x > cur) ?? 0;
    void put(pis.map((q, k) => (k === i ? { ...q, radius: r >= RADII[RADII.length - 1] ? 0 : r } : q)));
  };
  // a straight piece has no corners, so its elevation is its ends', moved by edgesLevel as in the selection bar
  const level = pis.length ? (pis.every((p) => p.level === pis[0].level) ? pis[0].level : null) : track && track.levelMin === track.levelMax ? track.levelMin : null;
  const setLevel = async (l: Level) => {
    if (pis.length) return void edit({ op: "edgeLevel", edge, level: l });
    const a = await edit({ op: "edgesLevel", edges: [edge], level: l }, true);
    if (!a.ok) say(`This track cannot go to elevation ${levelLabel(l)}.`, true);
  };
  return (
    <>
      <div class="h">
        <span data-tip="Add a corner between points, then drag its square on the map.">Corners</span>
      </div>
      <div class="blueprint-corners">
        {Array.from({ length: pis.length + 1 }, (_, i) => <>
          <button class="btn text add-corner" disabled={busy} onClick={() => add(i)}
            aria-label={pis.length ? `Add corner ${i + 1}` : "Add corner"}>Add corner</button>
          {i < pis.length && <fieldset class="corner" disabled={busy}>
            <legend>Corner {i + 1}</legend>
            <div class="corner-row">
              <span class="grow">
                {pis[i].used > 0 ? speed(pis[i].kmh) : "No curve"}
                {pis[i].radius > 0 ? "" : <span class="muted"> auto</span>}
              </span>
              <Stepper value={pis[i].used > 0 ? length(pis[i].used) : "none"} wide label={`corner ${i + 1} radius`} onStep={(d) => step(i, d)} />
            </div>
            <div class="corner-row">
              <span class="grow">Elevation</span>
              <Stepper value={levelLabel(pis[i].level)} label={`corner ${i + 1} elevation`} onStep={d => {
                const next = Math.max(-3, Math.min(3, pis[i].level + d));
                if (next !== pis[i].level) void put(pis.map((p, k) => k === i ? { ...p, level: next } : p));
              }} />
              <button class="btn text" aria-label={`Remove corner ${i + 1}`} onClick={() => void put(pis.filter((_, k) => k !== i))}>Remove</button>
            </div>
          </fieldset>}
        </>)}
      </div>
      <div class="h">Elevation of the stretch</div>
      <div class="group levels">
        {LEVELS.map((l) => (
          <button disabled={busy} class={"btn" + (l === level ? " on" : "")} onClick={() => void setLevel(l)}>
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
  const tip = "price" in answer ? `Double up to quad track for ${money(answer.price * 1e6)} (blueprint)\nThe express pair runs beside it to the next junction or track end each way\nThe price includes a crossover past each station where one fits` : answer.why;
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
        removeLabel={`Remove ${e.crossover ? "crossover" : "track"}`} removeTip={e.built ? "Nothing is refunded" : ""} />
      <Built built={e.built} />
      {!e.built && e.off && <p class="muted" data-tip="Will not be constructed unless enabled">Disabled</p>}
      <ul class="rows">
        {!e.crossover && (
          <li data-tip={e.hsr ? "High-speed trains run up to 320 km/h here and ordinary trains up to 160 km/h" : "Trains run up to 160 km/h here. Over 160 needs high-speed track."}>
            <span class="grow">Speed limit</span>
            <span class="val">{speed(e.hsr ? 320 : 160)}</span>
          </li>
        )}
        <Row label="Slowest curve">{speed(e.vminKmh)}</Row>
        <Row label="Cost to construct">{money(cost)}</Row>
        {!e.crossover && m && (
          <Row label="Upkeep">
            <span data-tip={`${upkeepText(m, e.built ? "Upkeep" : "Once constructed, upkeep")} The lines here share it by their trains a day.`}>
              {money(edgeUpkeep({ ...e, built: true }, m))} a day
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
