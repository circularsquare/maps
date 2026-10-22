import { distance, levelLabel, money, upkeepRule } from "../game/format";
import { edgeUpkeep, stationUpkeep } from "../game/regionMoney";
import { clearBpPicked, levelPicked, offTotals, pickedState, removePicked, switchPicked } from "../game/blueprintPick";
import { liveMoney, moneyState } from "../game/money";
import { inkOn, LEVEL_COLOURS } from "../game/palette";
import { tool } from "../game/ui";
import { LEVELS, type BlueprintItem, type Level } from "../game/types";
import { constructBlueprints } from "../game/lineActions";
import { edit, redo, say, undo, world } from "../game/world";
import { Icon } from "./widgets";

const LEVEL_NAME: Record<Level, string> = { 3: "High viaduct", 2: "Viaduct", 1: "Viaduct", 0: "Ground level", [-1]: "Cut and cover", [-2]: "Tunnel", [-3]: "Deep tunnel" };
const KINDS = ["Track", "Stations", "Junctions", "Flyovers", "Flat crossings", "Trains"];
/** a quote row's track count for high-speed track (sim `HSR_ROW`, T-817) */
const HSR_ROW = 3;

function description(r: BlueprintItem) {
  if (r.kind === 5) return `${r.quantity} new ${r.quantity === 1 ? "car" : "cars"}, after spare cars`;
  const quantity = r.kind === 0 ? distance(r.quantity) : `${r.quantity} ${r.kind === 1 ? (r.quantity === 1 ? "station" : "stations") : (r.quantity === 1 ? "item" : "items")}`;
  return [quantity, r.ramp ? "ramp" : "", r.tracks ? (r.tracks === 1 ? "single track" : r.tracks === HSR_ROW ? "high-speed track" : "double track") : "", r.wet ? "over water" : "on land"].filter(Boolean).join(", ");
}

/** Build's quote and undo controls. Every item is priced by the clock worker's track model. */
export function BlueprintControls() {
  const w = world.value;
  const items = w?.money.blueprintItems ?? [];
  const total = items.reduce((sum, r) => sum + r.cost, 0);
  const m = moneyState.value;
  // T-867: what the blueprint will cost a day once constructed, the pieces switched off left out
  let upkeep = 0;
  if (w && m && total > 0) {
    for (const e of w.edges) if (!e.built && !e.off) upkeep += edgeUpkeep({ ...e, built: true }, m);
    for (const s of w.save.stations) if (!s.built && !s.off) upkeep += stationUpkeep(m);
  }
  return (
    <div class="blueprint-controls">
      <div class="blueprint-head">
        <span class="b">Blueprint</span>
        <b class={"num" + (m?.mode !== "sandbox" && total > liveMoney.value.cash * 1e6 ? " bad" : "")} data-total={total} data-tip="Total construction cost">{money(total)}</b>
        <div class="group">
          <button class="btn icon" data-tip="Undo (Ctrl+Z)" aria-label="Undo" disabled={!w?.canUndo} onClick={() => void undo()}><Icon.undo /></button>
          <button class="btn icon" data-tip="Redo (Ctrl+Y)" aria-label="Redo" disabled={!w?.canRedo} onClick={() => void redo()}><Icon.redo /></button>
        </div>
      </div>
      {m && upkeep > 0 && (
        <p class="small muted blueprint-upkeep" data-upkeep={upkeep} data-tip={`Once constructed, upkeep is ${upkeepRule(m)}, whether trains run or not. High-speed track costs ${m.upkeepHsr} times as much and single track ${m.upkeepSingle} times.`}>
          then {money(upkeep)} a day upkeep
        </p>
      )}
      <button class="btn text blueprint-construct" data-tip="Construct every blueprint piece (C)" disabled={total <= 0} onClick={() => void constructBlueprints()}>Construct blueprints<span class="key muted">C</span></button>
    </div>
  );
}

const pieces = (e: number, st: number) => [e ? `${e} track` : "", st ? `${st} ${st === 1 ? "station" : "stations"}` : ""].filter(Boolean).join(", ");

/** How many blueprint pieces are switched off (T-333), whatever the tool, since Construct and the
 * total leave them out. Selecting, disabling and enabling are in the map's BlueprintBar (T-353). */
export function BlueprintPick() {
  const off = offTotals.value;
  const offN = off.edges + off.stations;
  if (!offN) return null;
  return (
    <div class="blueprint-pick">
      <p class="small muted" data-tip={`Disabled: ${pieces(off.edges, off.stations)}. Construct and the total leave these out`}>{offN} disabled</p>
    </div>
  );
}

/** T-353, T-815, T-829: while the Select tool has a selection, a bar at the map's top centre, the
 * selection's inspector: how many pieces and how much track, then for blueprint the elevation
 * buttons, Disable and Enable; Remove and Clear for either kind. Appears and goes in one step. */
export function BlueprintBar() {
  const s = pickedState.value;
  const n = s.edges + s.stations;
  if (tool.value !== "blueprint" || n === 0) return null;
  const piece = (s.built ? "Constructed: " : "Blueprint: ") + pieces(s.edges, s.stations);
  return (
    <div class="blueprint-bar" role="group" aria-label="Selection">
      <span class="count" data-tip={piece}>{n} selected</span>
      {s.edges > 0 && <span class="num" data-tip="Length of the selected track">{distance(s.lengthM)}</span>}
      {!s.built && s.edges > 0 && (
        <div class="group levels">
          {LEVELS.map((l) => (
            <button class={"btn lv" + (l === s.level ? " on" : "")} style={{ "--lc": LEVEL_COLOURS[l], "--lk": inkOn(LEVEL_COLOURS[l]) }}
              data-tip={`Put the selected track at ${LEVEL_NAME[l].toLowerCase()} (${levelLabel(l)})`} aria-label={`Elevation ${levelLabel(l)}`} onClick={() => void levelPicked(l)}>
              {levelLabel(l)}
            </button>
          ))}
        </div>
      )}
      {!s.built && (
        <div class="group">
          <button class="btn text" data-tip="Disable the selection (D): Construct and the total leave it out" disabled={!s.on} onClick={() => void switchPicked(true)}><Icon.switchOff />Disable</button>
          <button class="btn text" data-tip="Enable the selection (E)" disabled={!s.off} onClick={() => void switchPicked(false)}><Icon.switchOn />Enable</button>
        </div>
      )}
      <button class="btn text" data-tip={s.built ? "Remove the selection (Delete). Nothing is refunded" : "Remove the selection (Delete)"} onClick={() => void removePicked()}><Icon.bin />Remove</button>
      <button class="btn text" data-tip="Clear the selection (Esc)" onClick={() => clearBpPicked()}><Icon.cross />Clear</button>
    </div>
  );
}

export function BlueprintBreakdown() {
  const items = world.value?.money.blueprintItems ?? [];
  return (
    <div class="blueprint-panel">
      {items.length > 0 ? (
        <details class="blueprint-breakdown" open>
          <summary>Blueprint cost breakdown</summary>
          {KINDS.map((kind, k) => {
            const rows = items.filter((r) => r.kind === k);
            if (!rows.length) return null;
            return (
              <div class="blueprint-category">
                <div class="small b">{kind}</div>
                <ul class="rows blueprint-items">
                  {rows.map((r) => (
                    <li data-cost={r.cost}>
                      <span class="what">
                        <span>{r.kind === 5 ? "New cars" : `${LEVEL_NAME[r.level]} (${levelLabel(r.level)})`}</span>
                        <span class="small muted">{description(r)}</span>
                      </span>
                      <span class="val" data-tip={`$${Math.round(r.cost).toLocaleString()}`}>{money(r.cost)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
        </details>
      ) : <p class="small muted">No blueprint to construct.</p>}
    </div>
  );
}
