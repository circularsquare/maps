import { useEffect, useState } from "preact/hooks";
import { inkOn, LEVEL_COLOURS } from "../game/palette";
import type { Level } from "../game/types";
import { distance, levelLabel, levelName, money, moneyExact, plural, upkeepText } from "../game/format";
import { edgeUpkeep, stationUpkeep } from "../game/regionMoney";
import { clearBpPicked, levelPicked, offTotals, pickedState, removePicked, switchPicked } from "../game/blueprintPick";
import { liveMoney, moneyState } from "../game/money";
import { tool } from "../game/ui";
import type { BlueprintItem } from "../game/types";
import { constructBlueprints } from "../game/lineActions";
import { world } from "../game/world";
import { redo, undo } from "../game/fareUndo";
import { Icon, LevelButtons } from "./widgets";

const KINDS = ["Track", "Stations", "Junctions", "Flyovers", "Flat crossings", "Trains"];
/** a quote row's track count for high-speed track (sim `HSR_ROW`, T-817) */
const HSR_ROW = 3;

function description(r: BlueprintItem) {
  if (r.kind === 5) return `${r.quantity} new ${r.quantity === 1 ? "car" : "cars"}, after spare cars`;
  const quantity = r.kind === 0 ? distance(r.quantity) : plural(r.quantity, r.kind === 1 ? "station" : "item");
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
        <span class="bp-total">
          <b class={"num" + (m?.mode !== "sandbox" && total > liveMoney.value.cash * 1e6 ? " bad" : "")} data-tip="Total construction cost">{money(total)}</b>
          <OffCost />
        </span>
        <div class="group">
          <button class="btn icon" data-tip="Undo (Ctrl+Z)" aria-label="Undo" disabled={!w?.canUndo} onClick={() => void undo()}><Icon.undo /></button>
          <button class="btn icon" data-tip="Redo (Ctrl+Y)" aria-label="Redo" disabled={!w?.canRedo} onClick={() => void redo()}><Icon.redo /></button>
        </div>
      </div>
      {m && upkeep > 0 && (
        <p class="small muted blueprint-upkeep" data-tip={upkeepText(m, "Once constructed, upkeep")}>
          {money(upkeep)} / day upkeep
        </p>
      )}
      <button class="btn text blueprint-construct" data-tip="Construct every blueprint piece (C)" disabled={total <= 0} onClick={() => void constructBlueprints()}>Construct blueprints<span class="key muted">C</span></button>
    </div>
  );
}

const pieces = (e: number, st: number) => [e ? `${e} track` : "", st ? plural(st, "station") : ""].filter(Boolean).join(", ");

/** T-1296: what the disabled blueprint would cost, beside the total: everything unbuilt
 * (`blueprintCost`, disabled pieces in it) less the quote's rows other than new cars. */
function OffCost() {
  const w = world.value, off = offTotals.value;
  if (!w || !(off.edges + off.stations)) return null;
  const on = w.money.blueprintItems.reduce((sum, r) => sum + (r.kind === 5 ? 0 : r.cost), 0);
  const cost = w.money.blueprintCost - on;
  if (!(cost >= 0.5e6)) return null;
  return <span class="small muted num off-cost" data-tip={`Disabled blueprint: ${pieces(off.edges, off.stations)}, ${moneyExact(cost)}, left out of the total`}>{money(cost)} disabled</span>;
}

/** How many blueprint pieces are switched off (T-333), whatever the tool, since Construct and the
 * total leave them out. Selecting, disabling and enabling are in the map's BlueprintBar (T-353). */
export function BlueprintPick() {
  const off = offTotals.value;
  const offN = off.edges + off.stations;
  if (!offN) return null;
  return (
    <div class="blueprint-pick">
      <p class="small muted" data-tip={`${pieces(off.edges, off.stations)} will not be constructed`}>{offN} disabled</p>
    </div>
  );
}

/** T-353, T-815, T-829: while the Select tool has a selection, a bar at the map's top centre, the
 * selection's inspector: how many pieces and how much track, then for blueprint the elevation
 * buttons, Disable and Enable; Remove and Clear for either kind. Appears and goes in one step. */
export function BlueprintBar() {
  const s = pickedState.value;
  const n = s.edges + s.stations;
  const [levels, setLevels] = useState(false);
  useEffect(() => { if (n === 0) setLevels(false); }, [n]);
  if (tool.value !== "blueprint" || n === 0) return null;
  const piece = (s.built ? "Built: " : "Blueprint: ") + pieces(s.edges, s.stations);
  const levelTip = (l: Level) => `Put the selected track at ${levelName(l).toLowerCase()} (${levelLabel(l)})`;
  const canLevel = !s.built && s.edges > 0;
  return (
    <>
      <div class="blueprint-bar" role="group" aria-label="Selection">
        <span class="count" data-tip={`Selected: ${piece}`}><span class="bp-ico"><Icon.marquee /></span>{n}<span class="bp-label"> selected</span></span>
        {s.edges > 0 && <span class="num" data-tip="Length of the selected track">{distance(s.lengthM)}</span>}
        {canLevel && <div class="bp-levels"><LevelButtons value={s.level} tip={levelTip} onPick={(l) => void levelPicked(l)} /></div>}
        {canLevel && (
          <button class={"btn bp-level" + (s.level === null ? "" : " lv on")} style={s.level === null ? undefined : { "--lc": LEVEL_COLOURS[s.level], "--lk": inkOn(LEVEL_COLOURS[s.level]) }}
            data-tip={s.level === null ? "Elevation of the selected track: mixed" : `Elevation ${levelName(s.level)}`} aria-label="Elevation" aria-expanded={levels}
            onClick={() => setLevels(!levels)}>{s.level === null ? <Icon.autoLevel /> : levelLabel(s.level)}</button>
        )}
      {!s.built && (
        <div class="group">
          <button class="btn text" data-tip="Disable the selection so it will not be constructed (D)" disabled={!s.on} onClick={() => void switchPicked(true)}><Icon.switchOff /><span class="bp-label">Disable</span></button>
          <button class="btn text" data-tip="Enable the selection (E)" disabled={!s.off} onClick={() => void switchPicked(false)}><Icon.switchOn /><span class="bp-label">Enable</span></button>
        </div>
      )}
      <button class="btn text" data-tip={s.built ? "Remove the selection (Delete). Nothing is refunded" : "Remove the selection (Delete)"} onClick={() => void removePicked()}><Icon.bin /><span class="bp-label">Remove</span></button>
      <button class="btn text" data-tip="Clear the selection (Esc)" onClick={() => clearBpPicked()}><Icon.cross /><span class="bp-label">Clear</span></button>
      </div>
      {canLevel && levels && (
        <div class="blueprint-levels touch-levels">
          <LevelButtons value={s.level} tip={levelTip} onPick={(l) => { void levelPicked(l); setLevels(false); }} />
        </div>
      )}
    </>
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
                    <li>
                      <span class="what">
                        <span>{r.kind === 5 ? "New cars" : `${levelName(r.level)} (${levelLabel(r.level)})`}</span>
                        <span class="small muted">{description(r)}</span>
                      </span>
                      <span class="val" data-tip={moneyExact(r.cost)}>{money(r.cost)}</span>
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
