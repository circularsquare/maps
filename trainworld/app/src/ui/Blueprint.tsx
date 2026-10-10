import { km, levelLabel, money } from "../game/format";
import type { BlueprintItem, Level } from "../game/types";
import { edit, redo, say, undo, world } from "../game/world";
import { Icon } from "./widgets";

const LEVEL_NAME: Record<Level, string> = { 3: "High viaduct", 2: "Viaduct", 1: "Viaduct", 0: "Ground level", [-1]: "Cut and cover", [-2]: "Tunnel", [-3]: "Deep tunnel" };
const KINDS = ["Track", "Stations", "Junctions", "Flyovers", "Flat crossings", "Trains"];

function description(r: BlueprintItem) {
  if (r.kind === 5) return `${r.quantity} new ${r.quantity === 1 ? "car" : "cars"}, after spare cars`;
  const quantity = r.kind === 0 ? km(r.quantity) : `${r.quantity} ${r.kind === 1 ? (r.quantity === 1 ? "station" : "stations") : (r.quantity === 1 ? "item" : "items")}`;
  return [quantity, r.ramp ? "ramp" : "", r.tracks ? (r.tracks === 1 ? "single track" : "double track") : "", r.wet ? "over water" : "on land"].filter(Boolean).join(", ");
}

/** Build's quote and undo controls. Every item is priced by the clock worker's track model. */
export function BlueprintControls() {
  const w = world.value;
  const items = w?.money.blueprintItems ?? [];
  const total = items.reduce((sum, r) => sum + r.cost, 0);
  const construct = async () => {
    const a = await edit({ op: "constructAll" });
    if (a.ok) say(`Constructed for ${money(a.charge * 1e6)}.`);
  };
  return (
    <div class="blueprint-controls">
      <div class="blueprint-head">
        <span class="b">Blueprint</span>
        <b class="num" data-total={total} title="Total construction cost">{money(total)}</b>
        <div class="group">
          <button class="btn icon" title="Undo (Ctrl+Z)" disabled={!w?.canUndo} onClick={() => void undo()}><Icon.undo /></button>
          <button class="btn icon" title="Redo (Ctrl+Y)" disabled={!w?.canRedo} onClick={() => void redo()}><Icon.redo /></button>
        </div>
      </div>
      <button class="btn text blueprint-construct" disabled={total <= 0} onClick={construct}>Construct blueprints</button>
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
                      <span class="val" title={`$${Math.round(r.cost).toLocaleString()}`}>{money(r.cost)}</span>
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
