// T-4259, from omi-key's PR 35: a rail line's stations renamed together from its panel. The stops list turns into a
// box per station in the list's order, filled with the names they show; Rename sends the changed ones as one edit (one
// undo), each renamed as its own panel would (`renameStations` in workers/clockDemand.ts). A paste of several lines
// fills the box pasted into and those below it, so a list of names from elsewhere goes in at once.
// The rows are the line as it is now and the typing is kept by station, so an edit made while the boxes are open (an
// undo, a station deleted) takes only its own rows with it.

import { useEffect, useRef, useState } from "preact/hooks";
import { t as tr } from "../../i18n";
import { complexOf } from "../../game/complexView";
import { edit, stationNames, stationsById, world } from "../../game/world";
import { setListHot } from "../../map/hovered";

interface RenameRow {
  /** what the typing is kept by: an unbound station's id, or its complex's bound stations */
  key: string;
  /** the station renamed: the row's first station in the list */
  id: string;
  /** the name it shows now */
  name: string;
}

/** One box per station shown, in `order` (station ids top first); a bound station stands for its complex, so the
 * bound stations of one complex share a box. */
function renameRows(order: readonly string[]): RenameRow[] {
  const w = world.value;
  if (!w) return [];
  const names = stationNames(w), byId = stationsById(w), seen = new Set<string>(), rows: RenameRow[] = [];
  for (const id of order) {
    const s = byId.get(id);
    if (!s) continue;
    const key = s.unbound ? id : "c:" + complexOf(id).filter((m) => !byId.get(m)?.unbound).sort().join(",");
    if (seen.has(key)) continue;
    seen.add(key);
    rows.push({ key, id, name: names.get(id) ?? "" });
  }
  return rows;
}

export function RenameStops({ order, colour, onClose }: { order: readonly string[]; colour: string; onClose: () => void }) {
  const rows = renameRows(order);
  const [typed, setTyped] = useState<ReadonlyMap<string, string>>(new Map());
  const [busy, setBusy] = useState(false);
  const list = useRef<HTMLOListElement>(null);
  const box = (k: number) => list.current?.querySelectorAll("input")[k];
  useEffect(() => {
    box(0)?.focus();
    box(0)?.select();
    return () => setListHot(null);
  }, []);
  const text = (r: RenameRow) => typed.get(r.key) ?? r.name;
  const isChanged = (r: RenameRow) => { const v = text(r).trim(); return !!v && v !== r.name; };
  const changed = rows.filter(isChanged);
  const save = async () => {
    if (busy) return;
    const w = world.peek(), byId = w ? stationsById(w) : undefined;
    const nodes: number[] = [], names: string[] = [];
    for (const r of changed) {
      const n = byId?.get(r.id)?.num;
      if (n !== undefined) nodes.push(n), names.push(text(r).trim());
    }
    if (!nodes.length) return onClose();
    setBusy(true);
    const a = await edit({ op: "renameStations", nodes, names });
    setBusy(false);
    if (a.ok) onClose();
  };
  const paste = (k: number, e: ClipboardEvent) => {
    const lines = (e.clipboardData?.getData("text") ?? "").split(/\r?\n/).map((s) => s.trim());
    while (lines.length && !lines[lines.length - 1]) lines.pop();
    if (lines.length < 2) return;
    e.preventDefault();
    const into = rows.slice(k, k + lines.length);
    setTyped((m) => {
      const next = new Map(m);
      into.forEach((r, j) => lines[j] && next.set(r.key, lines[j]));
      return next;
    });
  };
  const key = (k: number, e: KeyboardEvent) => {
    if (e.isComposing || e.keyCode === 229) return;
    if (e.key === "Enter") {
      e.preventDefault();
      const next = box(k + 1);
      if (next && !(e.ctrlKey || e.metaKey)) {
        next.focus();
        next.select();
      } else void save();
    }
  };
  // Esc closes the boxes and nothing more, from a box or a button: the panel and the selection stay
  const esc = (e: KeyboardEvent) => {
    if (e.key !== "Escape" || e.isComposing) return;
    e.preventDefault();
    e.stopPropagation();
    onClose();
  };
  return (
    <div class="rename-stops" onKeyDown={esc}>
      <ol ref={list} class="stops rename" style={{ "--c": colour }}>
        {rows.map((r, k) => (
          <li key={r.key}>
            <input class={"rename-box" + (isChanged(r) ? " changed" : "")} value={text(r)} disabled={busy}
              aria-label={tr("renameStops.box", { name: r.name })} data-tip={isChanged(r) ? r.name : undefined}
              onInput={(e) => { const v = e.currentTarget.value; setTyped((m) => new Map(m).set(r.key, v)); }}
              onFocus={() => setListHot({ stations: r.key.startsWith("c:") ? complexOf(r.id) : [r.id] })} onBlur={() => setListHot(null)}
              onPaste={(e) => paste(k, e)} onKeyDown={(e) => key(k, e)} />
          </li>
        ))}
      </ol>
      <div class="line-stop-actions">
        <button class="btn text on" disabled={busy || !changed.length} onClick={() => void save()}>
          {tr("renameStops.save", { count: changed.length })}
        </button>
        <button class="btn text" disabled={busy} onClick={onClose}>{tr("renameStops.cancel")}</button>
      </div>
    </div>
  );
}
