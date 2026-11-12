// A line panel's title, rail, bus and ferry alike (T-2373): the badge and name, the editor for the
// name, symbol, shape (T-3319) and colour (T-670) with the colour a drag shows before the edit lands (T-672), and the
// group row.

import { useEffect, useRef, useState } from "preact/hooks";
import { LINE_PALETTE } from "../../game/palette";
import type { LineInput } from "../../game/types";
import { renameLinePrompt } from "../../game/ui";
import { edit, renameLine } from "../../game/world";
import { t as tr } from "../../i18n";
import { ColourPicker } from "../ColourPicker";
import { locateLine } from "../locate";
import { scrollerOf } from "../scroller";
import { Chip, Icon, ShapePicker } from "../widgets";
import { lineMode, modeShape, setLineShape, shapeOf } from "../../game/lineShapeState";
import { LineGroup } from "./LineGroup";
import { NameEdit } from "./NameEdit";

/** T-672: dragging a colour shows it at once (`setPreview`), and the line on the map follows with
 * one edit every 0.2 s at most; letting go sends the last colour. */
function useLiveColour(l: LineInput | undefined, setPreview: (hex: string | null) => void) {
  const live = useRef({ hex: null as string | null, timer: 0, last: 0, busy: false, gap: 200 });
  useEffect(() => () => clearTimeout(live.current.timer), []);
  const sendLive = () => {
    const s = live.current;
    s.timer = 0;
    if (s.hex === null || !l) return;
    if (s.busy) { s.timer = window.setTimeout(sendLive, 50); return; }
    const hex = s.hex;
    s.hex = null;
    s.busy = true;
    s.last = performance.now();
    void edit({ op: "lineLook", line: l.num, colour: hex }, true).then(() => {
      s.busy = false;
      s.gap = Math.max(200, 6 * (performance.now() - s.last)); // a slow machine follows less often, so edits take a fraction of its time
    });
  };
  const dragColour = (hex: string) => {
    setPreview(hex);
    const s = live.current;
    s.hex = hex;
    if (!s.timer) s.timer = window.setTimeout(sendLive, Math.max(0, s.gap - (performance.now() - s.last)));
  };
  const commitColour = (hex: string) => {
    if (!l) return;
    clearTimeout(live.current.timer);
    live.current.timer = 0;
    live.current.hex = null;
    setPreview(hex);
    void edit({ op: "lineLook", line: l.num, colour: hex }).then((a) => { if (!a.ok) setPreview(null); });
  };
  return { dragColour, commitColour };
}

/** T-1812: the symbol box's props. What is typed is kept until the box is left (or Enter) and then
 * sent alone as the symbol set by hand, so a rename landing meanwhile cannot reset the box and a
 * symbol equal to the name's own letter still sticks. A box the player did not type in sends
 * nothing; blank sends "", which gives the badge back to the name's letter. */
function useSymbolBox(l: LineInput | undefined) {
  const [draft, setDraft] = useState<string | null>(null);
  const typed = useRef(0), sent = useRef(-1);
  const commit = () => {
    if (draft === null || !l || sent.current === typed.current) return;
    sent.current = typed.current;
    const v = draft.trim(), at = typed.current;
    if (v === (l.symbol ?? "")) { setDraft(null); return; }
    void edit({ op: "lineLook", line: l.num, letter: v }).then(() => { if (typed.current === at) setDraft(null); });
  };
  return {
    value: draft ?? l?.letter ?? "",
    onInput: (e: { currentTarget: HTMLInputElement }) => { typed.current++; setDraft(e.currentTarget.value); },
    onBlur: commit,
    onKeyDown: (e: { key: string }) => { if (e.key === "Enter") commit(); },
  };
}

/** The title's editor state for line `id`; `shown` is its colour as drawn, a dragged one included. */
export function useLineTitle(id: string, l: LineInput | undefined) {
  const [editFor, setEditFor] = useState<string | null>(null);
  const [previewOf, setPreviewOf] = useState<{ id: string; hex: string } | null>(null);
  const editing = editFor === id;
  const setEditing = (open: boolean) => setEditFor(open ? id : null);
  const preview = previewOf?.id === id ? previewOf.hex : null;
  const setPreview = (hex: string | null) => setPreviewOf(hex === null ? null : { id, hex });
  const { dragColour, commitColour } = useLiveColour(l, setPreview);
  const symbolBox = useSymbolBox(l);
  useEffect(() => { if (preview !== null && l && l.colour.toLowerCase() === preview.toLowerCase()) setPreview(null); }, [l?.colour, preview]);
  // a click away, or Esc outside the name box (it handles its own), closes the editor
  useEffect(() => {
    if (!editing) return;
    const away = (e: PointerEvent) => { if (!(e.target as Element | null)?.closest?.("[data-line-edit]")) setEditing(false); };
    const esc = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || (e.target as Element | null)?.closest?.(".name-edit")) return;
      e.preventDefault();
      e.stopPropagation();
      setEditing(false);
    };
    window.addEventListener("pointerdown", away, true);
    window.addEventListener("keydown", esc, true);
    return () => {
      window.removeEventListener("pointerdown", away, true);
      window.removeEventListener("keydown", esc, true);
    };
  }, [editing]);
  // T-1894: the editor sits under the pinned title, so opening it scrolls the panel to its top
  const titleEl = useRef<HTMLDivElement>(null);
  useEffect(() => { if (editing) scrollerOf(titleEl.current)?.scrollTo({ top: 0 }); }, [editing]);
  return { id, editing, setEditing, shown: preview ?? l?.colour ?? "", dragColour, commitColour, symbolBox, titleEl };
}

export type LineTitleState = ReturnType<typeof useLineTitle>;

export function LineTitle({ l, t }: { l: LineInput; t: LineTitleState }) {
  return (
    <>
      <div class="title" data-line-edit ref={t.titleEl}>
        <button class={"badge-edit" + (t.editing ? " on" : "")} data-tip={tr("lineTitle.edit")} aria-label={tr("lineTitle.edit")}
          onClick={() => t.setEditing(true)} onDblClick={() => t.setEditing(true)}><Chip line={{ ...l, colour: t.shown }} size="lg" plain /></button>
        <NameEdit key={l.id} name={l.name} onRename={(n) => renameLine(l.id, n)} prompt={renameLinePrompt.value === t.id} onPrompted={() => { renameLinePrompt.value = null; }}
          control={{ open: t.editing, set: t.setEditing }}>
          <button class="btn icon" data-tip={tr("lineTitle.show")} aria-label={tr("lineTitle.show")} onClick={() => locateLine(l.stops)}>
            <Icon.locate />
          </button>
        </NameEdit>
      </div>
      {t.editing && (
        <div class="line-identity" data-line-edit>
          <label>{tr("lineTitle.symbol")} <input aria-label={tr("lineTitle.symbolLabel")} {...t.symbolBox} /></label>
          <div class="line-shape">
            <span>{tr("lineTitle.shape")}</span>
            <ShapePicker value={shapeOf(l.id)} colour={t.shown} onChange={(s) => setLineShape(l.id, s)}
              tips={(s) => (s === modeShape(lineMode(l.id)) ? tr("lineTitle.shapeDefault", { mode: lineMode(l.id) }) : undefined)} />
          </div>
          <ColourPicker value={l.colour} presets={LINE_PALETTE} onPreview={t.dragColour} onCommit={t.commitColour} />
        </div>
      )}
      <LineGroup l={l} />
    </>
  );
}
