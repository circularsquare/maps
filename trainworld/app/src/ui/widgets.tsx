// Small shared pieces: line chip, stepper, segmented buttons, checkbox, icons (inline SVG).

import type { ComponentChildren } from "preact";
import { useRef, useState } from "preact/hooks";
import { inkOn } from "../game/palette";
import { PLANE_PATH } from "../map/glyphs";
import { bulletEpoch, bulletShift } from "./bullet";
import type { LineInput } from "../game/types";
import { selection, type ListSort } from "../game/ui";
import type { Signal } from "@preact/signals";

export const Icon = {
  minus: () => <svg viewBox="0 0 16 16"><path d="M4 8h8" /></svg>,
  plus: () => <svg viewBox="0 0 16 16"><path d="M4 8h8M8 4v8" /></svg>,
  pencil: () => <svg viewBox="0 0 16 16"><path d="M10.5 3l2.5 2.5L6 12.5H3.5V10z" /></svg>,
  check: () => <svg viewBox="0 0 16 16"><path d="M3.5 8.5l3 3 6-7" /></svg>,
  // T-534: the station name's suggestions, round and round
  cycle: () => <svg viewBox="0 0 16 16"><path d="M12.8 6.5A5 5 0 004 5M3.2 9.5A5 5 0 0012 11M4 2v3h3M12 14v-3H9" /></svg>,
  // T-569: take the map to a station or line (a sight with its cross hairs)
  locate: () => <svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="3.5" /><path d="M8 1.5v3M8 11.5v3M1.5 8h3M11.5 8h3" /></svg>,
  pause: () => <svg viewBox="0 0 16 16"><path class="stroke2" d="M5.5 3.5v9M10.5 3.5v9" /></svg>,
  play1: () => <svg viewBox="0 0 16 16"><path class="solid" d="M5 3.5l7 4.5-7 4.5z" /></svg>,
  play2: () => <svg viewBox="0 0 16 16"><path class="solid" d="M2.5 4l5.5 4-5.5 4zM8 4l5.5 4L8 12z" /></svg>,
  play3: () => <svg viewBox="0 0 16 16"><path class="solid" d="M1.5 5l4 3-4 3zM6 5l4 3-4 3zM10.5 5l4 3-4 3z" /></svg>,
  track: () => <svg viewBox="0 0 16 16"><path d="M2.5 10.5l8-8M5.5 13.5l8-8M3 9l4 4M6 6l4 4M9 3l4 4" /></svg>,
  station: () => <svg viewBox="0 0 16 16"><path d="M1.5 8h3.5M11 8h3.5" /><circle cx="8" cy="8" r="3" /></svg>,
  // T-445: a run of segments, each starting where the last ended
  keepPlacing: () => <svg viewBox="0 0 16 16"><path d="M2 12.5l4-6 3 3 5-6.5" /><circle class="solid" cx="6" cy="6.5" r="1.4" /><circle class="solid" cx="9" cy="9.5" r="1.4" /><circle class="solid" cx="14" cy="3" r="1.4" /></svg>,
  // T-338: two parallel tracks joined by an X
  crossover: () => <svg viewBox="0 0 16 16"><path d="M1.5 3.5h13M1.5 12.5h13M4.5 3.5l7 9M11.5 3.5l-7 9" /></svg>,
  pointer: () => <svg viewBox="0 0 16 16"><path d="M4 2.5v10l3-2.8 2 4.3 1.8-.8-2-4.2 4-.3z" /></svg>,
  undo: () => <svg viewBox="0 0 16 16"><path d="M5.5 4L2.5 7l3 3M3 7h6.5a3.5 3.5 0 010 7H7" /></svg>,
  redo: () => <svg viewBox="0 0 16 16"><path d="M10.5 4l3 3-3 3M13 7H6.5a3.5 3.5 0 000 7H9" /></svg>,
  cross: () => <svg viewBox="0 0 16 16"><path d="M4.5 4.5l7 7M11.5 4.5l-7 7" /></svg>,
  bin: () => <svg viewBox="0 0 16 16"><path d="M2.5 4.5h11M6.5 4.5V2.5h3v2M4 4.5l.8 9h6.4l.8-9M6.8 7v4M9.2 7v4" /></svg>,
  // T-333: the blueprint selection tool (a dashed box), and switching the selection off or on
  marquee: () => <svg viewBox="0 0 16 16"><path d="M2.5 5.5v-3h3M10.5 2.5h3v3M13.5 10.5v3h-3M5.5 13.5h-3v-3M7 7l5 2-2 .8L9 12z" /></svg>,
  switchOff: () => <svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="5.5" /><path d="M4.1 11.9l7.8-7.8" /></svg>,
  switchOn: () => <svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="5.5" /><path d="M5.3 8.3l1.8 1.8 3.6-3.9" /></svg>,
  // T-252: the dock's tabs, and T-251's save button
  wrench: () => <svg viewBox="0 0 16 16"><path d="M13.2 4.6a3.3 3.3 0 01-4.4 3.6L3.9 13.1a1.3 1.3 0 01-1.9-1.9l4.9-4.9a3.3 3.3 0 013.6-4.4L8.6 4l.5 2.4 2.4.5z" /></svg>,
  route: () => <svg viewBox="0 0 16 16"><path d="M2.5 12.5l4-6 4 3 3-6.5" /><circle class="solid" cx="2.5" cy="12.5" r="1.5" /><circle class="solid" cx="6.5" cy="6.5" r="1.5" /><circle class="solid" cx="10.5" cy="9.5" r="1.5" /><circle class="solid" cx="13.5" cy="3" r="1.5" /></svg>,
  // a platform under its canopy, the track beside it
  building: () => <svg viewBox="0 0 16 16"><path d="M1.5 5.5h13l-2-3h-9zM4 5.5v5M12 5.5v5M1.5 10.5h13M1.5 13.5h13" /></svg>,
  money: () => <svg viewBox="0 0 16 16"><rect x="1.5" y="3.5" width="10" height="6.5" /><circle cx="6.5" cy="6.75" r="1.4" /><circle class="bg" cx="11.5" cy="11.5" r="3" /><circle cx="11.5" cy="11.5" r="1.2" /></svg>,
  save: () => <svg viewBox="0 0 16 16"><path d="M2.5 2.5h9l2 2v9h-11zM5 2.5v3.5h5v-3.5M4.5 13.5v-4.5h7v4.5" /></svg>,
  // T-332: the Lines and Stations tabs' scope toggle (the viewed place, the whole world)
  pin: () => <svg viewBox="0 0 16 16"><path d="M8 14.5s4.5-4.2 4.5-7.6a4.5 4.5 0 00-9 0c0 3.4 4.5 7.6 4.5 7.6z" /><circle cx="8" cy="6.8" r="1.6" /></svg>,
  globe: () => <svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="6" /><path d="M2 8h12M8 2c-2.2 2-2.2 10 0 12M8 2c2.2 2 2.2 10 0 12" /></svg>,
  // T-374: how commuters reach a station: on foot, or by a ride (car, bus, bike or lift)
  walk: () => <svg viewBox="0 0 16 16"><circle cx="9.5" cy="2.8" r="1.3" /><path d="M9.5 5.2l-1.8 3 2.3 2.3v3.7M7.7 8.2L5.3 9.4M9.6 6.4l2.4 1.8M7.7 8.2L6.2 14" /></svg>,
  ride: () => <svg viewBox="0 0 16 16"><path d="M1.8 11.2V8.4l1.6-3.4h9.2l1.6 3.4v2.8zM1.8 8.4h12.4" /><circle cx="4.8" cy="11.4" r="1.3" class="bg" /><circle cx="11.2" cy="11.4" r="1.3" class="bg" /></svg>,
  // T-378: an inter-region trip's modes (with `ride` for the car)
  train: () => <svg viewBox="0 0 16 16"><rect x="3.8" y="1.8" width="8.4" height="9.7" rx="2" /><path d="M3.8 7h8.4M6 14.2l1-2.7M10 14.2l-1-2.7" /></svg>,
  plane: () => <svg viewBox="0 0 16 16"><path d={PLANE_PATH} /></svg>,
  // T-685: a city's star in the Cities list
  star: () => <svg viewBox="0 0 16 16"><path class="solid" d="M8 1.5l1.9 4.1 4.5.5-3.4 3 1 4.4L8 11.3l-4 2.2 1-4.4-3.4-3 4.5-.5z" /></svg>,
  gear: () => (
    <svg viewBox="0 0 16 16">
      <path d="M6.83 3.14L7.09 1.26L8.91 1.26L9.17 3.14L10.61 3.74L12.13 2.59L13.41 3.87L12.26 5.39L12.86 6.83L14.74 7.09L14.74 8.91L12.86 9.17L12.26 10.61L13.41 12.13L12.13 13.41L10.61 12.26L9.17 12.86L8.91 14.74L7.09 14.74L6.83 12.86L5.39 12.26L3.87 13.41L2.59 12.13L3.74 10.61L3.14 9.17L1.26 8.91L1.26 7.09L3.14 6.83L3.74 5.39L2.59 3.87L3.87 2.59L5.39 3.74z" />
      <circle cx="8" cy="8" r="2.2" />
    </svg>
  ),
};

/** A line's bullet (T-521). Everywhere it is drawn a hover names the line and a click selects it,
 * except `plain` (the selected line's own panel, a search row that is the line). `pick` replaces
 * the selecting. `gone`: hidden but keeping its space (a station row's bullets past its "+", T-530). */
export function Chip({ line, size = "", plain, pick, gone }: { line: LineInput; size?: "" | "sm" | "lg"; plain?: boolean; pick?: (l: LineInput) => void; gone?: boolean }) {
  const click = plain ? undefined : (e: MouseEvent) => {
    e.stopPropagation();
    if (pick) pick(line);
    else selection.value = { kind: "line", line: line.id };
  };
  void bulletEpoch.value;
  return (
    <span class={"chip " + size + (plain ? "" : " link")} style={{ "--c": line.colour, "--ct": inkOn(line.colour), visibility: gone ? "hidden" : undefined }}
      data-tip={plain ? undefined : line.name} onClick={click}>
      <span style={{ transform: bulletShift(line.letter, size) }}>{line.letter}</span>
    </span>
  );
}

/** `jump`: Shift-click steps by 5 and Ctrl-click by 20 (T-516), the buttons' hover says so.
 * `type` (T-626): a click on the value opens a box for typing it; Enter or leaving the box applies
 * `set` with the number typed (anything else is dropped), Escape cancels. `text` is what the box
 * starts with, `tip` the value's hover. */
export function Stepper(p: { value: ComponentChildren; onStep: (d: number) => void; wide?: boolean; label: string; jump?: boolean; type?: { text: string; set: (v: number) => void; tip?: string }; tipDown?: string; tipUp?: string }) {
  const by = (e: MouseEvent) => (p.jump ? (e.ctrlKey ? 20 : e.shiftKey ? 5 : 1) : 1);
  const jump = p.jump ? "Shift-click for 5, Ctrl-click for 20" : undefined;
  const tip = (own?: string) => (own && jump ? own + "\n" + jump : own ?? jump);
  return (
    <span class={"stepper" + (p.wide ? " wide" : "")}>
      <button class="btn" aria-label={"Less " + p.label} data-tip={tip(p.tipDown)} onClick={(e) => p.onStep(-by(e))}>
        <Icon.minus />
      </button>
      {p.type ? <TypedValue label={p.label} type={p.type}>{p.value}</TypedValue> : <span class="v">{p.value}</span>}
      <button class="btn" aria-label={"More " + p.label} data-tip={tip(p.tipUp)} onClick={(e) => p.onStep(by(e))}>
        <Icon.plus />
      </button>
    </span>
  );
}

/** A stepper's value that opens a box on a click (T-626). */
function TypedValue(p: { label: string; type: { text: string; set: (v: number) => void; tip?: string }; children: ComponentChildren }) {
  const [draft, setDraft] = useState<string | null>(null);
  const done = useRef(false);
  const fresh = useRef(false);
  // focused as the box appears, so what is typed next goes into it
  const open = (el: HTMLInputElement | null) => {
    if (!el || !fresh.current) return;
    fresh.current = false;
    el.focus();
    el.select();
  };
  const commit = () => {
    if (done.current) return;
    done.current = true;
    const v = parseFloat((draft ?? "").replace(/[^0-9.\-]/g, ""));
    setDraft(null);
    if (Number.isFinite(v)) p.type.set(v);
  };
  if (draft === null) {
    return <button class="v typed" data-tip={p.type.tip} aria-label={"Type " + p.label} onClick={() => { done.current = false; fresh.current = true; setDraft(p.type.text); }}>{p.children}</button>;
  }
  return (
    <input class="v typing" ref={open} value={draft} inputMode="decimal" aria-label={p.label}
      onInput={(e) => setDraft(e.currentTarget.value)} onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") commit();
        if (e.key === "Escape") { done.current = true; setDraft(null); }
      }} />
  );
}

/** Buttons sharing borders, one of them on. */
export function Segmented<T>(p: { options: [T, ComponentChildren][]; value: T; onChange: (v: T) => void; cls?: string }) {
  return (
    <div class="group">
      {p.options.map(([v, label]) => (
        <button class={"btn " + (p.cls ?? "") + (v === p.value ? " on" : "")} onClick={() => p.onChange(v)}>
          {label}
        </button>
      ))}
    </div>
  );
}

export function Check(p: { checked: boolean; onChange: (v: boolean) => void; children: ComponentChildren }) {
  return (
    <label class="check">
      <input type="checkbox" checked={p.checked} onChange={(e) => p.onChange((e.target as HTMLInputElement).checked)} />
      {p.children}
    </label>
  );
}

/** T-310: a list's column headers. A click sorts by the column (numbers biggest first), a second
 * click reverses it, and the sorted column carries a small arrow. A column with a `cls` is a
 * right-aligned number column of that width; the arrow sits before its label so the label stays
 * over the figures. */
export function SortHead<K extends string>({ sort, cols }: { sort: Signal<ListSort<K>>; cols: { key: K; label: string; tip: string; first: 1 | -1; cls?: string; left?: boolean }[] }) {
  const s = sort.value;
  return (
    <div class="sort-head">
      {cols.map((c) => {
        const on = s.key === c.key;
        const arrow = <i class={"sort-arrow" + (on ? (s.dir > 0 ? " up" : " down") : "")} />;
        return (
          <button class={"sort-btn" + (c.cls ? " val " + c.cls : " grow") + (on ? " on" : "")} data-tip={c.tip}
            onClick={() => (sort.value = on ? { key: c.key, dir: s.dir === 1 ? -1 : 1 } : { key: c.key, dir: c.first })}>
            {c.cls && !c.left && arrow}{c.label}{(!c.cls || c.left) && arrow}
          </button>
        );
      })}
    </div>
  );
}

export function Row(p: { label: ComponentChildren; children?: ComponentChildren; bold?: boolean }) {
  return (
    <li class={p.bold ? "b" : ""}>
      <span class="grow">{p.label}</span>
      {p.children !== undefined && <span class="val">{p.children}</span>}
    </li>
  );
}
