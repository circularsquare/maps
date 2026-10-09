// Small shared pieces: line chip, stepper, segmented buttons, checkbox, icons (inline SVG).

import type { ComponentChildren } from "preact";
import { inkOn } from "../game/palette";
import type { LineInput } from "../game/types";

export const Icon = {
  minus: () => <svg viewBox="0 0 16 16"><path d="M4 8h8" /></svg>,
  plus: () => <svg viewBox="0 0 16 16"><path d="M4 8h8M8 4v8" /></svg>,
  pencil: () => <svg viewBox="0 0 16 16"><path d="M10.5 3l2.5 2.5L6 12.5H3.5V10z" /></svg>,
  pause: () => <svg viewBox="0 0 16 16"><path class="stroke2" d="M5.5 3.5v9M10.5 3.5v9" /></svg>,
  play1: () => <svg viewBox="0 0 16 16"><path class="solid" d="M5 3.5l7 4.5-7 4.5z" /></svg>,
  play2: () => <svg viewBox="0 0 16 16"><path class="solid" d="M2.5 4l5.5 4-5.5 4zM8 4l5.5 4L8 12z" /></svg>,
  play3: () => <svg viewBox="0 0 16 16"><path class="solid" d="M1.5 5l4 3-4 3zM6 5l4 3-4 3zM10.5 5l4 3-4 3z" /></svg>,
  track: () => <svg viewBox="0 0 16 16"><path d="M2.5 10.5l8-8M5.5 13.5l8-8M3 9l4 4M6 6l4 4M9 3l4 4" /></svg>,
  station: () => <svg viewBox="0 0 16 16"><path d="M1.5 8h3.5M11 8h3.5" /><circle cx="8" cy="8" r="3" /></svg>,
  pointer: () => <svg viewBox="0 0 16 16"><path d="M4 2.5v10l3-2.8 2 4.3 1.8-.8-2-4.2 4-.3z" /></svg>,
  undo: () => <svg viewBox="0 0 16 16"><path d="M5.5 4L2.5 7l3 3M3 7h6.5a3.5 3.5 0 010 7H7" /></svg>,
  redo: () => <svg viewBox="0 0 16 16"><path d="M10.5 4l3 3-3 3M13 7H6.5a3.5 3.5 0 000 7H9" /></svg>,
  cross: () => <svg viewBox="0 0 16 16"><path d="M4.5 4.5l7 7M11.5 4.5l-7 7" /></svg>,
  bin: () => <svg viewBox="0 0 16 16"><path d="M2.5 4.5h11M6.5 4.5V2.5h3v2M4 4.5l.8 9h6.4l.8-9M6.8 7v4M9.2 7v4" /></svg>,
  gear: () => (
    <svg viewBox="0 0 16 16">
      <circle cx="8" cy="8" r="2.2" />
      <path d="M8 1.8v2M8 12.2v2M1.8 8h2M12.2 8h2M3.6 3.6l1.4 1.4M11 11l1.4 1.4M3.6 12.4L5 11M11 5l1.4-1.4" />
    </svg>
  ),
};

export function Chip({ line, size = "" }: { line: LineInput; size?: "" | "sm" | "lg" }) {
  return (
    <span class={"chip " + size} style={{ "--c": line.colour, "--ct": inkOn(line.colour) }}>
      {line.letter}
    </span>
  );
}

export function Stepper(p: { value: ComponentChildren; onStep: (d: number) => void; wide?: boolean; label: string }) {
  return (
    <span class={"stepper" + (p.wide ? " wide" : "")}>
      <button class="btn" aria-label={"Less " + p.label} onClick={() => p.onStep(-1)}>
        <Icon.minus />
      </button>
      <span class="v">{p.value}</span>
      <button class="btn" aria-label={"More " + p.label} onClick={() => p.onStep(1)}>
        <Icon.plus />
      </button>
    </span>
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

export function Row(p: { label: ComponentChildren; children?: ComponentChildren; bold?: boolean }) {
  return (
    <li class={p.bold ? "b" : ""}>
      <span class="grow">{p.label}</span>
      {p.children !== undefined && <span class="val">{p.children}</span>}
    </li>
  );
}
