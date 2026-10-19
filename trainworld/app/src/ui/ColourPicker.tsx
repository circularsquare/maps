// A colour picker (T-672), after languagedots' swatch editor: a saturation and brightness area, a hue
// strip, a hex box and preset swatches, in the game's tokens. A drag only calls `onPreview`; `onCommit`
// comes once, when the pointer is released, a swatch is clicked, a key step is made or the hex is
// typed, so the game gets one edit and not one per pointer move.

import { useEffect, useRef, useState } from "preact/hooks";

type Hsv = [number, number, number];

function hexToHsv(hex: string): Hsv {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  if (!m) return [0, 0, 0.5];
  const [r, g, b] = [m[1], m[2], m[3]].map((v) => parseInt(v, 16) / 255);
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  let h = 0;
  if (d) h = mx === r ? ((g - b) / d + (g < b ? 6 : 0)) : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [h * 60, mx ? d / mx : 0, mx];
}

function hsvToHex([h, s, v]: Hsv): string {
  const f = (n: number) => {
    const k = (n + h / 60) % 6;
    return Math.round((v - v * s * Math.max(0, Math.min(k, 4 - k, 1))) * 255).toString(16).padStart(2, "0");
  };
  return "#" + f(5) + f(3) + f(1);
}

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const VALID = /^#?[0-9a-f]{6}$/i;

/** A strip or area dragged with the pointer: `at` gets the position in 0..1, `done` the release. */
function useDrag(at: (x: number, y: number) => void, done: () => void) {
  const dragging = useRef(false);
  return {
    onPointerDown: (e: PointerEvent) => {
      if (e.button !== 0) return;
      const el = e.currentTarget as HTMLElement;
      el.setPointerCapture(e.pointerId);
      dragging.current = true;
      const r = el.getBoundingClientRect();
      at(clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height));
      e.preventDefault();
    },
    onPointerMove: (e: PointerEvent) => {
      if (!dragging.current) return;
      const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
      at(clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height));
    },
    onPointerUp: () => {
      if (!dragging.current) return;
      dragging.current = false;
      done();
    },
  };
}

export function ColourPicker({ value, presets = [], onPreview, onCommit }: {
  value: string;
  presets?: readonly string[];
  /** every move of a drag, at most once a frame */
  onPreview?: (hex: string) => void;
  /** the colour chosen: once per drag, click or typed colour */
  onCommit: (hex: string) => void;
}) {
  const [hsv, setHsv] = useState<Hsv>(() => hexToHsv(value));
  const [draft, setDraft] = useState<string | null>(null);
  const cur = useRef(hsv);
  const frame = useRef(0);
  const sent = useRef(value);
  const hex = hsvToHex(hsv);
  // a colour changed from outside (a swatch elsewhere, an undo) moves the pointers; a hue kept at grey
  const active = useRef(false);
  useEffect(() => {
    if (active.current) return;
    if (value.toLowerCase() !== sent.current.toLowerCase() && value.toLowerCase() !== hsvToHex(cur.current)) {
      cur.current = hexToHsv(value);
      setHsv(cur.current);
    }
    sent.current = value;
  }, [value]);
  useEffect(() => () => cancelAnimationFrame(frame.current), []);
  const move = (next: Hsv) => {
    active.current = true;
    cur.current = next;
    setHsv(next);
    setDraft(null);
    if (!frame.current) frame.current = requestAnimationFrame(() => { frame.current = 0; onPreview?.(hsvToHex(cur.current)); });
  };
  const commit = (next: Hsv = cur.current) => {
    cancelAnimationFrame(frame.current);
    frame.current = 0;
    active.current = false;
    cur.current = next;
    setHsv(next);
    setDraft(null);
    const h = hsvToHex(next);
    sent.current = h;
    onCommit(h);
  };
  const area = useDrag((x, y) => move([cur.current[0], x, 1 - y]), () => commit());
  const strip = useDrag((x) => move([x * 360, cur.current[1], cur.current[2]]), () => commit());
  const step = (e: KeyboardEvent, hue: boolean) => {
    const d = e.shiftKey ? 0.1 : 0.02;
    const [h, s, v] = cur.current;
    const to: Hsv | null = hue
      ? (e.key === "ArrowLeft" ? [(h + 360 - d * 360) % 360, s, v] : e.key === "ArrowRight" ? [(h + d * 360) % 360, s, v] : null)
      : e.key === "ArrowLeft" ? [h, clamp(s - d), v] : e.key === "ArrowRight" ? [h, clamp(s + d), v]
      : e.key === "ArrowUp" ? [h, s, clamp(v + d)] : e.key === "ArrowDown" ? [h, s, clamp(v - d)] : null;
    if (!to) return;
    e.preventDefault();
    commit(to);
  };
  const typed = (text: string) => {
    if (VALID.test(text)) commit(hexToHsv(text.startsWith("#") ? text : "#" + text));
    else setDraft(null);
  };
  const [h, s, v] = hsv;
  return (
    <div class="cpick">
      <div class="cpick-area" data-picker-area role="slider" tabIndex={0} aria-label="Saturation and brightness" aria-valuetext={hex}
        style={{ "--hue": `hsl(${h}, 100%, 50%)` }} {...area} onKeyDown={(e) => step(e, false)}>
        <i class="cpick-dot" style={{ left: `${s * 100}%`, top: `${(1 - v) * 100}%` }} />
      </div>
      <div class="cpick-hue" data-picker-hue role="slider" tabIndex={0} aria-label="Hue" aria-valuemin={0} aria-valuemax={360} aria-valuenow={Math.round(h)}
        {...strip} onKeyDown={(e) => step(e, true)}>
        <i class="cpick-dot" style={{ left: `${(h / 360) * 100}%` }} />
      </div>
      <div class="cpick-row">
        <span class="cpick-now" style={{ background: hex }} />
        <input class="cpick-hex" aria-label="Colour as hex" spellcheck={false} maxLength={7} value={draft ?? hex}
          onInput={(e) => setDraft(e.currentTarget.value)}
          onBlur={(e) => { if (draft !== null) typed(e.currentTarget.value.trim()); }}
          onKeyDown={(e) => { if (e.key === "Enter") { typed(e.currentTarget.value.trim()); e.currentTarget.blur(); } }} />
      </div>
      {presets.length > 0 && (
        <div class="cpick-presets">
          {presets.map((c) => (
            <button class={"swatch" + (c.toLowerCase() === hex ? " on" : "")} style={{ "--c": c }} data-tip={c} aria-label={c} onClick={() => commit(hexToHsv(c))} />
          ))}
        </div>
      )}
    </div>
  );
}
