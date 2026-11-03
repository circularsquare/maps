import { diagramShown } from "../diagram/view";

/** T-2160: the network diagram in place of the map, and back (key M). */
export function DiagramButton() {
  const on = diagramShown.value;
  const tip = on ? "Show map (M)" : "Show network diagram (M)";
  return <button type="button" class={"btn icon map-diagram" + (on ? " on" : "")} data-tip={tip} aria-label={tip} aria-pressed={on}
    onClick={() => (diagramShown.value = !on)}>
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
      <path d="M2 4h5l4 4h3" />
      <path d="M2 12h3l4-4" />
      <circle cx="7" cy="4" r="1.4" fill="var(--bg)" />
      <circle cx="11" cy="8" r="1.4" fill="var(--bg)" />
    </svg>
  </button>;
}
