import { diagramProgress, diagramReady, diagramShown, saveDiagramPicture } from "../diagram/view";
import { BusyTag } from "./BusyTag";
import { LoadBar } from "./LoadingScreen";
import { BunnyTrain } from "./Spinner";
import { Icon } from "./widgets";

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

/** T-2161, T-2163: the map's upper left while the diagram is up: the save button, the first layout's
 * loading bar (T-2235, as the demand status's) and the busy tag. */
export function DiagramTopLeft() {
  const share = diagramProgress.value;
  return <div class="map-top-left">
    <div class="map-top-row">
      {diagramReady.value && <button type="button" class="btn icon diagram-save" data-tip="Save picture" aria-label="Save picture" onClick={saveDiagramPicture}><Icon.download /></button>}
      {share !== null && <div class="status-tag" role="status">
        <BunnyTrain small />
        <span class="demand-busy">Calculating diagram</span>
        <LoadBar fraction={share} step="Calculating diagram" thin tip={`${Math.floor(share * 100)}%`} />
      </div>}
      <BusyTag />
    </div>
  </div>;
}
