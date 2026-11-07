import { diagramProgress, diagramReady, diagramShown, diagramWater, saveDiagramPicture } from "../diagram/view";
import { t } from "../i18n";
import { BusyTag } from "./BusyTag";
import { LoadBar } from "./LoadingScreen";
import { Icon } from "./widgets";

/** T-2160: the network diagram in place of the map, and back (no key while it is a work in progress, T-2608). */
export function DiagramButton() {
  const on = diagramShown.value;
  const tip = on ? t("diagramButton.showMap") : t("diagramButton.showDiagram");
  return <button type="button" class={"btn icon map-diagram" + (on ? " on" : "")} data-tip={tip} aria-label={tip} aria-pressed={on}
    onClick={() => (diagramShown.value = !on)}>
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
      <path d="M1.5 3.5l4-1.5 5 2 4-1.5v9.5l-4 1.5-5-2-4 1.5z" />
      <path d="M5.5 2v9.5M10.5 4v9.5" />
    </svg>
  </button>;
}

/** T-2161, T-2163: the map's upper left while the diagram is up: the way back to the map (T-2416), the
 * save button, the first layout's loading bar (T-2235, as the demand status's) and the busy tag. */
export function DiagramTopLeft() {
  const share = diagramProgress.value;
  return <div class="map-top-left">
    <div class="map-top-row">
      <button type="button" class="btn icon diagram-exit" data-tip={t("diagramButton.showMap")} aria-label={t("diagramButton.showMap")} onClick={() => (diagramShown.value = false)}><Icon.cross /></button>
      <button type="button" class={"btn icon diagram-water" + (diagramWater.value ? " on" : "")} data-tip={t("diagramButton.showWater")} aria-label={t("diagramButton.showWater")} aria-pressed={diagramWater.value} onClick={() => (diagramWater.value = !diagramWater.value)}><Icon.waves /></button>
      {diagramReady.value && <button type="button" class="btn icon diagram-save" data-tip={t("diagramButton.savePicture")} aria-label={t("diagramButton.savePicture")} onClick={saveDiagramPicture}><Icon.download /></button>}
      {share !== null && <div class="status-tag" role="status">
        <span class="demand-busy">{t("diagramButton.calculating")}</span>
        <LoadBar fraction={share} step={t("diagramButton.calculating")} thin tip={`${Math.floor(share * 100)}%`} />
      </div>}
      <BusyTag />
    </div>
  </div>;
}
