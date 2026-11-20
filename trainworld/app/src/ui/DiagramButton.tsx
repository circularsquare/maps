import type { JSX } from "preact";
import { diagramModes, lineKinds, diagramProgress, diagramReady, diagramShown, diagramWater, diagramWidth, saveDiagramPicture, type DiagramMode } from "../diagram/view";
import { WIDTH_MAX, WIDTH_MIN } from "../diagram/draw";
import { t, type MessageKey } from "../i18n";
import { world } from "../game/world";
import { BusyTag } from "./BusyTag";
import { ShareButton } from "./Community";
import { LoadBar } from "./LoadingScreen";
import { Check, Icon } from "./widgets";
import { useEffect, useRef, useState } from "preact/hooks";
import { diagramCities } from "../game/diagramCities";
import { busStopsById } from "../game/world";
import { stationZone } from "../game/listScope";
import { zonesLoaded } from "../game/zones";
import { cityById, isCity } from "../game/cities";
import { tPlace } from "../i18n/places";

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

const MODES: { mode: DiagramMode; tip: MessageKey; icon: () => JSX.Element }[] = [
  { mode: "train", tip: "diagramButton.showTrains", icon: Icon.train },
  { mode: "tram", tip: "diagramButton.showTrams", icon: Icon.tram },
  { mode: "bus", tip: "diagramButton.showBuses", icon: Icon.bus },
  { mode: "ferry", tip: "diagramButton.showFerries", icon: Icon.ferry },
];

/** T-3360: a toggle per kind of line the network has, when it has more than one kind */
function ModeButtons() {
  const has = lineKinds(world.value);
  if (has.size < 2) return null;
  const on = diagramModes.value;
  return <>{MODES.filter((m) => has.has(m.mode)).map(({ mode, tip, icon: I }) =>
    <button key={mode} type="button" class={"btn icon diagram-mode" + (on[mode] ? " on" : "")} data-tip={t(tip)} aria-label={t(tip)} aria-pressed={on[mode]}
      onClick={() => (diagramModes.value = { ...on, [mode]: !on[mode] })}><I /></button>)}</>;
}

/** T-3409: the zones with a built station or a stop, by name; a zone chosen before stays listed so it can be unticked */
function listedCities(): string[] {
  const w = world.value, out = new Set<string>(diagramCities.value);
  void zonesLoaded.value;
  if (w) {
    for (const s of w.save.stations) { if (!s.built) continue; const z = stationZone(s); if (z && isCity(z)) out.add(z); }
    for (const s of busStopsById(w).values()) { const z = stationZone(s); if (z && isCity(z)) out.add(z); }
  }
  const name = (z: string) => (isCity(z) ? tPlace(z, cityById(z).name) : z);
  return [...out].sort((a, b) => name(a).localeCompare(name(b)));
}

/** T-3409: the cities the diagram is of: a list under its button, every city first (the whole network) */
function CityChoice() {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const away = (e: PointerEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    // Esc closes the list alone, not the diagram
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") (e.stopPropagation(), setOpen(false)); };
    document.addEventListener("pointerdown", away, true);
    document.addEventListener("keydown", esc, true);
    return () => { document.removeEventListener("pointerdown", away, true); document.removeEventListener("keydown", esc, true); };
  }, [open]);
  const cities = listedCities(), chosen = diagramCities.value;
  if (cities.length < 2 && !chosen.length) return null;
  const set = (next: string[]) => (diagramCities.value = [...next].sort());
  return <div class="diagram-cities" ref={box}>
    <button type="button" class={"btn icon" + (chosen.length ? " on" : "")} data-tip={t("diagramButton.chooseCities")} aria-label={t("diagramButton.chooseCities")} aria-expanded={open}
      onClick={() => setOpen(!open)}><Icon.cities /></button>
    {open && <ul class="diagram-cities-list">
      <li><Check checked={!chosen.length} onChange={() => set([])}>{t("cities.all")}</Check></li>
      {cities.map((c) => <li key={c}><Check checked={chosen.includes(c)} onChange={(on) => set(on ? [...chosen, c] : chosen.filter((x) => x !== c))}>{isCity(c) ? tPlace(c, cityById(c).name) : c}</Check></li>)}
    </ul>}
  </div>;
}

/** T-2161, T-2163: the map's upper left while the diagram is up: the way back to the map (T-2416), the
 * save button, the first layout's loading bar (T-2235, as the demand status's) and the busy tag. */
export function DiagramTopLeft() {
  const share = diagramProgress.value;
  return <div class="map-top-left">
    <div class="map-top-row">
      <button type="button" class="btn icon diagram-exit" data-tip={t("diagramButton.showMap")} aria-label={t("diagramButton.showMap")} onClick={() => (diagramShown.value = false)}><Icon.cross /></button>
      <ModeButtons />
      <CityChoice />
      <button type="button" class={"btn icon diagram-water" + (diagramWater.value ? " on" : "")} data-tip={t("diagramButton.showWater")} aria-label={t("diagramButton.showWater")} aria-pressed={diagramWater.value} onClick={() => (diagramWater.value = !diagramWater.value)}><Icon.waves /></button>
      <label class="slider diagram-width" data-tip={t("diagramButton.lineWidth")}>
        <input type="range" min={WIDTH_MIN} max={WIDTH_MAX} step={0.01} value={diagramWidth.value} aria-label={t("diagramButton.lineWidth")}
          onInput={(e) => (diagramWidth.value = Number((e.target as HTMLInputElement).value))} />
      </label>
      {diagramReady.value && <button type="button" class="btn icon diagram-save" data-tip={t("diagramButton.savePicture")} aria-label={t("diagramButton.savePicture")} onClick={saveDiagramPicture}><Icon.download /></button>}
      {diagramReady.value && <ShareButton />}
      {share !== null && <div class="status-tag" role="status">
        <span class="demand-busy">{t("diagramButton.calculating")}</span>
        <LoadBar fraction={share} step={t("diagramButton.calculating")} thin tip={`${Math.floor(share * 100)}%`} />
      </div>}
      <BusyTag />
    </div>
  </div>;
}
