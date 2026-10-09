// The dock: tabs, a tab area of fixed height that scrolls, and the inspector below it. The tab
// area never changes height when the tab changes, so the inspector's top edge stays put; the
// player can drag the dock's right edge to make it wider or narrower and the line between the tab
// area and the inspector up or down (T-085), both remembered per browser (game/ui.ts).

import type { JSX } from "preact";
import { useEffect, useRef } from "preact/hooks";
import { dockLayout, tab, type Tab } from "../game/ui";
import { live } from "../map/live";
import { Inspector } from "./inspector/Inspector";
import { BuildTab, CityTab, LinesTab, MoneyTab, StationsTab } from "./tabs";
import { SettingsTab } from "./SettingsTab";
import { Icon } from "./widgets";

const TABS: [Tab, string][] = [
  ["build", "Build"],
  ["lines", "Lines"],
  ["stations", "Stations"],
  ["money", "Money"],
  ["city", "City"],
];

const PANES: Record<Tab, () => JSX.Element | null> = {
  build: BuildTab,
  lines: LinesTab,
  stations: StationsTab,
  money: MoneyTab,
  city: CityTab,
  settings: SettingsTab,
};

/** Dock width limits, px: the tab row must fit; the map keeps most of the window. */
const DOCK_MIN = 280, DOCK_MAX = 560;
/** The tab area and the inspector each keep at least this much, px. */
const PANE_MIN = 110, INSPECTOR_MIN = 160, TABS_H = 36;

const dockWidth = (w: number) => Math.round(Math.max(DOCK_MIN, Math.min(DOCK_MAX, w, innerWidth * 0.45)));
const paneHeight = (h: number) => Math.round(Math.max(PANE_MIN, Math.min(h, innerHeight - TABS_H - INSPECTOR_MIN)));

/** Drag with the pointer captured on the handle; `f` gets the pointer position. */
function startDrag(e: PointerEvent, f: (x: number, y: number) => void) {
  e.preventDefault();
  const el = e.currentTarget as HTMLElement;
  el.setPointerCapture(e.pointerId);
  el.classList.add("on");
  document.body.classList.add("resizing");
  const move = (m: PointerEvent) => f(m.clientX, m.clientY);
  const up = () => {
    el.removeEventListener("pointermove", move);
    el.removeEventListener("pointerup", up);
    el.removeEventListener("pointercancel", up);
    el.classList.remove("on");
    document.body.classList.remove("resizing");
  };
  el.addEventListener("pointermove", move);
  el.addEventListener("pointerup", up);
  el.addEventListener("pointercancel", up);
}

export function Dock() {
  const t = tab.value;
  const Pane = PANES[t];
  const lay = dockLayout.value;
  const ref = useRef<HTMLElement>(null);
  // The map's canvas follows the dock's width.
  useEffect(() => {
    live.map?.resize();
  }, [lay.dockW]);
  const style: Record<string, string> = {};
  if (lay.dockW !== null) style["--dock-w"] = dockWidth(lay.dockW) + "px";
  if (lay.paneH !== null) style["--pane-h"] = paneHeight(lay.paneH) + "px";
  const dragWidth = (e: PointerEvent) => {
    const left = ref.current!.getBoundingClientRect().left;
    startDrag(e, (x) => (dockLayout.value = { ...dockLayout.value, dockW: dockWidth(x - left) }));
  };
  const dragDivider = (e: PointerEvent) => {
    const top = ref.current!.querySelector(".pane")!.getBoundingClientRect().top;
    startDrag(e, (_, y) => (dockLayout.value = { ...dockLayout.value, paneH: paneHeight(y - top) }));
  };
  return (
    <aside id="dock" ref={ref} style={style}>
      <nav class="tabs">
        {TABS.map(([id, label]) => (
          <button class={"tab" + (t === id ? " on" : "")} onClick={() => (tab.value = id)}>
            {label}
          </button>
        ))}
        <button class={"tab gear" + (t === "settings" ? " on" : "")} title="Settings" aria-label="Settings" onClick={() => (tab.value = "settings")}>
          <Icon.gear />
        </button>
      </nav>
      <section class="pane">
        <Pane />
      </section>
      <div class="pane-resize" title="Drag to resize" onPointerDown={dragDivider} />
      <Inspector />
      <div class="dock-resize" title="Drag to resize" onPointerDown={dragWidth} />
    </aside>
  );
}
