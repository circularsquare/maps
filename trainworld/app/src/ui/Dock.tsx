// The dock: tabs, a tab area of fixed height that scrolls, and the inspector below it. The tab
// area never changes height when the tab changes, so the inspector's top edge stays put; the
// player can drag the dock's right edge to make it wider or narrower and the line between the tab
// area and the inspector up or down (T-085), both remembered per browser (game/ui.ts).

import type { JSX } from "preact";
import { useEffect, useLayoutEffect, useRef } from "preact/hooks";
import { dockLayout, tab, uiScale, type Tab } from "../game/ui";
import { live } from "../map/live";
import { Inspector } from "./inspector/Inspector";
import { BuildTab, CitiesTab, LinesTab, MoneyTab, StationsTab } from "./tabs";
import { SavesPanel, savesOpen } from "./SavesPanel";
import { SettingsTab } from "./SettingsTab";
import { HelpPanel, helpOpen } from "./HelpPanel";
import { AccountButton, startAccountUi } from "./AccountPanel";
import { Icon } from "./widgets";

/** T-252: the tabs are icons, their names in the hover */
const TABS: [Tab, string, () => JSX.Element][] = [
  ["build", "Build", Icon.wrench],
  ["lines", "Lines", Icon.route],
  ["stations", "Stations", Icon.building],
  ["money", "Money", Icon.money],
  ["cities", "Cities", Icon.cities],
];

const PANES: Record<Tab, () => JSX.Element | null> = {
  build: BuildTab,
  lines: LinesTab,
  stations: StationsTab,
  money: MoneyTab,
  cities: CitiesTab,
  settings: SettingsTab,
};

/** Dock width limits, px: the tab row must fit; the map keeps most of the window. */
const DOCK_MIN = 280, DOCK_MAX = 560;
/** The tab area and the inspector each keep at least this much, px. */
const PANE_MIN = 110, INSPECTOR_MIN = 160, TABS_H = 36;

// The dock is zoomed by the UI scale (T-247): its sizes are in its own px, the window's in screen px.
const dockWidth = (w: number) => Math.round(Math.max(DOCK_MIN, Math.min(DOCK_MAX, w, innerWidth * 0.45 / uiScale.peek())));
const paneHeight = (h: number) => Math.round(Math.max(PANE_MIN, Math.min(h, innerHeight / uiScale.peek() - TABS_H - INSPECTOR_MIN)));

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
  useEffect(startAccountUi, []);
  // The map's canvas follows the dock's width.
  const scale = uiScale.value;
  useEffect(() => {
    live.map?.resize();
  }, [lay.dockW, scale]);
  // T-215: each tab keeps its own scroll position (this session), so a tab opens where it was
  // left, at its top the first time, never at another tab's scroll. The Build tab scrolls its
  // options list, the others the whole tab area.
  const paneRef = useRef<HTMLElement>(null);
  const scrolls = useRef<Partial<Record<Tab, number>>>({});
  const restoring = useRef(false);
  useLayoutEffect(() => {
    const pane = paneRef.current;
    if (!pane) return;
    const scroller = (pane.querySelector(".build-options") as HTMLElement | null) ?? pane;
    restoring.current = true;
    pane.scrollTop = 0;
    scroller.scrollTop = scrolls.current[t] ?? 0;
    restoring.current = false;
  }, [t]);
  const style: Record<string, string> = {};
  if (lay.dockW !== null) style["--dock-w"] = dockWidth(lay.dockW) + "px";
  if (lay.paneH !== null) style["--pane-h"] = paneHeight(lay.paneH) + "px";
  const dragWidth = (e: PointerEvent) => {
    const left = ref.current!.getBoundingClientRect().left;
    startDrag(e, (x) => (dockLayout.value = { ...dockLayout.value, dockW: dockWidth((x - left) / uiScale.peek()) }));
  };
  const dragDivider = (e: PointerEvent) => {
    const top = ref.current!.querySelector(".pane")!.getBoundingClientRect().top;
    startDrag(e, (_, y) => (dockLayout.value = { ...dockLayout.value, paneH: paneHeight((y - top) / uiScale.peek()) }));
  };
  return (
    <aside id="dock" ref={ref} style={style}>
      <nav class="tabs">
        {TABS.map(([id, label, Glyph]) => (
          <button class={"tab" + (t === id ? " on" : "")} data-tip={label} aria-label={label} aria-pressed={t === id} onClick={() => (tab.value = id)}>
            <Glyph />
          </button>
        ))}
        {/* T-251: the saved games open in their own panel, not as a tab */}
        <button class="tab gear save" data-tip="Saved games" aria-label="Saved games" onClick={() => (savesOpen.value = true)}>
          <Icon.save />
        </button>
        <AccountButton kind="tab" />
        <button class={"tab gear help" + (helpOpen.value ? " on" : "")} data-tip="Controls" aria-label="Controls" aria-pressed={helpOpen.value} onClick={() => (helpOpen.value = !helpOpen.peek())}>
          <Icon.help />
        </button>
        <button class={"tab gear" + (t === "settings" ? " on" : "")} data-tip="Settings" aria-label="Settings" aria-pressed={t === "settings"} onClick={() => (tab.value = "settings")}>
          <Icon.gear />
        </button>
      </nav>
      <SavesPanel />
      <HelpPanel />
      <section class={"pane" + (t === "build" ? " build-pane" : "")} ref={paneRef}
        onScrollCapture={(e) => { if (!restoring.current) scrolls.current[t] = (e.target as HTMLElement).scrollTop; }}>
        <Pane />
      </section>
      <div class="pane-resize"onPointerDown={dragDivider} />
      <Inspector />
      <div class="dock-resize"onPointerDown={dragWidth} />
    </aside>
  );
}
