// The dock: tabs, a tab area of fixed height that scrolls, and the inspector below it. The tab
// area never changes height when the tab changes, so the inspector's top edge stays put; the
// player can drag the dock's right edge to make it wider or narrower and the line between the tab
// area and the inspector up or down (T-085), both remembered per browser (game/ui.ts). On a phone
// held upright the dock is under the map, the inspector always beside the tab area (T-1885), and
// the strip above the tabs drags its height (T-1884).

import type { JSX } from "preact";
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { dockLayout, tab, uiScale, type Tab } from "../game/ui";
import { t, type MessageKey } from "../i18n";
import { live } from "../map/live";
import { Inspector } from "./inspector/Inspector";
import { BuildTab } from "./BuildTab";
import { CitiesList } from "./Cities";
import { AchievementsButton } from "./Achievements";
import { LinesTab } from "./LinesTab";
import { MoneyTab } from "./MoneyTab";
import { StationsTab } from "./StationsTab";
import { SavesPanel, savesOpen } from "./SavesPanel";
import { SettingsTab } from "./SettingsTab";
import { HelpPanel } from "./HelpPanel";
import { MareyPanel } from "./inspector/MareyChart";
import { startAccountUi } from "./AccountPanel";
import { Icon } from "./widgets";
import { tabKeyName } from "./toolKeys";

/** T-252: the tabs are icons, their names in the hover */
const TABS: [Tab, MessageKey, () => JSX.Element][] = [
  ["build", "dock.build", Icon.wrench],
  ["lines", "dock.lines", Icon.route],
  ["stations", "dock.stations", Icon.building],
  ["money", "dock.finances", Icon.money],
  ["cities", "dock.cities", Icon.cities],
];

const PANES: Record<Tab, () => JSX.Element | null> = {
  build: BuildTab,
  lines: LinesTab,
  stations: StationsTab,
  money: MoneyTab,
  cities: () => <><AchievementsButton /><CitiesList empty={<p class="muted">{t("dock.noStations")}</p>} /></>,
  settings: SettingsTab,
};

/** Dock width limits, px: the tab row must fit; the map keeps most of the window. */
const DOCK_MIN = 280, DOCK_MAX = 560;
/** The tab area and the inspector each keep at least this much, px. */
const PANE_MIN = 110, INSPECTOR_MIN = 160, TABS_H = 36;

// The dock is zoomed by the UI scale (T-247): its sizes are in its own px, the window's in screen px.
const dockWidth = (w: number) => Math.round(Math.max(DOCK_MIN, Math.min(DOCK_MAX, w, innerWidth * 0.45 / uiScale.peek())));
const paneHeight = (h: number) => Math.round(Math.max(PANE_MIN, Math.min(h, innerHeight / uiScale.peek() - TABS_H - INSPECTOR_MIN)));
/** the tab area's share of a phone's width beside the inspector (T-1259) */
const paneShare = (f: number) => Math.max(0.3, Math.min(0.7, f));
/** a phone's dock height as a share of the window: the map under the top bar keeps about 30%, the
 *  dock its tab row and a couple of rows of the tab (T-1884) */
const phoneShare = (f: number) => Math.max((TABS_H + 84) * uiScale.peek() / innerHeight, Math.min(0.58, f));
/** a phone held upright: the dock under the map (T-951), the same query as style.css */
const PHONE = matchMedia("(max-width: 600px) and (min-height: 560px)");

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
  const cur = tab.value;
  const Pane = PANES[cur];
  const lay = dockLayout.value;
  const ref = useRef<HTMLElement>(null);
  const [phone, setPhone] = useState(PHONE.matches);
  useEffect(() => {
    const f = () => setPhone(PHONE.matches);
    PHONE.addEventListener("change", f);
    return () => PHONE.removeEventListener("change", f);
  }, []);
  useEffect(startAccountUi, []);
  // The map's canvas follows the dock's width.
  const scale = uiScale.value;
  useEffect(() => {
    live.map?.resize();
  }, [lay.dockW, lay.phoneH, scale]);
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
    scroller.scrollTop = scrolls.current[cur] ?? 0;
    restoring.current = false;
  }, [cur]);
  const style: Record<string, string> = {};
  if (lay.dockW !== null) style["--dock-w"] = dockWidth(lay.dockW) + "px";
  if (lay.paneH !== null) style["--pane-h"] = paneHeight(lay.paneH) + "px";
  const dragWidth = (e: PointerEvent) => {
    const left = ref.current!.getBoundingClientRect().left;
    startDrag(e, (x) => (dockLayout.value = { ...dockLayout.value, dockW: dockWidth((x - left) / uiScale.peek()) }));
  };
  if (lay.paneW) style["--pane-w"] = Math.round(paneShare(lay.paneW) * 1000) / 10 + "%";
  const dragDivider = (e: PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    if (phone) startDrag(e, (x) => (dockLayout.value = { ...dockLayout.value, paneW: paneShare((x - r.left) / r.width) }));
    else {
      const top = ref.current!.querySelector(".pane")!.getBoundingClientRect().top;
      startDrag(e, (_, y) => (dockLayout.value = { ...dockLayout.value, paneH: paneHeight((y - top) / uiScale.peek()) }));
    }
  };
  if (lay.phoneH) style["--phone-h"] = Math.round(phoneShare(lay.phoneH) * 1000) / 10 + "dvh";
  const dragHeight = (e: PointerEvent) => {
    const off = ref.current!.getBoundingClientRect().top - e.clientY;
    startDrag(e, (_, y) => (dockLayout.value = { ...dockLayout.value, phoneH: phoneShare((innerHeight - y - off) / innerHeight) }));
  };
  return (
    <aside id="dock" ref={ref} style={style}>
      <div class="dock-grab" onPointerDown={dragHeight} />
      <nav class="tabs">
        {TABS.map(([id, key, Glyph]) => (
          <button class={"tab" + (cur === id ? " on" : "")} data-guide-tab={id} data-tip={t("dock.tabTip", { name: t(key), key: tabKeyName(id) })} aria-label={t(key)} aria-pressed={cur === id} onClick={() => (tab.value = id)}>
            <Glyph />
          </button>
        ))}
        {/* T-251: the saved games open in their own panel, not as a tab */}
        <button class="tab gear save" data-tip={t("dock.saves")} aria-label={t("dock.saves")} onClick={() => (savesOpen.value = true)}>
          <Icon.save />
        </button>
        <button class={"tab gear" + (cur === "settings" ? " on" : "")} data-tip={t("dock.settings")} aria-label={t("dock.settings")} aria-pressed={cur === "settings"} onClick={() => (tab.value = "settings")}>
          <Icon.gear />
        </button>
      </nav>
      <SavesPanel />
      <HelpPanel />
      <MareyPanel />
      <section class={"pane" + (cur === "build" ? " build-pane" : "")} ref={paneRef}
        onScrollCapture={(e) => { if (!restoring.current) scrolls.current[cur] = (e.target as HTMLElement).scrollTop; }}>
        <Pane />
      </section>
      <div class="pane-resize"onPointerDown={dragDivider} />
      <Inspector />
      <div class="dock-resize"onPointerDown={dragWidth} />
    </aside>
  );
}
