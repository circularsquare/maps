// T-728: the new game sheet, shared by the start menu and the saves panel. Normal or sandbox, then
// where to start: one of five suggested cities or the player's location (T-1132), any registry city
// or zone found by name, or the world map. In a sandbox, a city with a real network (T-731) can start with it.

import type { RefObject } from "preact";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "preact/hooks";
import { CITIES, cityById, isCity, zonesTouching, ZONE_REACH_M, type City } from "../game/cities";
import { zoneHolding } from "../game/realms";
import { cityMatch, matchScore, normalise } from "../game/search";
import type { GameMode, NewGameSpec } from "../workers/saveFile";
import { dataUrl } from "../game/dataBase";
import { display } from "../game/ui";
import { startTier } from "../game/startDifficulty";
import { newGameEnd, setStartAt } from "./locate";
import { t, tCountry } from "../i18n";
import { tPlace } from "../i18n/places";
import realNetworks from "../game/realNetworks.json";
import { patientFetch } from "../workers/patientFetch";

/** The suggested starts (T-728), easy starts (T-2690); Guangzhou for Toronto, for a start outside the West with
 * many cities near it (Anita, 2026-11-06). */
const START_CITIES = ["nyc", "lon", "par", "gua", "tyo"];
/** The cities with a real network a sandbox can start from (`packs/real/<id>.save`; the list is
 * written by pipeline/realauto.py ship). */
const REAL_NETWORKS = new Set<string>(realNetworks);
const MAX_HITS = 8;

function countryOf(c: City): string {
  const code = c.country;
  return code ? tCountry(code) : "";
}

/** A city's name in the current language (T-2681); zones keep theirs. */
const shown = (c: City) => tPlace(c.id, c.name);

/** Registry cities and zones with a word of their name or of an alias starting with the text,
 * ignoring case and accents ("hague" finds Rotterdam through The Hague, T-825): best match first
 * (search.ts cityMatch): an exact match first, then by name before by alias, then biggest. */
export function findStart(text: string, within?: ReadonlySet<string>): { city: City; matched?: string }[] {
  const q = normalise(text);
  if (!q) return [];
  return CITIES.filter((c) => !within || within.has(c.id)).flatMap((city) => {
    const own = shown(city);
    // a name in Japanese or Chinese is found as the Latin name is
    const loc = own !== city.name ? matchScore(normalise(own), q) : -1;
    const latin = cityMatch(city, q);
    const m = loc >= 0 && (!latin || loc <= latin.score) ? { score: loc, matched: undefined } : latin;
    // an exact name first; any other word start ranks by size, so "hague" finds Rotterdam before Haguenau's Bas-Rhin
    return m && m.score <= 2 ? [{ city, matched: m.matched, score: m.score ? 1 : 0 }] : [];
  })
    .sort((a, b) => a.score - b.score || Number(!!a.matched) - Number(!!b.matched) || (b.city.localTrips ?? 0) - (a.city.localTrips ?? 0))
    .slice(0, MAX_HITS);
}

/** T-2690: a search result's start tier, in light text after its name. */
function TierWord({ id }: { id: string }) {
  const tier = startTier(id);
  return <span class="new-game-tier muted small" style={{ marginLeft: "0.6em" }} data-tier={tier}>{t("newGame.tier", { tier })}</span>;
}

/** The world map centred on a location no zone is near: the continent round it in view. */
const LOCATED_WORLD_ZOOM = 4;
/** How long the location button waits for the browser before it says the location is unavailable. */
const LOCATE_WAIT_MS = 15_000;
/** Only where the browser can give a location (it asks the player itself). */
const CAN_LOCATE = typeof navigator !== "undefined" && "geolocation" in navigator && window.isSecureContext;

/** T-1132: the zone a start from (lon, lat) goes to: the one whose coarse outline holds it, else the
 * nearest by centre of those whose box comes within `ZONE_REACH_M` (the view's own reach, T-318);
 * null farther out, which starts on the map centred there. */
export function zoneForLocation(lon: number, lat: number): string | null {
  const held = zoneHolding(lon, lat);
  if (held) return held;
  const k = Math.cos((lat * Math.PI) / 180);
  let best: City | null = null, bd = Infinity;
  for (const c of zonesTouching(lon, lat, lon, lat, ZONE_REACH_M)) {
    const d = ((c.center[0] - lon) * k) ** 2 + (c.center[1] - lat) ** 2;
    if (d < bd) { bd = d; best = c; }
  }
  return best?.id ?? null;
}

/** CSS px kept below the sheet for the search's list: MAX_HITS rows and its border. */
const LIST_ROOM = MAX_HITS * 24 + 8;
/** the least gap kept between the pinned box and the window's top and bottom, CSS px */
const PIN_MARGIN = 16;

/** T-2748: the box holding the sheet keeps its top while the player types, so the search's list
 * lengthens it downward only (Anita, 2026-11-07). The top is where the box sat centred, raised if
 * need be so a full list still fits above the window's bottom; worked out again on a resize and when
 * the start menu's logo takes its size. The start menu centres its box (with the logo above it, or
 * not) and the saves panel is a centred dialog: the menu is given the offset of its first item as its
 * top padding and set to start there, the dialog a top margin. */
function usePinnedTop(sheet: RefObject<HTMLElement | null>) {
  useLayoutEffect(() => {
    const box = sheet.current?.closest<HTMLElement>(".start-box, dialog");
    if (!box) return;
    const menu = box.closest<HTMLElement>(".start-menu");
    const logo = menu?.querySelector<HTMLElement>(".start-logo") ?? null;
    const unpin = () => {
      if (menu) { menu.classList.remove("pinned"); menu.style.paddingTop = ""; }
      else { box.style.marginTop = ""; box.style.marginBottom = ""; }
    };
    const pin = () => {
      unpin();
      // CSS zoom scales the box and its margin as they draw
      const zoom = (box as HTMLElement & { currentCSSZoom?: number }).currentCSSZoom ?? 1;
      const lead = (logo ?? box).getBoundingClientRect().top, b = box.getBoundingClientRect();
      const top = Math.max(PIN_MARGIN, Math.min(lead, innerHeight - PIN_MARGIN - LIST_ROOM * zoom - (b.bottom - lead)));
      if (menu) {
        menu.style.paddingTop = `${top - menu.getBoundingClientRect().top}px`;
        menu.classList.add("pinned");
      } else {
        box.style.marginTop = `${top / zoom}px`;
        box.style.marginBottom = "auto";
      }
    };
    pin();
    addEventListener("resize", pin);
    // the logo is sized by its own effect, after this one
    const sized = logo ? new ResizeObserver(pin) : null;
    if (logo) sized!.observe(logo);
    return () => { removeEventListener("resize", pin); sized?.disconnect(); unpin(); };
  }, []);
}

/** The real network's save for a start city (T-731). */
async function realNetwork(id: string): Promise<ArrayBuffer> {
  const r = await patientFetch(dataUrl(`real/${id}.save`));
  if (!r.ok) throw new Error(`${r.status}`);
  return r.arrayBuffer();
}

/** The sheet. `onStart` gets the new game, with its real network downloaded when chosen. */
export function NewGameSheet({ onStart, onBack, heading = true }: { onStart: (spec: NewGameSpec) => void; onBack: () => void; heading?: boolean }) {
  const [mode, setMode] = useState<GameMode>("normal");
  // T-2123: New York is lit to begin with (the first suggestion); Choose on map is the world start
  const [start, setStart] = useState<string | null>(START_CITIES[0]);
  const [text, setText] = useState("");
  const [cursor, setCursor] = useState(0);
  const [real, setReal] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // T-1132: the start from the player's location, the browser asking for it; a refusal or no answer only marks the button
  const [here, setHere] = useState<"idle" | "asking" | "failed">("idle");
  const [located, setLocated] = useState<{ zone: string | null; at: [number, number] } | null>(null);
  const asked = useRef(0);
  const first = useRef<HTMLButtonElement>(null);
  const root = useRef<HTMLDivElement>(null);
  usePinnedTop(root);
  useEffect(() => first.current?.focus(), []);
  useEffect(() => () => { asked.current++; }, []);
  const hits = useMemo(() => (start && isCity(start) && text === shown(cityById(start)) ? [] : findStart(text)), [text, start]);
  const startId = located ? located.zone : start;
  const city = startId ? cityById(startId) : null;
  const realOffered = mode === "sandbox" && !!startId && REAL_NETWORKS.has(startId);
  // T-3225: the real networks by name, for the sandbox's list
  const realCities = useMemo(() => [...REAL_NETWORKS].filter((id) => isCity(id)).map((id) => cityById(id))
    .sort((a, b) => shown(a).localeCompare(shown(b))), []);
  const pick = (id: string | null) => {
    asked.current++;
    setHere("idle");
    setReal(false);
    setStart(id);
    setLocated(null);
    setText(id && !START_CITIES.includes(id) ? shown(cityById(id)) : "");
    setCursor(0);
    setError(null);
  };
  const locate = () => {
    if (located) return pick(null);
    const n = ++asked.current;
    setHere("asking");
    // an answer after the wait still counts, unless the player asked again or left the sheet
    const fail = () => { if (asked.current === n) setHere("failed"); };
    const timer = window.setTimeout(fail, LOCATE_WAIT_MS);
    navigator.geolocation.getCurrentPosition((p) => {
      window.clearTimeout(timer);
      if (asked.current !== n) return;
      const at: [number, number] = [p.coords.longitude, p.coords.latitude];
      pick(null);
      setLocated({ zone: zoneForLocation(at[0], at[1]), at });
      setHere("idle");
    }, () => { window.clearTimeout(timer); fail(); }, { timeout: LOCATE_WAIT_MS });
  };
  const go = async () => {
    if (busy) return;
    const spec: NewGameSpec = { mode, start: startId ?? undefined };
    if (realOffered && real) {
      setBusy(true);
      try {
        spec.real = await realNetwork(startId!);
      } catch {
        setBusy(false);
        setError(t("newGame.downloadFailed", { city: shown(city!) }));
        return;
      }
    }
    setStartAt(located ? { center: located.at, zoom: city ? city.zoom : LOCATED_WORLD_ZOOM } : null);
    // T-1441: a new game opens with the Destinations bubbles on; T-2685: Homes for the tutorial's
    display.value = { ...display.peek(), commuters: true, commuterEnd: newGameEnd(mode) };
    onStart(spec);
  };
  const modeBtn = (m: GameMode, label: string, tip: string | undefined, ref?: typeof first) => (
    <button ref={ref} class={"btn" + (mode === m ? " on" : "")} aria-pressed={mode === m} data-mode={m} data-tip={tip} onClick={() => setMode(m)}>{label}</button>
  );
  return (
    <div class="new-game" ref={root}>
      {heading && <h2 class="new-game-title">{t("newGame.title")}</h2>}
      <div class="group new-game-mode">
        {modeBtn("normal", t("newGame.normal"), undefined, first)}
        {modeBtn("sandbox", t("newGame.sandbox"), t("newGame.sandboxTip"))}
      </div>
      <div class="new-game-where">
        <span class="start-head">{t("newGame.startIn")}</span>
        <div class="new-game-cities">
          {START_CITIES.map((id) => (
            <button class={"btn text" + (start === id ? " on" : "")} aria-pressed={start === id} data-city={id} onClick={() => pick(id)}>{shown(cityById(id))}</button>
          ))}
          {CAN_LOCATE && <button class={"btn text new-game-here" + (located ? " on" : "") + (here === "failed" ? " muted" : "")} aria-pressed={!!located} data-state={located ? "located" : here} disabled={here === "asking"} onClick={locate}>
            {here === "asking" ? t("newGame.locating") : here === "failed" ? t("newGame.locationUnavailable") : t("newGame.currentLocation")}
          </button>}
        </div>
        <span class="new-game-or muted small">{t("newGame.or")}</span>
        <input type="text" class="new-game-find" aria-label={t("newGame.findCity")} placeholder={t("newGame.findCity")} value={text} maxLength={60}
          onInput={(e) => { setText((e.target as HTMLInputElement).value); setCursor(0); if (start && !START_CITIES.includes(start)) setStart(null); setLocated(null); }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown" && hits.length) { e.preventDefault(); setCursor((cursor + 1) % hits.length); }
            else if (e.key === "ArrowUp" && hits.length) { e.preventDefault(); setCursor((cursor + hits.length - 1) % hits.length); }
            else if (e.key === "Enter" && hits.length) { e.preventDefault(); pick(hits[cursor].city.id); }
            else if (e.key === "Escape" && text) { e.preventDefault(); e.stopPropagation(); pick(null); }
          }} />
        {hits.length > 0 && <ul class="new-game-hits" role="listbox">
          {hits.map(({ city: c, matched }, i) => (
            <li role="option" aria-selected={i === cursor} class={i === cursor ? "on" : ""} data-city={c.id} onMouseDown={(e) => { e.preventDefault(); pick(c.id); }}>
              <span>{shown(c)}{matched && <span class="muted small">{t("newGame.matched", { alias: matched })}</span>}<TierWord id={c.id} /></span><span class="muted small">{countryOf(c)}</span>
            </li>
          ))}
        </ul>}
        <span class="new-game-or muted small">{t("newGame.or")}</span>
        <button class={"btn text new-game-map" + (!startId && !located ? " on" : "")} aria-pressed={!startId && !located} onClick={() => pick(null)}>{t("newGame.chooseOnMap")}</button>
      </div>
      {/* T-3225: one list of every city with a real network; picking one makes it the start */}
      {mode === "sandbox" && realCities.length > 0 && <div class="new-game-real" data-tip={realOffered && real ? t("newGame.realTip", { city: shown(city!) }) : undefined}>
        <span>{t("newGame.realNetwork")}</span>
        <label class={"purpose-filter" + (realOffered && real ? " on" : "")}>
          <select aria-label={t("newGame.realNetwork")} value={realOffered && real ? startId! : ""} onChange={(e) => {
            const id = (e.target as HTMLSelectElement).value;
            if (!id) { setReal(false); return; }
            pick(id);
            setReal(true);
          }}>
            <option value="" />
            {realCities.map((c) => <option value={c.id}>{shown(c)}</option>)}
          </select>
        </label>
      </div>}
      {error && <p class="callout bad">{error}</p>}
      <div class="pane-foot new-game-foot">
        <button class="btn text start-new new-game-start" disabled={busy} onClick={() => void go()}>{city ? t("newGame.startInCity", { city: shown(city) }) : t("newGame.start")}</button>
        <button class="btn text" onClick={onBack}>{t("newGame.back")}</button>
      </div>
    </div>
  );
}
