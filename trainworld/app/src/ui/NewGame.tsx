// T-728: the new game sheet, shared by the start menu and the saves panel. Normal or sandbox, then
// where to start: one of five suggested cities or the player's location (T-1132), any registry city
// or zone found by name, or the world map. In a sandbox, a city with a real network (T-731) can start with it.

import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { CITIES, cityById, zonesTouching, ZONE_REACH_M, type City } from "../game/cities";
import { zoneHolding } from "../game/realms";
import { cityMatch, normalise } from "../game/search";
import type { GameMode, NewGameSpec } from "../workers/saveFile";
import { dataUrl } from "../game/dataBase";
import { display } from "../game/ui";
import { setStartAt } from "./locate";
import { t, tCountry } from "../i18n";

/** The suggested starts (T-728): the cities the demand model was checked on, each with a real
 * network. A fixed list, so easy, medium and hard starts can take its place later. */
const START_CITIES = ["nyc", "lon", "par", "tyo", "tor"];
/** The cities with a real network a sandbox can start from (`packs/real/<id>.save`,
 * pipeline/realnet.py ship). */
const REAL_NETWORKS = new Set(["nyc", "lon", "par", "tyo", "seo", "tor", "sdj"]);
const MAX_HITS = 8;

function countryOf(c: City): string {
  const code = c.country;
  return code ? tCountry(code) : "";
}

/** Registry cities and zones with a word of their name or of an alias starting with the text,
 * ignoring case and accents ("hague" finds Rotterdam through The Hague, T-825): best match first
 * (search.ts cityMatch): an exact match first, then by name before by alias, then biggest. */
function findStart(text: string): { city: City; matched?: string }[] {
  const q = normalise(text);
  if (!q) return [];
  return CITIES.flatMap((city) => {
    const m = cityMatch(city, q);
    // an exact name first; any other word start ranks by size, so "hague" finds Rotterdam before Haguenau's Bas-Rhin
    return m && m.score <= 2 ? [{ city, matched: m.matched, score: m.score ? 1 : 0 }] : [];
  })
    .sort((a, b) => a.score - b.score || Number(!!a.matched) - Number(!!b.matched) || (b.city.localTrips ?? 0) - (a.city.localTrips ?? 0))
    .slice(0, MAX_HITS);
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

/** The real network's save for a start city (T-731). */
async function realNetwork(id: string): Promise<ArrayBuffer> {
  const r = await fetch(dataUrl(`real/${id}.save`));
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
  useEffect(() => first.current?.focus(), []);
  useEffect(() => () => { asked.current++; }, []);
  const hits = useMemo(() => (start && text === cityById(start)?.name ? [] : findStart(text)), [text, start]);
  const startId = located ? located.zone : start;
  const city = startId ? cityById(startId) : null;
  const realOffered = mode === "sandbox" && !!startId && REAL_NETWORKS.has(startId);
  const pick = (id: string | null) => {
    asked.current++;
    setHere("idle");
    setStart(id);
    setLocated(null);
    setText(id && !START_CITIES.includes(id) ? cityById(id).name : "");
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
        setError(t("newGame.downloadFailed", { city: city!.name }));
        return;
      }
    }
    setStartAt(located ? { center: located.at, zoom: city ? city.zoom : LOCATED_WORLD_ZOOM } : null);
    // T-1441: a new game opens with the Destinations bubbles on
    display.value = { ...display.peek(), commuters: true, commuterEnd: "work" };
    onStart(spec);
  };
  const modeBtn = (m: GameMode, label: string, tip: string | undefined, ref?: typeof first) => (
    <button ref={ref} class={"btn" + (mode === m ? " on" : "")} aria-pressed={mode === m} data-mode={m} data-tip={tip} onClick={() => setMode(m)}>{label}</button>
  );
  return (
    <div class="new-game">
      {heading && <h2 class="new-game-title">{t("newGame.title")}</h2>}
      <div class="group new-game-mode">
        {modeBtn("normal", t("newGame.normal"), undefined, first)}
        {modeBtn("sandbox", t("newGame.sandbox"), t("newGame.sandboxTip"))}
      </div>
      <div class="new-game-where">
        <span class="start-head">{t("newGame.startIn")}</span>
        <div class="new-game-cities">
          {START_CITIES.map((id) => (
            <button class={"btn text" + (start === id ? " on" : "")} aria-pressed={start === id} data-city={id} onClick={() => pick(id)}>{cityById(id).name}</button>
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
              <span>{c.name}{matched && <span class="muted small">{t("newGame.matched", { alias: matched })}</span>}</span><span class="muted small">{countryOf(c)}</span>
            </li>
          ))}
        </ul>}
        <span class="new-game-or muted small">{t("newGame.or")}</span>
        <button class={"btn text new-game-map" + (!startId && !located ? " on" : "")} aria-pressed={!startId && !located} onClick={() => pick(null)}>{t("newGame.chooseOnMap")}</button>
      </div>
      {realOffered && <label class="check new-game-real" data-tip={t("newGame.realTip", { city: city!.name })}>
        <input type="checkbox" checked={real} onChange={(e) => setReal((e.target as HTMLInputElement).checked)} />
        <span>{t("newGame.realNetwork")}</span>
      </label>}
      {error && <p class="callout bad">{error}</p>}
      <div class="pane-foot new-game-foot">
        <button class="btn text start-new new-game-start" disabled={busy} onClick={() => void go()}>{city ? t("newGame.startInCity", { city: city.name }) : t("newGame.start")}</button>
        <button class="btn text" onClick={onBack}>{t("newGame.back")}</button>
      </div>
    </div>
  );
}
