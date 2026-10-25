// T-728: the new game sheet, shared by the start menu and the saves panel. Normal or sandbox, then
// where to start: one of five suggested cities or the player's location (T-1132), any registry city
// or zone found by name, or the world map. In a sandbox, a city with a real network (T-731) can start with it.

import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { CITIES, cityById, zonesTouching, ZONE_REACH_M, type City } from "../game/cities";
import { zoneHolding } from "../game/realms";
import { cityMatch, normalise } from "../game/search";
import type { GameMode, NewGameSpec } from "../workers/saveFile";
import { dataUrl } from "../game/dataBase";
import { setStartAt } from "./locate";

/** The suggested starts (T-728): the cities the demand model was checked on, each with a real
 * network. A fixed list, so easy, medium and hard starts can take its place later. */
const START_CITIES = ["nyc", "lon", "par", "tyo", "tor"];
/** The cities with a real network a sandbox can start from (`packs/real/<id>.save`,
 * pipeline/realnet.py ship). */
const REAL_NETWORKS = new Set(["nyc", "lon", "par", "tyo", "seo", "tor", "sdj"]);
const MAX_HITS = 8;

let regionNames: Intl.DisplayNames | null = null;
function countryOf(c: City): string {
  const code = c.country;
  if (!code) return "";
  try {
    regionNames ??= new Intl.DisplayNames(["en"], { type: "region" });
    return regionNames.of(code) ?? code;
  } catch {
    return code;
  }
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
  const [start, setStart] = useState<string | null>(null);
  const [text, setText] = useState("");
  const [cursor, setCursor] = useState(0);
  const [real, setReal] = useState(true);
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
        setError(`${city!.name}'s real network could not be downloaded.`);
        return;
      }
    }
    setStartAt(located ? { center: located.at, zoom: city ? city.zoom : LOCATED_WORLD_ZOOM } : null);
    onStart(spec);
  };
  const modeBtn = (m: GameMode, label: string, tip: string, ref?: typeof first) => (
    <button ref={ref} class={"btn" + (mode === m ? " on" : "")} aria-pressed={mode === m} data-mode={m} data-tip={tip} onClick={() => setMode(m)}>{label}</button>
  );
  return (
    <div class="new-game">
      {heading && <h2 class="new-game-title">New game</h2>}
      <div class="group new-game-mode">
        {modeBtn("normal", "Normal", "Construction costs money, and an edit is refused when the cash runs short", first)}
        {modeBtn("sandbox", "Sandbox", "Infinite money, and the cash is hidden. Fares and running costs are still counted.")}
      </div>
      <div class="new-game-where">
        <span class="start-head">Start in</span>
        <div class="new-game-cities">
          {START_CITIES.map((id) => (
            <button class={"btn text" + (start === id ? " on" : "")} aria-pressed={start === id} data-city={id} onClick={() => pick(start === id ? null : id)}>{cityById(id).name}</button>
          ))}
          {CAN_LOCATE && <button class={"btn text new-game-here" + (located ? " on" : "") + (here === "failed" ? " muted" : "")} aria-pressed={!!located} data-state={located ? "located" : here} disabled={here === "asking"} onClick={locate}>
            {here === "asking" ? "Locating" : here === "failed" ? "Location unavailable" : "Current location"}
          </button>}
        </div>
        <span class="new-game-or muted small">or</span>
        <input type="text" class="new-game-find" aria-label="Find a city" placeholder="Find a city" value={text} maxLength={60}
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
              <span>{c.name}{matched && <span class="muted small">, matched {matched}</span>}</span><span class="muted small">{countryOf(c)}</span>
            </li>
          ))}
        </ul>}
        <span class="new-game-or muted small">or</span>
        <button class={"btn text new-game-map" + (!startId && !located ? " on" : "")} aria-pressed={!startId && !located} onClick={() => pick(null)}>Choose on map</button>
      </div>
      {realOffered && <label class="check new-game-real" data-tip={`Start with ${city!.name}'s rail lines as they run today, built and running`}>
        <input type="checkbox" checked={real} onChange={(e) => setReal((e.target as HTMLInputElement).checked)} />
        <span>Real network</span>
      </label>}
      {error && <p class="callout bad">{error}</p>}
      <div class="pane-foot new-game-foot">
        <button class="btn text start-new new-game-start" disabled={busy} onClick={() => void go()}>{city ? `Start in ${city.name}` : "Start"}</button>
        <button class="btn text" onClick={onBack}>Back</button>
      </div>
    </div>
  );
}
