// T-728: the new game sheet, shared by the start menu and the saves panel. Normal or sandbox, then
// where to start: one of five suggested cities, any registry city or zone found by name, or none,
// which starts on the world map. In a sandbox, a city with a real network (T-731) can start with it.

import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { activeCity, CITIES, cityById, viewedZone, type City } from "../game/cities";
import { live } from "../map/live";
import { display } from "../game/ui";
import type { GameMode, NewGameSpec } from "../workers/saveFile";

/** The suggested starts (T-728): the cities the demand model was checked on, each with a real
 * network. A fixed list, so easy, medium and hard starts can take its place later. */
export const START_CITIES = ["nyc", "lon", "par", "tyo", "tor"];
/** The cities with a real network a sandbox can start from (`packs/real/<id>.save`,
 * pipeline/realnet.py ship). */
export const REAL_NETWORKS = new Set(["nyc", "lon", "par", "tyo", "seo", "tor", "sdj"]);
/** A start on the world map: the Americas, Europe and East Asia in view. */
const WORLD_VIEW = { center: [20, 30] as [number, number], zoom: 1.8 };
const MAX_HITS = 8;

const fold = (s: string) => s.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "");
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

/** Registry cities and zones whose name has a word starting with the text, ignoring case and
 * accents ("hague" finds "Rotterdam [The Hague]"), biggest first. */
export function findStart(text: string): City[] {
  const q = fold(text.trim());
  if (!q) return [];
  return CITIES.filter((c) => fold(c.name).split(/[^a-z0-9]+/).some((_, i, words) => words.slice(i).join(" ").startsWith(q)))
    .sort((a, b) => (b.localTrips ?? 0) - (a.localTrips ?? 0))
    .slice(0, MAX_HITS);
}

/** Move the camera to a new game's start: the city's registry view, or the world map. */
export function viewStart(start?: string) {
  const c = start ? CITIES.find((x) => x.id === start) : undefined;
  if (c && viewedZone.peek() === c.id && live.map && live.map.getZoom() > 9) return; // already there
  if (c) {
    activeCity.value = c.id;
    viewedZone.value = c.id;
  }
  live.map?.jumpTo(c ? { center: c.center, zoom: c.zoom } : WORLD_VIEW);
  // T-801: a start on the map is picked from the zones' zoomed-out bubbles, so Homes goes on
  if (!c) display.value = { ...display.peek(), commuters: true, commuterEnd: "home" };
}

/** The real network's save for a start city (T-731). */
export async function realNetwork(id: string): Promise<ArrayBuffer> {
  const r = await fetch(new URL(`packs/real/${id}.save`, document.baseURI));
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
  const first = useRef<HTMLButtonElement>(null);
  useEffect(() => first.current?.focus(), []);
  const hits = useMemo(() => (start && text === cityById(start)?.name ? [] : findStart(text)), [text, start]);
  const city = start ? cityById(start) : null;
  const realOffered = mode === "sandbox" && !!start && REAL_NETWORKS.has(start);
  const pick = (id: string | null) => {
    setStart(id);
    setText(id && !START_CITIES.includes(id) ? cityById(id).name : "");
    setCursor(0);
    setError(null);
  };
  const go = async () => {
    if (busy) return;
    const spec: NewGameSpec = { mode, start: start ?? undefined };
    if (realOffered && real) {
      setBusy(true);
      try {
        spec.real = await realNetwork(start!);
      } catch {
        setBusy(false);
        setError(`${city!.name}'s real network could not be downloaded.`);
        return;
      }
    }
    onStart(spec);
  };
  const modeBtn = (m: GameMode, label: string, tip: string, ref?: typeof first) => (
    <button ref={ref} class={"btn" + (mode === m ? " on" : "")} aria-pressed={mode === m} data-mode={m} data-tip={tip} onClick={() => setMode(m)}>{label}</button>
  );
  return (
    <div class="new-game">
      {heading && <h2 class="new-game-title">New game</h2>}
      <div class="group new-game-mode">
        {modeBtn("normal", "Normal", "Building costs money, and an edit is refused when the cash runs short", first)}
        {modeBtn("sandbox", "Sandbox", "Nothing is refused for money and the cash is hidden. Fares and running costs are still counted.")}
      </div>
      <div class="new-game-where">
        <span class="start-head">Start in</span>
        <div class="new-game-cities">
          {START_CITIES.map((id) => (
            <button class={"btn text" + (start === id ? " on" : "")} aria-pressed={start === id} data-city={id} onClick={() => pick(start === id ? null : id)}>{cityById(id).name}</button>
          ))}
        </div>
        <input type="text" class="new-game-find" aria-label="Find a city or region" placeholder="Find a city or region" value={text} maxLength={60}
          onInput={(e) => { setText((e.target as HTMLInputElement).value); setCursor(0); if (start && !START_CITIES.includes(start)) setStart(null); }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown" && hits.length) { e.preventDefault(); setCursor((cursor + 1) % hits.length); }
            else if (e.key === "ArrowUp" && hits.length) { e.preventDefault(); setCursor((cursor + hits.length - 1) % hits.length); }
            else if (e.key === "Enter" && hits.length) { e.preventDefault(); pick(hits[cursor].id); }
            else if (e.key === "Escape" && text) { e.preventDefault(); e.stopPropagation(); pick(null); }
          }} />
        {hits.length > 0 && <ul class="new-game-hits" role="listbox">
          {hits.map((c, i) => (
            <li role="option" aria-selected={i === cursor} class={i === cursor ? "on" : ""} data-city={c.id} onMouseDown={(e) => { e.preventDefault(); pick(c.id); }}>
              <span>{c.name}</span><span class="muted small">{countryOf(c)}</span>
            </li>
          ))}
        </ul>}
      </div>
      {realOffered && <label class="check new-game-real" data-tip={`Start with ${city!.name}'s rail lines as they run today, built and running`}>
        <input type="checkbox" checked={real} onChange={(e) => setReal((e.target as HTMLInputElement).checked)} />
        <span>Real network</span>
      </label>}
      {error && <p class="callout bad">{error}</p>}
      <div class="pane-foot new-game-foot">
        <button class="btn text start-new" disabled={busy} onClick={() => void go()}>{city ? `Start in ${city.name}` : "Start on the map"}</button>
        <button class="btn text" onClick={onBack}>Back</button>
      </div>
    </div>
  );
}
