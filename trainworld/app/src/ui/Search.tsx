// The search box at the map's upper right (T-148): stations, lines, cities, towns and
// neighbourhoods by name (game/search.ts; the view's zones' place lists load on first focus, T-177,
// T-400). Most of the
// time it is a small magnifying glass (T-211); a click on it, "/", Ctrl+K or Ctrl+F opens the box
// and focuses it, the arrow keys move through the results, Enter picks, and Escape, a pick or
// leaving the box empty closes it again. Picking moves the camera there at once (no flight, like
// every other state change) and, with no build tool on, selects the station or line; with a tool
// on it only moves the camera, so a half-drawn route or a line being edited carries on there.

import { h, render } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { activeCity } from "../game/cities";
import { zoneAt } from "../game/zones";
import { loadPlacesAt, places, widenPlaces } from "../game/placeList";
import { viewCentre } from "../game/listScope";
import { search, type SearchHit } from "../game/search";
import { ask, selection, tool } from "../game/ui";
import type { LineInput } from "../game/types";
import { world } from "../game/world";
import { live } from "../map/live";
import { keyIsTheirs } from "./keys";
import { locateLine, locateStation } from "./locate";
import { goToCity } from "./Cities";
import { Chip } from "./widgets";

/** T-400: a search showing fewer results than this asks for more zones' place lists. */
const WANT = 5;
/** The zoom a town or neighbourhood opens at (T-177), by its OSM kind: a town whole, a
 * neighbourhood at street level. */
const PLACE_ZOOM: Record<string, number> = {
  city: 12, town: 12.5, borough: 12, suburb: 13, quarter: 14, village: 13.5, neighbourhood: 14.5, hamlet: 14,
};
/** The words a place's row uses for its OSM kind. */
const PLACE_KIND: Record<string, string> = {
  city: "City", town: "Town", borough: "Borough", suburb: "Neighbourhood", quarter: "Neighbourhood",
  village: "Village", neighbourhood: "Neighbourhood", hamlet: "Village",
};

/** Move the camera to a result and select it. */
export function goTo(hit: SearchHit) {
  const map = live.map;
  const selecting = tool.peek() === "select";
  if (hit.kind === "city") {
    goToCity(hit.city);
    return;
  }
  if (hit.kind === "place") {
    map?.jumpTo({ center: [hit.lng, hit.lat], zoom: PLACE_ZOOM[hit.place.kind] ?? 14 });
    activeCity.value = zoneAt(hit.lng, hit.lat);
    // as for a city: a station or bubble selection elsewhere would hold the focus there (main.ts)
    const k = selection.peek()?.kind;
    if (selecting && (k === "station" || k === "commuters")) selection.value = null;
    return;
  }
  if (hit.kind === "station") {
    locateStation(hit.lng, hit.lat);
    if (!selecting) return;
    // a name shared by several platforms of one complex selects the member with the most lines;
    // the Station / Complex setting stays as the player had it (Anita, T-182)
    selection.value = { kind: "station", station: hit.target };
    return;
  }
  locateLine(hit.line.stops);
  if (selecting) selection.value = { kind: "line", line: hit.line.id };
}

export function Search() {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  // T-211: the box shows only while in use; otherwise a magnifying glass
  const [expanded, setExpanded] = useState(false);
  const [at, setAt] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  // a request to focus the box once it is drawn (it is not in the page while collapsed)
  const focusNext = useRef(false);
  void world.value; // results follow the network
  void places.value; // and the place lists as they arrive (fetched on first focus, T-177)
  const hits = open ? search(query) : [];
  const cur = Math.min(at, Math.max(0, hits.length - 1));
  // T-400: few results from the lists in so far: the next nearest zones' lists, until enough or all
  const wanting = open && query.trim().length >= 3 && hits.length < WANT;
  useEffect(() => {
    const c = viewCentre.peek();
    if (wanting && c) void widenPlaces(c[0], c[1]);
  }, [wanting, query, places.value.size]);

  const expand = () => {
    if (input.current) {
      input.current.focus();
      input.current.select();
    } else {
      focusNext.current = true;
      setExpanded(true);
    }
  };
  useEffect(() => {
    if (!expanded || !focusNext.current) return;
    focusNext.current = false;
    input.current?.focus();
  }, [expanded]);

  // "/", Ctrl+K or Ctrl+F from anywhere but a text field or an open question (a select lets them
  // through, ui/keys.ts). Ctrl+F is kept from the browser's find only then: in another text field
  // it is the browser's. In the box itself Ctrl+K and Ctrl+F select its text.
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      const slash = e.key === "/" && !e.ctrlKey && !e.metaKey && !e.altKey;
      const k = e.key.toLowerCase();
      const ctrl = (e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && (k === "k" || k === "f");
      if (!slash && !ctrl) return;
      if ((keyIsTheirs(e) && !(ctrl && e.target === input.current)) || ask.peek()) return;
      e.preventDefault();
      expand();
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, []);

  const close = () => {
    setQuery("");
    setOpen(false);
    setAt(0);
    setExpanded(false);
    input.current?.blur();
  };
  const pick = (hit: SearchHit | undefined) => {
    if (!hit) return;
    close();
    goTo(hit);
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape") {
      e.preventDefault();
      close();
    } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!hits.length) return;
      const n = hits.length;
      setAt((cur + (e.key === "ArrowDown" ? 1 : n - 1)) % n);
    } else if (e.key === "Enter") {
      e.preventDefault();
      pick(hits[cur]);
    }
  };

  // Collapsed: the magnifying glass only. Leaving an empty box collapses it (onBlur below); with
  // text in it the box stays, so the text is not lost.
  if (!expanded)
    return (
      <div class="map-search closed">
        <button class="btn icon search-open" aria-label="Search stations, lines and places" data-tip="Find a station, line or place (/ or Ctrl+F)"
          onClick={expand}>
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
            <circle cx="5.75" cy="5.75" r="4.25" />
            <path d="M8.9 8.9 L13 13" stroke-linecap="round" />
          </svg>
        </button>
      </div>
    );
  return (
    <div class="map-search">
      <input ref={input} type="text" value={query} placeholder="Find a station, line or place" aria-label="Search stations, lines and places"
        autocomplete="off" spellcheck={false} role="combobox" aria-expanded={open && !!query.trim()}
        aria-controls="map-search-results" aria-activedescendant={hits.length ? `map-search-${cur}` : undefined}
        onInput={(e) => { setQuery(e.currentTarget.value); setOpen(true); setAt(0); }}
        onFocus={() => { setOpen(true); const c = viewCentre.peek(); if (c) void loadPlacesAt(c[0], c[1]); }}
        onBlur={(e) => { setOpen(false); if (!e.currentTarget.value) setExpanded(false); }}
        onKeyDown={onKey} />
      {open && query.trim() && (
        <ul class="map-search-results" id="map-search-results" role="listbox">
          {hits.length === 0 && <li class="muted none">Nothing found</li>}
          {hits.map((hit, i) => (
            <li id={`map-search-${i}`} role="option" aria-selected={i === cur} class={i === cur ? "on" : ""}
              onMouseDown={(e) => e.preventDefault()} onMouseMove={() => i !== cur && setAt(i)} onClick={() => pick(hit)}>
              <Row hit={hit} pickLine={(line) => pick({ kind: "line", line, name: line.name })} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Row({ hit, pickLine }: { hit: SearchHit; pickLine: (l: LineInput) => void }) {
  if (hit.kind === "line")
    return <>
      <Chip line={hit.line} size="sm" plain />
      <span class="grow">{hit.name}</span>
      <span class="muted small">Line</span>
    </>;
  if (hit.kind === "city")
    return <>
      <i class="search-mark city" />
      <span class="grow">{hit.name}{hit.matched && <><br /><span class="muted small">matched {hit.matched}</span></>}</span>
      <span class="muted small">City</span>
    </>;
  if (hit.kind === "place")
    return <>
      <i class="search-mark place" />
      <span class="grow">{hit.name}{hit.tag && <><br /><span class="muted small">{hit.tag}</span></>}</span>
      <span class="muted small">{PLACE_KIND[hit.place.kind] ?? "Place"}</span>
    </>;
  return <>
    <i class="search-mark" />
    {/* a same-named result's tag goes on a line of its own, so a long name's ellipsis never hides it */}
    <span class={"grow" + (hit.built ? "" : " muted")}>{hit.name}{hit.tag && <><br /><span class="muted small">{hit.tag}</span></>}</span>
    {hit.lines.length ? chips(hit.lines).map((l) => <Chip line={l} size="sm" pick={pickLine} />) : <span class="muted small">Station</span>}
  </>;
}

/** A station's line chips, one per look (same badge and colour), at most four, so the name keeps
 * its room. */
function chips(lines: LineInput[]): LineInput[] {
  return [...new Map(lines.map((l) => [l.letter + l.colour, l])).values()].slice(0, 4);
}

/** Mount the search box in the map's container. */
export function startSearch(container: HTMLElement) {
  const root = document.createElement("div");
  container.appendChild(root);
  render(h(Search, null), root);
}
