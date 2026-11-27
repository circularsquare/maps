// T-3681: the search box at the top of Settings. It filters the rows the tab has already drawn, so each control
// keeps working in place and a new setting is found by its own label with no entry here; ALIASES only adds other
// words for some. Hidden rows carry `data-search-out` (an attribute, since Preact rewrites `class` and
// `.slider { display: flex }` beats `[hidden]`).

import { FindInput, SearchIcon } from "./FindInput";
import type { RefObject } from "preact";
import { useEffect, useLayoutEffect, useState } from "preact/hooks";
import { normalise } from "../game/search";
import { t, type MessageKey } from "../i18n";
import en from "../i18n/messages/en.json";

/** A row's label (matched by its whole shown text, so a heading's key aliases its whole section) and other
 * words for it. The English words and label always count too, so "language" finds the row in any language. */
const ALIASES: [MessageKey, MessageKey][] = [
  ["help.name", "settingsSearch.aliasHelp"],
  ["settingsTab.language", "settingsSearch.aliasLanguage"],
  ["settingsTab.bubbles", "settingsSearch.aliasBubbles"],
  ["settingsTab.day", "settingsSearch.aliasDay"],
  ["settingsTab.colors", "settingsSearch.aliasColors"],
  ["settingsTab.mapLabels", "settingsSearch.aliasMapLabels"],
  ["settingsTab.terrain", "settingsSearch.aliasTerrain"],
  ["settingsTab.game", "settingsSearch.aliasGame"],
  ["settingsTab.realNetwork", "settingsSearch.aliasRealNetwork"],
  ["settingsTab.sound", "settingsSearch.aliasSound"],
  ["settingsTab.volume", "settingsTab.sound"],
  ["settingsTab.globe", "settingsSearch.aliasGlobe"],
  ["settingsTab.satellite", "settingsSearch.aliasSatellite"],
  ["settingsTab.lineWidth", "settingsSearch.aliasLineWidth"],
  ["settingsTab.rotate", "settingsSearch.aliasRotate"],
  ["settingsTab.theme", "settingsSearch.aliasTheme"],
  ["settingsTab.uiScale", "settingsSearch.aliasUiScale"],
  ["settingsTab.units", "settingsSearch.aliasUnits"],
  ["settingsTab.currency", "settingsSearch.aliasCurrency"],
  ["settingsTab.performance", "settingsSearch.aliasPerformance"],
  ["settingsTab.lineShapes", "settingsSearch.aliasShapes"],
  ["settingsTab.stations", "settingsSearch.aliasStations"],
  ["accountPanel.account", "settingsSearch.aliasAccount"],
  ["accountPanel.signIn", "settingsSearch.aliasAccount"],
  ["feedbackPanel.open", "settingsSearch.aliasFeedback"],
  ["common.aboutGame", "settingsSearch.aliasAbout"],
];

/** As the map search's, with full-width letters (a Japanese or Chinese keyboard's "ｍｉｌｅｓ") made plain. */
const norm = (s: string) => normalise(s.normalize("NFKC"));

const OUT = "data-search-out";
const FIRST = "data-search-first";
/** Containers whose children are the rows; `data-search-alone` also keeps its rows out of the section above. */
const GROUP = ".checks, .stack, .shape-grid, [data-search-group]";

/** The text a reader sees in `el`, its pieces spaced apart, with the names of icon buttons and selects. */
function shownText(el: Element): string {
  const parts: string[] = [];
  const walk = document.createTreeWalker(el, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
  for (let n: Node | null = el; n; n = walk.nextNode()) {
    if (n.nodeType === Node.TEXT_NODE) parts.push(n.nodeValue ?? "");
    else { const a = (n as Element).getAttribute("aria-label"); if (a) parts.push(a); }
  }
  return parts.join(" ");
}

/** Hides the rows of `list` (the tab's body) that `query` does not find, and the headings left with none;
 * an empty query shows everything as drawn. Says whether anything is shown. */
export function filterSettings(list: HTMLElement, query: string): boolean {
  for (const el of list.querySelectorAll(`[${OUT}], [${FIRST}]`)) el.removeAttribute(OUT), el.removeAttribute(FIRST);
  const words = norm(query).split(" ").filter(Boolean);
  if (!words.length) return true;
  const aliases = ALIASES.map(([label, alias]) => [norm(t(label)), norm(`${t(alias)} ${en[alias]} ${en[label]}`)] as const);
  const hay = (el: Element) => {
    const labels = new Set<string>();
    for (const e of [el, ...el.querySelectorAll("*")]) {
      labels.add(norm(e.textContent ?? ""));
      const a = e.getAttribute("aria-label");
      if (a) labels.add(norm(a));
    }
    const aliasKey = el.getAttribute("data-search-alias") as MessageKey | null;
    return [
      norm(shownText(el)),
      aliasKey ? norm(`${t(aliasKey)} ${en[aliasKey]}`) : "",
      ...aliases.filter(([label]) => labels.has(label)).map(([, words]) => words),
    ].join(" ");
  };
  const found = (text: string) => words.every((w) => text.includes(w));
  /** Filters one run of rows under a heading's text; says whether any is shown. A `data-search-follow` row
   * (one with no label of its own, such as the ramp's colours) shows with the row before it. */
  const rows = (els: Element[], head: string): boolean => {
    let any = false, prev = false;
    for (const el of els) {
      // a row's open dialog (the side switch's list) is never hidden from under the player
      if (el.matches("dialog")) continue;
      const shown: boolean = el.matches(GROUP)
        ? rows([...el.children], head)
        : found(head + " " + hay(el)) || (prev && el.hasAttribute("data-search-follow"));
      if (!shown) el.setAttribute(OUT, "");
      prev = shown;
      any ||= shown;
    }
    return any;
  };
  // sections: the rows from one heading to the next; rows before the first heading or from a `data-search-alone`
  // one on have none
  let head: Element | null = null, run: Element[] = [], any = false;
  const close = () => {
    const shown = rows(run, head ? hay(head) : "");
    if (head && !shown) head.setAttribute(OUT, "");
    any ||= shown;
  };
  for (const el of list.children) {
    if (el.matches(".h")) { close(); head = el; run = []; }
    else if (el.hasAttribute("data-search-alone")) { close(); head = null; run = [el]; }
    else run.push(el);
  }
  close();
  // the first row shown sits right under the box, as the tab's first row does
  let first: Element | null = list.firstElementChild;
  while (first) {
    if (first.hasAttribute(OUT)) { first = first.nextElementSibling; continue; }
    first.setAttribute(FIRST, "");
    if (!first.matches(GROUP)) break;
    first = first.firstElementChild;
  }
  return any;
}

/** The box, and the line said when nothing matches; `list` is the tab's body it filters. */
export function SettingsSearch({ list }: { list: RefObject<HTMLElement | null> }) {
  const [query, setQuery] = useState("");
  const [none, setNone] = useState(false);
  useLayoutEffect(() => {
    const el = list.current;
    if (el) setNone(!filterSettings(el, query));
  }, [query]);
  // rows drawn while a search is up (a timer's figure, a row that appears, another language) are filtered too
  useEffect(() => {
    const el = list.current;
    if (!el || !query) return;
    const watch = new MutationObserver(() => setNone(!filterSettings(el, query)));
    watch.observe(el, { childList: true, subtree: true, characterData: true });
    return () => watch.disconnect();
  }, [query]);
  return (
    <>
      <FindInput type="search" wrap="settings-find" class="new-game-find" icon={<SearchIcon />} onClear={() => setQuery("")} aria-label={t("settingsSearch.placeholder")}
        maxLength={60} spellcheck={false} value={query} onInput={(e) => setQuery((e.currentTarget as HTMLInputElement).value)}
        onKeyDown={(e) => { if (e.key === "Escape" && query) { e.stopPropagation(); setQuery(""); } }} />
      <div class="settings-find-rule" />
      {none && <p class="muted settings-none">{t("settingsSearch.none")}</p>}
    </>
  );
}
