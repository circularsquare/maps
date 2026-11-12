// T-973: the leaderboards, a centred modal opened by the trophy at the top of the Cities tab, beside
// the achievements (`LeaderboardButton`, T-2179). Points only (Anita, 2026-10-23): the world board, then one per city, this
// game's cities first and then the others that have rows. T-3116 country boards (`country_<cc>`, a game's city points summed over the country) sit in the same picker, found by the country's name. The top 50, the player's own row under
// them when lower, and at the bottom Sign in to be listed, a name with Join, or Leave the
// leaderboards. Each person once per board, with their best normal game there at its current figures (T-3248; shonei-server's trainworld.go).

import { useEffect, useMemo, useState } from "preact/hooks";
import { normalise } from "../game/search";
import { tPlace } from "../i18n/places";
import { findStart } from "./NewGame";
import { CITIES, zoneCountry } from "../game/cities";
import { accountOpen, accountState, boardSetting, boards, boardsOpen, fetchBoards, session, setBoard, type BoardRow } from "../game/account";
import { exact } from "../game/format";
import { t, tCountry } from "../i18n";
import { countryName } from "../i18n/intl";
import { moneyState } from "../game/money";
import { POINTS_PER_KM, POINTS_PER_TRIP, pointsPast24h, pointsShort } from "../game/points";
import { Modal } from "./Modal";
import { CommunityButton } from "./Community";
import { Icon } from "./widgets";

export const WORLD = "points";
export const COUNTRY = "country_";
/** T-3132: Community's region places, `region_<realm id>` (the server's key too) */
export const REGION = "region_";
const cityName = (id: string) => CITIES.find((c) => c.id === id)?.name ?? id;
const collate = new Intl.Collator(undefined, { sensitivity: "base" }).compare;

function Row({ r, board, mine }: { r: BoardRow; board: string; mine: boolean }) {
  return <li class={mine ? "me" : ""}>
    <span class="lb-rank num">{r.rank}</span>
    <span class="lb-name">{r.name}</span>
    <span class="lb-value num" data-tip={t("leaderboard.valueTip", { points: exact(r.value), world: board === WORLD, city: board === WORLD ? "" : board.startsWith(COUNTRY) ? countryLabel(board.slice(COUNTRY.length)) : cityName(board.slice(WORLD.length + 1)) }) + "\n" + t("leaderboard.pointsRule", { trip: POINTS_PER_TRIP, km: POINTS_PER_KM })}>{t("leaderboard.value", { points: pointsShort(r.value) })}</span>
  </li>;
}

function Join() {
  const b = boardSetting.value;
  const [name, setName] = useState(b?.name ?? "");
  const [msg, setMsg] = useState("");
  const run = (f: () => Promise<unknown>) => { setMsg(""); f().then(() => fetchBoards()).catch((e) => setMsg(e.message)); };
  useEffect(() => { if (b?.name) setName(b.name); }, [b?.name]);
  return <div class="lb-foot">
    {moneyState.value?.mode === "sandbox" && <p class="small muted">{t("leaderboard.sandbox")}</p>}
    {!session.value
      ? <button class="btn text" onClick={() => { boardsOpen.value = false; accountOpen.value = true; }}>{t("leaderboard.signIn")}</button>
      : b?.on
        ? <button class="btn text" onClick={() => run(() => setBoard(false))}>{t("leaderboard.leave")}</button>
        : <form class="acct-row" onSubmit={(e) => { e.preventDefault(); if (name.trim()) run(() => setBoard(true, name)); }}>
          <input maxLength={24} placeholder={t("leaderboard.name")} aria-label={t("leaderboard.name")} value={name} onInput={(e) => setName((e.target as HTMLInputElement).value)} />
          <button class="btn text acct-go" type="submit">{t("leaderboard.join")}</button>
        </form>}
    {msg && <p class="acct-msg">{msg}</p>}
  </div>;
}

const MAX_SHOWN = 8;
const shownName = (id: string) => tPlace(id, cityName(id));
const countryLabel = (cc: string) => tCountry(cc.toUpperCase());
/** T-3116: a country board is named for the country, in the player's language or in English. */
const countryHit = (cc: string, q: string) => wordHit([countryLabel(cc), countryName("en", cc.toUpperCase())], q);
/** a name matches a typed query at the start of any of its words */
const wordHit = (names: string[], q: string) => names.some((n) => ` ${normalise(n)}`.includes(` ${normalise(q)}`));
/** Picker places beyond cities and countries (T-3132: Community's regions), each with its name. */
export interface MorePlaces { regions?: string[]; findCountries?: string[]; findRegions?: string[]; regionName?: (id: string) => string }
const boardLabel = (key: string, more?: MorePlaces) => key === WORLD ? t("leaderboard.world") : key.startsWith(COUNTRY) ? countryLabel(key.slice(COUNTRY.length))
  : key.startsWith(REGION) ? more?.regionName?.(key.slice(REGION.length)) ?? key.slice(REGION.length) : shownName(key.slice(WORLD.length + 1));

/** T-2813: the board picker works as the new game's city search does: type, and every matching
 * board is listed (arrows and Enter to choose); with nothing typed it lists the world and this game's cities. `onPick`: Community's, in place of loading that board.
 * `more` (Community, T-3132): regions offered like `countries`, and countries and regions found only when typed. */
export function CityPick({ boardKey, mine, all, countries, more, onPick }: { boardKey: string; mine: string[]; all: ReadonlySet<string>; countries?: string[]; more?: MorePlaces; onPick?: (key: string) => void }) {
  const [text, setText] = useState<string | null>(null);
  const [cursor, setCursor] = useState(0);
  const current = boardLabel(boardKey, more);
  const q = text?.trim() ?? "";
  const items = useMemo(() => {
    if (text === null) return [];
    const world = { key: WORLD, label: t("leaderboard.world"), sub: "" };
    const city = (id: string) => ({ key: `${WORLD}_${id}`, label: shownName(id), sub: "" });
    const country = (cc: string) => ({ key: COUNTRY + cc, label: countryLabel(cc), sub: "" });
    const regionName = more?.regionName ?? ((id: string) => id);
    const region = (id: string) => ({ key: REGION + id, label: regionName(id), sub: "" });
    const near = (more?.regions ?? []).slice(0, 2).map(region);
    if (!q) return [world, ...(countries ?? []).slice(0, 2).map(country), ...near, ...mine.slice(0, MAX_SHOWN - 3 - near.length).map(city)];
    const hits = findStart(q, all).map(({ city: c, matched }) => ({ key: `${WORLD}_${c.id}`, label: shownName(c.id), sub: matched ? t("newGame.matched", { alias: matched }) : "" }));
    const lands = [...new Set([...(countries ?? []), ...(more?.findCountries ?? [])])].filter((cc) => countryHit(cc, q)).sort((a, b) => collate(countryLabel(a), countryLabel(b))).map(country);
    const areas = [...new Set([...(more?.regions ?? []), ...(more?.findRegions ?? [])])].filter((id) => wordHit([regionName(id)], q)).sort((a, b) => collate(regionName(a), regionName(b))).map(region);
    return (normalise(t("leaderboard.world")).startsWith(normalise(q)) ? [world] : []).concat(lands, areas, hits).slice(0, MAX_SHOWN);
  }, [text, mine, all, countries, more]);
  const pick = (key: string) => { setText(null); if (onPick) onPick(key); else void fetchBoards(key); (document.activeElement as HTMLElement | null)?.blur(); };
  return <div class="lb-find">
    <input type="text" class="new-game-find lb-pick" aria-label={t("leaderboard.pick")} placeholder={t("newGame.findCity")} maxLength={60}
      value={text ?? current}
      onFocus={(e) => { setText(""); setCursor(0); (e.target as HTMLInputElement).select(); }}
      onBlur={() => setText(null)}
      onInput={(e) => { setText((e.target as HTMLInputElement).value); setCursor(0); }}
      onKeyDown={(e) => {
        if (e.key === "ArrowDown" && items.length) { e.preventDefault(); setCursor((cursor + 1) % items.length); }
        else if (e.key === "ArrowUp" && items.length) { e.preventDefault(); setCursor((cursor + items.length - 1) % items.length); }
        else if (e.key === "Enter" && items.length) { e.preventDefault(); pick(items[Math.min(cursor, items.length - 1)].key); }
        else if (e.key === "Escape" && text !== null) { e.preventDefault(); e.stopPropagation(); setText(null); (e.target as HTMLInputElement).blur(); }
      }} />
    {items.length > 0 && <ul class="new-game-hits lb-hits" role="listbox">
      {items.map((it, i) => (
        <li role="option" aria-selected={i === cursor} class={i === cursor ? "on" : ""} onMouseDown={(e) => { e.preventDefault(); pick(it.key); }}>
          <span>{it.label}{it.sub && <span class="muted small">{it.sub}</span>}</span>
        </li>
      ))}
    </ul>}
  </div>;
}

function Boards() {
  const st = boards.value;
  useEffect(() => { if (!st.at || Date.now() - st.at > 60_000) void fetchBoards(); }, [st.at]);
  const key = st.key, d = st.data && st.data.board === key ? st.data : null;
  // this game's cities, its most points first; then every other city with a board
  const mine = [...pointsPast24h.value].filter(([id, v]) => id && v > 0).sort((a, b) => b[1] - a[1]).map(([id]) => id);
  const others = Object.keys(st.list).filter((k) => k.startsWith(WORLD + "_")).map((k) => k.slice(WORLD.length + 1))
    .filter((id) => !mine.includes(id)).sort((a, b) => collate(cityName(a), cityName(b)));
  const lands = [...new Set([...mine.map((id) => zoneCountry(id)?.toLowerCase() ?? ""), ...Object.keys(st.list).filter((k) => k.startsWith(COUNTRY)).map((k) => k.slice(COUNTRY.length))].filter(Boolean))];
  const meIn = d?.me && d.rows.some((r) => r.rank === d.me!.rank && r.name === d.me!.name);
  return <>
    <CityPick boardKey={key} mine={mine} all={new Set([...mine, ...others])} countries={lands} />
    {!d ? <p class="small muted">{st.error || t("leaderboard.loading")}</p>
      : <ol class="lb-rows">
        {d.rows.length === 0 && <li class="muted">{t("leaderboard.nobody")}</li>}
        {d.rows.map((r) => <Row r={r} board={key} mine={!!meIn && r.rank === d.me!.rank && r.name === d.me!.name} />)}
        {d.me && !meIn && <><li class="lb-gap" /><Row r={d.me} board={key} mine /></>}
      </ol>}
    {st.error && d && <p class="acct-msg">{st.error}</p>}
    <Join />
  </>;
}

export function Leaderboard() {
  const open = boardsOpen.value;
  return <Modal open={open} onClose={() => (boardsOpen.value = false)} title={t("leaderboard.title")} cls="acct-panel lb-panel" head={<CommunityButton />}>
    {open && <Boards />}
  </Modal>;
}

/** The trophy beside the achievements; nothing until accounts are on. */
export function LeaderboardButton() {
  if (accountState.value !== "on") return null;
  return <button class="btn icon mini lb-open" data-tip={t("leaderboard.title")} aria-label={t("leaderboard.title")}
    onClick={(e) => { e.stopPropagation(); boardsOpen.value = true; }}><Icon.trophy /></button>;
}
