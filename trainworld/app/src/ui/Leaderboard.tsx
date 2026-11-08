// T-973: the leaderboards, a centred modal opened by the trophy at the top of the Cities tab, beside
// the achievements (`LeaderboardButton`, T-2179). Points only (Anita, 2026-10-23): the world board, then one per city, this
// game's cities first and then the others that have rows. The top 50, the player's own row under
// them when lower, and at the bottom Sign in to be listed, a name with Join, or Leave the
// leaderboards. Each person once per board, with their best normal game (shonei-server's trainworld.go).

import { useEffect, useState } from "preact/hooks";
import { CITIES } from "../game/cities";
import { accountOpen, accountState, boardSetting, boards, boardsOpen, fetchBoards, session, setBoard, type BoardRow } from "../game/account";
import { exact } from "../game/format";
import { t } from "../i18n";
import { moneyState } from "../game/money";
import { POINTS_PER_KM, POINTS_PER_TRIP, pointsPast24h, pointsShort } from "../game/points";
import { Modal } from "./Modal";
import { Icon } from "./widgets";

const WORLD = "points";
const cityName = (id: string) => CITIES.find((c) => c.id === id)?.name ?? id;
const collate = new Intl.Collator(undefined, { sensitivity: "base" }).compare;

function Row({ r, board, mine }: { r: BoardRow; board: string; mine: boolean }) {
  return <li class={mine ? "me" : ""}>
    <span class="lb-rank num">{r.rank}</span>
    <span class="lb-name">{r.name}</span>
    <span class="lb-value num" data-tip={t("leaderboard.valueTip", { points: exact(r.value), world: board === WORLD, city: board === WORLD ? "" : cityName(board.slice(WORLD.length + 1)) }) + "\n" + t("cities.pointsRule", { trip: POINTS_PER_TRIP, km: POINTS_PER_KM })}>{t("leaderboard.value", { points: pointsShort(r.value) })}</span>
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

function Boards() {
  const st = boards.value;
  useEffect(() => { if (!st.at || Date.now() - st.at > 60_000) void fetchBoards(); }, [st.at]);
  const key = st.key, d = st.data && st.data.board === key ? st.data : null;
  // this game's cities, its most points first; then every other city with a board
  const mine = [...pointsPast24h.value].filter(([id, v]) => id && v > 0).sort((a, b) => b[1] - a[1]).map(([id]) => id);
  const others = Object.keys(st.list).filter((k) => k.startsWith(WORLD + "_")).map((k) => k.slice(WORLD.length + 1))
    .filter((id) => !mine.includes(id)).sort((a, b) => collate(cityName(a), cityName(b)));
  const opt = (id: string) => <option value={`${WORLD}_${id}`} selected={key === `${WORLD}_${id}`}>{cityName(id)}</option>;
  const meIn = d?.me && d.rows.some((r) => r.rank === d.me!.rank && r.name === d.me!.name);
  return <>
    <select class="lb-pick" aria-label={t("leaderboard.pick")} onChange={(e) => void fetchBoards((e.target as HTMLSelectElement).value)}>
      <option value={WORLD} selected={key === WORLD}>{t("leaderboard.world")}</option>
      {mine.length > 0 && <optgroup label={t("leaderboard.thisGame")}>{mine.map(opt)}</optgroup>}
      {others.length > 0 && <optgroup label={t("leaderboard.otherCities")}>{others.map(opt)}</optgroup>}
    </select>
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
  return <Modal open={open} onClose={() => (boardsOpen.value = false)} title={t("leaderboard.title")} cls="acct-panel lb-panel">
    {open && <Boards />}
  </Modal>;
}

/** The trophy beside the achievements; nothing until accounts are on. */
export function LeaderboardButton() {
  if (accountState.value !== "on") return null;
  return <button class="btn icon mini lb-open" data-tip={t("leaderboard.title")} aria-label={t("leaderboard.title")}
    onClick={(e) => { e.stopPropagation(); boardsOpen.value = true; }}><Icon.trophy /></button>;
}
