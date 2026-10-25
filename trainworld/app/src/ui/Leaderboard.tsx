// T-973: the leaderboards, a centred modal opened by the trophy on the Cities tab's World row
// (`LeaderboardButton`). Points only (Anita, 2026-10-23): the world board, then one per city, this
// game's cities first and then the others that have rows. The top 50, the player's own row under
// them when lower, and at the bottom Sign in to be listed, a name with Join, or Leave the
// leaderboards. Each person once per board, with their best normal game (shonei-server's trainworld.go).

import { useEffect, useState } from "preact/hooks";
import { CITIES } from "../game/cities";
import { accountOpen, accountState, boardSetting, boards, boardsOpen, fetchBoards, session, setBoard, type BoardRow } from "../game/account";
import { exact } from "../game/format";
import { moneyState } from "../game/money";
import { pointsPast24h, pointsShort } from "../game/points";
import { Modal } from "./Modal";
import { Icon } from "./widgets";

const WORLD = "points";
const cityName = (id: string) => CITIES.find((c) => c.id === id)?.name ?? id;
const collate = new Intl.Collator(undefined, { sensitivity: "base" }).compare;

function Row({ r, board, mine }: { r: BoardRow; board: string; mine: boolean }) {
  const where = board === WORLD ? "in the world" : `in ${cityName(board.slice(WORLD.length + 1))}`;
  return <li class={mine ? "me" : ""}>
    <span class="lb-rank num">{r.rank}</span>
    <span class="lb-name">{r.name}</span>
    <span class="lb-value num" data-tip={`${exact(r.value)} points ${where} in the past 24 hours`}>{pointsShort(r.value)}</span>
  </li>;
}

function Join() {
  const b = boardSetting.value;
  const [name, setName] = useState(b?.name ?? "");
  const [msg, setMsg] = useState("");
  const run = (f: () => Promise<unknown>) => { setMsg(""); f().then(() => fetchBoards()).catch((e) => setMsg(e.message)); };
  useEffect(() => { if (b?.name) setName(b.name); }, [b?.name]);
  return <div class="lb-foot">
    {moneyState.value?.mode === "sandbox" && <p class="small muted">Sandbox games are not listed</p>}
    {!session.value
      ? <button class="btn text" onClick={() => { boardsOpen.value = false; accountOpen.value = true; }}>Sign in to be listed</button>
      : b?.on
        ? <button class="btn text" onClick={() => run(() => setBoard(false))}>Leave the leaderboards</button>
        : <form class="acct-row" onSubmit={(e) => { e.preventDefault(); if (name.trim()) run(() => setBoard(true, name)); }}>
          <input maxLength={24} placeholder="Name to show" aria-label="Name to show" value={name} onInput={(e) => setName((e.target as HTMLInputElement).value)} />
          <button class="btn text acct-go" type="submit">Join</button>
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
    <select class="lb-pick" aria-label="Leaderboard" onChange={(e) => void fetchBoards((e.target as HTMLSelectElement).value)}>
      <option value={WORLD} selected={key === WORLD}>World</option>
      {mine.length > 0 && <optgroup label="Cities in this game">{mine.map(opt)}</optgroup>}
      {others.length > 0 && <optgroup label="Other cities">{others.map(opt)}</optgroup>}
    </select>
    {!d ? <p class="small muted">{st.error || "Loading…"}</p>
      : <ol class="lb-rows">
        {d.rows.length === 0 && <li class="muted">Nobody here yet.</li>}
        {d.rows.map((r) => <Row r={r} board={key} mine={!!meIn && r.rank === d.me!.rank && r.name === d.me!.name} />)}
        {d.me && !meIn && <><li class="lb-gap" /><Row r={d.me} board={key} mine /></>}
      </ol>}
    {st.error && d && <p class="acct-msg">{st.error}</p>}
    <Join />
  </>;
}

export function Leaderboard() {
  const open = boardsOpen.value;
  return <Modal open={open} onClose={() => (boardsOpen.value = false)} title="Leaderboards" cls="acct-panel lb-panel">
    {open && <Boards />}
  </Modal>;
}

/** The trophy on the World row; nothing until accounts are on. */
export function LeaderboardButton() {
  if (accountState.value !== "on") return null;
  return <button class="btn icon mini lb-open" data-tip="Leaderboards" aria-label="Leaderboards"
    onClick={(e) => { e.stopPropagation(); boardsOpen.value = true; }}><Icon.trophy /></button>;
}
