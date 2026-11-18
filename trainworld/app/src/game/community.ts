// T-2887: Community, stage 1 (notes/T-2887.md; shonei-server's docs/trainworld-api.md "Community"). A player shares a
// game's schematic as drawing data; anyone can browse them, and signed-in players can like and report. Everything waits
// for `/noritetsu/config` to say `trainworldCommunity`: with the server's flag off nothing here is called or shown.

// T-3115: a share has a link (`?share=<id>`, `shareLink`) that opens the game on it in Community, and, while its
// player leaves "Let others open a copy" ticked (the default), its save goes up too so anyone can open a copy as a
// game of their own (game/sharedSave.ts checks it both ways).

import { effect, signal } from "@preact/signals";
import { accountConfig, accountState, api, ApiError, fetchBytes, gameForShare, loadBoardSetting, playingGameId, session, slotOfGame } from "./account";
import { diagramForShare } from "../diagram/view";
import { decodeShared, packShared, type DecodedShare, type SharedDrawing } from "../diagram/shared";
import { exportForShare, manageSaveSlot, openView, saveSlotsState } from "../workers/clockClient";
import { copyFromShare, CopyRefused, forShare, MAX_SHARED_SAVE } from "./sharedSave";
import { t, tDate } from "../i18n";
import { accountList } from "./accountGames";
import { playedText } from "./gameTime";

/** T-3132: Random, Biggest (`top`, by points) and Most liked */
export type ShareView = "random" | "top" | "liked";
export type ReportReason = "offensive" | "fake-score" | "other";

export interface ShareRow {
  id: string; rev: number; name: string; city: string; points: number; lines: number; stations: number; day: number; likes: number; at: number;
  /** what the sharer calls it (T-3144); none on a share from before titles */
  title?: string;
  /** T-3269: the sharer's flag, none while their name is hidden */
  flag?: string;
  sandbox?: boolean; copy?: boolean; liked?: boolean; mine?: boolean; rank?: number; game?: string; hidden?: unknown;
}
export interface ShareList { view: string; city: string; seed: string; count: number; rows: ShareRow[]; left?: number; perDay?: number }

export const communityOpen = signal(false);
/** the Share dialog is open */
export const shareAsk = signal(false);
/** this account's shares, for Share or Stop sharing on the game's card */
export const myShares = signal<ShareRow[]>([]);
/** T-3132: shares this account has left today, of how many a day (the server's count); null until read */
export const sharesLeft = signal<{ left: number; perDay: number } | null>(null);
/** counts this page's shares and stops, so an open Community list reloads after one */
export const shareTick = signal(0);
/** what a share is doing while it runs */
export const sharePhase = signal<"" | "drawing" | "sending">("");

// A server in preview (SHONEI_COMMUNITY=preview) says `trainworldCommunityPreview` instead, and Community shows only
// on a page opened with `?community` in its address; the server takes writes from its tester accounts alone.
const FLAG = /[?&]community(?:[=&]|$)/.test(location.search);
const available = (c: typeof accountConfig.value) => c?.trainworldCommunity === true || (FLAG && c?.trainworldCommunityPreview === true);
export const communityOn = () => accountState.value === "on" && available(accountConfig.value);
const on = () => accountState.peek() === "on" && available(accountConfig.peek());

const SHARE_ID = /^[a-z2-9]{8}$/;
/** The share a `?share=<id>` link opened, until Community shows it. The switch leaves the address bar at once, so a
 * reload, or the reload into a copy opened from it, does not open it again. */
export const linkedShare = signal<string | null>((() => {
  const q = new URLSearchParams(location.search), id = q.get("share");
  if (id === null) return null;
  const u = new URL(location.href);
  u.searchParams.delete("share");
  history.replaceState(history.state, "", u.href);
  return SHARE_ID.test(id) ? id : null;
})());
// Community opens on the linked share once the server says Community is there (preview: on a `?community` page only).
effect(() => { if (linkedShare.value && communityOn()) communityOpen.value = true; });

/** T-3685: the share the game in hand was opened from (T-3257's view), for a like button on the map. It lasts while the
 * network is unsaved and, once saved from the view, while that slot is the game in hand; any other game ends it. The
 * reload into a view carries it in sessionStorage. */
const VIEWED_KEY = "tw.community.viewed";
export const viewedShare = signal<{ id: string; seen: boolean; name: string; slot: string } | null>((() => {
  try {
    const id = sessionStorage.getItem(VIEWED_KEY);
    sessionStorage.removeItem(VIEWED_KEY);
    return id && SHARE_ID.test(id) ? { id, seen: false, name: "", slot: "" } : null;
  } catch { return null; }
})());
/** that share's card as the server has it: its likes, and whether this player liked it */
export const viewedRow = signal<ShareRow | null>(null);
effect(() => {
  const v = viewedShare.value, s = saveSlotsState.value;
  if (!v) return;
  if (s.viewing) { try { sessionStorage.removeItem(VIEWED_KEY); } catch { /* none */ } if (!v.seen || v.name !== s.viewing.name) viewedShare.value = { ...v, seen: true, name: s.viewing.name }; return; }
  if (!v.seen) return;
  const saved = v.slot ? null : s.slots.find((x) => x.key === s.active && x.name === v.name);
  if (saved) viewedShare.value = { ...v, slot: saved.key };
  else if (!v.slot || v.slot !== s.active) viewedShare.value = null;
});
effect(() => {
  const id = viewedShare.value?.id, live = communityOn(), who = session.value;
  if (!id || !live) { viewedRow.value = null; return; }
  void who;
  fetchShare(id).then((r) => { if (viewedShare.peek()?.id === id) viewedRow.value = r; }, () => { viewedRow.value = null; });
});

/** A share's link: this page's address and the share, and in preview the flag that shows Community. */
export function shareLink(id: string): string {
  return `${location.origin}${location.pathname}?share=${id}${accountConfig.peek()?.trainworldCommunity === true ? "" : "&community"}`;
}

/** One share's card, for a link; an Error to show when it is not there (hidden, banned or gone answer alike). */
export async function fetchShare(id: string): Promise<ShareRow> {
  try { return await api<ShareRow>("GET", `trainworld/community/${encodeURIComponent(id)}`); }
  catch (e) { throw plain(e, "share"); }
}

/** View a share's save as a game (T-3257): downloaded, checked (game/sharedSave.ts), then the page reloads into it,
 * kept nowhere until the player saves it, as `name`. Resolves with a reason to show, "" once the reload is under way,
 * null when the player chose to stay on the network they were viewing. */
export async function openSharedView(row: ShareRow, name: string): Promise<string | null> {
  let bytes: Uint8Array;
  try {
    const gz = await fetchBytes(`trainworld/community/${encodeURIComponent(row.id)}/${row.rev}/save`, MAX_SHARED_SAVE);
    bytes = await copyFromShare(gz, row.id);
  } catch (e) {
    if (e instanceof CopyRefused) return e.why === "version" ? t("community.errCopyVersion") : t("community.errCopyBad");
    const err = e as ApiError;
    if (err.code === "too-big") return t("community.errCopyBad");
    if (err.status === 404) return t("community.errGone");
    if (err.status === 429) return t("community.errBusy");
    return err.message;
  }
  const before = viewedShare.peek();
  viewedShare.value = { id: row.id, seen: false, name: "", slot: "" };
  try { sessionStorage.setItem(VIEWED_KEY, row.id); } catch { /* the button then waits for a game opened without a reload */ }
  const undo = () => { viewedShare.value = before; try { sessionStorage.removeItem(VIEWED_KEY); } catch { /* none */ } };
  try { const why = await openView(bytes, name); if (why !== "") undo(); return why; }
  catch { undo(); return t("community.errCopyBad"); }
}

/** What went wrong, in plain words, for each of the server's codes. */
function plain(e: unknown, act: "share" | "like" | "report"): Error {
  const err = e as ApiError, code = err.code;
  switch (code) {
    case "bad-upload": case "bad-drawing": case "bad-save": return new Error(t("community.errSend"));
    case "too-small": return new Error(t("community.errTooSmall"));
    case "need-name": return new Error(t("community.errNeedName"));
    case "bad-name": return new Error(t("community.errBadName"));
    case "bad-title": return new Error(t("community.errBadTitle"));
    case "too-new": {
      const sec = err.extra?.played ?? accountList.peek()?.played, known = typeof sec === "number" ? "yes" : "no", time = typeof sec === "number" ? playedText(sec) : "";
      return new Error(act === "share" ? t("community.errNewShare", { known, time }) : t("community.errNewLike", { known, time }));
    }
    case "preview": return new Error(t("community.errPreview"));
    case "banned": return new Error(t("community.errBanned"));
    case "too-many": return new Error(t("community.errTooMany"));
    case "too-big": return new Error(t("community.errTooBig"));
    case "daily": return new Error(t("community.errDaily"));
    case "full": return new Error(t("community.errFull"));
    case "broken": return new Error(t("community.errDown"));
    case "own": return new Error(t("community.errOwn"));
  }
  if (err.status === 404) return new Error(t("community.errGone"));
  if (err.status === 429) return new Error(t("community.errBusy"));
  if (err.status === 503) return new Error(t("community.errDown"));
  return new Error(err.message);
}

export async function refreshMyShares() {
  if (!on() || !session.peek()) { myShares.value = []; return; }
  try {
    const l = await api<ShareList>("GET", "trainworld/community/list?view=mine");
    myShares.value = l.rows ?? [];
    sharesLeft.value = l.left !== undefined && l.perDay ? { left: l.left, perDay: l.perDay } : null;
  }
  catch { /* keeps the last list; a 401 has signed out */ }
}

/** the playing game's share, read as a signal */
export function currentShare(): ShareRow | null {
  const rows = myShares.value;
  const id = playingGameId();
  return (id && rows.find((r) => r.game === id)) || null;
}

/** When a share went up, short, in the player's language: "11/10, 5:25 PM". */
export const sharedWhen = (row: ShareRow) => tDate(row.at * 1000, { month: "numeric", day: "numeric", hour: "numeric", minute: "2-digit" });

/** T-3144: the name of a copy opened from a share, "<title> - <their display name>"; either alone when the other is missing. */
export const copyName = (row: ShareRow) => [row.title, row.name].filter(Boolean).join(" - ");

/** the longest title the server keeps */
export const TITLE_MAX = 40;

// T-3132: Share again on one of the player's own shares whose game is another in this browser opens that game, and
// the Share dialog once it is playing. The wish rides the reload in sessionStorage.
const SHARE_AFTER = "tw.community.shareAfter";
/** Share again from an opened share of the player's own: the Share dialog for the playing game, else that game opened
 * first; "" when under way, or why not (a game not in this browser). */
export function shareAgain(row: ShareRow): string {
  const mine = myShares.peek().find((r) => r.id === row.id), game = mine?.game;
  if (game && game === playingGameId()) { shareAsk.value = true; return ""; }
  const slot = game ? slotOfGame(game) : null;
  if (!game || !slot) return t("community.openToShare");
  try { sessionStorage.setItem(SHARE_AFTER, game); } catch { /* opens the game only */ }
  communityOpen.value = false;
  manageSaveSlot("open", slot);
  return "";
}
// Read once as the page starts: the page that set it is still running its game until the reload, so it must not act on it.
let shareAfter = (() => {
  try { const v = sessionStorage.getItem(SHARE_AFTER); sessionStorage.removeItem(SHARE_AFTER); return v; } catch { return null; }
})();
effect(() => {
  void saveSlotsState.value;
  if (!shareAfter || !communityOn()) return;
  const id = playingGameId();
  if (!id) return;
  if (id === shareAfter) shareAsk.value = true;
  shareAfter = null;
});

/** Share or Share again on the playing game, `name` when the account has none, with its save when `copy` (Let others
 * open a copy), under `title` (the game's name when blank). Throws an Error to show. */
export async function shareGame(name: string, copy: boolean, title: string): Promise<void> {
  try {
    const game = await gameForShare();
    if (!game) throw new Error(t("community.errNoGame"));
    sharePhase.value = "drawing";
    const got = await diagramForShare();
    if (!got) throw new Error(t("community.errDraw"));
    const blob = await packShared(got.d, got.names);
    if (!blob) throw new Error(t("community.errTooBig"));
    let save: Uint8Array | "big" | null = null;
    if (copy) {
      const bytes = await exportForShare();
      if (!bytes) throw new Error(t("community.errSend"));
      save = await forShare(bytes);
      if (save === "big") throw new Error(t("community.errTooBig"));
      if (!save) throw new Error(t("community.errSend"));
    }
    sharePhase.value = "sending";
    // T-3114: the card counts what the saves list counts, every line and station, bus and ferry ones too
    const st = saveSlotsState.peek(), slot = st.slots.find((s) => s.key === st.active);
    const counts = slot && slot.lines !== undefined && slot.stations !== undefined ? { lines: slot.lines, stations: slot.stations } : {};
    // T-3144: the title is the game's name unless the player typed another
    const sent = (title.trim() || slot?.name || "").slice(0, TITLE_MAX);
    const form = new FormData();
    form.append("meta", JSON.stringify({ game, ...(name ? { name } : {}), copy, ...counts, ...(sent ? { title: sent } : {}) }));
    form.append("drawing", blob, "drawing.json.gz");
    if (save) form.append("save", new Blob([save as BlobPart]), "game.save");
    await api("POST", "trainworld/community", undefined, { form });
    await Promise.all([refreshMyShares(), loadBoardSetting()]);
    shareTick.value++;
  } catch (e) {
    throw e instanceof ApiError ? Object.assign(plain(e, "share"), { code: e.code }) : e;
  } finally {
    sharePhase.value = "";
  }
}

export async function stopSharing(id: string): Promise<void> {
  try {
    await api("DELETE", `trainworld/community/${encodeURIComponent(id)}`);
    await refreshMyShares();
    shareTick.value++;
  } catch (e) { throw plain(e, "share"); }
}

/** T-3224: the server's list takes a search (`q`, in titles and display names); an older one ignores it and the box is hidden */
/** T-3402: the server's Random list takes an `offset`, so every game is on some page */
export const communityPagesOn = () => accountConfig.value?.trainworldCommunityPages === true;
export const communitySearchOn = () => accountConfig.value?.trainworldCommunitySearch === true;
/** the longest search the server takes */
export const SEARCH_MAX = 40;

export async function fetchShares(view: ShareView, city: string, seed: string, search = "", offset = 0): Promise<ShareList> {
  const q = new URLSearchParams({ view, city, seed });
  if (offset) q.set("offset", String(offset));
  if (search) q.set("q", search.slice(0, SEARCH_MAX));
  const j = await api<ShareList>("GET", `trainworld/community/list?${q}`);
  return { ...j, rows: j.rows ?? [] };
}

export const newSeed = () => Math.random().toString(36).slice(2, 12);

/** decoded drawings, the newest kept; each row's URL changes with its rev */
const cache = new Map<string, Promise<DecodedShare>>();
const CACHE_SIZE = 60;
// T-3680: the server answers 429 past 60 drawings at once and 4 more a second, and a page is 12, so a few tab
// switches or shuffles in a row tripped it for about half the cards. Drawings are asked for four at a time, a 429
// holds the queue for a second, and a card nobody shows any more drops out of the queue unasked.
interface DrawingJob { key: string; row: { id: string; rev: number }; refs: number; tries: number; resolve: (s: DecodedShare) => void; reject: (e: unknown) => void }
const DRAWINGS_AT_ONCE = 4, DRAWING_TRIES = 10, DRAWING_HOLD_MS = 1000;
const waiting: DrawingJob[] = [], pending = new Map<string, DrawingJob>();
let drawingsActive = 0, drawingsHeldUntil = 0, drawingTimer: ReturnType<typeof setTimeout> | undefined;
function pumpDrawings() {
  while (drawingsActive < DRAWINGS_AT_ONCE && waiting.length) {
    const hold = drawingsHeldUntil - Date.now();
    if (hold > 0) { clearTimeout(drawingTimer); drawingTimer = setTimeout(pumpDrawings, hold); return; }
    const job = waiting.shift()!;
    if (job.refs <= 0) { pending.delete(job.key); job.reject(new Error("not wanted")); continue; } // i18n-ignore: never shown
    drawingsActive++;
    api<SharedDrawing>("GET", `trainworld/community/${encodeURIComponent(job.row.id)}/${job.row.rev}`, undefined, { auth: false }).then(decodeShared).then(
      (s) => { pending.delete(job.key); job.resolve(s); },
      (e) => {
        const status = e instanceof ApiError ? e.status : -1;
        if ((status === 0 || status === 429 || status >= 500) && ++job.tries < DRAWING_TRIES && job.refs > 0) {
          if (status === 429) drawingsHeldUntil = Date.now() + DRAWING_HOLD_MS;
          waiting.push(job);
        } else { pending.delete(job.key); job.reject(e); }
      },
    ).finally(() => { drawingsActive--; pumpDrawings(); });
  }
}
/** A card or the viewer asks for a drawing; pair each call with `releaseDrawing` when it stops looking. */
export function loadDrawing(row: { id: string; rev: number }): Promise<DecodedShare> {
  const key = `${row.id}/${row.rev}`;
  let p = cache.get(key);
  if (p) { const job = pending.get(key); if (job) job.refs++; return p; }
  p = new Promise<DecodedShare>((resolve, reject) => {
    const job: DrawingJob = { key, row, refs: 1, tries: 0, resolve, reject };
    pending.set(key, job);
    waiting.push(job);
  });
  p.catch(() => cache.delete(key));
  cache.set(key, p);
  if (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value!);
  pumpDrawings();
  return p;
}
export function releaseDrawing(row: { id: string; rev: number }) {
  const job = pending.get(`${row.id}/${row.rev}`);
  if (job) job.refs--;
}

export async function setLiked(id: string, liked: boolean): Promise<{ liked: boolean; likes: number }> {
  try { return await api(liked ? "PUT" : "DELETE", `trainworld/community/${encodeURIComponent(id)}/like`); }
  catch (e) { throw plain(e, "like"); }
}

export async function reportShare(id: string, reason: ReportReason): Promise<void> {
  try { await api("POST", `trainworld/community/${encodeURIComponent(id)}/report`, { reason }); }
  catch (e) { throw plain(e, "report"); }
}
