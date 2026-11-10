// T-2887: Community, stage 1 (notes/T-2887.md; shonei-server's docs/trainworld-api.md "Community"). A player shares a
// game's schematic as drawing data; anyone can browse them, and signed-in players can like and report. Everything waits
// for `/noritetsu/config` to say `trainworldCommunity`: with the server's flag off nothing here is called or shown.

import { signal } from "@preact/signals";
import { accountConfig, accountState, api, ApiError, gameForShare, loadBoardSetting, playingGameId, session } from "./account";
import { diagramForShare } from "../diagram/view";
import { decodeShared, packShared, type DecodedShare, type SharedDrawing } from "../diagram/shared";
import { t } from "../i18n";

export type ShareView = "random" | "best" | "top";
export type ReportReason = "offensive" | "fake-score" | "other";

export interface ShareRow {
  id: string; rev: number; name: string; city: string; points: number; lines: number; stations: number; day: number; likes: number; at: number;
  sandbox?: boolean; liked?: boolean; mine?: boolean; rank?: number; game?: string; hidden?: unknown;
}
export interface ShareList { view: string; city: string; seed: string; count: number; rows: ShareRow[] }

export const communityOpen = signal(false);
/** the Share dialog is open */
export const shareAsk = signal(false);
/** this account's shares, for Share or Stop sharing on the game's card */
export const myShares = signal<ShareRow[]>([]);
/** what a share is doing while it runs */
export const sharePhase = signal<"" | "drawing" | "sending">("");

// A server in preview (SHONEI_COMMUNITY=preview) says `trainworldCommunityPreview` instead, and Community shows only
// on a page opened with `?community` in its address; the server takes writes from its tester accounts alone.
const FLAG = /[?&]community(?:[=&]|$)/.test(location.search);
const available = (c: typeof accountConfig.value) => c?.trainworldCommunity === true || (FLAG && c?.trainworldCommunityPreview === true);
export const communityOn = () => accountState.value === "on" && available(accountConfig.value);
const on = () => accountState.peek() === "on" && available(accountConfig.peek());

/** What went wrong, in plain words, for each of the server's codes. */
function plain(e: unknown, act: "share" | "like" | "report"): Error {
  const err = e as ApiError, code = err.code;
  switch (code) {
    case "bad-upload": case "bad-drawing": return new Error(t("community.errSend"));
    case "too-small": return new Error(t("community.errTooSmall"));
    case "need-name": return new Error(t("community.errNeedName"));
    case "bad-name": return new Error(t("community.errBadName"));
    case "too-new": return new Error(act === "share" ? t("community.errNewShare") : t("community.errNewLike"));
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
  try { myShares.value = (await api<ShareList>("GET", "trainworld/community/list?view=mine")).rows ?? []; }
  catch { /* keeps the last list; a 401 has signed out */ }
}

/** the playing game's share, read as a signal */
export function currentShare(): ShareRow | null {
  const rows = myShares.value;
  const id = playingGameId();
  return (id && rows.find((r) => r.game === id)) || null;
}

/** Share or Share again on the playing game, `name` when the account has none. Throws an Error to show. */
export async function shareGame(name: string): Promise<void> {
  try {
    const game = await gameForShare();
    if (!game) throw new Error(t("community.errNoGame"));
    sharePhase.value = "drawing";
    const got = await diagramForShare();
    if (!got) throw new Error(t("community.errDraw"));
    const blob = await packShared(got.d, got.names);
    if (!blob) throw new Error(t("community.errTooBig"));
    sharePhase.value = "sending";
    const form = new FormData();
    form.append("meta", JSON.stringify(name ? { game, name } : { game }));
    form.append("drawing", blob, "drawing.json.gz");
    await api("POST", "trainworld/community", undefined, { form });
    await Promise.all([refreshMyShares(), loadBoardSetting()]);
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
  } catch (e) { throw plain(e, "share"); }
}

export async function fetchShares(view: ShareView, city: string, seed: string): Promise<ShareList> {
  const q = new URLSearchParams({ view, city, seed });
  const j = await api<ShareList>("GET", `trainworld/community/list?${q}`);
  return { ...j, rows: j.rows ?? [] };
}

export const newSeed = () => Math.random().toString(36).slice(2, 12);

/** decoded drawings, the newest kept; each row's URL changes with its rev */
const cache = new Map<string, Promise<DecodedShare>>();
const CACHE_SIZE = 60;
export function loadDrawing(row: { id: string; rev: number }): Promise<DecodedShare> {
  const key = `${row.id}/${row.rev}`;
  let p = cache.get(key);
  if (!p) {
    p = api<SharedDrawing>("GET", `trainworld/community/${encodeURIComponent(row.id)}/${row.rev}`, undefined, { auth: false }).then(decodeShared);
    p.catch(() => cache.delete(key));
    cache.set(key, p);
    if (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value!);
  }
  return p;
}

export async function setLiked(id: string, liked: boolean): Promise<{ liked: boolean; likes: number }> {
  try { return await api(liked ? "PUT" : "DELETE", `trainworld/community/${encodeURIComponent(id)}/like`); }
  catch (e) { throw plain(e, "like"); }
}

export async function reportShare(id: string, reason: ReportReason): Promise<void> {
  try { await api("POST", `trainworld/community/${encodeURIComponent(id)}/report`, { reason }); }
  catch (e) { throw plain(e, "report"); }
}
