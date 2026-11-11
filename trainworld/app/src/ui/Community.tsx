// T-2887: Community's modal (notes/T-2887.md). Random, Biggest and Most liked in a place (T-3132), twelve networks a page
// painted small with the schematic's own painter, one opened to pan and zoom with like and report; and the Share button
// and its dialog. Nothing shows unless the server's `trainworldCommunity` is on (game/community.ts). Names reach the
// page only through Preact's text and the canvas's fillText, never as HTML.

import { useEffect, useRef, useState } from "preact/hooks";
import { CITIES, zoneCountry } from "../game/cities";
import realmsJson from "../game/realms.json";
import { accountOpen, boardSetting, loadBoardSetting, session, shareableSlot } from "../game/account";
import { TITLE_MAX, communityOn, communityOpen, copyName, currentShare, fetchShare, fetchShares, linkedShare, loadDrawing, myShares, newSeed, openSharedCopy, refreshMyShares, reportShare, setLiked, shareAgain, shareAsk, shareTick, sharedWhen, sharesLeft, shareLink, sharePhase, shareGame, stopSharing, type ReportReason, type ShareRow, type ShareView } from "../game/community";
import { saveSlotsState } from "../workers/clockClient";
import { pointsPast24h, pointsShort } from "../game/points";
import { shade, theme as colourTheme } from "../game/ui";
import { diagramShareProgress, diagramTheme } from "../diagram/view";
import { paint, widened } from "../diagram/draw";
import { paintNames } from "../diagram/names";
import { fitView, type DecodedShare } from "../diagram/shared";
import { t } from "../i18n";
import { tPlace } from "../i18n/places";
import { CityPick, WORLD, type MorePlaces } from "./Leaderboard";
import { LoadBar } from "./LoadingScreen";
import { Modal } from "./Modal";
import { Icon } from "./widgets";
import "./community.css";

const PAGE = 12;
const cityName = (id: string) => tPlace(id, CITIES.find((c) => c.id === id)?.name ?? id);

// T-3132: the place picker's countries and regions. A region is a realm (game/realms.ts, named for players on the map)
// that is not just one country; a city's is its registry row's, else its country's, as the server's table has it.
interface RealmRow { id: string; name: string; countries?: string[]; units?: object }
const REALMS = (realmsJson as { realms: RealmRow[] }).realms;
const REALM_NAME = new Map(REALMS.map((r) => [r.id, r.name]));
const REALM_OF_COUNTRY = new Map(REALMS.flatMap((r) => (r.countries ?? []).map((cc) => [cc, r.id] as const)));
const REGIONS = REALMS.filter((r) => r.units || (r.countries?.length ?? 0) > 1).map((r) => r.id);
const COUNTRIES = [...new Set(CITIES.map((c) => c.country?.toLowerCase()).filter((cc): cc is string => !!cc))];
const regionOf = (id: string) => {
  const c = CITIES.find((x) => x.id === id), r = c?.realm && REALM_NAME.has(c.realm) ? c.realm : REALM_OF_COUNTRY.get(c?.country ?? "");
  return r && REGIONS.includes(r) ? r : null;
};
const MORE_PLACES: MorePlaces = { findCountries: COUNTRIES, findRegions: REGIONS, regionName: (id) => REALM_NAME.get(id) ?? id };
/** a picker key as the server's list takes it: "" for the world, a city id, country_<cc> or region_<id> */
const placeParam = (key: string) => key === WORLD ? "" : key.startsWith(WORLD + "_") ? key.slice(WORLD.length + 1) : key;

/** a drawing's lines never thinner than this many CSS px when zoomed out, so a card reads at its small size */
const MIN_LINE_PX = 1.5;

/** T-3115: Copy link, on an opened share and in the Share dialog. Where the clipboard is refused, the link shows in a
 * field to copy by hand. */
function CopyLink({ id, text }: { id: string; text?: boolean }) {
  const [state, setState] = useState<"" | "done" | "show">("");
  const url = shareLink(id);
  const copy = async () => {
    try { await navigator.clipboard.writeText(url); setState("done"); setTimeout(() => setState((s) => (s === "done" ? "" : s)), 2000); }
    catch { setState("show"); }
  };
  const label = state === "done" ? t("community.linkCopied") : t("community.copyLink");
  return <>
    {text
      ? <button type="button" class="btn text cm-copy-link" onClick={() => void copy()}><Icon.link />{label}</button>
      : <button type="button" class={"btn icon cm-copy-link" + (state === "done" ? " on" : "")} data-tip={label} aria-label={label} onClick={() => void copy()}><Icon.link /></button>}
    {state === "show" && <input class="cm-link-field" readOnly value={url} aria-label={t("community.copyLink")} ref={(el) => el?.select()} onFocus={(e) => (e.target as HTMLInputElement).select()} />}
  </>;
}

function paintShare(canvas: HTMLCanvasElement, s: DecodedShare, cx: number, cy: number, scale: number) {
  const dpr = devicePixelRatio || 1, w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) (canvas.width = w), (canvas.height = h);
  const ctx = canvas.getContext("2d");
  if (!ctx || !w || !h) return;
  const th = { ...diagramTheme(), border: undefined, water: undefined };
  const d = widened(s.d, Math.max(1, MIN_LINE_PX / (s.d.stroke * scale)));
  paint(ctx, d, w, h, cx, cy, scale * dpr, th);
  paintNames(ctx, s.names, s.d, w, h, cx, cy, scale * dpr, dpr, { ink: th.text, halo: th.bg, font: th.font });
}

/** A drawing for a row, null while it comes; `failed` when it could not be read. */
function useDrawing(row: ShareRow): { s: DecodedShare | null; failed: boolean } {
  const [state, setState] = useState<{ key: string; s: DecodedShare | null; failed: boolean }>({ key: "", s: null, failed: false });
  const key = `${row.id}/${row.rev}`;
  useEffect(() => {
    let live = true;
    loadDrawing(row).then((s) => { if (live) setState({ key, s, failed: false }); }, () => { if (live) setState({ key, s: null, failed: true }); });
    return () => { live = false; };
  }, [key]);
  return state.key === key ? state : { s: null, failed: false };
}

function PointsLine({ row }: { row: ShareRow }) {
  const city = row.city ? cityName(row.city) : "", points = row.points > 0 ? t("leaderboard.value", { points: pointsShort(row.points) }) : "";
  return <div class="small muted num cm-facts">
    {city && <span>{city}</span>}
    {points && <span>{points}</span>}
    {row.sandbox && <span class="save-sandbox small">{t("savesPanel.sandbox")}</span>}
  </div>;
}

function Likes({ row }: { row: ShareRow }) {
  return <span class="cm-likes small num" data-tip={t("community.likes", { count: row.likes })}><Icon.heart />{row.likes}</span>;
}

/** T-3144: the title in bold, then the sharer's name in grey; a share from before titles has the name alone */
function Heading({ row }: { row: ShareRow }) {
  if (!row.title) return <strong>{row.name || t("community.anon")}</strong>;
  return <span class="cm-heading"><strong>{row.title}</strong>{row.name && <span class="cm-by">{row.name}</span>}</span>;
}

function Card({ row, onOpen }: { row: ShareRow; onOpen: () => void }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const { s, failed } = useDrawing(row);
  const sh = shade.value, th = colourTheme.value;
  useEffect(() => {
    const c = canvas.current;
    if (!c || !s) return;
    const draw = () => { const v = fitView(s.d, c.clientWidth, c.clientHeight); paintShare(c, s, v.cx, v.cy, v.scale); };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(c);
    return () => ro.disconnect();
  }, [s, sh, th]);
  return <li class="cm-card">
    <button type="button" class="cm-open" onClick={onOpen} aria-label={row.title || row.name || t("community.anon")}>
      <canvas ref={canvas} class="cm-thumb" />
      {!s && <span class="cm-note small muted">{failed ? t("community.errLoad") : t("leaderboard.loading")}</span>}
      <div class="cm-card-info">
        <div class="cm-card-name">
          {row.rank !== undefined && <span class="lb-rank num">{row.rank}</span>}
          <Heading row={row} />
          <Likes row={row} />
        </div>
        <PointsLine row={row} />
        <div class="small muted num">{t("savesPanel.lines", { count: row.lines })} · {t("savesPanel.stations", { count: row.stations })}</div>
      </div>
    </button>
  </li>;
}

const MIN_PITCH_PX = 0.2, MAX_PITCH_PX = 80;

/** One network, to pan and zoom. */
function Detail({ row, onBack, onChange }: { row: ShareRow; onBack: () => void; onChange: (r: ShareRow) => void }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const { s, failed } = useDrawing(row);
  const sh = shade.value, th = colourTheme.value;
  const view = useRef({ cx: 0, cy: 0, scale: 0 });
  const [msg, setMsg] = useState("");
  const [reporting, setReporting] = useState(false);
  const [reported, setReported] = useState(false);
  useEffect(() => {
    const c = canvas.current;
    if (!c || !s) return;
    let frame = 0;
    const paintNow = () => { frame = 0; paintShare(c, s, view.current.cx, view.current.cy, view.current.scale); };
    const draw = () => { frame ||= requestAnimationFrame(paintNow); };
    const fit = () => { view.current = fitView(s.d, c.clientWidth, c.clientHeight, 24); };
    if (!view.current.scale) fit();
    const clamp = (sc: number) => Math.min(MAX_PITCH_PX / s.d.pitch, Math.max(MIN_PITCH_PX / s.d.pitch, sc));
    const zoomAt = (px: number, py: number, f: number) => {
      const v = view.current, next = clamp(v.scale * f), r = c.getBoundingClientRect(), ox = px - r.left - r.width / 2, oy = py - r.top - r.height / 2;
      v.cx += ox / v.scale - ox / next;
      v.cy -= oy / v.scale - oy / next;
      v.scale = next;
      draw();
    };
    const pointers = new Map<number, [number, number]>();
    let pinch = 0;
    const down = (e: PointerEvent) => { pointers.set(e.pointerId, [e.clientX, e.clientY]); c.setPointerCapture(e.pointerId); pinch = 0; };
    const move = (e: PointerEvent) => {
      const was = pointers.get(e.pointerId);
      if (!was) return;
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      if (pointers.size === 1) {
        view.current.cx -= (e.clientX - was[0]) / view.current.scale;
        view.current.cy += (e.clientY - was[1]) / view.current.scale;
        draw();
      } else if (pointers.size === 2) {
        const [[ax, ay], [bx, by]] = [...pointers.values()], d = Math.hypot(ax - bx, ay - by);
        if (pinch) zoomAt((ax + bx) / 2, (ay + by) / 2, d / pinch);
        pinch = d;
      }
    };
    const up = (e: PointerEvent) => { pointers.delete(e.pointerId); pinch = 0; };
    const wheel = (e: WheelEvent) => { e.preventDefault(); zoomAt(e.clientX, e.clientY, Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015))); };
    c.addEventListener("pointerdown", down);
    c.addEventListener("pointermove", move);
    c.addEventListener("pointerup", up);
    c.addEventListener("pointercancel", up);
    c.addEventListener("wheel", wheel, { passive: false });
    const ro = new ResizeObserver(draw);
    ro.observe(c);
    draw();
    return () => {
      if (frame) cancelAnimationFrame(frame);
      ro.disconnect();
      c.removeEventListener("pointerdown", down);
      c.removeEventListener("pointermove", move);
      c.removeEventListener("pointerup", up);
      c.removeEventListener("pointercancel", up);
      c.removeEventListener("wheel", wheel);
    };
  }, [s, sh, th]);
  /** signed out: the account panel, on top of this one */
  const needSignIn = () => { if (session.peek()) return false; accountOpen.value = true; return true; };
  const like = async () => {
    if (needSignIn()) return;
    setMsg("");
    try { const r = await setLiked(row.id, !row.liked); onChange({ ...row, liked: r.liked, likes: r.likes }); }
    catch (e) { setMsg((e as Error).message); }
  };
  const report = async (reason: ReportReason) => {
    setMsg("");
    try { await reportShare(row.id, reason); setReporting(false); setReported(true); }
    catch (e) { setMsg((e as Error).message); }
  };
  const reasons: [ReportReason, string][] = [["offensive", t("community.offensive")], ["fake-score", t("community.fakeScore")], ["other", t("community.somethingElse")]];
  const likeLabel = row.liked ? t("community.unlike") : t("community.like");
  const [opening, setOpening] = useState(false);
  // T-3115: a new game of the viewer's own, named after whose network it is; the page reloads into it
  const openCopy = async () => {
    setMsg("");
    setOpening(true);
    const why = await openSharedCopy(row, copyName(row) || (row.city ? cityName(row.city) : "") || t("savesPanel.copyMark"));
    if (why) { setMsg(why); setOpening(false); }
  };
  return <div class="cm-detail">
    <div class="cm-detail-head">
      <button type="button" class="btn icon" data-tip={t("community.back")} aria-label={t("community.back")} onClick={onBack}><Icon.goBack /></button>
      <div class="cm-detail-who">
        <Heading row={row} />
        <PointsLine row={row} />
        <div class="small muted num">{t("savesPanel.lines", { count: row.lines })} · {t("savesPanel.stations", { count: row.stations })}</div>
      </div>
      <div class="cm-detail-acts">
        {row.mine && <button type="button" class="btn text cm-share-again" onClick={() => setMsg(shareAgain(row))}><Icon.share />{t("community.again")}</button>}
        {row.copy && <button type="button" class="btn text cm-open-copy" disabled={opening} onClick={() => void openCopy()}><Icon.copy />{opening ? t("leaderboard.loading") : t("community.openCopy")}</button>}
        <CopyLink id={row.id} />
        {!row.mine && (reported
          ? <span class="small muted">{t("community.reported")}</span>
          : <button type="button" class={"btn text" + (reporting ? " on" : "")} aria-expanded={reporting} onClick={() => { if (!needSignIn()) setReporting(!reporting); }}><Icon.flag />{t("community.report")}</button>)}
        <button type="button" class={"btn text cm-like" + (row.liked || row.mine ? " on" : "")} disabled={row.mine} aria-label={likeLabel} aria-pressed={!!row.liked} data-tip={row.mine ? t("community.errOwn") : undefined} onClick={like}><Icon.heart /><span class="num">{row.likes}</span></button>
      </div>
    </div>
    {reporting && <div class="cm-reasons">{reasons.map(([r, label]) => <button type="button" class="btn text" onClick={() => void report(r)}>{label}</button>)}</div>}
    {msg && <p class="acct-msg bad">{msg}</p>}
    <div class="cm-canvas-box">
      <canvas ref={canvas} class="cm-canvas" style={{ touchAction: "none" }} />
      {!s && <span class="cm-note small muted">{failed ? t("community.errLoad") : t("leaderboard.loading")}</span>}
    </div>
  </div>;
}

function Browse() {
  const [view, setView] = useState<ShareView>("random");
  const [boardKey, setBoardKey] = useState(WORLD);
  const [seed, setSeed] = useState(newSeed);
  const [rows, setRows] = useState<ShareRow[] | null>(null);
  const [error, setError] = useState("");
  const [page, setPage] = useState(0);
  const [picked, setPicked] = useState<ShareRow | null>(null);
  // T-3115: a `?share=` link opens on its share; one that is not there says so above the Random page
  const [linkError, setLinkError] = useState("");
  const [linkWait, setLinkWait] = useState(false);
  const linked = linkedShare.value;
  useEffect(() => {
    if (!linked) return;
    linkedShare.value = null;
    setLinkWait(true);
    fetchShare(linked).then(setPicked, (e) => setLinkError((e as Error).message)).finally(() => setLinkWait(false));
  }, [linked]);
  const signedIn = session.value?.account.id ?? "";
  const city = placeParam(boardKey);
  // the player's own shares, for Share again on an opened one
  useEffect(() => { void refreshMyShares(); }, [signedIn]);
  useEffect(() => {
    let live = true;
    setRows(null);
    setError("");
    setPage(0);
    fetchShares(view, city, seed).then((l) => { if (live) setRows(l.rows); }, (e) => { if (live) setError((e as Error).message); });
    return () => { live = false; };
  }, [view, city, seed, signedIn, shareTick.value]);
  const mine = [...pointsPast24h.value].filter(([id, v]) => id && v > 0).sort((a, b) => b[1] - a[1]).map(([id]) => id);
  const lands = [...new Set(mine.map((id) => zoneCountry(id)?.toLowerCase() ?? "").filter(Boolean))];
  const areas = [...new Set(mine.map(regionOf).filter((r): r is string => !!r))];
  const change = (r: ShareRow) => { setPicked(r); setRows((all) => all && all.map((x) => (x.id === r.id ? r : x))); };
  if (picked) return <Detail key={picked.id} row={picked} onBack={() => setPicked(null)} onChange={change} />;
  if (linked || linkWait) return <p class="small muted">{t("leaderboard.loading")}</p>;
  const list = rows ?? [], pages = Math.ceil(list.length / PAGE), shown = list.slice(page * PAGE, (page + 1) * PAGE);
  const views: [ShareView, string][] = [["random", t("community.random")], ["top", t("community.biggest")], ["liked", t("community.mostLiked")]];
  return <>
    <div class="cm-bar">
      <div class="cm-views" role="tablist">
        {views.map(([v, label]) => <button type="button" role="tab" aria-selected={view === v} class={"btn text" + (view === v ? " on" : "")} onClick={() => setView(v)}>{label}</button>)}
      </div>
      <div class="cm-where">
        <span>{t("community.in")}</span>
        <CityPick boardKey={boardKey} mine={mine} all={ALL_CITIES} countries={lands} more={{ ...MORE_PLACES, regions: areas }} onPick={setBoardKey} />
      </div>
      {view === "random" && <button type="button" class="btn text cm-shuffle" onClick={() => setSeed(newSeed())}>{t("community.shuffle")}</button>}
    </div>
    {linkError && <p class="acct-msg bad">{linkError}</p>}
    {error ? <p class="acct-msg bad">{error}</p>
      : !rows ? <p class="small muted">{t("leaderboard.loading")}</p>
      : !list.length ? <p class="muted">{t("community.empty")}</p>
      : <ul class="cm-cards">{shown.map((r) => <Card key={r.id + r.rev} row={r} onOpen={() => setPicked(r)} />)}</ul>}
    {view === "top" && !error && rows && <p class="small muted cm-note-foot">{t("community.sandboxNote")}</p>}
    {pages > 1 && <div class="cm-pager">
      <button type="button" class="btn icon" aria-label={t("community.prev")} data-tip={t("community.prev")} disabled={page === 0} onClick={() => setPage(page - 1)}><Icon.goBack /></button>
      <span class="small muted num">{page + 1} / {pages}</span>
      <button type="button" class="btn icon" aria-label={t("community.next")} data-tip={t("community.next")} disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}><Icon.goOut /></button>
    </div>}
  </>;
}
const ALL_CITIES: ReadonlySet<string> = new Set(CITIES.map((c) => c.id));

export function Community() {
  const open = communityOpen.value && communityOn();
  return <Modal open={open} onClose={() => (communityOpen.value = false)} title={t("community.heading")} cls="community-panel" head={<ShareButton upload />}>
    {open && <Browse />}
  </Modal>;
}

/** Opens Community: in the head of the saves panel and of the leaderboards, and in Settings beside Controls (T-3132). */
export function CommunityButton({ settings }: { settings?: boolean }) {
  if (!communityOn()) return null;
  const cls = settings ? "btn icon" + (communityOpen.value ? " on" : "") : "btn icon save-head-mark cm-head";
  return <button type="button" class={cls} data-tip={t("community.title")} aria-label={t("community.title")} onClick={() => (communityOpen.value = true)}><Icon.people /></button>;
}

/** Share on the game being played: on its card in the saves panel and beside Save picture on the schematic with
 * Community's people, and in Community's own head with the upload arrow (`upload`). */
export function ShareButton({ upload }: { upload?: boolean }) {
  if (!communityOn() || !shareableSlot()) return null;
  void myShares.value;
  const row = currentShare(), tip = row ? t("community.lastShared", { when: sharedWhen(row) }) : upload ? t("community.shareOpen") : t("community.share");
  return <button type="button" class={"btn icon cm-share" + (upload ? " save-head-mark" : "") + (row ? " on" : "")} data-tip={tip} aria-label={tip}
    onClick={() => { if (session.peek()) shareAsk.value = true; else accountOpen.value = true; }}>{upload ? <Icon.share /> : <Icon.people />}</button>;
}

function ShareAsk() {
  const [name, setName] = useState("");
  const [ask, setAsk] = useState(false);
  const [msg, setMsg] = useState("");
  const [failed, setFailed] = useState(false);
  const [busy, setBusy] = useState(false);
  const setting = boardSetting.value;
  const row = currentShare(), left = sharesLeft.value;
  useEffect(() => { void loadBoardSetting(); void refreshMyShares(); }, []);
  const needName = ask || (setting !== null && !setting.name);
  const phase = sharePhase.value, progress = diagramShareProgress.value;
  // T-3115: Let others open a copy, ticked unless the share went up without it
  const [copyPick, setCopy] = useState<boolean | null>(null);
  const copy = copyPick ?? (row ? !!row.copy : true);
  const run = async (f: () => Promise<void>, done: string) => {
    setBusy(true);
    setMsg("");
    setFailed(false);
    try { await f(); setMsg(done); }
    catch (e) {
      setFailed(true);
      setMsg((e as Error).message);
      const code = (e as { code?: string }).code;
      if (code === "need-name" || code === "bad-name") setAsk(true);
    } finally { setBusy(false); }
  };
  // T-3144: the title starts as the game's name (or what the last share used) and may be changed
  const st = saveSlotsState.value, gameName = st.slots.find((s) => s.key === st.active)?.name ?? "";
  const [titlePick, setTitle] = useState<string | null>(null);
  const title = titlePick ?? row?.title ?? gameName;
  const go = () => run(() => shareGame(needName ? name.trim() : "", copy, title), t("community.shared"));
  return <>
    <p>{t("community.ask")}</p>
    {row && <p>{t("community.lastSharedOn", { when: sharedWhen(row) })}</p>}
    {left && <p>{left.left === left.perDay ? t("community.sharesToday", { count: left.left }) : t("community.sharesLeft", { count: left.left })}</p>}
    {row?.hidden ? <p class="small muted">{t("community.hidden")}</p> : null}
    <input maxLength={TITLE_MAX} placeholder={gameName || t("community.shareTitle")} aria-label={t("community.shareTitle")} value={title} disabled={busy} onInput={(e) => setTitle((e.target as HTMLInputElement).value)} />
    {needName && <input maxLength={24} placeholder={t("leaderboard.name")} aria-label={t("leaderboard.name")} value={name} disabled={busy} onInput={(e) => setName((e.target as HTMLInputElement).value)} />}
    <label class="check cm-copy-tick">
      <input type="checkbox" checked={copy} disabled={busy} onChange={(e) => setCopy((e.target as HTMLInputElement).checked)} />
      <span>{t("community.letCopy")}</span>
    </label>
    <div class="cm-ask-foot">
      <button type="button" class={"btn text" + (row ? "" : " save-open")} disabled={busy || (needName && !name.trim())} onClick={() => void go()}>{row ? t("community.again") : t("community.share")}</button>
      {row && <button type="button" class="btn text danger" disabled={busy} onClick={() => void run(() => stopSharing(row.id), t("community.stopped"))}>{t("community.stop")}</button>}
      {row && !row.hidden && !busy && <CopyLink id={row.id} text />}
      {!busy && <button type="button" class="btn icon cm-ask-browse" data-tip={t("community.title")} aria-label={t("community.title")} onClick={() => { shareAsk.value = false; communityOpen.value = true; }}><Icon.people /></button>}
    </div>
    {phase === "drawing" && <div class="cm-progress">
      <span class="small muted">{t("diagramButton.calculating")}</span>
      {progress !== null && <LoadBar fraction={progress} step={t("diagramButton.calculating")} thin tip={`${Math.floor(progress * 100)}%`} />}
    </div>}
    {phase === "sending" && <p class="small muted cm-progress">{t("community.sending")}</p>}
    {msg && <p class={"acct-msg" + (failed ? " bad" : "")}>{msg}</p>}
  </>;
}

/** The Share dialog. */
export function ShareDialog() {
  const open = shareAsk.value && communityOn();
  return <Modal open={open} onClose={() => (shareAsk.value = false)} title={t("community.share")} cls="acct-panel cm-ask">
    {open && <ShareAsk />}
  </Modal>;
}
