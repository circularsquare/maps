// T-2810: the list of what changed in each update, a Modal; opened from the start menu's update row and
// from Settings (T-3220). T-3466: each item has its area's icon, the list sorted by area, bug fixes lighter.
// T-3468: one page per x.y version (v0.3 holds v0.3, v0.3.1, ...), the newest open, links to each at the top; only
// the page shown is rendered, and nothing while closed.

import type { ComponentChildren, JSX } from "preact";
import { useRef, useState } from "preact/hooks";
import { Modal } from "./Modal";
import { AREAS, UPDATES, type Area, type Change, type Update } from "./updates";
import { Icon, Segmented } from "./widgets";
import { mobileDevice } from "../workers/deviceMemory";
import { t, tDate, tRich } from "../i18n";

const AREA_ICON: Record<Area, () => JSX.Element> = {
  track: Icon.track,
  demand: Icon.bubble,
  scheduling: Icon.marey,
  fares: Icon.money,
  schematic: Icon.foldMap,
  community: Icon.person,
  performance: Icon.computer,
  other: () => <svg viewBox="0 0 16 16"><circle class="solid" cx="8" cy="8" r="2" /></svg>,
  bug: Icon.bug,
};

/** by area, an area's fixes after its other items, otherwise as listed */
function sorted(changes: Change[]): Change[] {
  const rank = (c: Change) => AREAS.indexOf(c[1]) * 2 + (c[2] ? 1 : 0);
  return changes.map((c, i) => [c, i] as const).sort((a, b) => rank(a[0]) - rank(b[0]) || a[1] - b[1]).map(([c]) => c);
}

/** "v0.3" for v0.3, v0.3.1, v0.3.5 */
const pageOf = (version: string) => version.split(".").slice(0, 2).join(".");
/** the pages, newest first, each with its updates (UPDATES is newest first) */
const PAGES: [string, Update[]][] = [...UPDATES.reduce((m, u) => m.set(pageOf(u.version), [...(m.get(pageOf(u.version)) ?? []), u]), new Map<string, Update[]>())];

/** T-3585: *word* and **word** in an item's text are bold; a lone asterisk stays as typed. Everything else is plain
 * text (Preact escapes it), so item text can't inject markup. */
const BOLD_RE = /\*\*(?=\S)([^*\n]*?\S)\*\*|\*(?=\S)([^*\n]*?\S)\*/g;
export function withBold(s: string): ComponentChildren[] {
  const out: ComponentChildren[] = [];
  let at = 0;
  for (const m of s.matchAll(BOLD_RE)) {
    if (m.index! > at) out.push(s.slice(at, m.index));
    out.push(<strong>{m[1] ?? m[2]}</strong>);
    at = m.index! + m[0].length;
  }
  if (at < s.length) out.push(s.slice(at));
  return out;
}
const boldAll = (c: ComponentChildren): ComponentChildren => (Array.isArray(c) ? c.map(boldAll) : typeof c === "string" ? withBold(c) : c);

export function Changelog({ open, onClose }: { open: boolean; onClose: () => void }) {
  if (!open) return null;
  return <Modal open={open} onClose={onClose} title={t("startMenu.changelog")} cls="changelog"><Pages /></Modal>;
}

function Pages() {
  const [page, setPage] = useState(PAGES[0]?.[0] ?? "");
  const top = useRef<HTMLDivElement>(null);
  const updates = PAGES.find(([p]) => p === page)?.[1] ?? [];
  return <>
    {PAGES.length > 1 && <div class="changelog-pages" ref={top}>
      <Segmented options={PAGES.map(([p]) => [p, p])} value={page} onChange={(p) => { setPage(p); top.current?.closest(".saves-inner")?.scrollTo(0, 0); }} />
    </div>}
    {updates.map((u) => <section key={u.version}>
      <h3 class="changelog-head">{u.version}<span class="muted small">{tDate(Date.parse(u.at), { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" })}</span></h3>
      <ul class="changelog-list">{sorted(u.changes).filter(([k]) => !(mobileDevice() && k === "updates.v01DeleteKeys")).map(([k, area, fix]) => {
        const fig = u.nerfs?.[k];
        return <li key={k} class={fix ? "fix" : undefined}>
          <span class={`cl-area cl-${area}`} aria-hidden="true">{AREA_ICON[area]()}</span>
          <div>{fig ? boldAll(tRich(k, { figure: <span class="nerf">{t(fig)}</span> })) : withBold(t(k))}{u.sub?.[k] && <ul>{u.sub[k]!.map((s) => <li key={s}>{withBold(t(s))}</li>)}</ul>}</div>
        </li>;
      })}</ul>
    </section>)}
  </>;
}
