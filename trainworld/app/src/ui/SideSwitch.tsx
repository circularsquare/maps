// T-3297: the running side, one per game for trains, trams and buses: picked on the new game sheet (set from the start
// country, game/runningSide.ts), and switched in Settings once the game has track, after a question, with a list of
// what the switch fixed and what it left to fix, each row a click from the place.

import type { ComponentChildren } from "preact";
import { useState } from "preact/hooks";
import { ask, selection, tool } from "../game/ui";
import { flipSites, rightHand, stationsById, world } from "../game/world";
import type { Selection, WorldView } from "../game/types";
import { flipSide } from "../workers/clockClient";
import { locateLine, locateNodes } from "./locate";
import { Modal } from "./Modal";
import { Chip, Segmented } from "./widgets";
import { t } from "../i18n";

/** The two sides as a pair of buttons with the row's label. */
export function SideSwitch({ right, onPick, tip, cls = "slider" }: { right: boolean; onPick: (right: boolean) => void; tip?: string; cls?: string }) {
  return (
    <div class={cls + " side-switch"} data-tip={tip}>
      <span>{t("sideSwitch.label")}</span>
      <Segmented<"left" | "right"> options={[["left", t("sideSwitch.left")], ["right", t("sideSwitch.right")]]} value={right ? "right" : "left"} onChange={(v) => onPick(v === "right")} />
    </div>
  );
}

/** One place the last switch fixed or left: `kind` as the sim's `FLIP_*`, `id` its edge, line or node. */
interface Site { kind: number; id: number }
const FIXED = new Set([1, 2, 6]);

/** Settings' row: the game's side; picking the other asks first once there is track, then shows the list. */
export function SideRow() {
  const [open, setOpen] = useState(false);
  const right = rightHand.value;
  const w = world.value;
  const last = flipSites.value;
  const toFix = last ? last.sites.filter((s) => !FIXED.has(s.kind)).length : 0;
  const flip = async () => {
    const out = await flipSide();
    if (!out) return;
    const sites: Site[] = [];
    for (let i = 0; i + 1 < out.length; i += 2) sites.push({ kind: out[i], id: out[i + 1] });
    flipSites.value = { right: !right, sites };
    if (sites.length) setOpen(true);
  };
  const pick = (to: boolean) => {
    if (to === right) return;
    if (!w || !w.edges.length) { void flip(); return; }
    ask.value = { text: t("sideSwitch.ask", { side: to ? "right" : "left" }), yes: t("sideSwitch.switch"), no: t("sideSwitch.keep"), run: () => void flip() };
  };
  return (
    <>
      <SideSwitch right={right} onPick={pick} />
      {last && last.sites.length > 0 && (
        <div class="game-mode">
          <span class="muted">{toFix ? t("sideSwitch.toFixCount", { count: toFix }) : t("sideSwitch.allFixed")}</span>
          <button class="btn text" data-side-sites onClick={() => setOpen(true)}>{t("sideSwitch.showSites")}</button>
        </div>
      )}
      {open && last && <SiteList right={last.right} sites={last.sites} onClose={() => setOpen(false)} />}
    </>
  );
}

const pick = (s: Selection) => { if (tool.peek() === "select") selection.value = s; };

/** A site's row: what it is, and where a click takes the map. */
function row(w: WorldView, s: Site): { label: ComponentChildren; go: () => void } | null {
  if (s.kind === 2 || s.kind === 3) {
    const l = w.save.lines.find((x) => x.id === String(s.id));
    if (!l) return null;
    return {
      label: <><Chip line={l} size="sm" plain /> {t(s.kind === 2 ? "sideSwitch.moved" : "sideSwitch.broken", { name: l.name })}</>,
      go: () => { locateLine(l.stops); pick({ kind: "line", line: l.id }); },
    };
  }
  if (s.kind === 1) {
    const e = w.edges.find((x) => x.id === s.id);
    if (!e) return null;
    return { label: t("sideSwitch.reversed"), go: () => { locateNodes([e.a, e.b]); pick({ kind: "track", edge: e.id }); } };
  }
  if (!w.nodes.has(s.id)) return null;
  // T-3332: a one-track station keeps its platform's track, which now carries the other way
  if (s.kind === 6) {
    const st = stationsById(w).get(String(s.id));
    if (!st) return null;
    return { label: t("sideSwitch.oneTrack", { name: st.name }), go: () => { locateNodes([s.id]); pick({ kind: "station", station: String(s.id) }); } };
  }
  return {
    label: t(s.kind === 4 ? "sideSwitch.flat" : "sideSwitch.unusable"),
    go: () => { locateNodes([s.id]); if (s.kind === 4) pick({ kind: "junction", node: s.id }); },
  };
}

/** What the switch fixed and what it left, each a click from its place (the list closes on a click). */
function SiteList({ right, sites, onClose }: { right: boolean; sites: Site[]; onClose: () => void }) {
  const w = world.value;
  if (!w) return null;
  const rows = (fixed: boolean) => sites.filter((s) => FIXED.has(s.kind) === fixed).map((s) => ({ s, r: row(w, s) })).filter((x) => x.r);
  const left = rows(false), done = rows(true);
  const list = (items: typeof left) => (
    <ul class="side-sites">
      {items.map(({ s, r }) => <li key={`${s.kind}-${s.id}`}><button class="btn text" onClick={() => { r!.go(); onClose(); }}>{r!.label}</button></li>)}
    </ul>
  );
  return (
    <Modal open onClose={onClose} title={t("sideSwitch.done", { side: right ? "right" : "left" })} cls="side-sites-panel">
      <div class="h">{t("sideSwitch.toFix")}</div>
      {left.length ? list(left) : <p class="muted">{t("sideSwitch.nothing")}</p>}
      {done.length > 0 && <><div class="h">{t("sideSwitch.fixed")}</div>{list(done)}</>}
    </Modal>
  );
}
