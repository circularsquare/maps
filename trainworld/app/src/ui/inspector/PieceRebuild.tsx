// T-3408: the Select bar's rebuild (game/rebuild.ts) for the one built piece in the track panel: its level,
// single or double track, and regular, high-speed or tram track. Same quote, hover, question and edit as the bar's
// pick of that piece. A level or a change of family (rail and tram) is offered only where the track model would
// take it for this piece alone: the track joined to it often ties both.

import type { ComponentChildren } from "preact";
import { useEffect, useState } from "preact/hooks";
import { levelLabel, levelName } from "../../game/format";
import { t, lowerWord } from "../../i18n";
import { LEVELS, type Level, type TrackEdge } from "../../game/types";
import { canRebuild, fetchRebuildQuote, rebuildPieces, rebuildTipFor, rebuildWhole, type RebuildChoice, type RebuildKind, type RebuildQuote } from "../../game/rebuild";
import { HSR_COST } from "../../game/trackKinds";
import { world } from "../../game/world";
import { mobileDevice } from "../../workers/deviceMemory";
import { TOOLS } from "../toolKeys";
import { Actions, Icon, LevelButtons, TrackCountButtons } from "../widgets";

const NONE: RebuildChoice = { level: null, kind: null, tracks: null };

export function PieceRebuild({ e, children }: { e: TrackEdge; children: ComponentChildren }) {
  const [choice, setChoice] = useState<RebuildChoice>(NONE);
  // the quote with the choice it is for, so a quote still on its way for an older choice is never offered
  const [quoted, setQuoted] = useState<{ key: string; q: RebuildQuote } | null>(null);
  // the levels and the other family's kinds this piece can take alone
  const [open, setOpen] = useState<{ levels: Level[]; kinds: RebuildKind[] } | null>(null);
  const version = world.value?.version;
  const key = `${choice.level}/${choice.kind}/${choice.tracks}`;
  const chosen = choice.level !== null || choice.kind !== null || choice.tracks !== null;
  const target = { edges: [e.id], stations: [] };
  const ownLevel = e.levelMin === e.levelMax ? e.levelMin : null;
  const ownKind: RebuildKind = e.kind ? "tram" : e.hsr ? "hsr" : "regular";
  const ownTracks = e.tracks === 1 ? 1 : 2;
  useEffect(() => {
    if (!chosen) return;
    let on = true;
    void fetchRebuildQuote(target, choice).then((q) => { if (on && q) setQuoted({ key, q }); });
    return () => { on = false; };
  }, [e.id, key, version]);
  useEffect(() => {
    let on = true;
    void (async () => {
      const levels: Level[] = [], kinds: RebuildKind[] = [];
      for (const level of LEVELS) {
        if (level === ownLevel) continue;
        if (rebuildWhole(await fetchRebuildQuote(target, { ...NONE, level }))) levels.push(level);
        if (!on) return;
      }
      for (const kind of (e.kind ? ["regular", "hsr"] : ["tram"]) as RebuildKind[]) {
        if (rebuildWhole(await fetchRebuildQuote(target, { ...NONE, kind }))) kinds.push(kind);
        if (!on) return;
      }
      setOpen({ levels, kinds });
    })();
    return () => { on = false; };
  }, [e.id, version]);
  const q = quoted?.key === key && chosen ? quoted.q : null;
  // picking what the piece has already drops that choice, as the bar's track count does
  const pick = (c: Partial<RebuildChoice>) => setChoice({
    level: c.level === undefined ? choice.level : c.level === ownLevel ? null : c.level,
    kind: c.kind === undefined ? choice.kind : c.kind === ownKind ? null : c.kind,
    tracks: c.tracks === undefined ? choice.tracks : c.tracks === ownTracks ? null : c.tracks,
  });
  const shownKind = choice.kind ?? ownKind;
  // the piece's own family's kinds, and the other family's where open; the group only when it offers a change
  const kinds = (["regular", "hsr", "tram"] as RebuildKind[]).filter((k) => (k === "tram") === !!e.kind || open?.kinds.includes(k));
  const levels = open?.levels.length ? LEVELS.filter((l) => l === ownLevel || open.levels.includes(l)) : [];
  const levelTip = (l: Level) => t("blueprint.levelTip", { level: lowerWord(levelName(l)), label: levelLabel(l) });
  const selectKey = TOOLS.find(([id]) => id === "blueprint")?.[3] ?? "4";
  const kindLabel = (k: RebuildKind) => t(k === "hsr" ? "blueprint.highSpeed" : k === "tram" ? "toolBar.tram" : "blueprint.regular");
  // the choices, then Rebuild on the panel's row of actions with `children` (Remove track) on the right; greyed
  // Rebuild has no hover, since the bar's ("Pick a level, ...") may name a level this piece has no buttons for
  return (
    <>
      <div class="piece-rebuild">
        {levels.length > 0 && <LevelButtons value={choice.level ?? ownLevel} levels={levels} tip={levelTip} onPick={(level) => pick({ level })} />}
        <TrackCountButtons value={choice.tracks ?? ownTracks} onPick={(tracks) => pick({ tracks })} />
        {kinds.length > 1 && (
          <div class="group" role="group" aria-label={t("blueprint.trackType")}>
            {kinds.map((k) => (
              <button class={"btn" + (shownKind === k ? " on" : "")} aria-pressed={shownKind === k} data-tip={k === "hsr" ? t("blueprint.hsrTip", { factor: HSR_COST }) : undefined}
                onClick={() => pick({ kind: k })}>{kindLabel(k)}</button>
            ))}
          </div>
        )}
      </div>
      <Actions>
        <button class="btn text" data-tip={q ? rebuildTipFor(q, choice, 1) : undefined} disabled={!canRebuild(q)}
          onClick={() => rebuildPieces(target, choice, q, () => setChoice(NONE))}>
          <Icon.wrench />{t("blueprint.rebuild")}
        </button>
        {children}
      </Actions>
      <p class="small muted piece-rebuild-hint">{t("trackInspector.rebuildHint", { touch: mobileDevice(), key: selectKey })}</p>
    </>
  );
}
