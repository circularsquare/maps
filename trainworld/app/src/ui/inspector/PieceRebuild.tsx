// T-3408: the Select bar's rebuild (game/rebuild.ts) for the one built piece in the track panel: single or
// double track, and regular or high-speed track (rail only: one piece cannot change family alone, and the level
// stays with the Select bar). Same quote, hover, question and edit as the bar's pick of that piece.

import type { ComponentChildren } from "preact";
import { useEffect, useState } from "preact/hooks";
import { t } from "../../i18n";
import type { TrackEdge } from "../../game/types";
import { canRebuild, fetchRebuildQuote, rebuildPieces, rebuildTipFor, type RebuildChoice, type RebuildQuote } from "../../game/rebuild";
import { HSR_COST } from "../../game/trackKinds";
import { world } from "../../game/world";
import { mobileDevice } from "../../workers/deviceMemory";
import { TOOLS } from "../toolKeys";
import { Actions, Icon, TrackCountButtons } from "../widgets";

type Kind = "regular" | "hsr";

export function PieceRebuild({ e, children }: { e: TrackEdge; children: ComponentChildren }) {
  const [kind, setKind] = useState<Kind | null>(null);
  const [tracks, setTracks] = useState<1 | 2 | null>(null);
  // the quote with the choice it is for, so a quote still on its way for an older choice is never offered
  const [quoted, setQuoted] = useState<{ key: string; q: RebuildQuote } | null>(null);
  const version = world.value?.version;
  const choice: RebuildChoice = { level: null, kind, tracks };
  const key = `${kind}/${tracks}`;
  const target = { edges: [e.id], stations: [] };
  useEffect(() => {
    if (kind === null && tracks === null) return;
    let on = true;
    void fetchRebuildQuote(target, choice).then((q) => { if (on && q) setQuoted({ key, q }); });
    return () => { on = false; };
  }, [e.id, key, version]);
  const q = quoted?.key === key && (kind !== null || tracks !== null) ? quoted.q : null;
  const ownKind: Kind = e.hsr ? "hsr" : "regular";
  const ownTracks = e.tracks === 1 ? 1 : 2;
  // picking what the piece has already drops the choice, as the bar's track count does
  const pickKind = (k: Kind) => setKind(k === ownKind ? null : k);
  const pickTracks = (n: 1 | 2) => setTracks(n === ownTracks ? null : n);
  const shownKind = kind ?? ownKind;
  const selectKey = TOOLS.find(([id]) => id === "blueprint")?.[3] ?? "4";
  // the choices, then Rebuild on the panel's row of actions with `children` (Remove track) on the right; greyed
  // Rebuild has no hover, since the bar's ("Pick a level, ...") names a level this panel has no buttons for
  return (
    <>
      <div class="piece-rebuild">
        <TrackCountButtons value={tracks ?? ownTracks} onPick={pickTracks} />
        {!e.kind && (
          <div class="group" role="group" aria-label={t("blueprint.trackType")}>
            <button class={"btn" + (shownKind === "regular" ? " on" : "")} aria-pressed={shownKind === "regular"} onClick={() => pickKind("regular")}>{t("blueprint.regular")}</button>
            <button class={"btn" + (shownKind === "hsr" ? " on" : "")} aria-pressed={shownKind === "hsr"} data-tip={t("blueprint.hsrTip", { factor: HSR_COST })} onClick={() => pickKind("hsr")}>{t("blueprint.highSpeed")}</button>
          </div>
        )}
      </div>
      <Actions>
        <button class="btn text" data-tip={q ? rebuildTipFor(q, choice, 1) : undefined} disabled={!canRebuild(q)}
          onClick={() => rebuildPieces(target, choice, q, () => { setKind(null); setTracks(null); })}>
          <Icon.wrench />{t("blueprint.rebuild")}
        </button>
        {children}
      </Actions>
      <p class="small muted piece-rebuild-hint">{t("trackInspector.rebuildHint", { touch: mobileDevice(), key: selectKey })}</p>
    </>
  );
}
