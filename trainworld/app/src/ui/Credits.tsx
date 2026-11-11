// T-924's data credits live on the game's about page, which lists every source and the notices their
// licences ask for (Anita, 2026-11-06: the list in Settings shortened to a link). The map's own
// OpenStreetMap / OpenMapTiles attribution stays on the map (map/mapView.ts).

import { useState } from "preact/hooks";
import { t } from "../i18n";
import { Changelog } from "./Changelog";

// absolute, so it works from dev servers and 8800 too
const ABOUT_URL = "https://anita.garden/projects/trainsworld/";

/** The game's about page on Anita's site, in a new tab so the game stays open. */
export function AboutLink({ class: cls }: { class?: string }) {
  return <a class={cls} href={ABOUT_URL} target="_blank" rel="noopener">{t("common.aboutGame")}</a>;
}

/** The foot of Settings: the about page, which has the data sources, and the changelog (T-3220). */
export function Credits() {
  const [changelog, setChangelog] = useState(false);
  return (
    <div class="pane-foot">
      <AboutLink class="btn text" />
      <button class="btn text" onClick={() => setChangelog(true)}>{t("startMenu.changelog")}</button>
      <Changelog open={changelog} onClose={() => setChangelog(false)} />
    </div>
  );
}
