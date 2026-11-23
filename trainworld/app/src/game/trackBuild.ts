// T-4126: the track branch's build that 8800 serves beside the main one (tools/ops/snaptrack.py builds it with
// TW_CHANNEL=track; http://localhost:8800/trainworld/?track opens it). Both are on one origin, so the track build keeps
// its saves, settings and locks under names of its own and has no account: a game it saves never reaches the main
// build's saves list or the player's account.

declare const __TW_TRACK__: boolean | undefined;
export const TRACK_BUILD = typeof __TW_TRACK__ !== "undefined" && __TW_TRACK__ === true;
/** The saves database: the main build's, or the track build's own. */
export const SAVE_DB = TRACK_BUILD ? "anitabuilder-track" : "anitabuilder";
/** The main build's saves database, which the track build only reads (`copyTrackGames`). */
export const MAIN_SAVE_DB = "anitabuilder";
