---
name: publish
description: Publish a trainworld update, or do one of the jobs that go with it (shonei-server deploy, open-source sync, PR reply, R2 purge). Run it when an update is about ready, before asking Anita for the publish yes
argument-hint: "[version]"
---

For the trainworld orchestrator. Every step that publishes, deploys or pushes the website happens only on Anita's
word for that exact action, never overnight. The permission check refuses deploys and ssh unless her latest message
says yes to that action. Commands run from the maps root.

## Before asking for the yes

1. **Changelog review.** A fresh Sonnet agent reads every shown item not yet reviewed and asks as a player "would i
   ever care about this / want to read it in a changelog". It hides each no (`changelog.py hide <id>`), may tighten
   wording, and lists what it hid and why in `notes/changelog-review.md` (ids reviewed, so the next pass skips
   them). Tell her what was hidden in one line. Run this also after a big batch of landings.
2. **Sort it** before her final check: by area in the game's order (`AREAS` in tools/changelog.py), within an area
   features before fixes and the biggest first, hidden items last (`changelog.py move <id> --before <id>`).
3. **Highlights** (T-3591): when an update has more than about 20 shown items, pick 1-8 of the biggest changes for a
   player and mark them (`changelog.py highlight <id>`, `--off` to clear; a top-level item or a group). The game
   lists them first, then the rest by area, and she adjusts the star on the board. Under about 20 items, none.
4. **She edits the changelog on the board.** Put the items she will see on the board's Changelog view (hide what
   won't ship, set the start-menu line). She almost always wants to edit it, unless the update has only a couple of
   items. Publishing empties the board's list.
5. **Her OK on new player text.** A publish waits for her OK on every new player-visible line that she hasn't
   already acknowledged, in-game strings and changelog lines alike. Her yes to the release is not an OK on the wording, and neither is the board's changelog
   review. Put the exact strings in the message as the one blocking question, and hold the publish until she
   answers.
6. **Playtest list.** Give her one list covering everything shipping, from the unpublished changelog and the
   landings since the last publish (`git log` since the last publish commit), hidden items included. General steps,
   not exact ones ("make a bus line and add buses", "paint a fare zone, save and reload"), grouped by area, the
   riskiest first, on 8800. An update once went out too hastily without one. Only at this point, not throughout.
7. **New zones image** (T-3614): if the publish adds or grows zones, run
   `venv\Scripts\python.exe trainworld/tools/release/new_zones_map.py --version vX.Y.Z` (about 12 s; it compares the
   live game's zones with data/packs) and give her `artifacts/trainworld/review/<version>-new-zones.png`. Nothing
   new, no image.
8. `checks/smoke8800.cmd` (light then dark; logs `artifacts/trainworld/smoke8800-*.log`).

## Publishing the game

Commands in full: notes/T-924.md "Commands" (`tools/deploy.py --dry-run --version X`, then `--smoked`, website
push, `--verify`). Every publish gets a version and an entry in `app/src/ui/updates.ts`, written from the board's
changelog (T-3424):

1. Re-read the items against what ships.
2. `tools/changelog.py line "update vX.Y.Z: ..."`
3. `changelog.py publish --version vX.Y.Z --dry-run`, then without `--dry-run`. It writes her wording, hidden items
   left out, sorted by area, groups' items under them. It refuses an item with no area: `changelog.py set <id>
   --area ...`.
4. Translate the new `updates.vXYZ*` keys.
5. deploy.py. Its rclone can saturate her Wi-Fi upload and slow her game.

## Wiki pages for the website (T-3447)

**On hold** (her 11-21: a player made a fan wiki; the link and pages are out of the website until she decides; the
last build sits in artifacts/trainworld/wikisite/site-hold). When she picks it up: `venv\Scripts\python.exe trainworld/tools/wiki_site.py` rebuilds the player wiki into the website clone (`projects/trainsworld/wiki/`, the wiki as of the newest release's changelog commit), then tell her it waits in ..\website for her push. Notes: notes/T-3447.md.

## Open-source copy (T-2913, notes/T-2913.md)

Synced at each game publish, under the same yes: `venv\Scripts\python.exe trainworld/tools/opensource/export.py`,
then `git push` in ..\trainsworld-export.

A PR is applied in maps by an agent (`gh pr diff`, `git apply`), checked like agent work, committed with the
contributor as co-author once she has seen it, and closed with a link after the next sync.

Replies on a PR are written for a person, whether or not the PR looks LLM-written: thanks first, plain words, what
we took and what we didn't with the main reason for each, short; internals only where they need them to act. Offer
rather than ask ("if you share the save, we can ..."), never a request of their time. She sees each reply before it
is posted.

## Servers and storage

- shonei-server: `.\deploy\deploy.ps1` from that repo, alone in its own PowerShell call; then `git push`
  separately, which her yes to the deploy covers.
- Old R2 versions: only through `python trainworld/tools/r2_purge.py <versions> [--yes]` (allowed by a permission
  rule; it refuses the live version, the newest 3 and anything not dated). Never a bare `rclone purge`.

## Right after a publish

Don't go idle. A publish tends to be followed by a quiet stretch while a lot is still open. Take a fresh look at
everything not done (TASKS.md by priority, her open asks, follow-ups agents filed, player reports since the
update), start what can go ahead, and send her a few bullets of what you are starting and what you'd do next.
