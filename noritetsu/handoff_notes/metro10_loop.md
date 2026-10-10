# Paris Métro 10's one-way loop (fr agent, 2026-10-08)

HANDOFF 2026-10-08 item 6. Line 10 (m3328741) shipped as one chain of 22 sections with both
directions' loop stations threaded together (Mirabeau - Église d'Auteuil 0.135 km ...
Michel-Ange-Molitor - Porte d'Auteuil 0.057 km), so the strip diagram had no loop to draw.

## Cause: build_model.place_stations' distance snap

Not the fr register and not station merging: the two OSM route relations (123768 eastbound,
3328740 westbound) list the right stops and every stop resolves to its own station. But
place_stations walks each variant's path against the stations of EVERY variant, and a station
whose stop node is not on this path is snapped to it by distance if within STOP_SNAP_M (400 m).
The loop's arms are 150-330 m apart, so each direction took the other arm's three stations:

    eastbound path: ... BJJ, [Porte d'Auteuil 333 m], MAM, [MAA 309 m], Chardon-Lagache,
                    [Église d'Auteuil 153 m], Mirabeau, Javel ...
    westbound path: ... Javel, [Mirabeau 29 m], Église d'Auteuil, [Chardon-Lagache 229 m],
                    MAA, [MAM 325 m], Porte d'Auteuil, BJJ ...

The snap exists for express variants on separate parallel track (place_stations' docstring),
where the skipped station's track lies a few metres off. Here the stop node lies on the OTHER
arm's track, which does not run alongside.

No hook in rules/fr.py or fr_register.py reaches place_stations, so the fix is a shared diff.

## The diff: `handoff_notes/metro10_loop.diff` (applies to build_model.py as of 2026-10-08)

`git apply handoff_notes/metro10_loop.diff` from noritetsu/ (checked with `--check`).

- A station whose stop node lies on another variant's assembled path is snapped onto this
  variant's path only where that track runs alongside it (`_alongside`): the stop node within
  ALONGSIDE_M (100 m) of this path, and ALONGSIDE_WIN_M (150 m) along the stop node's track
  either way never more than ALONGSIDE_SPREAD_M (15 m) further off than the stop node is.
  Stop nodes on no variant's track (platform nodes beside it) snap as before.
- Gated on `SNAP_ALONGSIDE = True` in rules/<cc>.py; only rules/fr.py sets it (already added,
  a no-op until the diff lands). With the flag off the only change is that build() assembles
  each line's runs once before its variant loop instead of inside it (same function, same
  input), so other countries build byte-for-byte as before: no trial needed beyond fr.
- Logs how many snaps were refused and on which lines.

Thresholds come from every snap in the fr build whose station's stop node is on another
variant's track (491 of 1,191 snaps; scratch probe): parallel track kept 0-6 m of spread over
the window (Lille-Europe 42 m off, Le Mans 40 m, Gare du Nord 79 m, Asnières 9-25 m); arms,
branches and crossings moved 20-90 m (Mirabeau: 2 m off at the stop node, the westbound track
crossing it, 38 m within 150 m; Fontenay-sous-Bois 11 m / 47 m) or lay 120-400 m off.

## Trial (fr only; ab.py baseline and patched build into private folders)

`tools/ab.py fr` on the current tree: 0 lines differ from dist (dist is the current code's
build). Patched build (`build_model_patched.py` run in place of build_model.py, `--out` a temp
folder) against dist: **22 OSM lines differ, 0 register lines, station ids 5468 -> 5468 (0 gone,
0 new)**; foot.json and ways.json differ (they follow the sections). 118 snaps refused on 29
lines. Every changed line checked by its section diff and its pieces/ends:

| line | before -> after | verdict |
|---|---|---|
| Métro 10 | 22 -> 23 sections, 11.53 -> 14.73 km | the loop: Javel - Église d'Auteuil 0.78 - Michel-Ange-Auteuil 0.37 - Porte d'Auteuil 0.38 - Boulogne-Jean Jaurès 1.67; Boulogne-Jean Jaurès - Michel-Ange-Molitor 1.64 - Chardon-Lagache 0.45 - Mirabeau 0.37 - Javel 0.53. Both arms are real one-way track, so both count |
| RER A | 104.9 -> 115.0 km | right: A4 Vincennes - Val-de-Fontenay (no Fontenay-sous-Bois), A3 Achères-Ville - Maisons-Laffitte (no Achères Grand Cormier, A5's), Cergy/Poissy Nanterre-Préfecture - Houilles (Nanterre-Université is A1's); three false chords gone |
| RER C | 159.9 -> 164.0 | right: Champ de Mars - Javel replaces the cross-Seine chord Javel - Kennedy; Villeneuve-le-Roi - Choisy replaces Villeneuve-le-Roi - Les Saules; Versailles-Chantiers - Versailles Château RG chord gone |
| RER D | 197.2 -> 198.3 | right: Corbeil - Moulin-Galant (Malesherbes branch) replaces Moulin-Galant - Essonnes-Robinson |
| Transilien J | 181.0 -> 182.3 | right: Asnières - Houilles (Mantes trains) replaces Bois-Colombes - Houilles |
| TER 04 (Cannes) | 82.1 -> 85.3 | right: Mandelieu - Cannes-la-Bocca and Cannes - Le Bosquet (Grasse branch) replace chords through Le Bosquet |
| trams: Brest A, Montpellier 3 and 5, Saint-Étienne T1 and T3, Nantes 1, Lyon T3, T8, Clermont A, Caen T2 and T3, Rouen Métro | ±0.2-1.5 km each | snapped chords between a one-way pair's or a branch's stops gone (Lyon T3's Meyzieu Lycée Beltrame - Les Panettes 1.46 km; Rouen's Avenue de Caen - Europe; Nantes' Jamet - Jean Moulin; Caen T3's Château Quatrans - Bernières). No line split into pieces; Nantes 1, Lyon T3 and Caen T3 gain an end where a false chord had closed a ring |
| TGV Bruxelles/Paris/Lyon - Perpignan/Toulouse (3 named trains) | +149 km each | worse on paper, named trains only (not counted): the Nîmes Centre variants no longer take Nîmes Pont du Gard (265 m off, on the bypass) as a stop, so Valence TGV - Nîmes Centre (161 km) overlaps Valence - Nîmes Pont du Gard; before, a 12.7 km chord Nîmes Pont du Gard - Nîmes Centre stood in. No station at the junction to split them |
| TGV InOui 261B | 390.1 -> 383.0 | right: Lille-Europe - Lille-Flandres and Lille-Europe - Croix-Wasquehal chords gone |

Métro 7bis's loop (Botzaris - Place des Fêtes - Pré-Saint-Gervais - Danube - Botzaris) was
already a loop and is unchanged; Valenciennes T1/T2 at La Briquette refuse snaps but their
sections do not change.

Headless Chrome (private port, the app at localhost:8800 with France's json files swapped for
the trial's by a fetch rewrite): line 10's diagram draws the loop, Javel splitting into a side
lane of Église d'Auteuil, Michel-Ange-Auteuil, Porte d'Auteuil rejoining at Boulogne-Jean
Jaurès, beside Mirabeau, Chardon-Lagache, Michel-Ange-Molitor
(`handoff_notes/metro10_loop_after.png`). Before: one straight lane of 23 stops.

## After landing

    python tools/compare_lines.py save fr
    python tools/slot.py 2 -- python tools/rebuild.py -j 1 fr
    python tools/compare_lines.py diff fr
    python check_model.py --region fr
    python tools/build_regions.py      # managing session

Expect the 22 lines above to move and nothing else. Making SNAP_ALONGSIDE the default
everywhere would very likely fix the same kind of chord in other cities' one-way loops and
branches; that needs `ab.py --all` and a read of the moved lines first.
