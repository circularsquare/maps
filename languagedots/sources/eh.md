# Western Sahara: closed, drawn inside Morocco (`ma`)

Closed 2026-10-05 (session edd42a8c-eh). No `countries/eh.py`, no mapping, no geography: the
territory is already on the map as part of `countries/ma.py`, from the same source the queue row
named, and a separate entry would draw the same ground twice.

## Why there is nothing to build

- **Same table.** The queue's lead for `eh` is HCP's RGPH 2024 indicator workbook
  (https://www.hcp.ma/file/242671/), the one `sources/ma_rgph.py` already normalises. Its
  Western Sahara communes are rows of `data/normalized/ma.csv`.
- **Same ground.** `sources/ma_geo.py` re-keys religiondots' Morocco hex layer, which has four
  Western Sahara units (EH01-EH04, COD-AB Western Sahara cut to Natural Earth's B19, so west of
  the berm only). 30 communes sit on those units in `data/geo/ma/ma_lookup.csv`
  (EH01 Boujdour 4, EH02 Es-Semara 7, EH03 Laâyoune and the Tarfaya strip 10, EH04 Dakhla and
  Aousserd 9), three of them as `EH0x-rest` remainders.
- **Same handling as religiondots.** religiondots has no `countries/eh.py`; its `ma` entry draws
  the territory "as far as Morocco administers it", west of the berm, under its ask 031 (ruled
  2026-09-15). languagedots' `ma` follows that ruling and says so in `note_public`. Nothing here
  goes beyond it, so no §7 ask.

## What `ma` draws on the four EH units (checked from `data/normalized/ma.csv`)

600,991 people in households, all `derived` (multi-answer scaling):

| language | people |
|---|---:|
| Darija | 392,972 |
| Hassania | 123,777 |
| Tachelhit | 61,382 |
| Tamazight | 21,284 |
| Tarifit | 1,576 |

The EH03 figure includes five Tarfaya-province communes (Tarfaya, Daoura, El Hagounia,
Akhfennir, Tah; 15,218 people) because religiondots puts the strip north of 27°40'N in EH03.
Lagouira, Aghouinite, Zoug and Mijik have no language figures and are in `ma`'s `gap`; garrisons
the census files under communes beyond the berm (Tifariti, Gleibat El Foula, Mijik...) are drawn
on their province's towns, as in religiondots.

## Not covered by any source here

The strip east of the berm and the Sahrawi refugee camps near Tindouf (in Algeria) are outside
the Moroccan census. No language source for either was searched for; if one turns up, it would
be a new `eh` entry for the area east of the berm only.
