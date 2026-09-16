# 037 — mz: Mozambique: districts are drawn where the insurgency has spread beyond Cabo Delgado

Summary: Mozambique is drawn at 2007 districts in Niassa and Nampula, where insurgents have attacked Mecula, Memba and Eráti (a Catholic mission burned in 2022). Leave as built, or draw those provinces whole like Cabo Delgado? I'd leave it.

*Filed 2026-09-15 by session `cb8b206e-rev7` (reviewer). Anita's call; nothing is waiting on it.*

## What I did

Left it as built: Niassa's 16 and Nampula's 21 districts drawn from 2007 shares fitted to 2017, and Cabo Delgado whole. The builder's §14 note (`sources/mz.md` §7.5) covers Cabo Delgado only and does not mention attacks in the two provinces beside it.

## What it costs to reverse

Drawing Niassa and Nampula whole: a few lines in `countries/mz.py` (add `MZ01` and `MZ03` to `_MZ_WHOLE`, drop their districts before the partition check), the note's Niassa sentences rewritten, and the build tail.

## Why it is yours rather than mine

Spec §14: the map shows, district by district, where Christians live in places where armed Islamists have attacked Christians, at units smaller than the ones ask 018 ruled on ("these provinces and regions seem pretty big").

## The detail

- **The attacks outside Cabo Delgado.** Mecula, Niassa: attacks from late 2021 (VOA, "Officials Say Insurgency in Northern Mozambique Is Spreading"), and one on the Mariri Environmental Centre that the Islamic State claimed (Africa Defense Forum, June 2025). Eráti and Memba, Nampula, 2 to 7 September 2022: at least 17 killed, among them an Italian nun at the Chipene Catholic mission, which was burned, and about 65,000 displaced (allAfrica; OCHA on ReliefWeb). ACLED's Mozambique monitor still reports on Niassa in August 2026.
- **What the map draws there** (`data/normalized/mz_districts.csv`, 2017 people):

  | district | people | Muslim | Catholic |
  |---|---:|---:|---:|
  | Mecula (Niassa) | 20,888 | 94.8% | 3.0% |
  | Memba (Nampula) | 328,460 | 65.5% | 27.9% |
  | Eráti (Nampula) | 387,713 | 42.3% | 50.8% |

- **Against ask 018.** Burkina Faso's 45 provinces averaged about 310,000 people in 2006. Memba and Eráti are that size. Mecula is a fifteenth of it, but its Catholics are about 630 people, under one dot at 1:1,000.
- **Your options:**
  - **(a) Leave as built. Recommended.** The Nampula districts are the size of units you already allowed, Mecula's minority is under one dot, the shares are INE's own printed 2007 figures, and dots inside a district follow population, not religion.
  - **(b)** Draw Niassa and Nampula whole from 2017, like Cabo Delgado. This loses the Niassa detail in the note (Lago's Anglicans, Nipepe's Catholics) and Nampula's contrast between the Muslim coast and the Catholic interior.
  - **(c)** Coarsen only the attacked districts, grouped with neighbours (ask 018's option b). About one session; the builder records which attacks it counted and on what date.
