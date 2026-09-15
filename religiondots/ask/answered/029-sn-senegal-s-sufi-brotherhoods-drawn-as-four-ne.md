# 029 — sn: Senegal's Sufi brotherhoods drawn as four new Islam nodes

Summary: Senegal 1988 is drawn with the Tijaniyya, Mourides, Qadiriyya and Layène as four new nodes under islam; two are Senegal-only legend rows. Reversing folds them into islam: a mapping edit and a rescatter.

*Filed 2026-09-15 by session `d743fc47-sn`. Anita's call; nothing is waiting on it.*

## What I did

Drew Senegal's 1988 census with its four Sufi brotherhoods as four new nodes, `islam.tijaniyya`,
`islam.mouride`, `islam.qadiriyya` and `islam.layene`, in the "Sufi orders" group under `islam`
beside Bektashi. Muslims who named none of the four (5.1%) are on the bare `islam`.

## What it costs to reverse

Point four MAP entries in `taxonomy/sn1988.py` at `islam`, delete the four nodes in
`taxonomy/branches.py`, run `taxonomy/build_tree.py`, rescatter `sn` in both editions and let the
next build tail run: about 15 minutes, nothing else moves.

## Why it is yours rather than mine

AGENT_BRIEF §3: a single-country node that adds a legend row nobody else uses. No other census on
this map counts any of the four, and the Mourides and the Layène are Senegalese foundings that no
other country is likely to supply. Tonga's four Methodist churches were flagged on the same bar.

## The detail

The 1988 household form's religion item (P11) asks every Muslim for their brotherhood: 1 Khadr, 2
Layène, 3 Mouride, 4 Tidiane, 5 Muslim of none of these, then Catholic, other Christian and other.
The national report's Tableau 1.15 prints it by région, one decimal (`sources/sn.md`).

| node | people, 1988 | share | where it stands out |
|---|---:|---:|---|
| Tijaniyya | 3,260,497 | 47.3% | 80.2% of Saint-Louis (Matam included), 65.4% of Kaolack |
| Mouride | 2,047,728 | 29.7% | 91.5% of Mbacké (Touba), 45.9% of Louga, 44.7% of Thiès |
| Qadiriyya | 806,271 | 11.7% | 32.0% of Ziguinchor, 26.0% of Kolda |
| Layène | 41,681 | 0.6% | 74.9% of them in the Dakar région |

Folded into `islam`, Senegal would be one green over 94% of its dots: the brotherhoods are the whole
of what this census adds. Senegal's legend is seven rows with them and four without.

Two things inside this you might want differently. The Layène is the smallest (about 42 dots at
1:1,000) and could fold into `islam` alone while the other three stay. And the four sit under
`islam`, not `islam.sunni`: the form asks the order and not the school, as Albania's does for
Bektashi, and the Layène's founder proclaimed himself the Mahdi, which makes a Sunni filing arguable
for one of them. The other three are Sunni and Maliki in Senegal, and moving them under
`islam.sunni` is an id change in the same two files.

The Tijaniyya and Qadiriyya are pan-West-African; Afrobarometer's card offers them (and the
Mourides) in several countries, so a later survey build could reuse the nodes. None does yet.


---

## Ruled 2026-09-15 by Anita: keep the four nodes for now

*"ah cool. maybe lets leave them as is and make a more final decision once we have a better grasp on
what the maghreb reports. i think leaving them separate is fine for now."*

The four brotherhoods stay as four nodes, as built. Not final: revisit once more countries' sources
show what they report on the orders.
