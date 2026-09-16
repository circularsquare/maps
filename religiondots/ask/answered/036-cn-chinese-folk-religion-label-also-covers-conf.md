# 036 — cn: Chinese folk religion label also covers Confucianism and Falun Gong

Summary: Renaming the node to 'Chinese folk religion' also labels Indonesia's Khonghucu (Confucianism), New Zealand's Falun Gong and other countries' Confucianism rows on it. Keep them under that label, or give them their own node?

*Filed 2026-09-15 by session `cb8b206e` (supervisor), from review `cb8b206e-rev5`. Anita's call; nothing is waiting on it.*

## What I did

Left every row where it was. The rows other countries had on the node now show under the new name "Chinese folk religion", as they did under the old one.

## What it costs to reverse

A new node in `taxonomy/branches.py`, the affected rows re-pointed in each country's mapping, those countries rescattered and a build tail. About one agent session.

## Why it is yours rather than mine

The rename was your call, and this is what the legend calls people. Confucianism as an official religion in Indonesia, and Falun Gong, are not folk religion, so the label may now describe them wrongly.

## The detail

- The review of the China, Taiwan and Hong Kong redraw (`sources/folk_practice.md` §9) found these rows on the renamed node: Indonesia's Khonghucu (Confucianism, one of its six recognised religions), New Zealand's Falun Gong, and Confucianism rows in several other countries.
- Country mappings that put rows on the node: au, ca, cz, id, mu, nz, sg, uk, vn, as well as cn, hk and tw themselves. Not every one of those rows is Confucianism or Falun Gong; the review has the list.
- **Your options:** keep them under "Chinese folk religion"; or give Confucianism (and Falun Gong) a sibling node under the same Chinese religions branch, so the colour family stays but the name is right.
