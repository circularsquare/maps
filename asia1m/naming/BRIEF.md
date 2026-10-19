# Naming a batch of asia1m regions

You are proposing names for one batch of regions on Anita's asia1m map ("Asia divided into regions
of one million people"). She picks the final names herself, so your job is a short, well-sourced
list of candidates per region, with what each name means. Read this whole file before starting.

Keep the effort proportionate. Many regions are easy: one county or city makes up most of it and
its name works. Give those one or two candidates and move on. Spend research on the regions where
nothing obvious fits.

## What is in the batch folder

`asia1m/naming/batches/<batch>/`:

- `sources.md`: **read this first.** Gathered by script, so reading it costs no searches. Per
  region: the historical commanderies, prefectures and abolished counties whose seats lay inside it
  (CHGIS, with years), then Chinese Wikipedia's lead, history, name and geography sections for the
  main counties and prefectures. That is where 古称, 别称, 因…得名 and the local rivers and mountains
  usually are.
- `facts.json`: one entry per region, numbered `n` = 1, 2, .... For each region:
  - `people`: population (Kontur 2023).
  - `units`: admin units by `people_share` (share of the region's people living in that unit) and
    `cover` (share of the unit's own people that this region holds). A unit with high share and
    high cover is what the region mostly *is*.
  - `parents`: prefectures and provinces the same way.
  - `towns` (China): townships (街道/镇/乡) by people. For China, `units` are built from these
    townships, so county names are current. `parents` still come from an older county layer and
    can predate 2002-2004 boundary changes; when they disagree with `units`, trust `units`.
  - `cities`: places of 100k+ inside (simplemaps). Chinese figures are usually whole-prefecture
    totals, so a "1.1M" Guyuan is not a 1M city; judge the city rule from the urban population.
  - `neighbours`: adjacent regions by key, with `name` where Anita has already picked one.
- `labels_<t>.png`: the area as drawn, with Anita's text layer. Settled names are written here in
  lowercase pixel text. Long names are split over two or three lines ("beng / he" = Benghe). Small
  city regions carry a number with a numbered list beside them. A label sometimes sits partly
  outside its region. Small coloured patches over borders are there to keep text readable.
- `numbers_<t>.png`: the same area with each batch region's `n` in a red box, others faded.
  `facts.json` → `tiles` says which regions each picture shows.

## Step 1: what is already named

Compare `labels_*` with `numbers_*`. For every batch region, decide whether it already has a
settled name on the map; if so record it as `settled` and propose nothing. Note neighbours' names
too (from the pictures and `neighbours[].name`).

## Step 2: research, cheapest first

1. `sources.md` and `facts.json`.
2. `python asia1m/naming/lookup.py hits <name> <province>` for the recognisability test on names
   that aren't current admin units (Chinese Wikipedia full-text hits and top titles). The context
   is the province only (河南, 山西), never the city or county: a reader roughly knows the
   province, not the prefecture, so "210 hits with 开封" overstates how known a name is. And `lookup.py page <title>` to read another article.
   These cost almost nothing; use them instead of web searches. Run them one at a time, not in
   parallel (Wikipedia rate-limits; the script retries).
3. WebFetch for a specific page you already know you need (a county government's 历史沿革 page,
   a province's place-name series such as Ningxia's "探寻宁夏地名" on mca.nx.gov.cn).
4. WebSearch last, only for a gap the above cannot fill. **At most 15 WebSearch calls for the
   whole batch**; the session cap is 200 and shared by every agent. Baidu Baike blocks fetches.

Never put Anita's name or email in any request.

## Step 3: choose candidates

Anita's rules:

- Names should feel like standalone regions. Avoid names that suggest part of a whole, like
  "North Dakota" (北/南/东/西 + an existing name, "Upper X"). A direction word inside a well-known
  established name is fine: Xixia (the Western Xia dynasty) and Hebei are good names.
- No fabricated names. Nothing super obscure: a small amount of recognisability is necessary. If
  `lookup.py hits` shows the name is mostly about somewhere else, or barely appears, it is weak.
  Don't run the check for current county, city or prefecture names: those always score well.
  Use it for historical, river, mountain and other names.
- An old prefecture form of a county (Fenyang → Fenzhou) is a good name when the prefecture also
  covered the region's other counties, so it fits the whole region better than one county does,
  and the -zhou name points to this place. Avoid it when the same -zhou name is famous somewhere
  else in China. Say in `fit` which counties the prefecture covered.
- When a county was renamed (Mi county 密县 became Xinmi 新密), its old name (Mixian) or the
  mountain or river it was named after is often the more rooted name: list it if it is still
  known.
- A one-character county is never a reason to reject a name: it takes its usual English form
  with -xian (Linxian, Mianxian, Yangxian).
  `lookup.py en <name>` shows whether English Wikipedia has it and what an English search finds;
  note that many old prefectures have short English stubs ("Fen Prefecture"), which alone
  doesn't make them known.
- A prefecture name is fine for the region that holds the prefecture's seat (and ideally what it
  is named after), even when the prefecture spans other regions too: Shangluo for the region with
  Shangzhou and the upper Luo River.
- A landscape name doesn't claim the whole feature (the state of Colorado and the Colorado River):
  a lake, river or range is a good name for the region at its heart even when it extends into
  other regions. Anita picked Dongting for the south Dongting plain.
- Lean away from names that mean pacifying, subduing or settling a frontier or its peoples
  (镇, 靖, 绥, 定, 宁, 安, 平 with 边, 远, 番, 夷, 羌, 蛮 and the like: Jingbian, Dingbian, Zhenyuan,
  Suining-type names), especially where many people are not Han. Don't drop them when they are
  plainly the name the place is known by, but when an alternative is about as good, rank it first,
  and say what the name means so Anita can judge.
- A big region made of many small counties is a prime place for a geographical name: a river,
  basin, range or plain that spans the region or stands for it, rather than one county's name.
- A mix of name origins across the area: cities, historical names, current region names, rivers,
  mountains, plains. Don't give two adjacent regions the same kind of name where a good
  alternative exists, and never repeat a name already on the map nearby. A repeat means the same
  Chinese characters; two different places that only share a spelling (项城 and 襄城, both
  Xiangcheng) are fine.
- Spelling in the most commonly spoken local language. In China that is usually Mandarin, so use
  Hanyu Pinyin without tone marks, as one word, with the apostrophe where pinyin needs it (Xi'an,
  Yan'an). For a name of Tibetan, Uyghur, Mongolian or other origin in an area where Mandarin is now
  what most people speak, use the Mandarin pinyin of that name. Where another language is what
  most people speak, use that language's usual romanisation and say which one you used.
- For a city of over 1M, the most central region (holding the old centre) is named after the city
  itself; the city's other regions need other names.
- Give the meaning of every name you suggest. If one unit makes up a large part of the region's
  people and you don't suggest it, add it as a `rejected` entry with its meaning and why.

How well a name fits matters too. A county name fits when the region holds most of that county's
people and the county is most of the region. A river fits when it runs mostly through the region,
not just past it. A historical unit fits when its seat and territory overlap most of the region's
people. Say this briefly in `fit`.

**At most 4 entries per region, including any rejected one.** Once you have two good candidates,
stop. Best first.

**No source, no candidate.** Every candidate except a current admin unit name needs a
`source_url` that names this place: a Wikipedia article from `sources.md` or `lookup.py` counts.
If you could not confirm a name, leave it out.

## Step 4: write `candidates.json`

In the batch folder, as UTF-8 JSON:

```json
{
  "batch": "ningxia",
  "regions": [
    {
      "n": 4,
      "key": "#d0eec2",
      "settled": null,
      "summary": "Yellow River plain south of Yinchuan: Litong, Lingwu, Qingtongxia, Yongning.",
      "candidates": [
        {
          "name": "Lingzhou",
          "local": "灵州",
          "origin": "historical",
          "meaning": "\"Spirit isle\": a Han county on a Yellow River island said never to flood.",
          "fit": "Tang prefecture seated at today's Litong; covers the two largest parts.",
          "recognisability": "Own zh-wiki article, 203 hits with 宁夏, top ones about this place.",
          "source_url": "https://zh.wikipedia.org/wiki/灵州",
          "source_note": "治所在回乐县（今宁夏吴忠市西北古城镇）",
          "rating": 4,
          "status": "suggested"
        }
      ],
      "notes": ""
    }
  ]
}
```

- `origin`: one of `city`, `county`, `prefecture`, `historical`, `river`, `mountain`, `lake`,
  `plain`, `ethnic`, `other`.
- `status`: `suggested`, or `rejected` for a dominant unit you are explaining but not recommending.
- `rating`: your overall view of the name for this region, 1 to 5 (5 = an obviously good name
  that fits well and is known; 3 = workable; 1 = poor, listed only for its meaning). Omit it on
  rejected entries.
- `lat`, `lon` (optional): for a river, mountain, plain or historical name, a point if a source
  you read gives one. A script fills in the rest from CHGIS, Wikipedia and OSM for the picture.
- For settled regions write `"settled": "<name as written>"` and an empty `candidates` list.
- Include every region in the batch, in order of `n`. One sentence per field; `notes` only when
  there is something Anita should know (a conflict with a neighbour, a data error).

## Round two

If the batch folder has `feedback.txt`, Anita has reviewed the candidates. Each line names a
region (by `n`) and picks a name, rules names out ("not X"), asks for more ideas, or comments. For
regions that want more ideas, add up to 3 new candidates (mark ruled-out ones `rejected`); leave
every other region as it is. Rewrite `candidates.json` in place.

## When you finish

Reply in a few lines: regions named vs already settled, WebSearch and WebFetch calls used, and
anything that worried you. Don't edit anything outside the batch folder.
