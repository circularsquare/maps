# National estimates: the queue and the brief for short agents (spec §15)

**One item per agent, and stop when it is done.** Read spec §15.3, §15.4, §15.4b and §15.11 first,
then the docstring at the top of `estimates_hand.py`, which defines a row. Take the first item marked
`open`: change its marker to `taken <date> <your id>` before you start, and to `done <date>` or
`parked <date>: <why>` when you stop. Edit only your own line. Other agents are editing this file too,
so match on the item's text, never on its line number.

## How to add a figure

1. **Find it, self-identification first.** A census table, or a national survey that asks the
   question: ESS, ISSP, Afrobarometer, Arab Barometer, LAPOP, DHS, a national social survey. A
   compiler (Pew's specialised reports, the World Religion Database) only where nothing
   self-identified exists, with the reason in `note`. **Look in the archive before the web**:
   `sources.md`, `sources/*.md` and `data/raw/` already hold figures researched for country builds.
2. **Add one row per (country, node) to `estimates_hand.py`.** Shares of the whole country. Where good
   sources disagree, `low` and `high` span them; never a midpoint. Set `within` when Pew counts these
   people inside one of its seven families but the taxonomy keeps them apart, as with Alevis and
   Pew's Muslims.
3. **`python estimates.py --dry`.** Your rows appear under "hand rows" as shown or not shown, with the
   reason. "Not shown: the country's own source measures ..." is §15.3 working, not a failure. Note it
   against the item and move on.
4. **`python estimates.py`** writes the two files. It is quick, and nothing but the viewer reads them.
5. **Record each figure in `sources/estimates.md`**, under a heading for the religion: URL, table or
   question, year, and why that figure.

## Rules

- **Nothing from the World Religion Project.** §15.4b found it wrong in ways no flag catches.
  `tools/scan_estimates.py` may point at a country worth looking at; its numbers are never copied.
- **A figure for citizens only is not a national estimate.** Anita, 2026-09-14, on Bahrain: *"only
  citizens is pretty different from what we're trying."* Where non-citizens are a large share of
  residents and nobody measures them, leave the country off rather than ship the citizens' figure.
- **No new taxonomy nodes.** A religion with no node in `taxonomy/religions.json` is a taxonomy
  change: record it here as `parked` and file it with `python tools/ask.py new <cc> --title "..."`.
- **§14 still applies.** A national figure adds no spatial resolution, but a figure for a persecuted
  group whose only source is an interested party, or whose publication could matter to its safety,
  goes to Anita through `tools/ask.py`, with the bar in `AGENT_BRIEF.md` §3. The row ships anyway
  unless the question is whether it should exist at all.
- **Do not edit** `index.html`, `countries.py`, `todo.txt` or another agent's queue line.
- **Do not commit.**

## Queue

Markers: `open`, `taken <date> <id>`, `done <date>`, `parked <date>: <why>`.

### Islam's branches

- `done 2026-09-14` **Shia and Sunni, first batch: the Gulf and Yemen.** `python tools/scan_estimates.py
  islam.shia -v` lists the 73 countries the World Religion Project would have outlined; use it only
  to find countries worth sourcing. Later batches get their own line here, about ten countries each,
  by region. Yemen from the Arab Barometer (self_id); Saudi Arabia, Kuwait, Qatar, the UAE and Oman
  from Pew 2009 (estimate), with no Sunni row for Oman. Rows use the new `of="islam"`.
  `sources/estimates.md` has the write-up.
- `done 2026-09-14` **Bahrain's Shia and Sunni: left off, Anita's call.** The only self-identified
  figure covers citizens (47% of residents), and Pew 2009's share exceeds every citizen when carried to
  2020. Anita: *"only citizens is pretty different from what we're trying."* `sources/estimates.md`,
  Bahrain.
- `done 2026-09-14` **Shia and Sunni, third batch: Pew 2012 Q31 for Pakistan, Bangladesh,
  Egypt, Jordan, Kenya, Ghana (Anita, 2026-09-14).** 11 rows, all shown, plus Ghana's Ahmadiyya.
  Kenya's 4% Ahmadiyya, printed as "something else" on Pew 2012 p. 30, is left for the Ahmadiyya
  item. `sources/estimates.md` has the write-up.
- `open` **Shia and Sunni, second batch: Iran, Azerbaijan, Afghanistan, Syria, Lebanon, Tajikistan.**
  Read the Gulf section of `sources/estimates.md` first. The Arab Barometer covers Lebanon (and its
  national sect shares may survive the quota that closed its governorates, §11al, or may not) and none
  of the others; WVS wave 7 has Iran. **Iran now draws `islam.sunni`** (Masaili's province
  estimates, 2026-09-15, `sources/ir.md` §9), so a national Sunni row for Iran would restate a drawn
  node (spec §15.3); a Shia row would not, since the rest of Iran's Muslims stay on `islam`.
- `open` **Ahmadiyya.** The World Religion Project records it in Indonesia alone. Pakistan's census
  counts Ahmadis. Check which built countries already draw `islam.ahmadiyya` before sourcing any.
  §14 applies.
- `open` **Alevis outside Türkiye**, Germany first. Bulgaria's Shia dots are already its Kazalbash
  Alevis (its note in `countries.py`), so Bulgaria needs nothing.

### Religions inside Pew's "other religions"

- `open` **Daoism, Taiwan.** Pew's 2025 methodology cites its own Taiwan survey on people raised
  Daoist.
- `open` **Shinto, Japan.** `sources.md` records JGSS self-identification near 0.7%. The Agency for
  Cultural Affairs figures count shrine catchment, not adherents, and are not usable.
- `open` **Zoroastrianism, Iran.** Iran's 2016 census is in the UN Demographic Yearbook table on disk,
  `data/raw/unsd/dyb_table28_values.zip`.
- `open` **Sikhism**, countries that are not built.
- `open` **Jainism**, countries that are not built.
- `open` **Baha'i.**
- `open` **Yazidism.**
- `open` **Druze, Syria.**
- `open` **Confucianism and Chinese folk religion, Taiwan.**

### Sweeps

- `open` **UN Demographic Yearbook table 28 against built countries.** For each built country, list
  the religions its latest national census tabulation names that its drawn table does not. Jamaica
  is the known case: its Baha'is, Hindus, Muslims and Jews are absent from the parish tables. Those
  become hand rows from the country's own census. `sources/estimates.md` has the file's layout and
  the seven country names that need aliases.

### Decided, not for agents

- The header keeps counting built countries only, and the about panel says nothing about the layer;
  the control's tooltip is the explanation. Anita, 2026-09-14.
