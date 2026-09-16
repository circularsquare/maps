# 042 — ss: South Sudan: download the World Bank's HFSSS files, and at what grain to draw it during the war (section 14)

Summary: Only state-level religion source: World Bank HFSSS 2015-16 household heads, free login. Section 14: civil war, 2.3M refugees abroad; Jonglei, Unity, Upper Nile never sampled. Download and draw sampled states, one national share, or leave undrawn?

*Filed 2026-09-15 by session `cb8b206e-ss`. Anita's call; nothing is waiting on it.*

## What I did

Parked South Sudan at checkpoint A, with nothing built. The only religion source below the nation is the World Bank's High Frequency South Sudan Survey, waves 1 and 2 (2015-16, household heads), whose files need a free World Bank Microdata Library login; every open route I found is national only or asks no religion. If you download the files, my plan is to draw the states the survey sampled, at former-state grain, and leave Jonglei, Unity and Upper Nile empty because nobody sampled them.

## What it costs to reverse

Nothing is built. Declining leaves South Sudan blank, as now. After a download, the grain is a choice inside one build session (sampled states only, or one national share in all ten).

## Why it is yours rather than mine

AGENT_BRIEF §3: an account in your name (the World Bank login), and §14 (whether a country in an ethnic war with mass displacement is drawn, and at what resolution).

## The detail

- **The files.** World Bank Microdata Library, *South Sudan - High Frequency Survey 2015, Wave 1* (`SSD_2015_HFS-W1_v02_M`, https://microdata.worldbank.org/index.php/catalog/2778) and *High Frequency Survey 2016, Wave 2* (`SSD_2016_HFS-W2_v02_M`, https://microdata.worldbank.org/index.php/catalog/2777). From each, the household file `hhq` (wave 1: 3,550 households; wave 2: 1,189) and the member file `hhm`. Access is "Public Use": sign in, accept the terms (statistical use only, no redistribution, no attempt to identify anyone). They go in `data/raw/ss/`.
- **What they hold.** The household head's religion (`C_9_hhh_religion1`) with `state`, `ea` and a population weight. Unweighted answers: wave 1 3,492 (Christianity 3,117, Traditional African Religion 231, Islam 132); wave 2 1,156 (1,073, 31, 50). Wave 1 sampled six states, urban and rural: Central, Eastern and Western Equatoria, Northern and Western Bahr el Ghazal, and Lakes. Wave 2 adds urban Warrap.
- **What they miss.** Jonglei, Unity and Upper Nile, 4.68 million of the 12.39 million in the 2022 estimate (37.7%), were never sampled. They are where the war that began in Juba in December 2013 "quickly spread" (South Sudan Law Society and UNDP, *Search for a New Beginning*, 2015). Warrap (10.4%) is urban only.
- **The situation.** UNHCR, 2024: 2,290,622 South Sudanese refugees abroad, 944,631 internally displaced, 404,744 refugees returned. In the 2015 survey above, 63% of respondents said a close family member had been killed, and the report names the Dinka, Nuer and Shilluk as the groups most associated with the conflict. Religion is not that line. A state is 660,000 to 2.0 million people, and nothing would be placed below a state.
- **Levels.** Surveys that ask put traditional religion far below Pew's estimate: HFSSS wave 1 6.6% of heads, IRI's 2013 national poll 7%, against Pew 2020's 32.8% "other religions". The map would show what people call themselves.
- **Refugees in South Sudan** (ask 033). UNHCR 2024: 514,794 refugees (487,652 of them from Sudan), 2,677 asylum seekers and 18,000 stateless, about 4.1% on the 2022 estimate, for the not-drawn part of the bar.
- **Precedent.** Chad (017), Burkina Faso and Mali (018) drawn at their published tiers despite attacks; DR Congo's east at province (039) and Sudan at one national share (041), both open; Ecuador's Galápagos, never measured, drawn empty (2026-09-08).
- `sources/ss.md` §1-§3.

Options:

1. **Download, and draw the sampled states only** (my recommendation): six or seven former states at their own shares; Jonglei, Unity and Upper Nile empty and named in the note.
2. **Download, and draw one national share in all ten states**: nothing points at a state, but figures nobody measured would sit on the three war states.
3. **Leave South Sudan undrawn**, and skip the download.
