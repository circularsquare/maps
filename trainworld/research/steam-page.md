# Steam page for trainsworld (researched 2026-11-18)

Sources are Valve's Steamworks docs (partner.steamgames.com/doc), read 2026-11-18. "Unverified" means I found no Valve page saying it.

## 1. Checklist to a live "Coming soon" page

1. **Steamworks account.** Sign up as a partner, then sign the digital agreements. [Onboarding](https://partner.steamgames.com/doc/gettingstarted/onboarding)
2. **Identity, bank and tax.** You give legal identification, bank details and a tax questionnaire. The bank account holder name must match the name you give. Tax verification by a third party can take up to 30 days; you can't change tax info until it finishes. Plan for 2 to 4 weeks. (An older Valve partner page says 15-30 days for paperwork.) Whether you can sign up as an individual with no company: unverified.
3. **Steam Direct fee: $100 USD per game.** Paid by an Admin on the account, not with Steam Wallet. Not refundable. It is recouped in the payment after the game reaches $1,000 in revenue, so a free game never gets it back (no revenue). Valve's page says nothing special about free games. [Fee](https://partner.steamgames.com/doc/gettingstarted/appfee)
4. **21-day wait.** For a first few titles there is a 21-day wait between paying the fee and releasing. It runs alongside the Coming soon period. [Onboarding](https://partner.steamgames.com/doc/gettingstarted/onboarding)
5. **Create the app.** Tick "This is a free product" under Options; Steam makes a hidden free package. [Free to play](https://partner.steamgames.com/doc/store/freetoplay)
6. **Fill the Store Presence checklist.** All items are required before posting: capsules, screenshots, descriptions, tags, content survey, and so on. A trailer is not needed for Coming soon. [Coming soon](https://partner.steamgames.com/doc/store/coming_soon)
7. **Mark as ready for review.** Valve takes 3-5 business days; submit at least 7 business days before you want the page live. [Review process](https://partner.steamgames.com/doc/store/review_process)
8. **Click "Post as Coming Soon"** after approval. The page is public.
9. **Wait at least 2 weeks** with the page listed as Coming soon. The 14 days you heard is right. [Releasing](https://partner.steamgames.com/doc/store/releasing)
10. Before release only: upload a near-final build, pass build review (also 3-5 business days), upload a trailer, click Release App yourself.

Fastest realistic path: the page can be live a week or so after submitting, if tax and fee are done. Tax may be the slowest step.

## 2. Assets (all sizes from [Store graphical assets](https://partner.steamgames.com/doc/store/assets/standard) and [overview](https://partner.steamgames.com/doc/store/assets))

Valve's August 2024 notice says old dimensions are no longer accepted. Check the templates in Steamworks when uploading.

Store, all required: header capsule 920x430, small capsule 462x174, main capsule 1232x706, vertical capsule 748x896. Optional: page background 1438x810.
Library, required: library capsule 600x900, library hero 3840x1240 (png, art only), library logo 1280 wide and/or 720 tall (png, transparent), library header capsule 920x430. Also: app icon 184x184 (jpg), shortcut icon 256x256 (png or ico).
Screenshots: at least 5, 1920x1080 or larger, 16:9. Mark at least 4 as suitable for all ages.
Trailer: needed to release, not for Coming soon. 16:9, up to 1920x1080, 30 or 60 fps, 5,000+ Kbps, mp4/mov/wmv, H.264 + AAC preferred. [Trailers](https://partner.steamgames.com/doc/store/trailer)
Unverified: which library assets are checked at Coming soon vs release. Treat them all as needed.

Capsule rules ([Graphical asset rules](https://partner.steamgames.com/doc/store/assets/rules)): only artwork, the game name and an official subtitle. No review scores, awards, sale text, calls to action (wishlist, "coming soon"), other products, or other text. Library hero: no words. Library logo: logo only. Art must be PG-13. The logo must be readable.

## 3. Text fields

- **Short description:** plain text, "a few hundred characters" (no exact number given; I'd keep it under 300). No dates, no "now available". [Descriptions](https://partner.steamgames.com/doc/store/page/description)
- **About this game:** no length limit. Headers, bold and images allowed; each image under 5 MB, all images under 15 MB. No links, QR codes or URLs (use the link fields), nothing that imitates Steam's UI. Textless images preferred.
- **Tags:** at least 5 before launch, up to 20; top 5 matter most. [Tags](https://partner.steamgames.com/doc/store/tags)
- **Languages:** "Game support only" languages are listed on the page and searchable. If none of Steam's full-support languages apply, English must still have content. Store text can stay English only. [Languages](https://partner.steamgames.com/doc/store/localization/languages)
- **System requirements:** Valve's pages I read don't cover it (unverified). Suggested: Windows 10 64-bit, a GPU with WebGL 2, RAM and disk to be measured later, and "Requires an internet connection (map data loads from the web)".
- **Content survey:** three parts: general content, mature content, generative AI. [Content survey](https://partner.steamgames.com/doc/gettingstarted/contentsurvey) The game has none of the mature topics as far as I know. Answers can later only be changed through Steam Support.
- **AI disclosure:** Valve asks about content that ships with the game and players consume: art, sound, narrative, localization. "Efficiency gains" from AI tools are explicitly not the focus. So the AI-made translations count (pre-generated, ships with the game; Valve reviews it like any other content). AI-written code is not addressed by name and looks out of scope (unverified). Any AI-written in-game text and any AI art or sound count. Live-generated AI (made while playing): the game has none.

## 4. The downloadable build, later (no work now)

- A desktop wrapper around the web game: Electron or Tauri (WebView2 on Windows). Electron is the safer fit for a WebGL/WASM game since it bundles Chromium; Tauri is smaller but depends on the system webview. Not verified against Valve docs.
- The game streams map data from the web, so it needs internet. Say so in the requirements. Offline play would mean bundling the map data (size unknown).
- A free game with no achievements, cloud saves or overlay needs no Steamworks SDK code. A build is uploaded with SteamPipe (a depot and a build script). [Uploading](https://partner.steamgames.com/doc/sdk/uploading) (not read in detail)
- Saves live in browser storage today; a wrapper has its own storage, so existing web saves would not carry over (check before building).

## 5. Drafts

**Short description (about 230 characters)**
Build rail on the real world map. Draw track, place stations, run lines and watch riders choose your trains over cars. Free, and the whole world is playable, from small towns to Cairo and New York.

**Long description**
trainsworld is a free game about building rail on a real map of the world.

Pick a city. Lay track, place stations and set up lines. Trains run on schedules. People want to make trips between homes, jobs and schools, and they take your trains if the trip is good.

What you can do
- Build track, with one way track, junctions, tram, bus and ferry lines
- Set fares, headways and weekday and weekend timetables
- See where trips go and why people ride or drive
- Play any city in the world, or link several cities and the trips between them
- Share your network and see other players' networks
- Play in many languages (translations are made with AI)

The game is free and has no ads. It needs an internet connection to load the map.

Screenshot ideas (1920x1080, no text overlays):
1. A big city network at night-time zoom, lines in colour
2. The schematic diagram view of the same network
3. Track tool mid-build with the track tag showing type and number of tracks
4. A line panel with the 30-day money chart
5. A demand view showing trips between zones
6. Several cities linked by intercity rail
7. A station area at high zoom with trains arriving
8. The Community tab with shared networks

(Check the real feature names against the live game before shooting; these come from the changelog and SPEC.)

## 6. Who does what

Anita only: Steamworks account, agreements, identity, bank and tax forms, the $100 payment, content survey and AI answers (she is the legal party), marking ready for review, Post as Coming Soon, later Release App.
Agents can prepare: all capsule and library art at the sizes above (needs a logo and a style call from her), screenshots at 1920x1080, text drafts, tag list, system requirements, a trailer from screen recordings. Agents can't upload, since that needs her login.

Open question for her: whether she signs up as an individual or under a business name, since the bank name must match. Valve's pages I read don't settle which is allowed.
