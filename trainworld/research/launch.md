# Launch: where to share trainsworld and what to do first (T-2537)

Research for Anita, 2026-11-05, agent launchprep. Read only: nothing was posted, signed up for or
changed apart from this file and the T-2537 line. Every claim about a rule, a price or a launch has a
link. Where a page could not be read (Reddit blocks automated reads; several press sites answered
403), it says so, and the item is marked **not read**.

The game is at https://anita.garden/trainsworld/ (the old /trainworld/ redirects there).

## Checklist

Before launch
- [ ] Add link preview tags and a 1200x600 `preview.png` to `trainworld/app/index.html` (the source deploy copies). The live page has none today, so a shared link shows only "trainsworld". Section 4.1.
- [ ] Add a Cloudflare Cache Rule so `anita.garden/trainsworld/assets/*` is cached. The .wasm and .json files skip Cloudflare's cache today and come from GitHub Pages on every visit (about 3 MB each); 100k visitors would be about 300 GB against Pages' 100 GB a month. Section 5.2.
- [ ] Open the game on a real iPhone and a real Android phone. An empty New York game takes 1.4 GB in desktop Chrome and iOS closes tabs that use too much. Decide what a phone visitor sees. Section 4.2.
- [ ] Time a first visit on a slow connection (Chrome DevTools, Fast 4G, cache off) from opening the link to the first track. The first load is about 22 MB.
- [ ] Check what Start does on a first visit. If it still lands on the whole world, make New York (or the visitor's city) the default: idea 1 in `research/first-ten-minutes.md`, the one of its top three not done yet.
- [ ] Send one message through the in-game feedback form on the live site and check it reaches you. The server says feedback is on.
- [ ] Record one clip, 20-40 s, that opens on trains already running in a real city and ends on the network diagram. Also take 3-4 stills, one of them a diagram export.
- [ ] Write the tweet, the Show HN title and first comment, and one short Reddit text ahead of time. Drafts are in section 7.
- [ ] Delete old data versions on R2 and keep two. Each is 2.9 GB and the free tier is 10 GB.
- [ ] Read the rules page of each subreddit you plan to use. I could not read them; the links are in section 1.2.

Launch day
- [ ] Twitter first: the clip and one or two plain lines, with the link in the first reply.
- [ ] Same post on Bluesky.
- [ ] Show HN on the weekend: Saturday 9am-3pm ET or Sunday 6-11am ET did best in a 2025 study. Post the first comment yourself and stay 3-4 hours to answer.
- [ ] Don't ask anyone to upvote or comment on HN. It is against the rules and gets posts buried.
- [ ] Keep an eye on https://market.anita.garden/trainworld/stats, the R2 page in the Cloudflare dashboard, and the feedback mail.

The week after
- [ ] Reddit, one subreddit a day: r/WebGames, r/playmygame, r/indiegames (Promotion flair), r/TransitDiagrams (a diagram image), r/MapPorn (a map image).
- [ ] r/gamedev only as a comment in its weekly Screenshot Saturday thread.
- [ ] Stay out of other games' spaces: r/SubwayBuilder, r/NIMBYRails, r/CitiesSkylines and those games' Discords. If players post it there themselves, that is fine.
- [ ] Write to 3-5 people from section 2 with a short note, the link and the clip. Start with Miles in Transit, Alan Fisher, kottke.org, Streetsblog tips and PC Gamer tips.
- [ ] Fix what people report and reply in the same threads when it is fixed.
- [ ] If the Show HN got no notice, one repost later is allowed.
- [ ] Optional: an itch.io page that frames the anita.garden page. Section 1.4.

---

## 1. Where to post, and each place's rules

### 1.1 Hacker News (Show HN)

**Rules** ([showhn.html](https://news.ycombinator.com/showhn.html), [FAQ](https://news.ycombinator.com/newsfaq.html)):

- "Show HN is for something you've made that other people can play with." The title begins with
  "Show HN". The project must be something you made and "are around to discuss".
- "Please make it easy for users to try your thing out, ideally without barriers such as signups."
  trainsworld needs no account, which fits.
- "Please don't ask friends to upvote or comment." The FAQ: "We penalize or ban submissions,
  accounts, and sites that break this rule."
- "Share something that is deeply personal and interesting to you. Explain how and why." "A Show HN
  needn't be complicated or look slick."
- Reposts: "a small number of reposts is ok" if the story has had no significant attention in about
  a year; don't delete and repost.

**When.** Myriade's July 2025 analysis of 157k Show HN posts on BigQuery, counting a "breakout" as 30+
points ([source](https://www.myriade.ai/blogs/when-is-it-the-best-time-to-post-on-show-hn)): Sunday
11.75% and Saturday 11.08% broke out, weekdays 9.45-9.90%. Best slots: Sunday 00:00-02:00 UTC (up to
15.7%), Sunday 11:00-16:00 UTC (12-14%), Saturday 14:00-20:00 UTC (12-14%). In Eastern time after
1 November (UTC-5): Saturday 9am-3pm, Sunday 6-11am, or Saturday 7-9pm. Sturdy Statistics found
Show HN posts in 2025 did worse overall than in earlier years (about an 11% chance of passing 10
points) ([source](https://blog.sturdystatistics.com/posts/show_hn)), so a quiet result is common and
not a verdict.

**How transit things have done there** (HN's own search, [hn.algolia.com](https://hn.algolia.com/)):

| Post | Points | Date |
|---|---|---|
| [Trains.fyi, a live map of passenger trains in the US and Canada](https://news.ycombinator.com/item?id=38434574) | 503 | 2023-11 |
| [Subway Builder: A realistic subway simulation game](https://news.ycombinator.com/item?id=45530744) (not a Show HN; posted the day its beta opened) | 303 | 2025-10 |
| [Ubähnchen, animated subway map of Berlin](https://news.ycombinator.com/item?id=32647227) | 278 | 2022-08 |
| [I built a transit travel time map](https://news.ycombinator.com/item?id=36847960) | 255 | 2023-07 |
| [NYC Subway Simulator and Route Designer](https://news.ycombinator.com/item?id=44490588) (buildmytransit.nyc) | 201 | 2025-07 |

What the two game threads said:
- **NYC Subway Simulator** (a weekend project on OSM data): praise for the sandbox; one crash
  report; "This is sadly not mobile friendly in an amusing way" (controls covered the map on a
  phone); realism nitpicks ("Your default dwell times are wayyyyy too short", fixed versus moving
  block); requests for colour by line and a time slider.
- **Subway Builder**: the main complaints were the $30-40 price, US-only cities (its demand comes
  from the US Census, so non-US readers said they would wait), no free demo, and early-beta UI. Many
  had heard of it on Twitter: "Saw a lot of buzz about this on Twitter". A commenter said Miles in
  Transit had streamed it.

What that means for trainsworld: it answers both of Subway Builder's top HN complaints (free, any
city in the world). Say so plainly in the first comment, with the data sources. Expect realism
questions from people who know rail; answering them well is what keeps a thread going.

### 1.2 Reddit

Reddit refused my reader (403, and reading through mirror sites was blocked as bot-check
circumvention), so **I could not read any subreddit's current rules**. What is below comes from
search snippets and secondary guides, and each line says how sure it is. Before posting, open the
subreddit's rules page (`https://www.reddit.com/r/<name>/about/rules`) and the sidebar; it takes a
minute each.

Site-wide: the old "10% of your posts" self-promotion guideline is retired as an official rule, but
subreddits keep its spirit and moderators judge accounts that only post their own links
([redship guide, 2026](https://redship.io/blog/reddit-self-promotion-rules-2026), a secondary
source). An account with ordinary history beyond the game is the safe kind.

| Subreddit | Fit | What I could confirm | What to post |
|---|---|---|---|
| [r/WebGames](https://www.reddit.com/r/WebGames/about/rules) | Good: free games that run in a browser | **Not read.** Search found nothing on its rules | A link post straight to the game, plain title |
| [r/playmygame](https://www.reddit.com/r/playmygame/about/rules) | Good: made for developers | From a mirrored rules post ([2024](https://lr.ggtyler.dev/r/playmygame/comments/1eg08dh/how_to_avoid_having_your_posts_removed_version_2)): the game must be free to play, you must be on the team, give a direct link, say what the game is; titles carry a platform tag such as "[Web]"; check /new after posting, the spam filter catches posts | "[Web] trainsworld" + a comment saying what it is |
| [r/indiegames](https://www.reddit.com/r/indiegames/about/rules) | Good | A guide says promotion is allowed with the right flair ("Promotion", "Video") ([Steam Page Analyzer](https://www.steampageanalyzer.com/blog/reddit-indie-game-marketing)); an Aug 2024 mod post bans self-promotion dressed up as asking for feedback ([mirror](https://redlib.belloworld.it/r/indiegames)) | The clip, Promotion flair, link in a comment |
| [r/gamedev](https://www.reddit.com/r/gamedev/about/rules) | Only in weekly threads | Mirrored wiki pages: self-promotion goes in the weekly threads (Screenshot Saturday and others); standalone screenshot posts are removed ([posting FAQ mirror](https://red.127.pp.ua/r/gamedev/wiki/posting_guidelines_faq)) | A comment in Screenshot Saturday |
| [r/TransitDiagrams](https://www.reddit.com/r/TransitDiagrams/about/rules) | Good for the diagram export | **Not read.** A Jan 2025 meta thread discusses a "Rule 8" with set days for some kinds of post ([mirror](https://redlib.hbubli.cc/r/TransitDiagrams/comments/1hzcplq/we_should_update_rule_8)); check which day a game or tool post belongs on | One exported diagram as an image, "made with my game" and the link in a comment |
| [r/MapPorn](https://www.reddit.com/r/MapPorn/about/rules) | Good for a map image, not for a game link | Its FAQ (mirror): selling products is not allowed; any map can be submitted and the community decides ([FAQ mirror](https://rl.klein.ruhr/r/MapPorn/wiki/faq)). The free game is not selling | A striking map image of a built network, link in a comment |
| [r/transit](https://www.reddit.com/r/transit/about/rules) | Audience fits | **Not read.** Game posts do appear there (a [Jan 2025 post](https://axebps-redlib.hf.space/r/transit/comments/1i85nu1/experimenting_with_3d_in_my_subway_building_game) about a subway building game) | Only after reading the rules |
| [r/urbanplanning](https://www.reddit.com/r/urbanplanning/about/rules) | Weak: a discussion sub | **Not read** | Probably skip |
| [r/trains](https://www.reddit.com/r/trains/about/rules) | Weak: real trains and photos | **Not read** | Probably skip |
| [r/InternetIsBeautiful](https://www.reddit.com/r/InternetIsBeautiful/about/rules), [r/simulationgames](https://www.reddit.com/r/simulationgames/about/rules), [r/tycoon](https://www.reddit.com/r/tycoon/about/rules) | Possible extra fits | **Not read**; search found nothing on their rules | Read first |

**Poaching.** Posting in another game's subreddit or Discord reads as advertising to its players, and
these communities usually have a rule against other games' promotion (the common Discord pattern is
promotion only in a set channel, if at all). Stay out of r/SubwayBuilder, r/NIMBYRails,
r/CitiesSkylines, r/MiniMetro, r/OpenTTD, r/TransportFever and those games' Discords. A player
posting it there is fine.

### 1.3 Bluesky

Urbanists moved to Bluesky in late 2024 and early 2025. Brent Toderian (about 44k followers there)
says his X views fell from 20-40k a post to about 2k, and a thread that got 2,500 views on X drew
160 quote replies on Bluesky ([Planning magazine, Feb 2025](https://planning.org/planning/2025/feb/what-makes-bluesky-the-new-it-space-for-urbanists/)).
Starter packs are how people find each other there: Toderian's "Cities, Urbanism & City-Building"
and "Better Transportation & Mobility" packs ([bluefacts](https://bluefacts.app/starter-pack/brenttoderian.bsky.social)),
and a public transport, rail and urbanism pack by @railmaps.bsky.social, `go.bsky.app/NbrwCsd`
([via @transit.bsky.social](https://www.bushdrum.com/bsky/transit.bsky.social)). Directory of
transport packs: [blueskystarterpack.com/transport](https://blueskystarterpack.com/transport).
Post the same clip and text as on Twitter.

### 1.4 itch.io

- Uploads: at most 500 MB and 1,000 files after unzipping, 200 MB a file
  ([itch.io HTML5 docs](https://itch.io/docs/creators/html5)). trainsworld's data (2.9 GB, 14,248
  files) cannot go there, and does not need to.
- Pointing at a game hosted elsewhere: itch's docs don't cover it. Forum answers say to upload an
  `index.html` that holds an iframe of your own site, set the kind to HTML and tick "play in
  browser"; the iframe URL must be https ([forum](https://itch.io/t/28193/html-game-question-can-i-have-an-iframe-to-a-game-im-hosting-elsewhere)).
  The page inside the frame stays on anita.garden, so R2's CORS (which allows anita.garden) still
  works. Not tested: whether GitHub Pages allows framing; try it before relying on it.
- Turn "Mobile friendly" off unless phones work: with it off, itch warns phone visitors the game may
  not work ([docs](https://itch.io/docs/creators/html5)).
- Worth it? Rail Route ran its free open alpha on itch for two years before Steam (section 3). For a
  free browser game the gain is a listing in itch's browse pages; it is a week-after item.

### 1.5 Discords and forums

- **The Urbanist Discord**, `discord.gg/urbanist`, linked from Alan Fisher's own link page
  ([linktr.ee/alanthefisher](https://linktr.ee/alanthefisher)). Read its rules channel for a
  self-promotion channel before posting. **Not read.**
- **Tildes ~transport** ([tildes.net/~transport](https://tildes.net/~transport)) has railfans
  ([thread](https://tildes.net/~transport/1669/any_tildesian_railfans)); Tildes is invite only.
- Other games' Discords (NIMBY Rails' official server, where its developer answers players, per its
  [Steam page](https://store.steampowered.com/app/1134710/NIMBY_Rails/), and Subway Builder's): poaching, see 1.2.
- I found no rules pages for transit-game Discords through search (results were generic templates).

### 1.6 Twitter

Her home ground. Two points with sources:
- Clip first. Chris Zukowski's 2026 notes on game tweets that spread: post clips, not trailers;
  open on the strongest moment and cut to gameplay; "If everyone leaves after two seconds, something
  is probably wrong with the opening"; raw gameplay "can often feel more honest"
  ([howtomarketagame.com](https://howtomarketagame.com/2026/05/07/2026-tips-for-making-your-tweets-go-viral/)).
- Link in a reply. Musk said in 2024 that posts with links get less reach and suggested putting the
  link in a reply; he later said links have not been throttled for over a year
  ([ppc.land](https://ppc.land/x-drops-year-old-link-penalty-musk-tells-zuckerberg-on-platform/),
  [TechRound](https://techround.co.uk/news/musk-limiting-external-links-x)). The sources disagree,
  and a reply link costs nothing, so put it in the first reply.

## 2. People who might feature it

Only contact routes they publish. "Read" means I read the page; "not read" means the site blocked my
reader and the link is where the route should be. Write to the few that fit best, each with a short
personal note, rather than all of them.

| Who | Why them | Published route | |
|---|---|---|---|
| Miles in Transit (Miles Taylor, YouTube and blog) | Plays transit games: wrote about [Mini Metro](https://milesintransit.com/?p=918), and an HN commenter says he streamed Subway Builder | Form at [milesintransit.com/contact](https://milesintransit.com/contact/): "Just have some cool transit thing to tell me about?" | read |
| Alan Fisher, Armchair Urbanist (YouTube) | Big urbanist channel; on collaboration he says it "cross-pollinates" ([Streetsblog](https://sf.streetsblog.org/2026/04/24/train-tubers-a-talk-with-the-youtubes-transit-warriors)) | Email listed as "My Email" on [linktr.ee/alanthefisher](https://linktr.ee/alanthefisher) | read |
| kottke.org | Links to small web projects | [kottke.org/about](https://kottke.org/about/): "If you have a project or video you're proud of, send it along" (an email link) | read |
| Streetsblog | Runs transit videos and wrote about transit YouTubers | tips@streetsblog.org, [contact page](https://nyc.streetsblog.org/contact) | read |
| PC Gamer | Covered Slow Roads, a free browser game ([article](https://www.pcgamer.com/explore-endless-roads-in-this-beautiful-driving-game-that-runs-in-your-browser)), and Subway Builder | tips@pcgamer.com, [pcgamer.com/tips](https://www.pcgamer.com/tips) | from search of its own pages |
| Maps Mania (Keir Clarke) | Daily posts about web maps; covered [Metro Memory](https://googlemapsmania.blogspot.com/2023/10/the-london-tube-memory-game.html) and a [New York subway quiz](https://googlemapsmania.blogspot.com/2023/12/how-well-do-you-know-new-yorks-subway.html) | No tip address on the blog, only post comments ([blog](https://googlemapsmania.blogspot.com/)) | read |
| Planetizen | Covered Subway Builder ([Oct 2025](https://www.planetizen.com/news/2025/10/136214-nerd-alert-subway-builder-video-game-lets-you-build-your-dream-transit-system)) | [planetizen.com/contact](https://www.planetizen.com/contact) | not read |
| Greater Greater Washington | Covered Subway Builder players ([article](https://ggwash.org/view/101667/subway-builder-turns-metro-enthusiasts-into-transit-designers)) | [ggwash.org/contact](https://ggwash.org/contact) | not read |
| Not Just Bikes (YouTube) | The biggest channel in the space, "about a million subscribers" ([Streetsblog](https://sf.streetsblog.org/2026/04/24/train-tubers-a-talk-with-the-youtubes-transit-warriors)) | About tab of [youtube.com/notjustbikes](https://www.youtube.com/notjustbikes) | not read |
| CityNerd (Ray Delahanty, YouTube) | Former planner, ranks cities and transit | About tab of [youtube.com/CityNerd](https://www.youtube.com/CityNerd) | not read |
| Trains Are Awesome (Thom de Boo) | Transit YouTuber, @Thom-TRA ([Streetsblog](https://sf.streetsblog.org/2026/04/24/train-tubers-a-talk-with-the-youtubes-transit-warriors)) | About tab of [youtube.com/@Thom-TRA](https://www.youtube.com/@Thom-TRA) | not read |
| Car Free Keith (Keith Davis) | Transit YouTuber, @CarFreeKeith (same article) | About tab of [youtube.com/@CarFreeKeith](https://www.youtube.com/@CarFreeKeith) | not read |
| City Planner Plays (YouTube) | A planner who plays city builders ([wiki](https://youtube.fandom.com/wiki/City_Planner_Plays)) | Channel About tab | not read |
| Rock Paper Shotgun | Reviewed NIMBY Rails, "NIMBY Rails has made me a train person" (per [Wikipedia](https://en.wikipedia.org/wiki/NIMBY_Rails)) | Contact page on rockpapershotgun.com | not read |
| The Overhead Wire (newsletter) | Daily urbanism links | [theoverheadwire.com](https://www.theoverheadwire.com/) | not read (403) |
| Next City | Covered a free NYC subway design browser game in 2016 ([article](https://nextcity.org/daily/entry/nyc-subway-design-game-robert-moses)) | [nextcity.org/pitch](https://nextcity.org/pitch) takes freelance pitches and op-eds only, not tips, and says it does not accept pitches from AI agents | read; not a tip route |

City press picks up "your city as a game" stories: Metro Memory got Time Out, LBC, Londonist and
The Week ([Time Out](https://www.timeout.com/london/news/how-to-play-londons-viral-tube-metro-memory-game-102423),
[The Week](https://www.theweek.com/tech/why-the-metro-memory-game-is-so-addictive)); Brand New Subway
got Next City, NBC New York, 6sqft and Viewing NYC. Their tip routes were not read.

## 3. How similar games launched

| Game | Where it first got noticed | What to copy |
|---|---|---|
| **Mini Metro** (2013-15) | Made at Ludum Dare 26 (April 2013), partly in Unity because a browser entry would draw more attention; the free web alpha (Sept 2013) had the web player "front and center" on the site and the team credits its "virality" to how easy it was to get into; then Greenlight, Early Access (Aug 2014), and "a good chunk" of free alpha players moved to the paid beta. They doubt the studio "would be operating now" without the web build. Its subway-map idea read at a glance from one screenshot ([postmortem](https://gamedeveloper.com/audio/postmortem-dinosaur-polo-club-s-i-mini-metro-i-)) | Play with no barrier, and a picture that explains the game without words |
| **Subway Builder** (2025-26) | Colin Miller posted on 10 Jan 2025 that he was making "a game where you can build a subway network"; a video of the early demo drew enough attention that he made it a full game ([Wikipedia](https://en.wikipedia.org/wiki/Subway_Builder), [The Banner](https://www.thebanner.com/community/transportation/subway-builder-baltimore-transit-system-VPYKWAWVZZGZ7BSAQ6H6J6ZFVQ)). Before the 9 Oct 2025 beta, hype ran on X: players' and the developer's screenshots of their cities ("rush hour in Boston") ([Punished Backlog](https://punishedbacklog.com/subway-builder-internet-hype/)). HN 303 points on beta day. Press: NY1, Fast Company, PC Gamer, GGWash, Planetizen | Short clips of real cities while it is being made; players posting their own cities did the rest |
| **NIMBY Rails** (2020-21) | Revealed mid-March 2020 in a development retrospective on the developer's [devblog](https://carloscarrasco.com/); Steam Early Access 26 Jan 2021 with a VICE article the same day; Rock Paper Shotgun review July 2021 ([Wikipedia](https://en.wikipedia.org/wiki/NIMBY_Rails)). The developer answers players on Discord and the Steam forums ([Steam](https://store.steampowered.com/app/1134710/NIMBY_Rails/)) | A regular devblog, and being there to answer |
| **Rail Route** (2019-21) | Free open alpha on itch.io from Aug 2019 (v0.6), Steam Early Access 23 June 2021 ([devlog](https://bitrich.itch.io/railroute/devlog/94582/rail-route-06-now-available-on-itchio), [EA launch](https://bitrich.itch.io/railroute/devlog/266656/rail-route-officially-launched-on-steam)); over 500 reviews, 94% positive, about half a year in ([interview](https://intoindiegames.com/reviews/previews/rail-route-interview-build-a-rail-network-and-become-a-train-dispatcher/)) | A free build gathers players early |
| **NYC Subway Simulator** (buildmytransit.nyc, 2025) | Show HN, 201 points, a long-weekend project on OSM data ([thread](https://news.ycombinator.com/item?id=44490588)) | Show HN works for a transit sim; phones and realism come up first |
| **Metro Memory** (2023) | Launched 20 Oct 2023 (London); over half a million players within weeks; then Paris, Berlin, Munich, Hamburg and Mexico City editions, each with local press ([Ilford Recorder](https://ilfordrecorder.co.uk/news/23879238.london-metro-memory-game-play), [The Week](https://www.theweek.com/tech/why-the-metro-memory-game-is-so-addictive), [Sopitas](https://www.sopitas.com/noticias/metro-memory-juego-nombres-estaciones-metro-cdmx/)) | Each city is its own story for that city's press and subreddit |
| **Slow Roads** (2022) | The developer posted a video of it on Twitter; PC Gamer and Jalopnik wrote it up within days; the publisher says over a million players in two weeks ([Jalopnik](https://www.jalopnik.com/slow-roads-best-browser-driving-game-1850304751/), [G2A](https://www.g2a.com/news/game-spotlights/slow-roads-the-viral-browser-game-where-you-just-drive-forever-launches-on-steam-today/), a publisher claim) | One good video on Twitter plus a free browser game is enough for games press |
| **Brand New Subway** (2016) | A free browser NYC subway designer from a game design contest; Next City, NBC New York, 6sqft, Viewing NYC ([Next City](https://nextcity.org/daily/entry/nyc-subway-design-game-robert-moses)) | NYC local press covers NYC transit toys |
| **Metro Dreamin'** | I found no account of its launch; search only turned up user maps on [metrodreamin.com](https://metrodreamin.com) | |

The pattern: every one of these got noticed through a short video or a screenshot of a real place
that explains itself, and the free ones spread because playing took one click.

## 4. Before launch: first impression

### 4.1 Link previews

Checked 2026-11-05: the live `website/trainsworld/index.html` (and `curl https://anita.garden/trainsworld/`)
has a `<title>trainsworld</title>` and `<meta name="description" content="Build trains around the
world!">` and no `og:` or `twitter:` tags at all. Her other maps have them (`website/nycriders/index.html`:
`og:image`, `twitter:card summary_large_image`, `twitter:image`).

The tags belong in `trainworld/app/index.html`: `tools/deploy.py` copies the built index.html into
the website repo, so a hand edit there is lost at the next deploy. The image goes in `app/public/`
so the build copies it beside the page. Image URLs must be absolute:

```html
<meta property="og:title" content="trainsworld" />
<meta property="og:description" content="(her wording)" />
<meta property="og:image" content="https://anita.garden/trainsworld/preview.png" />
<meta property="og:url" content="https://anita.garden/trainsworld/" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:image" content="https://anita.garden/trainsworld/preview.png" />
```

X's large card: 2:1, at least 300x157, at most 4096x4096, under 5 MB, JPG/PNG/WEBP/GIF, no SVG; a
smaller image falls back to the small card; X caches cards for a while, so get it right before the
first post ([X docs as quoted](https://dev.twitter.com/cards/types/summary-large-image), via search).
1200x600 is the safe size. A map of a built network or a diagram export, with no text on it, will
read at thumbnail size. Not checked: Bluesky's own card rules; it builds cards from the page's tags.
To test, paste the link into a Bluesky or Discord draft and look before sending.

### 4.2 What a phone visitor sees

Many clicks from Twitter and Bluesky will come from phones. Known so far:
- `research/first-ten-minutes.md` (S8): at 390x844 the game is playable, but the map keeps half the
  screen and the first tool is two taps away. That was headless Chrome, not a phone.
- `notes/T-2472.md`: an empty New York game uses 1.4-1.5 GB in desktop Chrome (renderers 1.0 GB).
- iOS closes a tab based on system memory pressure, not a fixed limit, and reloads it without
  warning when WebAssembly memory grows too far; one report found only about 300-500 MB reliably
  available to WebAssembly pages on iOS 17 devices ([WebKit bug 269777](https://bugs.webkit.org/show_bug.cgi?id=269777),
  [bug 284752](https://bugs.webkit.org/show_bug.cgi?id=284752)).

So the real risk is an iPhone visitor getting a reload loop and leaving. Test on a real iPhone and
a mid-range Android. If it fails, a short note for phones ("best on a computer for now") is better
than a crash; whether to add one is her call.

### 4.3 Time to the first track

- Poki: "Drop-off during loading is brutally underestimated"; skip title screens and level selects
  ([Poki guide](https://developers.poki.com/guide/easy-access)), quoted in `research/first-ten-minutes.md`.
- T-2087 walked the start: a New York game was playable 3 s after Start; the first load is about
  22 MB (`notes/T-2472.md`), the first line pulls 14.4 MB more.
- Of the T-2087 ideas, "opens on the Build tab" and "trains at every level" are done (T-2092). "Start
  in a city by default" (idea 1) is not in the archive as done; if Start still lands on the world,
  that is the cheapest fix for a launch crowd.
- Measure it once like a stranger: a new Chrome profile, DevTools network "Fast 4G", cache off,
  stopwatch from opening the link to the first track drawn.

### 4.4 Clip or screenshots

Both: the clip leads, the stills follow. Every launch in section 3 spread on a short video or a
picture of a real place. For the clip: start on trains already running on a network in a city
people know, no title card, no slow zoom; show one track being drawn and the riders count moving;
end on the auto-drawn network diagram, which no other transit game has. 20-40 s, recorded at full
quality (Zukowski: no blur, stutter or odd compression). Stills: one city map, one diagram export,
one non-US city (it answers HN's main Subway Builder complaint).

### 4.5 Feedback

The in-game form exists (T-1673): it posts to `market.anita.garden/trainworld/feedback`, which mails
it to her, at most 10 an hour per IP; `python tools/feedback.py` in shonei-server lists reports. The
live config answers `"trainworldFeedback":true`, so it is on. Send one test from the live site. A
public place to talk (a Discord, a subreddit) is optional; NIMBY Rails and Mini Metro both grew one,
but the launch threads themselves will carry the first week.

## 5. Hosting cost in a spike

### 5.1 Cloudflare R2

Prices ([R2 pricing](https://developers.cloudflare.com/r2/pricing/)), Standard storage:
- Storage $0.015 per GB-month; Class A (writes, lists) $4.50 per million; Class B (reads) $0.36 per
  million; egress free.
- Free each month: 10 GB-month storage, 1 million Class A, 10 million Class B. The free tier is for
  the whole account, so religiondots and the other maps share it.
- GetObject and HeadObject are Class B. The pricing page bills per request and does not mention Range
  requests, so a Range read is one GetObject, one Class B operation, whatever its size. That is an
  inference from the page, not a stated rule; no search found it stated either way.
- Failed auth (401, signature 403) is not billed.

Limits ([R2 limits](https://developers.cloudflare.com/r2/platform/limits/)): r2.dev has a variable
rate limit (429s at hundreds of requests a second) and is "not intended for production"; the game
uses the custom domain `data.anita.garden`, which has no stated read limit. No per-object read limit
is listed.

What reaches R2: `data.anita.garden` answers `cf-cache-status: DYNAMIC` (her "data bypass" Cache
Rule, memory note "Publishing a map"), so every request a player makes goes to R2 and counts. The
files carry `Cache-Control: public, max-age=31536000, immutable` (checked on `trainworld/2026-11-03b/nyc.json`),
so a returning player's browser asks again only for new files.

Requests per visitor are **estimated, not measured**: T-2472 counted bytes, not requests. A New York
first load reads whole bins, Range reads of other bins, pack JSON, outlines, terrain, boundaries and
places, a few files per zone in view, which I put at 100-200 requests; a session that visits a few
cities and builds lines, about 300; a long one, up to 1,000.

| Visitors in a day | Class B at 300 each | Cost | At 1,000 each | Cost | Bytes out of R2 |
|---|---|---|---|---|---|
| 1,000 | 0.3 M | $0 (free tier) | 1 M | $0 | about 22-50 GB, free |
| 10,000 | 3 M | $0 (free tier) | 10 M | $0 to $3.60 | about 220-500 GB, free |
| 100,000 | 30 M | about $7 | 100 M | about $32 | about 2-5 TB, free |

Storage: 2.87 GB per data version (T-2472). Two kept versions plus the other maps may pass 10 GB;
each GB over costs $0.015 a month, so a forgotten old version is about 4 cents a month. Not a worry,
but delete old versions anyway (`rclone lsf r2:anitamaps/trainworld/ --dirs-only` lists them).

To check one session's request count: DevTools Network tab, filter `data.anita.garden`, read the count
at the bottom.

### 5.2 The surprise: GitHub Pages, not R2

The code half is on GitHub Pages behind Cloudflare. Checked 2026-11-05 with `curl -I`:

| File | cf-cache-status | So |
|---|---|---|
| `assets/index-*.js` | EXPIRED (cached, revalidated) | Cloudflare caches .js |
| `assets/sim_bg-*.wasm` | DYNAMIC | every visit goes to GitHub |
| `assets/market-rows-*.json` (1.75 MB gzipped) | DYNAMIC | every first line goes to GitHub |

Cloudflare does not cache JSON or HTML by default, and .wasm is not on its default list
([default cache behaviour](https://developers.cloudflare.com/cache/concepts/default-cache-behavior/)).
That is about 3 MB from GitHub per visitor (WASM 1.2 MB, market rows 1.75 MB): 3 GB a day at 1k
visitors, 30 GB at 10k, 300 GB at 100k. GitHub Pages has a soft limit of 100 GB a month; past it
GitHub may rate-limit with HTTP 429, may be unable to serve the site, or sends "a polite email"
([Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)).
The other maps on anita.garden would be hit too.

Fix: a Cache Rule on the anita.garden zone, "Eligible for cache", matching
`anita.garden/trainsworld/assets/*`. Vite gives those files hashed names, so a long edge TTL is safe.
Leave `index.html` uncached so a deploy shows at once. The wrangler token cannot edit cache rules
(memory note), so this is a dashboard step for her.

### 5.3 Other services a spike touches

- **OpenFreeMap** (basemap, 0.8-1.1 MB per city view per T-2472): "no limits on the number of map
  views or requests", commercial use allowed, attribution required (MapLibre adds it), donations
  asked for ([openfreemap.org](https://openfreemap.org/)). At 100k visitors that is a few hundred GB
  of their bandwidth; a donation would be a kind gesture if it takes off.
- **market.anita.garden** (her server): one config read per page load and a playtime ping about once
  a minute per open game (shonei-server CLAUDE.md), so 5,000 people playing at once is roughly 80
  requests a second. Plays and minutes per day: `curl https://market.anita.garden/trainworld/stats`.
- **Terrain tiles** come from the AWS open data bucket (`basemap.ts`); not checked for limits.
- **Pages vs R2** (memory note): never move a Range-read file to the website repo; Pages behind
  Cloudflare breaks Range requests and the symptom is missing data, not an error.

## 6. Notes on what I could not do

- Reddit rules: blocked (403; mirrors refused as bot-check circumvention). Every Reddit row says
  how sure it is.
- Contact pages at GGWash, Planetizen, Londonist, Rock Paper Shotgun and The Overhead Wire answered
  403 or refused my reader; YouTube About tabs did not render. They are marked "not read".
- R2 requests per visitor are an estimate; one DevTools session would replace it with a count.
- No Discord's rules were readable from outside.

## 7. Drafts (for her to reword)

Show HN title (under 80 characters):

    Show HN: trainsworld, a free browser game to build rail lines in any city

First comment:

    you draw track and stations in any real city, and riders come from open
    data (OSM, GHSL and others) about where people live and where they go. it
    runs in the browser, rust compiled to wasm with maplibre on top. it also
    draws a schematic map of your network by itself, which you can save as an
    image.

    it's free and needs no account. happy to answer questions about the demand
    model or the data.

(Add how long it took and why she made it; Show HN asks for "how and why". Add a line about phones
once she knows how they do.)

Tweet (link in the first reply):

    i made a train game. build lines in any city in the world, and see who rides them

Each draft follows her text rules (no dashes between phrases, no hype words); she rewords.
