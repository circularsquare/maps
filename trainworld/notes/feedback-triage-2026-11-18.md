# Feedback triage, 2026-11-18

The 78 open reports with no reply from you, checked against v0.4.0, what landed since, TASKS.md and DECISIONS.md.
Reports from the same player on the same thing are merged (4 pairs), so there are 74 blocks.

| Group | Reports | Blocks |
|---|---|---|
| 1. Close with no reply | 4 | 4 |
| 2. Reply, then close | 49 | 47 |
| 3. Reply and keep open | 11 | 11 |
| 4. Needs your decision first | 14 | 12 |

How I sorted: group 1 is only where a reply tells the player nothing (one word, too vague, handled on Discord).
Group 2 is everything where nothing more is needed from the player: a bug that is fixed, a request that has a task
("it's on the list"), a how-to, or something you already said no to. Group 3 is only where the player has to answer
(does it still happen on v0.4, which device, a save). Group 4 is where the reply depends on a call you haven't made.
If you'd rather keep open bugs with a task open until they're fixed, move those group 2 blocks to group 3; they say
"(open task)" in the status line.

"fixed" below means a changelog or archive line names the same failure. Replies don't promise dates, and say "on the
list" only where a task exists. Control names in translations are the game's own strings for that language.

---

## 1. Close with no reply

**1794764329-b77884c3**, anon (the French player on Discord), English
- Reported: "report on discord with not loading save game".
- Status: handled with you on Discord (notes/jordan-save-investigation-2026-11-15.md); v0.3.5 fixed the Safari save
  bug; save recovery is T-3395.

**1794759131-ef1c8bb5**, kt, Japanese
- Reported: one word, "backup".
- Status: nothing to answer. Backups are T-3395; v0.4.0 keeps the last undamaged copy of a damaged save.

**1794314418-62f0b3a8**, qwas, Japanese, Safari 16, 2026-11-10
- Reported: "data could not be loaded", nothing else.
- Status: too vague to act on, from before v0.3. Old Safari fixes went out since (v0.3.3, v0.3.5).

**1794177285-8de1f1ac**, anon, Japanese, 2026-11-08
- Reported: "values don't update", nothing else.
- Status: from before v0.1; frozen figures were fixed in T-3313 (v0.3.x).

---

## 2. Reply, then close

### Fixed in v0.4.0

**1794923198-73ff6bb0**, anon, Korean
- Reported: turning single tram track into double track is hard.
- Status: v0.4.0 put single/double and Rebuild in the track panel of any built piece (T-3408). The sim has no tram
  refusal for single to double, but I didn't try it in the game, so the reply asks to hear back if it fails. Repeat
  request: 4 reports asked for single to double.
- Reply: thanks. since v0.4 you can click a built piece of track and switch it between single and double right in
  its panel, then press Rebuild. if that doesn't work on your tram track, tell me.
- Korean: 고마워요. v0.4부터는 건설한 선로를 클릭하면 선로 패널에서 단선과 복선을 바로 바꾸고 "다시 짓기"를 누를 수 있어요. 트램 선로에서 안 되면 알려 주세요.

**1794617845-10206e23**, syazanie_amirin, English
- Reported: wants to upgrade single track to double, can't find a way.
- Status: fixed: the Select tool's track count since v0.3.3, and the track panel's Rebuild in v0.4.0 (T-3408).
- Reply: thanks. since v0.4 you can click a built piece of track and switch it between single and double right in
  its panel, then press Rebuild. for a longer stretch, pick it with the Select tool, choose the two line icon, then
  Rebuild.

**1794915126-6da18fba**, Christian, English
- Reported: loves the game; auto station names could take the closest real station's name.
- Status: added in v0.4.0 (a station within 150 m of a real station takes its name, T-3571).
- Reply: thank you <3 v0.4 does this now: a new station placed near a real world station takes its name. it only
  looks within about 150 m, so a station far from any real one still gets the usual name.

**1794752820-5af81eb7**, anon, Japanese
- Reported: split a double track into two singles on separate routes, and one-direction stations on the singles
  can't be set up.
- Status: fixed in v0.4.0 (one-way station fixes for split lines, T-3565 and others).
- Reply: thanks for building that out. v0.4 fixed this: when a double track line splits into two single track
  lines, trains now stop correctly at one way stations on the single track parts.
- Japanese: 作り込んでくれてありがとうございます。v0.4で直しました。複線の路線を単線の路線2本に分けたとき、単線区間にある一方向の駅に列車が正しく停車するようになりました。

**1794616312-f39422e3**, simon, English
- Reported: Clydebank, Dumbarton and Bonhill outside Glasgow show no data.
- Status: fixed in v0.4.0 (Glasgow rebuilt with West Dunbartonshire, T-617).
- Reply: thanks for spotting it. v0.4 added the missing demand in Clydebank, Dumbarton and the rest of West
  Dunbartonshire.

**1794604978-c087fb80**, anon, English
- Reported: the real world overlay could show station dots and names.
- Status: added in v0.4.0 (T-3570).
- Reply: added in v0.4. in Settings, under Real world network, turn on Show stations. hover a station to see its
  name.

**1794498765-b68f53c7**, loopo, English
- Reported: trips are recalculated on every click of new track even while paused; each click waits about 5 s.
- Status: fixed in v0.4.0 (setting to hold off recalculating until you unpause, plus faster edits).
- Reply: thanks. v0.4 has a setting for this: pause, then turn off Recalculate trips while paused in Settings.
  trips are calculated again once you unpause. edits also show up faster than before.

**1794291447-bf0c4ad8**, anon, English
- Reported: more riders on intercity lines, please.
- Status: v0.4.0 fixed the bug where local lines lowered intercity train use (T-3426). A common request (6+ reports).
- Reply: v0.4 fixed a bug where local lines were lowering intercity train use, so most intercity lines carry a lot
  more now. feeder lines into the intercity stations help too.

**1794462461-b0dd8cff**, anon, Japanese
- Reported: wants extra fares for limited express trains or reserved seats.
- Status: v0.4.0 added a fare multiple per line, 0x to 3x (T-3567). Repeats other reports asking for express fares.
- Reply: v0.4 added a fare multiple for each line, from 0x to 3x, in the line's panel. you can set an express line
  to charge more.
- Japanese: v0.4で路線ごとの運賃倍率を追加しました。路線の画面で0倍から3倍まで設定できるので、特急の路線だけ運賃を高くできます。

**1794667196-f7d45679**, anon, Japanese
- Reported: limited express cars (fewer seats, higher fare), and more long-distance riders.
- Status: fare part and more intercity riders are in v0.4.0. Cars with fewer seats are a design question with no
  task; the reply leaves it out.
- Reply: thanks. two things in v0.4 help here: a fare multiple for each line (0x to 3x, in the line's panel), so an
  express line can charge more, and a bug fix that raised intercity train use a lot.
- Japanese: ありがとうございます。v0.4の2つの変更が役に立つと思います。路線ごとの運賃倍率（路線の画面で0倍から3倍）で特急の路線だけ運賃を高くできるようになったのと、都市間列車の利用を減らしていたバグを直して、利用が大きく増えました。

**1794720389-9984d28e**, shuangnagi, Chinese (with English)
- Reported: Chengdu Tianfu Airport sits in Jianyang; the game shows very few trips between it and Chengdu.
- Status: the rule is built: an airport in a zone outside its city takes its passengers from the city, up to its
  published passengers; it comes with the next data rebuild (docs/demand-plain.md, T-3245). v0.4.0 also fixed the
  airport hover leaving out trips from other zones (T-3500), which may be part of what he saw; not confirmed.
- Reply: thanks for the detail. airports that sit outside the city they serve, like Tianfu, getting their trips from
  that city is on the list, with a later data update. v0.4 also fixed the airport hover leaving out trips from other
  zones, so it may already look better.
- Chinese: 谢谢你写得这么详细。像天府机场这样位于所服务城市之外的机场，让它的出行来自那座城市，这已经在计划里了，会在之后的数据更新中加入。另外v0.4修复了机场悬停信息漏算其他区域出行的问题，所以现在可能已经好一些了。

**1794825926-17cb49d2**, Kaeruotoko (蛙男), Japanese
- Reported: long-distance ferries need better treatment.
- Status: v0.4.0 made ferries faster (45 km/h, was 32). A second pass on long-distance ferries is T-3435 (open task).
- Reply: v0.4 made ferries faster, 45 km/h instead of 32. long distance ferries are on the list for another look.
- Japanese: v0.4でフェリーを速くしました（時速32 kmから45 kmに）。長距離フェリーはもう一度見直す予定のリストに入っています。

**1794827081-b548bc07**, akou, Japanese, Safari 18.5
- Reported: the game crashes soon after building track; asks for more capacity.
- Status: likely memory (device unknown). v0.4.0 uses less memory (about 130 MB less on phones, 50-170 MB less on
  desktop); not a full fix.
- Reply: sorry about the crashes. v0.4 uses less memory, which should help. big networks can still run out of
  memory, mostly on phones and tablets; the game works best on a computer.
- Japanese: 落ちてしまってごめんなさい。v0.4でメモリの使用量を減らしたので、少しましになると思います。ただ、大きな路線網だと、特にスマホやタブレットではまだメモリが足りなくなることがあります。パソコンでのプレイがいちばん快適です。

**1794413129-2219ec18**, anon, Japanese
- Reported: too heavy; hard to build lines by tapping.
- Status: speed and memory work in v0.4.0 (less memory on phones, faster loads and edits); not a full fix.
- Reply: sorry it's slow. v0.4 uses less memory on phones and loads and edits faster, so it should feel lighter. the
  game still works best on a computer.
- Japanese: 重くてごめんなさい。v0.4でスマホのメモリ使用量を減らし、読み込みや編集も速くなったので、前より軽くなっていると思います。それでも、パソコンでのプレイがいちばん快適です。

**1794851474-5cefd3cd**, Ethan, English
- Reported: weekends make big places crowded; the game should reflect that.
- Status: already in the game (Saturday shopping up 40%, Sunday outings up two thirds, commuting down;
  docs/demand-plain.md), and v0.4.0 added weekend schedules.
- Reply: thanks. weekends already have their own trips in the game: on saturdays shopping is up by about 40% and
  commuting is down, and on sundays outings are up. v0.4 added weekend schedules too, so you can run a different
  number of trains on weekends and holidays.

### Fixed earlier (the first two only partly)

**1794469823-6ca3ebcb**, anon, Japanese
- Reported: where two opposite singles (a de facto double) meet a double, trains from the double cross to the
  other track with no crossover, ignoring one-way.
- Status: partly fixed: T-3344 (v0.3.3) fixed this drawing order; more wrong-side joins are fixed in the next update
  (T-3609), and "two single tracks into a double" was still listed as a known bug on 11-17 (notes/T-3527.md). Not
  all cases are known fixed.
- Reply: thanks. several cases of this were fixed in v0.3.3 and v0.4, where a single track joined the wrong track of
  a double, and the next update fixes a few more. if trains still cross over where there's no crossover, a screenshot
  would help.
- Japanese: ありがとうございます。単線が複線の反対側の線路につながってしまうケースを、v0.3.3とv0.4でいくつか直しました。次のアップデートでもさらにいくつか直します。それでも渡り線のない所で反対側に入るようなら、スクリーンショットを送ってもらえると助かります。

**1794547102-2b67ef81** and **1794536610-e7c57cf7**, soleurs, English (merged)
- Reported: singles joined to the end of a double look connected but lines can't run on them; can't get two singles
  to turn into one double.
- Status: partly fixed (T-3344 in v0.3.3; a sideways shift at such joins in v0.4.0; T-3609 in the next update).
  "Two single tracks into a double" was still a known bug on 11-17 (notes/T-3527.md), and drawing it a different
  way clears some cases (T-3618, your word 11-18). The track rework (T-3444) covers the rest.
- Reply: thanks. several cases of this were fixed in v0.3.3 and v0.4, and the next update fixes a few more. if a
  transition still won't work, drawing the double track first and then the singles off its end often helps. a
  screenshot would help me too.

**1794477454-621202f1**, Oh-my-gah (おーまいがー), Japanese
- Reported: showing Hyogo turns into Osaka.
- Status: fixed 2026-11-12 (T-3318, out in v0.3.3): the Kansai region showed Osaka city's name in Japanese; it
  shows Kansai now.
- Reply: thanks. the region covering Kyoto, Osaka, Hyogo and Nara was showing Osaka city's name in japanese. it shows
  as Kansai now.
- Japanese: ありがとうございます。京都・大阪・兵庫・奈良をまとめた地域が、日本語では大阪市の名前で表示されていました。今は「関西」と表示されます。

**1794402162-f7f13cac**, anon, Japanese, Safari 27
- Reported: station rider counts stuck at 0 and not changing.
- Status: fixed (T-3313: riders at 0 on builds before v0.3).
- Reply: thanks. this was a bug in builds before v0.3, where the numbers could freeze. it's fixed now.
- Japanese: ありがとうございます。v0.3より前のバージョンで、数字が止まってしまうバグでした。今は直っています。

**1794376285-71a5d5ab**, kurimeshi, Japanese
- Reported: after raising the train count, every train vanishes on reaching its terminal.
- Status: fixed (T-3313: trains vanishing at terminals after a count raise, builds before v0.3).
- Reply: thanks. this was a bug in builds before v0.3. it's fixed now, trains keep running after you add more.
- Japanese: ありがとうございます。v0.3より前のバージョンのバグでした。今は直っていて、本数を増やしても列車は走り続けます。

**1794363667-94733a92**, rinkalk, Japanese
- Reported: the construct button for track and stations stopped coming back.
- Status: fixed (T-3313: "Construct gone" on builds before v0.3).
- Reply: thanks. this was a bug in builds before v0.3. it's fixed now.
- Japanese: ありがとうございます。v0.3より前のバージョンのバグでした。今は直っています。

**1794475501-3eb3c3f2**, anon, English
- Reported: wants to switch which track trains use, like Australia's left-hand running.
- Status: added in v0.3.3 (left-hand running, per save).
- Reply: you can do this now: in Settings, under Game, set the side trains run on to Left. it's kept per save.

### How-to

**1794158060-18669327**, anon, English, 2026-11-08
- Reported: "can't even delete a track".
- Status: works (Delete tool, Select tool, delete keys since v0.1); it's in the help list.
- Reply: sorry it wasn't clear. use the Delete tool on one piece, or pick a stretch with the Select tool and click
  Remove. it's also in the help list under the ? button.

**1794467174-685efe8b**, anon, Chinese
- Reported: "why can't it recognise a loop line?"
- Status: not a bug (T-3317): a loop closes only when asked, by right clicking its first stop.
- Reply: a loop line closes only when you ask it to: while adding stops, right click the stop the line should loop
  back to (on a phone, hold it).
- Chinese: 环线只会在你指定时闭合：添加停靠站时，右键点击线路要绕回的那一站（手机上长按）。

**1794751467-6a7ac139**, anon (匿名, "anonymous"), Japanese
- Reported: built a station shared by 3 lines and wants through-running between them; how?
- Status: works: a line runs on any connected track; copy a line and edit its stops.
- Reply: thank you, glad you're enjoying it. a line can run over any track that's connected, so for through running,
  use Copy line in one line's panel, then Edit stops to carry it on to the other line's stations. the tracks need to
  join where the lines meet.
- Japanese: いつも遊んでくれてありがとうございます。路線はつながっている線路ならどこでも走れるので、直通運転は、片方の路線の画面で「路線を複製」を押して、「停車駅編集」でもう一方の路線の駅まで停車駅を延ばしてください。路線同士が出会う所で線路がつながっている必要があります。

**1794635355-e5d99426**, uriekarch, English
- Reported: wants to select a station and rename it right away; renaming many is tedious.
- Status: works: the pencil or a double click on the name in the station panel; a triple click in the Stations list.
- Reply: you can rename from the station's panel: click the pencil or double click the name. in the Stations list, a
  triple click on a name renames it too, which is quicker for many stations.

**1794477119-ec914dbc**, anon, Chinese (Traditional)
- Reported: 1. build on top of real lines without paying for them, 2. choose the real network at new game, 3. more
  accurate map, 4. a satellite map mode?
- Status: real networks are in Sandbox (v0.3.3); satellite map is in (v0.3.3). Point 1 for normal games has no task
  and the reply doesn't promise it.
- Reply: thanks. real world networks are in Sandbox: start a new game in sandbox mode and pick one, and you can add
  your own lines to it. a satellite map is in Settings too (Satellite).
- Chinese (Traditional): 謝謝。真實路網在沙盒模式裡：用沙盒模式開新遊戲並選一個真實路網，就能在上面加你自己的路線。衛星地圖也在設定裡（衛星）。

**1794535499-e57eacc9**, Gabriel, English
- Reported: wants trains per day per line and demand by day and time of day; offers to help as a developer.
- Status: weekend schedules and timetables in v0.4.0; more service periods are T-3542 (open task). The code is on
  GitHub (v0.3.3), so he can send a pull request. (There is a GabrielMoraez with PR 13; can't tell if it's him.)
- Reply: thank you, glad it made your week. v0.4 added weekend schedules and a timetable, and more periods in the
  day are on the list. the code is on github now, at circularsquare/trainsworld.

**1794159849-7bcd307d**, qqq, English
- Reported: wants a server to give feedback in; an option for English station names (building in Korea).
- Status: Discord link is in Settings (v0.1.1); English station names are T-2586 (open task); v0.4.0 names stations
  near a real station after it in Latin letters.
- Reply: thanks! there's a discord now, the link is in Settings. since v0.4 a station placed near a real station
  takes its name in latin letters, and an option for english station names everywhere is on the list.

### On the list

**1794852386-a26aeec0** and **1794759555-92f1270b**, Yunagi Fuji (夕凪フジ), Japanese (merged: two topics)
- Reported: wants quad track paired by direction, like a highway's two lanes each way; Kobe Airport's demand is
  offset from the airport.
- Status: quad paired by direction is T-3348 under the track rework T-3444 (open task; 4 reports ask for it). Kobe
  Airport is T-3433 (open task).
- Reply: thanks for both. quad track paired by direction is on the list with the track rework, and so is moving
  Kobe Airport's trips to the terminal.
- Japanese: 2件ともありがとうございます。方向別複々線は線路の作り直しと一緒にやる予定のリストに入っています。神戸空港の需要の位置をターミナルに合わせるのもリストに入っています。

**1794711765-0024a40c**, iyonotami, Japanese
- Reported: wants quad track paired by direction.
- Status: T-3348 under T-3444 (open task).
- Reply: thanks. quad track paired by direction is on the list, with the track rework.
- Japanese: ありがとうございます。方向別複々線は、線路の作り直しと一緒にやる予定のリストに入っています。

**1794485386-c65a527c**, massan16, Japanese
- Reported: wants quad track paired by direction, not only by line.
- Status: T-3348 under T-3444 (open task).
- Reply: thanks. quad track paired by direction is on the list, with the track rework.
- Japanese: ありがとうございます。方向別複々線は、線路の作り直しと一緒にやる予定のリストに入っています。

**1794589078-277bb8c7**, anon, Japanese
- Reported: enjoying the new features; wants one-direction double track and stations to build quad paired by
  direction; at passing stations, locals sometimes leave before the express.
- Status: quad by direction is T-3348/T-3444; locals leaving first is a known report under T-3346 (open tasks).
- Reply: thank you! quad track paired by direction is on the list with the track rework, and so is the express
  leaving first when both stop at a passing station.
- Japanese: ありがとうございます！方向別複々線は線路の作り直しと一緒にやる予定のリストに入っています。待避駅で両方が停車しているときに速達列車を先に発車させるのも、リストに入っています。

**1794605160-b253985b**, anon, English
- Reported: wants a three-track option or custom track counts.
- Status: any number of parallel tracks is part of the track rework, T-3444 (open task).
- Reply: thanks. more track counts are on the list, as part of a track rework.

**1794576939-7d14052e**, anon, English (machine translated, source language not given)
- Reported: wants five demand levels, to add trains in the morning only.
- Status: T-3542, four or five service periods (open task).
- Reply: thanks. more periods in the day, like a separate morning, are on the list.

**1794780449-1c973e41**, myworld3, English
- Reported: Seattle to Victoria ferry has no riders and shows no route.
- Status: T-3431 (open task, after T-2216): trips between two landmasses with no car or plane link aren't offered yet.
- Reply: thanks. trips between Seattle and Victoria aren't in the game yet, since they're on two landmasses with no
  road link. that's on the list, and the ferry will get riders once it's in.

**1794410532-56c10e5e**, Kamematsudo (亀松堂), Japanese
- Reported: in Sendai's real network, the Senzan Line skips Toshogu and starts oddly from Kita-Sendai.
- Status: fix is written, the Sendai rebuild is T-3325 (open task, parked for a later batch).
- Reply: thanks. the Senzan Line stopping at Toshogu is on the list.
- Japanese: ありがとうございます。仙山線が東照宮に停車するように直すのは、リストに入っています。

**1794498601-e683bb03**, anon, Japanese
- Reported: trains pass each other on single track.
- Status: reported on a v0.3/v0.3.1 build; v0.3.3 "fixed trains on single track drawn passing through each other"
  (T-3202). What's left of the drawing is T-3264 (open task).
- Reply: thanks. this was fixed in v0.3.3: trains on single track are no longer drawn passing through each other.
  they always took turns on the track, it was only the drawing. a few leftover drawing cases are on the list.
- Japanese: ありがとうございます。v0.3.3で直しました。単線で列車がすれ違って見えることはなくなりました。実際にはずっと交互に線路を使っていて、表示だけの問題でした。残っている表示の問題もリストに入っています。

**1794637913-ba90914d**, anon, Japanese
- Reported: a train stands a long while at the passing section before single track though it could go on.
- Status: the drawing's hold, not the sim (notes/T-3264.md); T-3264 (open task).
- Reply: thanks. that's how the train is drawn, not how it runs: it's held on screen longer than it is in the game.
  a fix for the drawing is on the list.
- Japanese: ありがとうございます。それは列車の表示の問題で、実際の運行ではありません。画面上だけ必要以上に長く止まって見えています。表示の修正はリストに入っています。

**1794498471-5174140f**, anon, Japanese
- Reported: extending single track from single track shifts sideways; from each track of a double, only one single
  can be extended, the second is refused.
- Status: v0.4.0 fixed a sideways shift where single track joins a double; the second single refused is T-3353
  (open task).
- Reply: thanks. v0.4 fixed a sideways shift where single track joins a double. a second single from the other
  track of a double being refused is on the list.
- Japanese: ありがとうございます。v0.4で、単線が複線につながる所で線路が横にずれる問題を直しました。複線のもう一方の線路から2本目の単線を延ばせない問題は、リストに入っています。

**1794478237-f832483c**, anon, Japanese
- Reported: trips to Hakone feel very low; Hakone is a major Kanto tourist spot (about 20 Odakyu limited expresses a
  day); review other tourist spots too.
- Status: fixed the same day (T-3280, 2026-11-12, Japan's market packs rerun; out in v0.3.3 as "resort towns get
  higher traffic"): 79 resort towns count their published visitors; Hakone went from 1,090 to 9,144 trips a weekday
  each way. (docs/demand-plain.md still says this waits for a rebuild; that line looks stale.)
- Reply: thanks, this was fixed in v0.3.3. Hakone and about 80 other resort towns in Japan now get trips from their
  published visitor numbers, and Hakone gets about 8 times as many as before.
- Japanese: ありがとうございます。v0.3.3で直しました。箱根をはじめ日本の約80の観光地は、公表されている観光客数をもとに移動需要が入るようになり、箱根は以前の約8倍になりました。

**1794478439-efe6b4ba**, anon, Japanese
- Reported: wants long tunnels through mountains to ease gradients.
- Status: slopes don't slow trains today (they only add to track cost); slope effects are T-3027 and T-3445 (open
  tasks, waiting on your terrain rework call).
- Reply: thanks. slopes don't slow trains in the game yet, they only make track cost more. how slopes should work is
  on the list.
- Japanese: ありがとうございます。今のところ勾配で列車が遅くなることはなく、建設費が高くなるだけです。勾配の扱いはリストに入っています。

**1794302659-76c9ee63**, Fran, English
- Reported: an optional maximum slope, with more upkeep or lower speed or length on steep track (built 1 km at 7%).
- Status: T-3027, slopes slow trains (open task, design first).
- Reply: thanks. how slopes should affect trains is on the list.

**1794737457-b530376d**, anon, English
- Reported: buses make no profit and trams are far better on the same route; buses should feed heavier lines.
- Status: v0.4.0 cut the bus boarding penalty (6 to 4 minutes); bus and tram balance is T-3378 (open task, design
  for you first).
- Reply: thanks. v0.4 made buses a bit more attractive by cutting their boarding penalty, and bus and tram balance is
  on the list.

**1794782422-e256b35f**, anon, English
- Reported: track lying completely under a station can't be edited or deleted, only the pieces around it.
- Status: likely a bug, not checked; no task yet (listed at the end).
- Reply: thanks, noted.

### Already decided against

**1794481509-d9a2f1c8**, anon, English
- Reported: freight trains and cargo, commuters by day and freight at night.
- Status: you said no to freight the player runs (2026-10-26, TASKS.md "Not wanted: ... flights or freight you
  run"). Change the reply if you've changed your mind.
- Reply: thanks for the idea. freight isn't planned for now, the game sticks to passengers.

**1794467134-bdfeb1a5**, anon, English
- Reported: 320 km/h isn't enough for Jakarta to Bandung, where the real trains go faster.
- Status: you said no to 350 km/h (2026-11-12, trainworld/CLAUDE.md).
- Reply: thanks. high speed tops out at 320 km/h in the game for now.

---

## 3. Reply and keep open

**1794952390-78ae0270**, Francesco S., English
- Reported: Milan to Bologna has trips and a high-speed line (TAV), but the trip panel says "no transit route serves
  both ends in this period" and the line has few riders (2 pictures looked at: 2% transit, TAV 278 riders).
- Status: same shape as T-3432 (no intercity trip between New York and Philadelphia; needs a save). v0.4.0 fixed the
  bug that lowered intercity riders, which may be most of it.
- Reply: thanks Francesco, glad you're enjoying it. v0.4, out now, fixed a bug where local lines were lowering
  intercity train use, so the TAV should carry a lot more. if the trip panel still says there's no transit route on
  v0.4, could you share the game in Community and tell me its name? that would let me see what's going on.

**1794788348-183b1e0c**, boblovespi, English
- Reported: two lines between New York and Philadelphia, but the trip explorer finds no inter-city trip.
- Status: T-3432 (open task, not reproduced, needs his save).
- Reply: thanks. v0.4 fixed a bug that lowered intercity train use, which might be part of this. if you still see no
  trip between New York and Philadelphia on v0.4, could you share the game in Community and tell me its name?

**1794894256-30911903**, Rynco Maekawa, English (game in Chinese)
- Reported: a one-platform station can't go on a one-way single track: "a station needs a track going through it"
  (picture looked at: Beijing, the single track runs beside line 1 near Jianguomen Bridge).
- Status: v0.4.0 fixed this refusal at the end of single track (T-3411); his is mid-run beside another line, which
  T-3411 says already worked. Not known to be fixed; bug with no task if it still happens.
- Reply: thanks for the picture. v0.4 fixed one case of this, at the end of single track. yours looks like the middle
  of a one way single track beside another line. could you try again on v0.4 and tell me if it's still refused?

**1794891212-17240a38**, Fudemame (フデ豆), Japanese, iPad
- Reported: on an iPad, Kansai or Tokyo says "Kansai (Tokyo) could not load" and no track can be built; the same game
  works on an Android phone.
- Status: bug, no task (listed at the end). The message is the zone in view failing to load; cause unknown.
- Reply: sorry about that, and thanks for checking on two devices. i don't know the cause yet. could you try again
  on v0.4 and tell me if it still happens, and which iPad model you have?
- Japanese: ご迷惑をおかけしてごめんなさい。2つの端末で確かめてくれてありがとうございます。原因はまだわかっていません。v0.4でもう一度試して、まだ起きるかどうかと、お使いのiPadの機種を教えてもらえますか？

**1794957009-e4db8ec6**, anon, Japanese, Safari 27.2
- Reported: after playing a while, Tokyo doesn't load, so no riders and no money.
- Status: same as the one above, bug with no task.
- Reply: sorry about that. i don't know the cause yet. could you try again on v0.4 and tell me if it still happens,
  and which device you play on?
- Japanese: ご迷惑をおかけしてごめんなさい。原因はまだわかっていません。v0.4でもう一度試して、まだ起きるかどうかと、どの端末で遊んでいるかを教えてもらえますか？

**1794656835-c742b993**, anon, Japanese, Safari 27
- Reported: intercity trips are on but stay at 0 (picture looked at: a line from London across Asia to Japan, the
  setting "Calculate intercity trips" on, memory 2.5 GB of 3 GB).
- Status: a small device near its memory cap; may be the market never opening. Not known to be fixed. v0.4.0 uses
  less memory and fixed a bug that lowered intercity riders.
- Reply: thanks for the picture. v0.4 uses less memory on phones and tablets and fixed a bug that lowered intercity
  riders. is it still 0 on v0.4? if so, which device are you playing on?
- Japanese: 画像ありがとうございます。v0.4ではスマホやタブレットのメモリ使用量を減らし、都市間の乗客を減らしていたバグも直しました。v0.4でもまだ0人のままですか？そうなら、どの端末で遊んでいるか教えてください。

**1794736223-fb6d9c3d**, grantchen668, Chinese
- Reported: a one-way loop line is capped by "single track near the station is full" and can't run more trains (2
  pictures looked at: Shanghai loop, alert near Yongning Street, 4 trains an hour both ways).
- Status: likely fixed by v0.4.0 ("fixed a station turning one-way single track into two-way track"), not
  confirmed. Workaround: set that single track one way.
- Reply: thanks for the pictures. v0.4 fixed a station turning one way single track back into two way track, which
  is likely what capped your loop. does it still happen on v0.4? if it does, select that single track and turn on Run
  trains one way in its panel.
- Chinese: 谢谢截图。v0.4修复了车站把单向单线变回双向轨道的问题，你的环线很可能就是因为这个被限制了班次。更新到v0.4后还会这样吗？如果还会，选中那段单线，在面板里打开“列车单向行驶”。

**1794657859-7a5dceae**, anon, Japanese, 2026-11-14
- Reported: trains, buses and ferries all stopped and people pile up.
- Status: unknown. On an 11-13 build (v0.3.2 or v0.3.3), after the T-3313 freeze fixes, so not known to be fixed.
- Reply: sorry about that. does it still happen on v0.4? if it does, please send a report from the game while it's
  stuck. the report carries what the game was doing, which helps a lot.
- Japanese: ご迷惑をおかけしてごめんなさい。v0.4でもまだ起きますか？起きたら、止まっている間にゲームからレポートを送ってください。レポートにはゲームの状態が入るので、とても助かります。

**1794470946-cb0435de**, mizushin, Japanese, 2026-11-12
- Reported: on the same device, money, station names and transit share don't update.
- Status: unclear whether it's frozen figures (fixed in T-3313, v0.3.2) or a save not keeping changes.
- Reply: sorry for the slow reply. does this still happen on v0.4? if it does, could you tell me a bit more: do the
  numbers stop changing while you play, or does a game lose your changes when you open it again?
- Japanese: 返信が遅くなってごめんなさい。v0.4でもまだ起きますか？起きる場合、もう少し詳しく教えてください。遊んでいる最中に数字が動かなくなるのか、それとも開き直すと変更が消えているのか、どちらでしょう？

**1794742880-a24c85bc**, abcd, Chinese
- Reported: the Songjiang tram in Shanghai's real network is recognised wrongly.
- Status: possibly T-3514 (real network saves build tram lines as train track; open task); not confirmed.
- Reply: thanks. what looks wrong with the Songjiang tram? one thing i know: real network saves build tram lines as
  regular train track for now, and that's on the list. if it's something else, a screenshot would help.
- Chinese: 谢谢。松江有轨电车具体哪里不对？我知道的一点是：目前真实路网的存档会把有轨电车线路建成普通铁路轨道，这个已经在计划里了。如果是别的问题，发张截图会很有帮助。

**1794635717-5c59d7c8**, coyotecountry, English
- Reported: bus lines don't make good connections and show weird arrows at times, mostly on loops.
- Status: unclear. The arrows are probably one-way chevrons (one-way loops, one-way streets); the next update draws
  bus chevrons thinner and further out.
- Reply: hi! the arrows on bus lines show which way a one way loop or one way street runs, that part is normal. if a
  bus line isn't joining up where you expect, could you send a screenshot?

---

## 4. Needs your decision first

Each block has the question, my recommendation, and a reply for that recommendation.

**1794966656-01a11c4b**, technologytree123, English (on v0.4.0)
- Reported: wants separate options for metro and for long-distance or conventional rail.
- Question: a metro kind of line or track, apart from regular rail? No task. Similar to an earlier commuter or
  regional request.
- Recommend: no new kind now; raise it in the track rework talk (T-3444).
- Reply if so: thanks for the idea, noted.

**1794811284-cd6b3be1**, anon, Japanese
- Reported: wants to join track to a station from its front end, not only along its side (Hiroshima Station as the
  example); more real networks in Japan.
- Question: track entering a station at its end is a station model question (T-3444); more Japanese real networks
  have no task.
- Recommend: fold the first into T-3444's talk; more Japan real networks in a later real-network batch if you want.
- Reply if so: thanks, both noted.
- Japanese: ありがとうございます。2つとも参考にします。

**1794807176-15962b46**, adrc1, English
- Reported: please let bus stops go anywhere near each other and near stations (as a toggle).
- Question: let bus stops go closer together, at least when placed by hand? A repeat (also 1794733621-93499520).
  I didn't find the rule that stops them, so it's worth checking what refuses them first. Buses are settled unless
  you ask.
- Recommend: yes for hand-placed stops only, no toggle, if the limit turns out to be a simple distance.
- Reply if yes: thanks. letting hand placed bus stops go closer together is on the list.
- Reply if no: thanks for the idea. it's noted, but not planned for now.

**1794733621-93499520**, abc82392216, Chinese
- Reported: on a loop bus line where the two directions run close, one-direction stops; and free placement of
  hand-placed stops even near other routes' stops.
- Question: same spacing call as adrc1, plus one-way stops on two-way bus lines.
- Recommend: same as adrc1 for spacing; one-way stops on two-way bus lines not now.
- Reply if yes to spacing: thanks for the ideas. letting hand placed bus stops go closer together is on the list.
  one way stops on a two way bus line aren't planned for now.
- Chinese: 谢谢你的建议。让手动放置的公交站可以靠得更近，已经在计划里了。双向公交线路上的单向车站目前还没有计划。
- Reply if no: thanks for the ideas. they're noted, but not planned for now.
- Chinese: 谢谢你的建议，已经记下了，不过目前还没有计划。

**1794720154-8340b1ec** and **1794784060-a02f0dd3**, anon, Turkish (merged)
- Reported: a history of the notices that pop up after an edit, to see what each choice did (picture: the "+41k
  trips/day, profit -72.3M/day" notice with the lines it touched).
- Question: keep a list of the recent edit notices? No task; came up before (notes/T-3527.md lists "notification
  list" as a request).
- Recommend: yes, low priority: the last 10-20 notices in a small panel.
- Reply if yes: thanks for the picture, that helps. a history of these notices is on the list.
- Turkish: Ekran görüntüsü için teşekkürler, çok yardımcı oldu. Bu bildirimlerin geçmişi yapılacaklar listesinde.

**1794768764-01013111**, leni_v, English
- Reported: when editing a bus line, right click shouldn't break the line; it should remove stops, like Cities:
  Skylines.
- Question: change right click on a bus line's points? It's taken: on a middle point it loops the line back there,
  on the first point it closes the loop, on a ring it opens the ring there, and only on the last point it deletes
  (docs/ui-building.md, the bus tool's table). "Breaks the line" is likely the ring opening.
- Recommend: no.
- Reply if no: thanks! right click on a bus line's stops is used for loops: it closes a loop, or opens one at that
  stop, which is what breaks the line. so it can't remove stops too. sorry, i know the skylines habit is hard to
  break.

**1794744888-2d57c9d9** and **1794744778-ef2051d1**, anon, Japanese (merged, 2 minutes apart)
- Reported: a way to hide lines; a way to hide the "when it was saved" text at the top.
- Status: hiding lines is in v0.4.0 (an eye button on the Lines tab, T-3420).
- Question: let players hide the saved-time text? No task.
- Recommend: no.
- Reply if no: thanks. v0.4 added hiding lines: press the eye button on the Lines tab. the saved time stays for now.
- Japanese: ありがとうございます。v0.4で路線を非表示にできるようになりました。路線タブの目のボタンを押してください。保存時刻の表示は、今のところそのままにします。

**1794739805-22b5bd83**, hydro, English
- Reported: add malls that create more destinations.
- Question: a buildable destination? No task; closest is T-454 (destinations that respond to the network, P4).
- Recommend: no.
- Reply if no: thanks for the idea. it's not planned for now.

**1794642959-426e58d0**, anon, English (uses the Japanese term shingojo)
- Reported: wants a signal station (a passing point where passengers can't board), or something to turn trains
  around mid-journey, since reversing on the line congests the track. Picture not looked at.
- Question: a passing loop with no platform? Fits the track rework (T-3444) and waypoints (T-3079).
- Recommend: raise it in the T-3444 talk.
- Reply if so: thanks. passing points without platforms are noted for the track rework.
- Japanese: ありがとうございます。ホームのない信号場は、線路の作り直しの検討に入れておきます。

**1794562640-0a9f2bc7**, pino, Japanese
- Reported: a way to view only the route diagrams shown in the community, and to export them as images.
- Question: community diagram view and image export? No task; read-only shared games is T-1446 (P3).
- Recommend: image export of the diagram as a small P3 if you like it; otherwise no.
- Reply if yes: thanks. saving the network diagram as an image is on the list.
- Japanese: ありがとうございます。路線図を画像として保存できるようにするのは、リストに入っています。
- Reply if no: thanks for the idea. it's noted, but not planned for now.
- Japanese: アイデアありがとうございます。参考にしますが、今のところ予定はありません。

**1794499082-0877c9eb**, anon, Japanese
- Reported: building together, with a rule that you can't touch lines someone else is working on; sharing lines with
  chosen people by URL.
- Question: shared building? No task. Read-only sharing by link is T-1446 (P3).
- Recommend: no to building together; mention read-only links only if you want T-1446 said out loud.
- Reply if no: thanks for the idea. building together isn't planned for now. you can share a network in Community.
- Japanese: アイデアありがとうございます。共同で作る機能は今のところ予定はありません。路線網はコミュニティで共有できます。

**1794491739-53fd3171**, anon, English
- Reported: import GeoJSON or GPX files for bus routes.
- Question: file import for routes? No task.
- Recommend: no.
- Reply if no: thanks for the idea. it's not planned for now.

---

## Bugs with no task (for the orchestrator)

- **Big Japanese cities fail to load on iPad and iOS Safari**: "Kansai (Tokyo) could not load" (`statusTag.placeFailed`,
  the zone in view failing); the same game works on an Android phone. 1794891212-17240a38 (iPad) and
  1794957009-e4db8ec6 (Safari 27.2). Reach: possibly many, since Japanese players are a big share and Tokyo and
  Kansai are the biggest packs; cause unknown, memory is a guess.
- **One-platform station refused mid-run on one-way single track beside another line**: "a station needs a track
  going through it". 1794894256-30911903 (Beijing, Jianguomen Bridge, v0.3.5 build). T-3411 fixed only the end of
  single track. Few players; file once he says it still happens on v0.4.0.
- **Track lying completely under a station can't be picked to edit or delete**. 1794782422-e256b35f. Few players.
- **Intercity stays at 0 on a 3 GB device with "Calculate intercity trips" on**: 1794656835-c742b993 (Safari 27,
  2.5 of 3 GB used, a line across Eurasia). Maybe the market never opens near the memory cap. File once the player
  answers.
- If they write back that it still happens on v0.4.0: the one-way loop capped as two-way single track
  (1794736223-fb6d9c3d) and everything standing still (1794657859-7a5dceae).
