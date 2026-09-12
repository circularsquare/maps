# -*- coding: utf-8 -*-
"""The yearbook, parsed, plus the one thing it cannot supply: which name means
which line.

The 철도통계연보 names its lines three different ways depending on which table
you are in -- `경부고속본선` in the distance table, `경부고속` in the station
roster, `경부고속선` in the traffic table -- and OSM has a fourth opinion
(`경부본선` alongside `경부선` on the same route). None of the four is derivable
from the others, so LINES below is a hand table, in the manner of japanriders'
build_names.py. Everything else here is read out of the workbook.

    python lines.py          # print the resolved table and what is missing
"""

import io
import os
import re
import sys
import zipfile

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
# Which edition of the 철도통계연보 everything reads. **2023 as of 2026-09-08**;
# it was 2022 until then and much of README.md still quotes 2022 figures, which
# are marked where they matter. The readers cope with both -- see `_member_key`
# below and `frequency.SEP` -- so
#
#     KOREARIDERS_YEARBOOK=data/korail_yearbook_2022_excel.zip python solve.py
#
# reproduces the older map exactly. Anything that quotes the year takes it from
# `year()` rather than hardcoding it, so switching does not leave the page or
# the reports describing themselves wrongly.
YEARBOOK = os.environ.get(
    "KOREARIDERS_YEARBOOK",
    os.path.join(D, "korail_yearbook_2023_excel.zip"))


def year():
    """The edition's year, off the filename.

    The page and the reports quote it in a dozen places -- "no trains ran on
    this section in 2022" and so on -- and hardcoding it there means changing
    edition silently leaves the map describing itself as the wrong year.
    `write_geojson` puts this in the file and `index.html` substitutes it.
    """
    m = re.search(r"(19|20)\d{2}", os.path.basename(YEARBOOK))
    return int(m.group(0)) if m else None

PASSENGER = "1.지역간철도/4. 수송(여객)_완.xlsx"
FACILITY = "1.지역간철도/8. 시설_완.xlsx"

# canonical -> (traffic-table name, distance-table name, roster name, OSM names)
# A None means that table has no row for the line; the pipeline copes as long as
# the distance and OSM entries are there.
LINES = {
    "경부고속선": ("경부고속선", "경부고속본선", "경부고속", ["경부고속선"]),
    "호남고속선": ("호남고속선", "호 남 고 속 본 선", "호남고속", ["호남고속선"]),
    "수서고속선": ("수서고속선", "수서평택 고속선", "수서평택선", ["수서평택고속선"]),
    "경부선": ("경부선", "경부선", "경부선", ["경부선", "경부본선"]),
    "호남선": ("호남선", "호남선", "호남선", ["호남선"]),
    "전라선": ("전라선", "전라선", "전라선", ["전라선"]),
    "장항선": ("장항선", "장항선", "장항선", ["장항선"]),
    "중앙선": ("중앙선", "중앙선", "중앙선", ["중앙선"]),
    "경전선": ("경전선", "경전선", "경전선", ["경전선"]),
    "동해선": ("동해선", "동해선", "동해선", ["동해선", "동해본선"]),
    "영동선": ("영동선", "영동선", "영동선", ["영동선", "영동본선"]),
    "태백선": ("태백선", "태백선", "태백선", ["태백선"]),
    "충북선": ("충북선", "충북선", "충북선", ["충북선"]),
    "경북선": ("경북선", "경북선", "경북선", ["경북선"]),
    "경원선": ("경원선", "경원선", "경원선", ["경원선"]),
    "경의선": ("경의선", "경의선", "경의선", ["경의선"]),
    "경춘선": ("경춘선", "경춘선", None, ["경춘선"]),
    "광주선": ("광주선", "광주선", "광주선", ["광주선"]),
    "대구선": ("대구선", "대구선", "대구선", ["대구선"]),
    "정선선": ("정선선", "정선선", "정선선", ["정선선"]),
    "강릉선": ("강릉선", "경강선(원주-강릉)", "경강선(강릉선)", ["경강선"]),
    "중부내륙선": ("중부내륙선", "중부내륙선", "중부내륙선", ["중부내륙선"]),
    "삼척선": (None, "삼척선", "삼척선", ["삼척선"]),
    "진해선": (None, "진해선", "진해선", ["진해선"]),
    "여천선": (None, "여천선", "여천선", ["여천선"]),
    "문경선": (None, "문경선", "문경선", ["문경선"]),
}

# The distance table's endpoints are the legal extent of the line, which is not
# always where trains run to or what OSM calls the place. Overrides win.
#
# Moving an end also moves what the line's 영업거리 measures, and the yearbook
# publishes no station-to-station distances to replace it with -- so a line whose
# ends are overridden keeps OSM's own chainage instead of being rescaled to a
# figure describing a different piece of track. See `scaled` in resolve().
ENDS = {
    # 영동선 legally ends at a signal box; 강릉 is where the track and the
    # station data stop.
    "영동선": ("영주", "강릉"),
    # 중앙선's 모량 end is a junction with 동해선 and has no station; 경주 is the
    # last station with traffic.
    "중앙선": ("청량리", "경주"),
    # The distance table still says 장항, but 장항선 has run through to 익산
    # since 군산선 was absorbed -- the roster and station table both agree.
    "장항선": ("천안", "익산"),
    # 광주선's own table calls the junction end 동송정, which is a signal point.
    "광주선": ("광주송정", "광주"),
    # 강릉선 branches off 중앙선 at 서원주, not at 원주 station.
    "강릉선": ("서원주", "강릉"),
    # 동해선's 부산진 end is a freight junction; 부전 is the passenger terminus.
    "동해선": ("부전", "영덕"),
    # 호남선 legally starts at 대전조차장, which is not a passenger station and
    # has no OSM node; 서대전 is the first one trains call at.
    "호남선": ("서대전", "목포"),
    # 태백선 ends at 백산 junction, again no station; 태백 is the last stop.
    "태백선": ("제천", "태백"),
    # 대구선 starts at 가천 junction, no OSM node; 동대구 is where trains start.
    "대구선": ("동대구", "영천"),
    # 수서평택고속선 ends at a junction with 경부고속선, not at 평택 station.
    "수서고속선": ("수서", "평택지제"),
    # 경부고속선 keeps 서울 even though its own metals start south of 광명: the
    # corridor search bridges 서울~광명 over 경부선 at a penalty, and 서울 is the
    # line's largest single source of traffic -- dropping it cost a third of the
    # reconstruction.
    "경부고속선": ("서울", "부산"),
    # 호남고속선 is deliberately NOT extended north to 용산, though that is where
    # its passengers board and it was tried. Two reasons, the second decisive.
    # `pick()` anchors the corridor search on the line's own track, so 용산 lands
    # on the nearest 호남고속선 metal, which is at 오송: the chain came out with
    # 용산 drawn 0.74 km from 오송 against a real 106, and a zero-length first
    # segment. Forcing it would need `pick()` loosened, which is the one thing
    # stopping 영동선 setting off down 중앙선.
    #
    # And it should not be forced, because those passengers are not on this line
    # yet -- they ride 경부선 and 경부고속선 as far as 오송. Extending the chain
    # would paint 호남고속선's load over 160 km of track its riders are not on.
    # The line's extent is already right; only its level is wrong, so the fix
    # belongs in the entry flow -- see ENTRY_SHARE.
}


# What sizes a line's entry flow, where its passengers join it from another line
# and never touch a station of its own.
#
# 호남고속선 is the case. It runs 오송-광주송정 and nearly everyone on it boards
# at 용산, 서울 or 광명, none of which is on it, so the reconstruction sees 4.2M
# of a published 22.4M and the entry flow has nothing to size it -- the
# smallness prior simply pushes it down. The published counts do size it:
# 경부고속선 runs 177 trains a day into 오송 and 127 out the far side, and the 50
# that vanish are exactly 호남고속선's own 오송-익산 count. So the entry is that
# share of what the feeding line carries in.
#
# Trains rather than seats, so this is a prior and not a measurement -- a Honam
# KTX-산천 seats 363 against a Gyeongbu KTX-1's 935. The independent check is
# the published line totals: 호남고속선's 통과인원 is 23.6% of 경부고속선's,
# against a 28.2% share of trains, so the two disagree by a fifth and not by a
# factor. The fit had it at 12.4%.
#
#     line -> (the line it joins, the junction it joins at)
ENTRY_SHARE = {
    "호남고속선": ("경부고속선", "오송"),
}


# A station whose riders of one train type ride a line that does not call
# there, joining it at a station that does.
#
# 용산 is the case, and it is the largest single hole in the map: 10.75M KTX a
# year, 3.3 % of everything the yearbook counts, reaching no segment at all.
# The station sits on 경부선's and 경원선's chains and neither declares KTX, so
# `check_orphans.py` reports the whole row as unreachable. The row itself is
# fine and unfiltered -- every KTX-declaring line's `flows_by_kind` already
# holds it -- and only solve.py's chain test keeps it out.
#
# Where they go is not in doubt. Every KTX out of 용산 is a 호남 or 전라
# service; it runs down 경부선's metals to 광명, rides 경부고속선 from there to
# 오송, and turns off. So the passengers join 경부고속선 at 광명 and the
# injection belongs at that stop rather than at the chain's start -- they never
# ride 서울-광명.
#
# The type is not in doubt either, which took a second source to establish.
# KRIC's 노선별 table (see kric_stats.py) shows 강릉선's and 중앙선's high-speed
# traffic is *entirely* KTX-이음, and its 역별 table shows 용산 taking no
# KTX-이음 in any of the twelve months of 2023. So no 강릉선 or 중앙선 service
# touches 용산 and there is no third destination to divide against. README.md,
# "용산's KTX split is published".
#
# Extending 경부고속선's chain to 용산 is the wrong move and OVER cannot do it
# anyway: 서울 and 용산 are two termini on different approaches, and a linear
# chain cannot fork.
#
#     (station, train type) -> (the line that carries them, where they join it)
OFF_CHAIN = {
    ("용산", "KTX"): ("경부고속선", "광명"),
}


# Where a line hands its traffic to another at a place the receiving line's
# chain does not name.
#
# Junction conservation pairs lines by station name, so it only works where the
# handover point is a stop on both chains. Where it is not, the traffic does not
# go anywhere -- it stops existing. 수서고속선 ends at 평택지제, where the SRT
# join 경부고속선, and 평택지제 is 2.93 km from 경부고속선's corridor against a
# 0.30 km snap radius, so it never will be a stop on it. The line delivers
# 47,006 passengers a day to that end and every one of them vanishes:
# 경부고속선 reads 105,490 a day on 광명-천안아산 and *less*, 101,907, on
# 천안아산-오송, when the SRT should have joined in between.
#
# The receiving stop is 천안아산 rather than 광명 because the SRT join south of
# 광명: a step at 천안아산 puts them on 천안아산-오송 and not on 광명-천안아산,
# which is right. It does leave them off the short stretch between the real
# junction and 천안아산, which nothing published can fix -- 평택분기점 is not a
# station and has no 승하차 row.
#
#     (line, its own end) -> (receiving line, the stop that takes the step)
#
# 호남선's start is the same fault. It legally begins at 대전조차장, a freight
# yard with no station and no 승하차 row, so ENDS moves it to 서대전 -- which no
# other chain names, leaving the traffic that branches off 경부선 there with no
# line to have come from.
#
# The receiving stop is 신탄진, not 대전, and 대전 is the one that looks right:
# 대전조차장 sits between them, so a step at 대전 would put the traffic on
# 대전-신탄진 where most of it belongs, while 신탄진 leaves that segment short.
# Two things rule 대전 out. It is called by 경부고속선 as well, so the pool would
# be three-way and the fit could take the traffic off the high-speed line
# instead of the conventional one; and 대전 has a published count of its own,
# 41 to 49 in chain order, whose sign rule penalises a negative step at weight
# 12 -- the exact opposite of what a handover needs. The 31 trains that actually
# branch do so at 대전조차장, whose own boundary is 80 to 49, and that station
# cannot be drawn. 신탄진 has no published boundary, so the sign rule does not
# apply, and only 경부선 calls there.
HANDOVER = {
    ("수서고속선", "평택지제"): ("경부고속선", "천안아산"),
    ("호남선", "서대전"): ("경부선", "신탄진"),
    # 태백선's trains do not terminate at 태백: six 무궁화 a day run 8.7 km on to
    # 백산, leave the line there and rejoin 영동선 at 동백산, and the published
    # section counts show the same six arriving (영동선 철암-동백산 4, 동백산-동해
    # 10). Added, reverted and added again -- the first attempt gave it a 4.7x
    # directional imbalance for a service that is symmetric by construction, and
    # the arbitrariness propagated through junction conservation. It is only
    # safe with solve.W_HOSYM holding the two directions equal; see README.md,
    # "A handover's two directions are the same trains".
    ("태백선", "태백"): ("영동선", "동백산"),
}

# Where a handover physically happens, for drawing only -- the fit still hangs
# the step on HANDOVER's receiving stop, because that is the only place with
# rows to fit against. `write_geojson` cuts the receiving segment here and gives
# the far side the through flow, so the map does not credit a whole segment with
# traffic that joins partway along it.
#
# Without an entry the cut point is the giving line's own end station, which is
# right for 수서고속선 -- 평택지제 is 1.4 km from 평택분기점 and the cut lands
# where the SRT really join. It is wrong for 호남선. Its end is 서대전, which
# sits 3 km *southwest* of 대전, so the nearest point on 경부선's 대전-신탄진 is
# that segment's own first point; `split_at_junction` then refuses the cut for
# landing inside a kilometre of the end, and the whole 14.5 km segment was drawn
# short of the 호남선 traffic that rides its northern 10 km.
#
# 대전조차장 is not in `data/osm_stations.json` because that pull filtered to
# `railway=station|halt` and a marshalling yard is neither; it is OSM node
# 7640162489, `railway=yard`, `wikidata=Q188837`. One coordinate is cheaper to
# record here than a second pull is to maintain.
#
#     (line, its own end) -> (lat, lon) of the junction
HANDOVER_POINT = {
    ("호남선", "서대전"): (36.3710255, 127.4218344),   # 대전조차장
    ("수서고속선", "평택지제"): (36.95148, 127.07057),  # 평택분기점
    # 백산 is OSM node 368637144, `railway=service_station` -- which is why the
    # station pull missed it, exactly as `railway=yard` hid 대전조차장.
    ("태백선", "태백"): (37.1383871, 129.0337917),      # 백산
}

# Where each of those lines' own metals stop, which is not where its last
# station is. Both handover lines run on past their last platform to the
# junction, and the corridor -- routed between the two end *stations* -- stopped
# at the platform, so the map drew the line ending in mid-air pointing at
# nothing. 수서고속선 was 7.9 km short of its 영업거리 and 호남선 5.8 km, and both
# gaps are exactly this.
#
#     (line, its own end) -> draw on to this line's HANDOVER_POINT
RUN_ON = {
    ("수서고속선", "평택지제"),
    ("호남선", "서대전"),
    # The 8.7 km on to 백산, which is 태백선's whole -8.4 % gap against its
    # 영업거리. A run-on has no load without a handover to give it one, so this
    # only works alongside the HANDOVER entry above.
    ("태백선", "태백"),
}

# Lines that reach an end station over another line's metals, and whose corridor
# has to be extended along that line to get there.
#
# 경부고속선 is the case. Its high-speed track begins near 금천구청 and KTX reach
# 서울 over 경부선's rails, so the corridor stopped 14.29 km short of 서울역 and
# the station was drawn at the metals' end instead -- giving 서울-광명 a length
# of 2.9 km where the two stations are 22 km apart, and crediting 79,545 riders
# a day with an eighth of the distance they travel. It is about 2 % of the
# network's passenger-km.
#
# **Why this needs naming rather than searching.** `data/osm_railways.json` is
# `way[railway=rail][name]`, so unnamed track -- junction throats, crossovers,
# the connections between lines -- is not in the graph at all. Routing from the
# high-speed metals to 서울역 therefore takes 111.96 km for a 14.29 km gap, and
# 광주선's 1.77 km gap routes 402 km. A generic bounded stub is in `build.py`
# and correctly refuses all of these; it earns its keep on 영동선's 강릉 and
# 중앙선's 경주, where the approach happens to be named track. Where it cannot
# reach, the honest fix is to say which line carries the trains and take that
# line's own corridor, which is already built and already trusted.
#
# The two lines then overlap on the shared stretch, which is what they do in
# life -- japanriders draws shared track the same way, one geometry per line,
# stacked.
#
#     (line, its own end) -> the line whose metals carry it there
OVER = {
    ("경부고속선", "서울"): "경부선",
}

# End stations the roster calls this line's own, where trains nonetheless run
# straight through onto another line. The anchor asserts that everything
# alights at the last stop, and at these it is false.
#
# 광주송정 is 호남고속선's on the facility roster, so it passed the anchor test
# and the fit was told that every high-speed passenger empties out there. They
# do not: `6. 운전` has 42 high-speed trains a day arriving 익산-광주송정 and 28
# leaving it again on 호남선 towards 목포.
#
# Suppressing it does not by itself put those passengers on the right line: the
# fit carries them north up 호남선's conventional metals instead, lifting
# 서대전-계룡 from 16,574 a day to 24,904. Both builders rise by the same 8,330,
# which is the leaking traffic and not a change of method.
#
# This removes a false assertion; on its own it does not change the answer, and
# the reason is worth keeping. Suppressing the anchor was expected to let the
# 목포 traffic PART_TYPES gives 호남선 arrive the way it really does, down
# 호남고속선 and through the junction. It does not, because nothing then pushes
# the through flow *up*: the smallness prior pushes it to zero, carrying the
# traffic north up 호남선's conventional metals instead, and both answers
# satisfy junction conservation equally. Cost, every line's mirror and every
# figure in the report came back identical to four decimal places.
#
# Treating the end as a junction outright does move it, by handing the line's
# level to 통과인원 -- 호남고속선 goes to a plausible 40,697 from 5,774. But it
# overcorrects: a hard 22.4M target pulls on every neighbour through 오송 and
# 익산, and the network's median weighted mirror went from 1.9% to 4.1% and its
# cost from 82 to 190, with 광주선 at 43.6% and 장항선 at 30.1%.
#
# What would settle it is the section counts' *magnitude* rather than their
# sign. The KTX count across 광주송정 drops 19 to 2, so the great majority of
# the high-speed service south of it does not continue north, and the step is
# not merely permitted but roughly sized. README.md records that this cannot
# help 대구선 or 경북선; 호남선 is one of the four lines whose boundaries are
# informative, and this is the case that wants it.
#
# 태백선's 태백 is here on the same evidence: six trains a day run straight
# through it to 백산 and on to 영동선, so the anchor's assertion that everyone
# alights at 태백 is false. It was left out until the fit could pin the through
# flow, which solve.W_HOSYM now does.
THROUGH_ENDS = {
    "호남고속선": {"광주송정"},
    "태백선": {"태백"},
}

# Which train types run on each line. The point is the parallel pairs: 경부선 and
# 경부고속선 share 서울, 대전, 동대구 and 부산, and sheet 8's combined counts
# cannot tell which set of metals a passenger rode. Only the high-speed services
# use the 고속선, so the type split separates them.
#
# It does not separate everything. 서울's KTX arrivals are 경부, 호남, 전라 and
# 강릉 KTX together, and this table hands all of them to every line that lists
# KTX -- see README.md.
ALL_TYPES = ["KTX", "SRT", "새마을", "ITX-새마을", "무궁화", "통근"]
CONVENTIONAL = ["새마을", "ITX-새마을", "무궁화", "통근"]
TYPES = {
    "경부고속선": ["KTX", "SRT"],
    "호남고속선": ["KTX", "SRT"],
    "수서고속선": ["SRT"],
    "강릉선": ["KTX"],
    "중부내륙선": ["KTX"],
    "경부선": CONVENTIONAL,
    "경의선": CONVENTIONAL,
    "경원선": CONVENTIONAL,
    "경춘선": CONVENTIONAL,
    "충북선": CONVENTIONAL,
    "경북선": CONVENTIONAL,
    "대구선": CONVENTIONAL,
    "정선선": CONVENTIONAL,
    "태백선": CONVENTIONAL,
    "영동선": CONVENTIONAL,
    "장항선": CONVENTIONAL,
    "호남선": CONVENTIONAL,     # plus high-speed south of 광주송정 -- see PART_TYPES
    # These carry high-speed services over their own conventional metals.
    "전라선": ALL_TYPES,
    "중앙선": ALL_TYPES,
    "경전선": ALL_TYPES,
    "동해선": ALL_TYPES,
    "광주선": ALL_TYPES,
}

# A train type a line carries over only part of its length, and the stations at
# which it may be counted.
#
# 호남선 south of 광주송정 is the one stretch of conventional metals in the
# country with no high-speed line beside it. 호남고속선 *ends* at 광주송정, and
# the KTX and SRT that carry on to 목포 run on 호남선's own track -- `6. 운전`
# says so outright, 광주송정-목포 being KTX 19 + SRT 9 against 무궁화 7 +
# 새마을 4. With the line restricted to conventional types none of them reached
# the map: 목포's 1.55M KTX and 0.59M SRT a year were dropped and the segment
# was drawn at 743 a day.
#
# Granting the types line-wide instead does far more damage than the bug it
# fixes, and both mechanisms are worth recording. The types then reach 서대전,
# 익산 and 정읍, whose high-speed traffic is 호남고속선's; and worse, the line's
# type set becomes the complete one, which flips its 통과인원 residual from a
# one-sided ceiling to two-sided equality -- so the fit is actively driven to
# reach a published 16.8M that counts the through traffic as well. It duly took
# it from the only place available. 호남선 went to a 수송밀도 of 28,009, ahead
# of 경부선, and 호남고속선 collapsed from 5,774 to 2,412 with 정읍-광주송정 at
# zero and a negative segment. `full_types` below keeps the ceiling one-sided.
#
# 광주송정 itself is deliberately *not* in the span. Its 1.95M KTX alightings
# are people off 호남고속선, and letting 호남선 into the share group there is a
# smaller version of the same theft. The passengers who ride through it to 목포
# never appear in its 승하차 at all -- they arrive as the junction step, which
# is exactly what the step is for and which the section count licenses, 21
# trains dropping to 11 across the station.
#
# The span is listed rather than derived because nothing published orders a
# line's stations; if a station is ever added south of 광주송정 it has to be
# added here too, or it will quietly carry no high-speed traffic.
PART_TYPES = {
    "호남선": (["KTX", "SRT"],
              {"나주", "다시", "함평", "무안", "몽탄", "일로", "임성리", "목포"}),
}

# The inverse of PART_TYPES: a station where a train type belongs to exactly one
# of the lines calling there, whatever the others declare. `STATION_HOME` says
# that of a whole row; this says it of one type at one platform.
#
# 경주 is the case that needs it, and it only became one when 신경주 was aliased
# on. Three lines call there and all three declare KTX and SRT -- 중앙선 and
# 동해선 because they really do carry high-speed services, but not within tens of
# km of 경주, and `6. 운전` says exactly where:
#
#     중앙선  영주-안동    고속열차 9      <- its high-speed ends at 안동
#     중앙선  안동-영천    무궁화 5
#     중앙선  영천-모량    무궁화 17, 새마을 1
#     동해선  북울산-신경주 무궁화 15, 새마을 1
#     동해선  신경주-모량   (no train of any type)
#     동해선  모량-부조    고속열차 16     <- the 포항 KTX, off 경부고속선 at
#     동해선  부조-포항    고속열차 16        건천연결선, never through 경주
#
# So the only high-speed trains at that platform are 경부고속선's, and the
# 4.14M KTX and SRT a year there are its passengers outright.
#
# Without this the 신경주 alias is a net harm rather than a fix. Measured: the
# alias alone moved 3,015 riders a day onto 중앙선 경주-아화 and carried the
# rise the length of the line to 안동 and beyond, while 경부고속선 동대구-경주
# gained 47. The share prior does not stop it the way README.md's "The share
# prior already knows a branch line is small" found it stopping 경북선 and
# 대구선: those lines are a fraction of a per cent of their junction's traffic,
# where 중앙선's 11.75M 통과인원 against 경부고속선's 100.69M is a tenth of it,
# and the fit then moved the share well past that prior.
#
# This is deliberately one station and not the general per-type gate, which was
# built and rejected the same day for having no measurable effect anywhere the
# prior already handled and for overturning curated attributions at 광주송정,
# 익산 and 순천 where it did. Nothing here is curated: 경주 had no high-speed
# attribution at all until the alias created one.
#
# 강릉 is the second entry and is there for the 광주송정 reason rather than a
# 운전 one. 영동선's KTX span reaches it, because the eight 고속열차 a day on
# 동해-청량신호소 really do run to 강릉 -- but 강릉 is where 강릉선 *ends*, and
# its 2.78M KTX a year are that line's entire traffic against 영동선's eight-train
# tail of it. Sharing a terminus's whole ridership on those terms is the theft
# 호남선 is kept away from at 광주송정. The riders who continue to 동해 are not
# lost by this: they arrive on 영동선 as the junction step at 강릉.
#
#     station -> {train type: the only line whose trains carried them}
TYPE_HOME = {
    "경주": {"KTX": "경부고속선", "SRT": "경부고속선"},
}


def _member_key(name):
    """A member name reduced to something stable across editions.

    Every filename in the bundle changed between 2022 and 2023 without a single
    sheet moving:

        2022  1.지역간철도/4. 수송(여객)_완.xlsx
        2023  1. 지역간 철도/4. 수송(여객).xlsx
        2022  2.도시철도/도시철도-3.수송실적_완.xlsx
        2023  2. 도시철도/3. 수송실적.xlsb

    Spaces come and go, the `_완` suffix was dropped, the 도시철도 files stopped
    repeating their section in the basename, and four workbooks changed from
    xlsx to xlsb. So match on what is actually stable: the section and sheet
    numbers with their titles, spaces and suffix and extension removed.

    Matching on the numbers alone would be simpler and is wrong -- part 1 has
    two sheet 6s, `6. 운전(1~6)` and `6. 운전(7~9)`.
    """
    part, _, base = name.replace("\\", "/").rpartition("/")
    base = base.rsplit(".", 1)[0]                     # drop the extension
    part = re.sub(r"\s+", "", part)
    base = re.sub(r"\s+", "", base).replace("_완", "")
    # 2022 repeats the section name inside each basename; 2023 does not.
    section = re.sub(r"^\d+\.", "", part)
    if section:
        base = re.sub(r"^%s-" % re.escape(section), "", base)
    return "%s/%s" % (part, base)


def _open(member, book=None):
    """The member's bytes, out of `book` or the current edition.

    `book` exists for the one reader that must not follow `YEARBOOK`:
    `yearbook_extra.py` reads the 도시철도 and 광역철도 volumes, and those went
    `.xlsb` in the 2023 bundle, which openpyxl cannot open at all. Left to
    follow the edition it fails with "File contains no valid workbook part",
    which reads as a corrupt download rather than a format change.
    """
    want = _member_key(member)
    with zipfile.ZipFile(book or YEARBOOK) as z:
        for i in z.infolist():
            n = i.filename
            if not (i.flag_bits & 0x800):
                n = n.encode("cp437").decode("cp949")
            if _member_key(n) == want:
                return io.BytesIO(z.read(i.filename))
    raise SystemExit("no %s in the yearbook zip" % member)


# Every reader below wants values and nothing else, and openpyxl spends all of
# its time on the part none of them read. The yearbook's workbooks are 90 %
# formatting -- 수송(여객) is 13.2 MB of parts of which `xl/styles.xml` is 11.8,
# holding 53,810 named styles -- and `apply_stylesheet` expands every one of
# them on open. Measured on 수송(여객): the open is 10.4 s and iterating the rows
# afterwards is 0.02 s.
#
# `apply_stylesheet` returns quietly when the part is absent (`except KeyError:
# return wb`), so dropping it from the archive skips the whole expansion. The
# five readers here go from 20.7 s to 0.28 s and return identical dicts; the fat
# workbooks are the ones every builder opens, so the same trick is worth having
# anywhere else in the tree that reads one.
#
# Only safe for `read_only=True`, which is why that is not a parameter. Without
# the stylesheet `wb._cell_styles` holds openpyxl's single default, and a normal
# load binds every cell to `_cell_styles[style_id]` -- an IndexError against ids
# running to 53,809. Read-only worksheets never look, so `values_only` rows are
# unaffected. A caller that needs a font or a number format has to open the
# workbook the slow way.
def _book(member, book=None):
    """The workbook, opened read-only with its stylesheet left out."""
    src = zipfile.ZipFile(_open(member, book))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as dst:
        for i in src.infolist():
            if i.filename != "xl/styles.xml":
                dst.writestr(i, src.read(i.filename))
    buf.seek(0)
    return openpyxl.load_workbook(buf, read_only=True, data_only=True)


def _num(v):
    return 0.0 if v in (None, "-", "") or isinstance(v, str) else float(v)


# The passenger tables and OSM do not always spell a station the same way, and
# a name that does not match is a station whose traffic reaches no line at all.
# 김천구미 is the plain case -- OSM writes 김천(구미), the station is already on
# 경부고속선's chain, and 1.2M high-speed passengers a year were falling down
# the gap between the two spellings.
#
# Merged rather than overwritten, since a target name may already have a row of
# its own. Anything not listed is passed through untouched.
STATION_ALIAS = {
    "김천구미": "김천(구미)",
    # A typo in the 2023 edition, not a different station: 여수엑수포 for
    # 여수엑스포. It matters more than a typo should, because that station is
    # 전라선's clean terminus and therefore its anchor -- unmatched, the one
    # line in the network that passes every test would silently anchor on
    # nothing. Harmless against 2022, which spells it correctly and so never
    # has this key to rewrite.
    "여수엑수포": "여수엑스포",
    # One station, two labels, in the *same* edition. 2022's sheet 9 (KTX)
    # writes 서대구 and its sheet 10 (SRT) writes 서대구('22.3.31~), annotating
    # the day the station opened -- so the KTX row reaches 경부고속선's chain and
    # the SRT row does not, and 184,595 passengers a year have been going
    # nowhere. It is the 김천구미 fault again with a date instead of a bracket.
    # 2023 spells both plainly, which is how this was found at all.
    #
    # Listed rather than derived: stripping any bracketed annotation would also
    # merge 판교(경기) into 판교, and those really are two stations 200 km apart.
    "서대구('22.3.31~)": "서대구",
    # 2023 disambiguates a name 2022 left bare. 판교(충남) is 장항선's, and is
    # the one the chain means; 판교(경기) is the 경강선 station near Seoul, on no
    # chain here, and must stay separate. Two stations 200 km apart that happen
    # to share a stem, which is the opposite of the 신경주 case below.
    "판교(충남)": "판교",
}

# One station under two names in the *same* edition, and the largest single hole
# `check_orphans.py` finds. 신경주 became 경주 when the old 경주역 closed in
# December 2021, and the 2023 tables took the new name up unevenly: sheet 9
# files 2.15M KTX a year under 경주 and sheet 10 files 1.99M SRT under 신경주.
# Same platform, same trains. Unmerged, the SRT row reaches no line at all.
#
# **Only safe from 2023 on, and this file used to say it was never safe.** The
# alias rewrites a flow key, so it is right exactly when OSM's 경주 node sits on
# the chain of the line those trains ran on -- 경부고속선's. Whether it does
# depends on the roster, since the roster is what names a chain's stops:
#
#   2023  `8. 시설` says 경주 and so does OSM, so 경부고속선's chain has 23 stops
#         with 경주 between 동대구 and 울산, and the SRT land on the high-speed
#         leg where they belong.
#   2022  the roster still says 신경주, which OSM has no node for, so 경부고속선
#         skips the station outright -- 22 stops, 동대구 straight to 울산 -- and
#         the 경주 node is on 중앙선's and 동해선's chains alone. Aliasing there
#         would hand 3.6M of KTX and SRT to two lines whose trains never carried
#         them. That is the trap README.md's Geometry section describes, and it
#         is still live against 2022.
#
# Both editions were checked by building the chains under each rather than
# reasoned about, and the stop counts above are what the check printed.
#
# `>=` rather than `== 2023` because the rename is permanent. A later edition
# either spells both rows 경주, in which case this key never occurs and the
# entry is inert, or repeats 2023's split and needs exactly this.
if (year() or 0) >= 2023:
    STATION_ALIAS["신경주"] = "경주"


# One name, two stations, and the operator cannot separate them because both are
# Korail's. This is a judgement and it is here rather than in the code so that
# it is visible and arguable.
#
# 양원 is the only case. There is a 양원역 in 봉화 on 영동선 (36.964/129.091,
# `railway=halt`) and a 양원역 in 서울 중랑구 on the 중앙선 (37.607/127.108,
# `network=수도권 전철`), 189.5 km apart, and the yearbook has one 양원 row.
# It is 영동선's, on three grounds:
#
#   - The yearbook counts 일반열차 only. Seoul's 양원 is served by the 경의중앙선
#     광역전철 and by nothing else, so it cannot be the row.
#   - The row carries 새마을 as well as 무궁화 (643/234 하행, 711/292 상행),
#     which is the signature its neighbours 분천, 승부 and 석포 have -- all three
#     within a few km of it on 영동선, and all three in 영동선's roster.
#   - `serves()` would otherwise give 중앙선 a station 189 km from the one
#     영동선 has, and `solve.py` keys junctions by name: the two lines were
#     being made to conserve through flow at a station neither shares.
#
# `8. 시설` sheet 2 would settle it and does not -- 양원 is in no line's roster,
# which is a reminder that the roster misses small halts even where the 승하차
# table has them. `check.py`'s junction check is what flags a case like this;
# if it ever reports another, the answer belongs here.
#
#     station -> the only line whose station the 승하차 row describes
STATION_HOME = {
    "양원": "영동선",
}


def _alias(out, name, v):
    nm = STATION_ALIAS.get(name, name)
    p = out.get(nm)
    out[nm] = v if p is None else tuple(p[i] + v[i] for i in range(4))


def station_flows():
    """역별 승하차 -> {station: (하행승차, 하행하차, 상행승차, 상행하차)}."""
    wb = _book(PASSENGER)
    ws = wb["8"]
    out = {}
    for row in ws.iter_rows(min_row=7, max_row=ws.max_row,
                            min_col=2, max_col=10, values_only=True):
        st = row[0]
        if not st or not str(st).strip() or str(st).strip() == "합계":
            continue
        v = [_num(x) for x in row[1:]]
        _alias(out, str(st).strip(), (v[0], v[1], v[4], v[5]))
    wb.close()
    return out


# Sheet 9 carries KTX and SRT in one table, split by a 열차종 label in column A
# and shifted one column right of the others; sheets 10-13 are one train type
# each, station name in column A.
TYPE_SHEETS = {"9": None, "10": "새마을", "11": "ITX-새마을",
               "12": "무궁화", "13": "통근"}


def station_flows_by_type():
    """{train type: {station: (하행승차, 하행하차, 상행승차, 상행하차)}}.

    Sheet 8 is the total over all train types, which is what breaks the trunk
    lines: 서울's arrivals are KTX riding 경부고속선 *and* 무궁화 riding 경부선,
    and cumulating the sum along either one is meaningless. Splitting by train
    type separates the parallel pair, since only the high-speed services use the
    고속선.
    """
    wb = _book(PASSENGER)
    out = {}
    for sheet, fixed in TYPE_SHEETS.items():
        ws = wb[sheet]
        col = 2 if fixed is None else 1        # sheet 9 has 열차종 in column A
        kind = fixed
        for row in ws.iter_rows(min_row=5, max_row=ws.max_row,
                                min_col=1, max_col=col + 7, values_only=True):
            if fixed is None and row[0] and str(row[0]).strip():
                kind = str(row[0]).strip()
            st = row[col - 1]
            if not st or not str(st).strip() or str(st).strip() == "합계":
                continue
            v = [_num(x) for x in row[col:col + 7]]
            _alias(out.setdefault(kind, {}), str(st).strip(),
                   (v[0], v[1], v[4], v[5]))
    wb.close()
    return out


def line_passing():
    """선별 통과인원 -> {traffic-table name: 명/년}."""
    wb = _book(PASSENGER)
    ws = wb["5"]
    out = {}
    for row in ws.iter_rows(min_row=9, max_row=ws.max_row,
                            min_col=1, max_col=2, values_only=True):
        if row[0] and str(row[0]).strip():
            out[str(row[0]).strip()] = _num(row[1])
    wb.close()
    return out


def rosters():
    """노선 -> {station, ...} from the facility table."""
    wb = _book(FACILITY)
    ws = wb["2"]
    out, cur = {}, None
    for row in ws.iter_rows(min_row=6, max_row=ws.max_row,
                            min_col=3, max_col=4, values_only=True):
        ln, st = row
        if ln and str(ln).strip():
            cur = str(ln).strip()
        if cur and st and str(st).strip():
            out.setdefault(cur, set()).add(str(st).strip())
    wb.close()
    return out


def distances():
    """영업선로별 철도거리 -> {name: (from, to, km)}, km summed over track types."""
    wb = _book(FACILITY)
    ws = wb["4"]
    out = {}
    for row in ws.iter_rows(min_row=8, max_row=ws.max_row,
                            min_col=1, max_col=11, values_only=True):
        name = row[0]
        if not name or not str(name).strip():
            continue
        name = " ".join(str(name).split())
        a, b = row[3], row[5]
        km = sum(_num(x) for x in row[7:11])
        if a and b and km > 0:
            out[name] = ("".join(str(a).split()), "".join(str(b).split()), km)
    wb.close()
    return out


def bad_anchor(station, roster, flows):
    """Can this end station anchor the reconstruction?

    The anchor asserts that everything alights here in 하행 and boards here in
    상행, which is only true at a station belonging to this line and nowhere
    else. The roster assigns every station exactly one *home* line, so an end
    station whose home is some other line is a junction shared with it -- 익산's
    home is 호남선, and its 591k arrivals are mostly 호남선's, not 장항선's.
    A station with no 승하차 row at all cannot anchor anything either.
    """
    if station not in flows:
        return "no 승하차 row"
    if roster and station not in roster:
        return "belongs to another line"
    return None


def resolve():
    """Everything the build needs, per canonical line name.

    `last` is the end the reconstruction anchors on, so a true terminus is
    preferred; if only one end is a junction the pair is swapped to put the
    clean one last. `clean_end` false means both ends are junctions and the
    profile's level will have to be solved rather than measured.
    """
    flows, passing = station_flows(), line_passing()
    by_type = station_flows_by_type()
    rost, dist = rosters(), distances()

    out = {}
    for canon, (fname, dname, rname, osm) in LINES.items():
        if dname not in dist:
            out[canon] = {"error": "no distance row named %r" % dname}
            continue
        legal_a, legal_b, km = dist[dname]
        a, b = ENDS.get(canon, (legal_a, legal_b))
        # 영업거리 measures the legal extent. Where ENDS moves an end it stops
        # describing the track this build draws -- 대구선's row is 가천-영천,
        # 26.1 km, while trains run 동대구-영천 and OSM draws 32.3. Rescaling the
        # chainage to it then shrinks every segment by 24 %, and the rescale is
        # what hides that: the stations stay in the right proportion of the line
        # and only the kilometres are wrong. Nothing published replaces it, so
        # these lines keep OSM's chainage, which the lines that *do* match show
        # to be good to a few tenths of a per cent.
        #
        # OVER is the same fault arriving the other way. 경부고속선's legal ends
        # really are 서울-부산, so this test passes and it used to be rescaled --
        # but its 영업거리 of 398.2 km measures the high-speed metals only, and
        # once the corridor is extended 18.7 km up 경부선 to reach 서울역 it draws
        # 417.3 km, which is the distance a KTX actually runs. Rescaling then
        # squeezed 417 km of railway into 398 and took the difference out of
        # every other segment, so 서울-광명 came out right and 대전-동대구 came
        # out short. Same rule: a corridor covering track the 영업거리 does not
        # keeps OSM's own chainage.
        scaled = ((a, b) == (legal_a, legal_b)
                  and not any(L == canon for L, _ in OVER))
        roster = rost.get(rname, set()) if rname else set()
        # Sum only the train types that use this line's metals, and a type the
        # line runs over only part of its length only at the stations on that
        # part -- see PART_TYPES.
        declared = TYPES.get(canon, ALL_TYPES)
        part_kinds, part_stations = PART_TYPES.get(canon, ((), None))
        kinds = list(declared) + [k for k in part_kinds if k not in declared]

        def ours(st, k):
            """Is this station's traffic of this type on this line's trains?

            Both filters drop a row rather than move it, so a type barred here
            has to be somebody's at the same platform or its passengers vanish
            -- see TYPE_HOME, whose every entry names the line that takes them.
            """
            if part_stations is not None and k in part_kinds \
                    and st not in part_stations:
                return False
            return TYPE_HOME.get(st, {}).get(k, canon) == canon

        lf = {}
        for k in kinds:
            for st, v in by_type.get(k, {}).items():
                if not ours(st, k):
                    continue
                p = lf.get(st, (0.0, 0.0, 0.0, 0.0))
                lf[st] = tuple(p[i] + v[i] for i in range(4))
        # Keep the split as well as the sum. Allocating a shared station between
        # the lines calling there has to be done a train type at a time -- the
        # KTX at 광주송정 are 호남고속선's and 광주선's to divide, and have
        # nothing to do with 호남선's 무궁화 standing at the same platforms.
        lfk = {}
        for k in kinds:
            if k not in by_type:
                continue
            lfk[k] = {st: v for st, v in by_type[k].items() if ours(st, k)}
        bad_a, bad_b = (bad_anchor(a, roster, lf), bad_anchor(b, roster, lf))
        # Put the usable end last, since that is the one the anchor reads. That
        # can leave the chain running 종점 -> 기점, i.e. against 하행, and the
        # 승하차 columns are labelled by the line's own 기점 -> 종점 -- so anyone
        # cumulating along the chain has to know and swap them. 경부선, 중앙선
        # and 수서고속선 are the three it happens to.
        rev = False
        if bad_b and not bad_a:
            a, b, bad_a, bad_b = b, a, bad_b, bad_a
            rev = True
        out[canon] = {
            "first": a, "last": b, "length_km": km, "reversed": rev,
            "legal_ends": (legal_a, legal_b), "scaled": scaled,
            "clean_end": bad_b is None, "why": bad_b,
            # The last stop is this line's own, but trains run through it -- so
            # it may not anchor, while the level still comes from the profile
            # rather than from 통과인원. See THROUGH_ENDS.
            "through_end": b in THROUGH_ENDS.get(canon, ()),
            "ways": osm,
            "roster": roster,
            "passing": passing.get(fname, 0.0) if fname else 0.0,
            "types": kinds,
            # Whether 통과인원 -- which counts everyone on the line's metals --
            # is comparable with the rebuild, which sums only this line's types.
            # A part type does not make a line fully typed: 호남선 runs the
            # high-speed services over a fifth of its length and the published
            # figure still counts through traffic the rebuild cannot see, so it
            # stays a ceiling rather than an equality.
            "full_types": set(declared) == set(ALL_TYPES) and not part_kinds,
            "flows": lf,
            "flows_by_kind": lfk,
        }
    return out, flows


def main():
    table, flows = resolve()
    print("%-11s %9s %8s %-9s %-9s %8s %12s"
          % ("line", "length", "anchor", "first", "last", "roster", "통과인원"))
    print("-" * 74)
    for canon in LINES:
        r = table[canon]
        if "error" in r:
            print("%-11s  %s" % (canon, r["error"]))
            continue
        # A line whose ends were moved is not measured by its own 영업거리 any
        # more, so say what the figure actually covers rather than printing it
        # beside a pair of stations it does not run between.
        note = r["why"] or ""
        if not r["scaled"]:
            moved = "영업거리 covers %s-%s" % r["legal_ends"]
            note = "%s; %s" % (note, moved) if note else moved
        print("%-11s %8.1f %8s %-9s %-9s %8d %12.0f  %s"
              % (canon, r["length_km"], "clean" if r["clean_end"] else "junction",
                 r["first"][:9], r["last"][:9], len(r["roster"]), r["passing"],
                 note))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
