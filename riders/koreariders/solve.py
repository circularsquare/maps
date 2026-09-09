# -*- coding: utf-8 -*-
"""Solve the whole network at once, instead of each line on its own.

Cumulating one line in isolation assumes every passenger who boards on it also
alights on it. They do not. Someone riding 서울 to 목포 by 무궁화 boards at 서울,
alights at 목포, and leaves 경부선 at 대전조차장 *without appearing in any
station's 하차* -- they simply carry on down 호남선. A per-line cumulation has no
way to represent that, so it carries them all the way to 부산, and 경부선 ends up
with negative loads once the real alightings are subtracted.

So the load on line L is modelled as

    load_i = base_i + E_L + sum of tau over the junctions at or before stop i

where `base` is the plain cumulation of that line's allocated station flows, E_L
is the flow entering the line at its first stop, and each `tau` is the traffic
stepping on or off the line at a junction without using a platform. Conservation
makes the taus at one junction sum to zero: what leaves one line joins another.

Unknowns, all solved together:

    E_L        one per line per direction -- entry flow
    tau        one per (line, junction, direction) -- interchange step
    share      how a shared station's 승하차 splits between the lines calling
               there; 익산's 무궁화 passengers are 호남선's, 전라선's and 장항선's
               and the yearbook reports one number

Constraints, as weighted residuals for scipy's least_squares:

    anchor     at a true terminus everything alights, so the last segment's load
               is known outright -- this is what made the single-line version
               work and it still carries most of the information
    positive   no segment may carry fewer than nobody; the hard evidence that a
               reconstruction has gone wrong
    passing    each line's rebuilt 통과인원 against the published count
    junction   taus at a junction sum to zero, per direction
    mirror     하행 and 상행 profiles should agree, as they do to ~2% wherever
               the single-line method was trustworthy
    prior      taus small unless the data demands them, shares near their
               traffic-weighted split -- these only pick among the solutions the
               constraints above leave open

    python solve.py            # solve and report
    python solve.py --line 경부선
"""

import argparse
import collections
import io
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

import build as B
import frequency as FQ
import lines as LN
import membership as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "segments.geojson")

# Residual weights. The first three are evidence, the last two are taste; the
# ordering matters more than the values.
W_ANCHOR = 3.0
W_POSITIVE = 6.0
W_CEILING = 20.0
W_PASSING = 2.0
W_JUNCTION = 2.0
W_SHARESUM = 6.0
W_MIRROR = 1.0
W_TAUSYM = 2.0
W_HOSYM = 30.0   # a handover's two directions, which the same trains make
W_FREQ = 12.0
W_NOTRAIN = 12.0
W_CONTAIN = 20.0
W_ENTRY = 2.0    # 1.0 loses a third of 호남고속선 and buys no mirror back
W_TAU = 0.15
W_SHARE = 0.30

SCALE = 1e6         # work in millions of passengers/year, so residuals compare


def host_slice(shape, station, reach):
    """The host line's own metals from `reach` back to `station`.

    Returned running outward from `reach` -- the borrowing line's corridor end --
    towards the station, which is the orientation `build.graft` wants.
    """
    poly, cum, raw = shape.get("poly"), shape.get("cum"), shape.get("raw") or {}
    if not poly or station not in raw:
        return None
    d, k_join = B.project(reach, poly, cum)
    if d > 1.0:                 # the host does not actually pass the junction
        return None
    seg = B.slice_corridor(poly, cum, k_join, raw[station])
    return seg if len(seg) >= 2 else None


def build_chains(table, serves, g, named, corridors=None):
    """Each line's stops in running order, keeping every station that calls.

    Two passes. `lines.OVER` names lines whose corridor has to be extended along
    another line's metals to reach an end station, and that host corridor only
    exists once it has been built -- so everything is built once, and the
    borrowers are then rebuilt with the slice in hand.
    """
    chains, shapes = {}, {}

    # A handover line runs on past its last platform to the junction, and
    # lines.RUN_ON says which; the point is the one HANDOVER_POINT already
    # records for drawing the receiving line's step.
    beyond = {}
    for L, end in LN.RUN_ON:
        jpt = LN.HANDOVER_POINT.get((L, end))
        if jpt:
            beyond.setdefault(L, {})[end] = jpt

    def one(canon, spec, over=None):
        # Membership by track proximity plus a train type that stops there,
        # rather than the roster's one-home-line-per-station.
        spec = dict(spec)
        spec["roster"] = set()
        shape = {}
        stops, err = B.order_stations(spec, g, named, spec["flows"], shape,
                                      over, beyond.get(canon))
        if err:
            print("%-11s %s" % (canon, err))
            return
        keep = [(km, nm) for km, nm in stops
                if canon in serves.get(nm, [canon]) or nm in (spec["first"],
                                                              spec["last"])]
        if len(keep) >= 3:
            chains[canon] = keep
            shapes[canon] = shape

    # Ends lines.OVER governs, reserved on the first pass so the generic stub
    # cannot take them before the host corridor exists to be sliced. See the
    # `over` handling in build.order_stations.
    reserved = {}
    for (L, end) in LN.OVER:
        reserved.setdefault(L, {})[end] = None

    for canon, spec in table.items():
        if "error" in spec or spec["passing"] <= 0:
            continue
        one(canon, spec, reserved.get(canon))

    for (canon, end), hostname in LN.OVER.items():
        if canon not in shapes or hostname not in shapes:
            continue
        poly = shapes[canon].get("poly")
        if not poly:
            continue
        # Which end of this corridor is the one that stops short.
        at = 0 if end == table[canon]["first"] else -1
        seg = host_slice(shapes[hostname], end, poly[at])
        if not seg:
            print("%-11s no %s corridor to reach %s over" % (canon, hostname,
                                                             end))
            continue
        print("%-11s reaching %s over %s: %.1f km"
              % (canon, end, hostname,
                 sum(B.haversine(a, b) for a, b in zip(seg, seg[1:]))))
        one(canon, table[canon], {end: seg})

    if corridors is not None:
        corridors.update(shapes)
    return chains


class Network(object):
    def __init__(self, table, chains, serves, flows_by_line):
        self.table, self.chains = table, chains
        self.flows = flows_by_line
        self.lines = sorted(chains)
        self.rev = {L: bool(table[L].get("reversed")) for L in self.lines}

        # Published trains per section. Where the count does not change across a
        # station, no service joins or leaves there and a junction step is not
        # merely unlikely but impossible -- 경부선 runs 66 conventional trains a
        # day from 서울 unchanged as far as 천안, so the 3.7M step the fit kept
        # putting at 용산 cannot exist however well it suited the 승하차.
        # Stored in chain order, which is backwards from the sheet's for a
        # reversed line.
        self.freq = {}
        for L in self.lines:
            ch = FQ.changes(L, table[L]["types"])
            if ch is not None:
                self.freq[L] = {nm: ((b, a) if self.rev[L] else (a, b))
                                for nm, (a, b) in ch.items()}

        # And the count per segment, not just the change across a station. A
        # section running no trains of this line's types carries no passengers
        # of them either -- 경원선 north of 청량리 is 전동차 only and north of
        # 소요산 is shut altogether, so the 16,694 a day the fit drew all the way
        # to 백마고지 on the DMZ was not a small error.
        self.runs = {}
        for L in self.lines:
            order = [nm for _, nm in chains[L]]
            rr = FQ.runs_along(L, table[L]["types"],
                               order[::-1] if self.rev[L] else order)
            if rr is not None:
                self.runs[L] = rr[::-1] if self.rev[L] else rr

        # A blank section row is not the same claim as a zero, and the
        # passenger sheet can tell them apart.
        #
        # The rule above reads "no trains of this line's types" off the 운전
        # volume, and a blank cell reads as no trains. That is right where the
        # railway really was shut -- 경원선 north of 소요산, closed for the 전철
        # works -- and it is the whole reason the rule exists. It is wrong where
        # the sheet simply did not report, and the 2023 edition does that: it
        # leaves 동해선's 포항-영덕 blank where 2022 gave it 무궁화 5, and blanks
        # 정선선 outright. Taken at face value the tail of 동해선 collapses to
        # -28, -3, 19 and 64 a day, two of them impossible.
        #
        # The other sheet settles it. Passengers who boarded had to board
        # something, so a section with no trains must have no riders on it, and
        # the 승하차 rows are published independently of the 운행횟수 ones.
        # Across both editions there are twelve blank segments and the test
        # splits them cleanly, with no ambiguous case:
        #
        #   경원선 ...-백마고지          nobody         really shut, trust it
        #   정선선 아우라지-구절리        nobody         shut since 2004, trust it
        #   동해선 포항-...-영덕      127,058/yr        the sheet contradicts itself
        #   정선선 민둥산-...-아우라지  20,256/yr        likewise
        #
        # Per segment, not per run of them, and that distinction is the whole
        # difficulty. 광운대 and 동두천 report 1,762 and 1,466 conventional
        # riders a year -- nine a day between them, on a stretch the sheet gives
        # no intercity train -- so a rule that frees a whole blank run as soon
        # as any station on it reports somebody frees 경원선 all the way to the
        # DMZ and reinstates exactly the error this constraint was written for.
        # Judged one segment at a time, those two free their own segment and
        # nothing beyond it, which is both harmless and arguably right: the
        # stations are drawn grey with no rider figure either way, since that
        # flag is read off the section counts and not off this.
        #
        # The evidence is the station the segment *leads to* along the chain.
        # `resolve()` puts the clean anchor last, so chain order runs from the
        # junction end outward and stops[i+1] is the one further out -- the one
        # that can only be reached across this segment. The near end is shared
        # with the section before, which may be running trains perfectly well;
        # 포항 has three million riders and says nothing about 포항-월포.
        self.notrain = {}
        for L, rr in self.runs.items():
            stops, f = chains[L], flows_by_line[L]
            idx = []
            for i, n in enumerate(rr):
                if n:
                    continue
                v = f.get(stops[i + 1][1])
                if v and sum(abs(x) for x in v):
                    continue
                idx.append(i)
            self.notrain[L] = idx

        # Which lines call at each station, restricted to those actually mapped.
        on = collections.defaultdict(set)
        for L, stops in chains.items():
            for _, nm in stops:
                on[nm].add(L)
        self.on = on
        shared = set(nm for nm, ls in on.items() if len(ls) > 1)

        # Handovers at a place the receiving chain does not name -- see
        # lines.HANDOVER. Both ends have to be brought into play: the handing
        # line's own end so it may carry a through flow at all, and the
        # receiving stop so a step exists there to take it. Neither becomes a
        # share group, since only one line calls at either.
        self.handover = {}
        for (L, end), (Mm, stop) in LN.HANDOVER.items():
            if L not in chains or Mm not in chains:
                continue
            order = [nm for _, nm in chains[Mm]]
            if end not in (chains[L][0][1], chains[L][-1][1]):
                continue
            if stop not in order or not 0 < order.index(stop) < len(order) - 1:
                continue
            self.handover[(L, end)] = (Mm, stop)
            shared.update((end, stop))
        self.ho_recv = set(v for v in self.handover.values())
        self.shared = sorted(shared)

        # Lines whose published 통과인원 counts trains the model cannot claim.
        #
        # 통과인원 counts everyone who used a line's metals. The 승하차 sheets
        # record six train types, and a line is modelled with the subset that
        # runs on it -- so where the 운전 volume shows a line carrying a type
        # that is *not* in that subset, the published figure includes passengers
        # no station row can ever supply to this line, and it is a ceiling
        # however clean the terminus is.
        #
        # 경원선 is the case. Its only intercity section is 용산-청량리, running
        # ITX-청춘 26 and 고속열차 18 a day, and its types are CONVENTIONAL. The
        # 18 고속열차 are 7.50M passengers a year -- 571 a train, against a
        # KTX-1's 935 seats -- and they board at 용산, whose KTX row is 9.0M of
        # 호남선 and 전라선 traffic that never touches 경원선. There is no
        # published way to separate them, so the fit was being driven to a
        # number it cannot construct, and it took the difference out of the
        # 용산-청량리 entry flow one direction harder than the other. That was
        # 경원선's 24.5 % mirror.
        #
        # The test is whether the line has *any* claimable intercity service at
        # all, not whether some train is unclaimed. Firing on any unclaimed
        # train was tried and is far too blunt: 영동선 runs 33 claimable trains
        # a day against 7 it cannot claim, its 통과인원 is mostly reachable, and
        # freeing its level took it from a 2.1 % mirror to 19.0 % and put a
        # negative segment on the network. 경원선 has **no** claimable service on
        # any section -- the 고속열차 are not its types and the ITX-청춘 have no
        # 승하차 row (FQ.NO_FLOW) -- so its 7.50M is entirely unreachable.
        #
        # Fires on 경원선 and 경춘선; 경춘선 has a clean terminus so its
        # 통과인원 was already a one-sided ceiling and nothing moves there.
        #
        # The set now does three jobs, and they are the same statement said
        # three ways. It exempts the line from the 통과인원 target below; it
        # bars the line from every share group and from `alloc`, because a line
        # that can claim no train can claim no passenger either; and
        # `write_geojson` draws it from the published total instead, since with
        # no allocation left there is nothing to cumulate. Adding the second
        # without the third would have drawn 경원선 at nil.
        #
        # A line every one of whose section rows is *blank* is not evidence of
        # anything and must not land here. 정선선 in 2023 is the case: its one
        # row is empty, `claimable` duly returns nothing for it, and without
        # this guard the line is declared to have no claimable service, takes no
        # station allocation, and draws nil the length of it -- while its own
        # stations report 20,256 riders. Untyped has to be read off rows that
        # actually report trains, which is what makes it a finding about 경원선
        # rather than about the sheet's gaps.
        self.untyped = set()
        for L in self.lines:
            secs = FQ.by_line().get(L)
            if not secs or not any(runs for _, _, runs in secs):
                continue
            kinds = table[L]["types"]
            if not any(FQ.claimable(runs, kinds) for _, _, runs in secs):
                self.untyped.add(L)

        # Parameter layout.
        self.idx, n = {}, 0
        for L in self.lines:
            for d in (0, 1):
                self.idx[("E", L, d)] = n
                n += 1
        # A tau only exists where it can move a segment. Stop 0's entry is E,
        # and traffic joining at the very last stop rides no segment of this
        # line at all -- allocating one there gives the junction constraint a
        # free variable that satisfies it without touching any profile, which
        # is how 오송 came to inject 307k passengers into 충북선.
        self.taus = []
        for L in self.lines:
            for i, (_, nm) in enumerate(chains[L][:-1]):
                if i > 0 and nm in self.shared:
                    for d in (0, 1):
                        self.idx[("T", L, nm, d)] = n
                        self.taus.append((L, nm, d, i))
                        n += 1
        # Shares are per train type, not per line. A station's row for one type
        # is divided between the lines there that actually run it: the KTX at
        # 광주송정 are 호남고속선's and 광주선's to divide and have nothing to do
        # with 호남선's 무궁화 at the same platforms, and 부산's 무궁화 are
        # 경부선's alone however many KTX also terminate there. Dividing per line
        # instead let 광주선 -- which carries every type -- bridge the two, and
        # scaled the high-speed lines down against traffic that was never theirs.
        #
        # An `untyped` line takes no share of anything. It runs no train any
        # station row can be attributed to -- that is what put it in the set --
        # so every share it held was traffic belonging to whichever line at the
        # platform *does* run those trains. 경원선 was taking about a third of
        # 용산's 새마을, ITX-새마을 and 무궁화, 1.1M passengers a year, off
        # 경부선, on a section the 운전 volume gives 고속열차 18 and ITX-청춘 26
        # and no conventional train at all. Where that leaves one line holding
        # the platform the group disappears and it takes the row outright, which
        # is the right answer and not a fallback.
        self.kinds = {L: list(table[L]["types"]) for L in self.lines}
        self.fk = {L: table[L].get("flows_by_kind", {}) for L in self.lines}

        # Gating a share *per train type* on what the section table says runs
        # at that platform was tried on 2026-09-08 and is not worth having --
        # see README.md, "The share prior already knows a branch line is
        # small". It bars 19 memberships, and 경북선, 대구선, 정선선 and 충북선
        # come out identical to the passenger, because the traffic-weighted
        # prior below already gives a branch line a fraction of a per cent of a
        # trunk junction's rows. Its only live effect is on 광주송정, 익산 and
        # 순천, where it would overturn curated high-speed attributions on the
        # strength of one sheet's line naming.
        self.groups = []
        for nm in self.shared:
            for k in LN.ALL_TYPES:
                ls = [L for L in sorted(on[nm])
                      if k in self.kinds[L] and L not in self.untyped
                      and any(self.fk[L].get(k, {}).get(nm, (0,) * 4))]
                if len(ls) > 1:
                    self.groups.append((nm, k, ls))

        self.shares = []
        for nm, k, grp in self.groups:
            for L in grp:
                self.idx[("S", nm, k, L)] = n
                self.shares.append((nm, k, L))
                n += 1
        self.n = n

        # Traffic-weighted prior for the shares: a busier line takes more of a
        # shared station's passengers. Crude, but it only breaks ties.
        self.share_prior = {}
        for nm, k, grp in self.groups:
            w = {L: max(table[L]["passing"], 1.0) for L in grp}
            tot = sum(w.values())
            for L in grp:
                self.share_prior[(nm, k, L)] = w[L] / tot

        # Precomputed allocation terms, so a residual pass is a short walk over
        # a list rather than a dictionary lookup per station per train type.
        #
        # Dropping a line from the groups above is not enough on its own, and
        # getting it wrong is worse than leaving it alone: a station with no
        # group falls through to a share of 1.0, so 경원선 would have gone from
        # a third of 용산 to all of it. A line that may not share a row may not
        # allocate it either, and the two tests have to be the same one.
        self.alloc = collections.defaultdict(list)
        for L in self.lines:
            if L in self.untyped:
                continue
            for _, nm in chains[L]:
                for k in self.kinds[L]:
                    v = self.fk[L].get(k, {}).get(nm)
                    if v and any(v):
                        self.alloc[(L, nm)].append(
                            (self.idx.get(("S", nm, k, L), -1), v))

        # Riders whose station is on no chain of the line they ride, allocated
        # where they actually join it -- see lines.OFF_CHAIN.
        #
        # At a fixed share of 1.0 rather than a fitted one. A share variable
        # would need a group, and a group needs two lines calling at the
        # station that both declare the type; 용산 has neither, which is
        # exactly why the row was orphaned. Nothing else can claim these
        # passengers, so there is nothing to divide.
        # Kept per direction as well, because the entry-share residual below
        # has to know how much of a feeding line's load is this traffic: it is
        # measured and it all turns off at the junction, so it is not part of
        # what the train-count share divides. See the residual for what
        # happens without that.
        self.offchain = collections.defaultdict(float)
        for (st, k), (L, join) in LN.OFF_CHAIN.items():
            if L not in chains or k not in self.kinds[L]:
                continue
            if join not in set(nm for _, nm in chains[L]):
                continue
            v = self.fk[L].get(k, {}).get(st)
            if v and any(v):
                self.alloc[(L, join)].append((-1, v))
                for d in (0, 1):
                    b, a = self._cols(L, d, v)
                    self.offchain[(L, d)] += abs(b - a) / SCALE

        # A part type runs over only a span of its line, so its passengers ride
        # only that span: whatever boards inside it has to leave at the
        # boundary rather than carry on up the rest of the line. That is a
        # consequence of lines.PART_TYPES rather than a new assumption, and
        # nothing else in the fit says it. 호남선's 목포 KTX and SRT rode north
        # past 광주송정 instead of transferring to 호남고속선, lifting
        # 서대전-계룡 by 8,330 a day; suppressing 호남고속선's false anchor there
        # frees the transfer but does not compel it, and the smallness prior
        # then leaves it at zero.
        #
        # Held as the index of the last segment inside the span, against which
        # the first one outside is compared. Only a span reaching one end of
        # the chain is handled -- 호남선's runs 나주 to 목포 -- since a span in
        # the middle would have two boundaries and no line needs that yet.
        self.contain = []
        for L in self.lines:
            kinds, span = LN.PART_TYPES.get(L, ((), None))
            if not kinds:
                continue
            order = [nm for _, nm in chains[L]]
            inside = [i for i, nm in enumerate(order) if nm in span]
            if len(inside) < 2:
                continue
            lo, hi = min(inside), max(inside)
            if hi != len(order) - 1 or lo < 2:
                continue
            terms = []
            for i in inside:
                for k in kinds:
                    v = self.fk[L].get(k, {}).get(order[i])
                    if v and any(v):
                        terms.append(
                            (self.idx.get(("S", order[i], k, L), -1), v))
            if terms:
                self.contain.append((L, lo - 1, terms))

        # Entry flows sized from the published train counts -- see
        # lines.ENTRY_SHARE. Held as the feeding line's segment arriving at the
        # junction and the share of its service that turns off there.
        #
        # The share comes from the feeding line's own count either side of the
        # junction, not from dividing one line's count by the other's. Per
        # segment the counts do not line up: 천안아산-오송 spans SR분기, where
        # the SRT join, so runs_along gives it the 117 from the near side while
        # 177 arrive at 오송 -- which would have made 호남고속선's share 50/117
        # rather than 50/177, half again too big. Read at the station the split
        # actually happens at, 177 to 127, the 50 that leave are its own count
        # and the arithmetic closes.
        self.entry = []
        for L, (M, J) in LN.ENTRY_SHARE.items():
            if L not in chains or M not in chains:
                continue
            order = [nm for _, nm in chains[M]]
            if J not in order or order.index(J) < 1:
                continue
            step = FQ.match(J, self.freq.get(M) or {})
            if not step or step[0] <= step[1]:
                continue
            self.entry.append((L, M, order.index(J) - 1,
                               float(step[0] - step[1]) / float(step[0])))

    def x0(self):
        """Start from the single-line answer, with every tau at zero."""
        x = np.zeros(self.n)
        for nm, k, L in self.shares:
            x[self.idx[("S", nm, k, L)]] = self.share_prior[(nm, k, L)]
        for L in self.lines:
            stops, f = self.chains[L], self.flows[L]
            prof = {d: B.reconstruct(stops, f, d == 0, self.rev[L])
                    for d in (0, 1)}
            # build.py's levelling: no clean terminus means the anchor read the
            # junction's whole traffic and lifted both profiles by a constant,
            # and 통과인원 is what pins it back down.
            spec = self.table[L]
            if not spec["clean_end"] and spec["passing"] > 0:
                users = (prof[0][0] + prof[1][-1]
                         + sum(f.get(nm, (0, 0, 0, 0))[0]
                               + f.get(nm, (0, 0, 0, 0))[2]
                               for _, nm in stops[1:-1]))
                shift = (spec["passing"] - users) / 2.0
                prof = {d: [v + shift for v in prof[d]] for d in (0, 1)}
            # E is the first segment's load, which is what build.py just gave.
            for d in (0, 1):
                x[self.idx[("E", L, d)]] = prof[d][0] / SCALE
        return x

    def profile(self, x, L, d):
        """Segment loads for one line and direction, in millions/year."""
        stops = self.chains[L]
        f = self.flows[L]
        # 상행 trains run the other way along the same stop order, so a boarding
        # at stop i adds to the load *behind* it, not ahead. build.reconstruct
        # gets this free by walking backwards; cumulating forwards must carry
        # the sign, and dropping it makes the up profile the near-negative of
        # the down one -- which the mirror residual then flattens both to kill.
        sgn = 1.0 if d == 0 else -1.0
        # Stop 0 stays out of the cumulation in both directions, so E is the
        # load on the first segment outright. It has to be exactly that for the
        # junction constraint below to read a through flow off it.
        base, cur = [0.0], 0.0
        for _, nm in stops[1:-1]:
            b, a = self._cols(L, d, self._flow(x, L, nm))
            cur += sgn * (b - a) / SCALE
            base.append(cur)
        # Entry flow, then a step at each junction.
        out, step = [], x[self.idx[("E", L, d)]]
        for i, val in enumerate(base):
            k = self.idx.get(("T", L, stops[i][1], d))
            if k is not None and i > 0:
                step += x[k]
            out.append(val + step)
        return np.array(out)

    def _flow(self, x, L, nm):
        """This line's allocated 승하차, as (하행승, 하행하, 상행승, 상행하)."""
        b0 = b1 = b2 = b3 = 0.0
        for j, v in self.alloc.get((L, nm), ()):
            s = 1.0 if j < 0 else x[j]
            b0 += s * v[0]
            b1 += s * v[1]
            b2 += s * v[2]
            b3 += s * v[3]
        return b0, b1, b2, b3

    def _cols(self, L, d, v):
        """(boardings, alightings) for trains running the way this d does.

        d == 0 is stop 0 -> stop n along the chain, which is 하행 only when the
        chain runs 기점 -> 종점. lines.resolve() flips it to put the clean anchor
        last, and the 승하차 columns then have to be swapped to match -- 경부선,
        중앙선 and 수서고속선 are the three lines that need it, and all three are
        ones the single-line build gave up on.
        """
        i = 0 if (d == 0) != self.rev[L] else 2
        return v[i], v[i + 1]

    def residuals(self, x):
        r = []
        prof = {}
        for L in self.lines:
            for d in (0, 1):
                prof[(L, d)] = self.profile(x, L, d)

        for L in self.lines:
            spec, stops = self.table[L], self.chains[L]
            f = self.flows[L]
            down, up = prof[(L, 0)], prof[(L, 1)]

            # Anchor -- only where the far end is a genuine terminus. It takes
            # the line's share of the platform, not all of it: 익산's conventional
            # alightings are 호남선's, 전라선's and 장항선's together, and handing
            # every line the whole figure anchors each of them to all three.
            # A through end is the line's own station but not a terminus: 28 of
            # the 42 high-speed trains a day reaching 광주송정 carry straight on
            # to 목포, so asserting that everything alights there is false, and
            # it is the assertion that blocks them. Junction conservation then
            # carries the level instead -- the end segment's load less the
            # platform alightings is the through flow, and it balances against
            # the step 호남선 takes at the same station.
            if spec["clean_end"] and not spec["through_end"]:
                v = self._flow(x, L, stops[-1][1])
                r.append(W_ANCHOR * (down[-1] - self._cols(L, 0, v)[1] / SCALE))
                r.append(W_ANCHOR * (up[-1] - self._cols(L, 1, v)[0] / SCALE))

            # positivity
            r.extend(W_POSITIVE * np.minimum(down, 0.0))
            r.extend(W_POSITIVE * np.minimum(up, 0.0))

            # mirror
            r.extend(W_MIRROR * (down - up))

            # No trains of this line's types on the section, no passengers --
            # except where the 승하차 sheet says otherwise, see `self.notrain`.
            for i in self.notrain.get(L, ()):
                r.append(W_NOTRAIN * down[i])
                r.append(W_NOTRAIN * up[i])

            # 통과인원. It counts everyone who touched the line's metals, while
            # the reconstruction sums only the train types that line runs, so a
            # rebuild may legitimately come in under the published figure -- but
            # never over it. That makes it a ceiling on every line without
            # exception, and leaking traffic in from a junction is precisely
            # what breaks it, so it is the sharpest test available and worth far
            # more than the one-residual-per-line it used to get against four
            # hundred-odd mirror terms.
            #
            # Equality is only expected where the line carries every train type,
            # or where it has no clean terminus and takes its level from here.
            if spec["passing"] > 0:
                interior = sum(sum(self._flow(x, L, nm)[0::2])
                               for _, nm in stops[1:-1]) / SCALE
                users = down[0] + up[-1] + interior
                over = users - spec["passing"] / SCALE
                r.append(W_CEILING * max(over, 0.0))
                # And not where the published figure counts trains this line's
                # types cannot claim -- see `self.untyped`. An earlier version
                # of this exemption keyed on 광역전철 sections instead, on the
                # belief that 통과인원 counts 전동차 riders; the 광역철도 volume
                # says it does not (경원선's 전동차 alone carry 141M against a
                # 통과인원 of 7.5M), and that version fired on lines this one
                # leaves alone. See README.md, "경원선's 통과인원 is a KTX
                # figure".
                if ((not spec["clean_end"] or spec["full_types"])
                        and L not in self.untyped):
                    r.append(W_PASSING * over)

        # A part type's passengers stay inside its span. Everything of those
        # types that boards within the span is riding the last segment inside
        # it, so the first segment outside must be lighter by at least that
        # much -- one-sided, since the conventional step at the same station is
        # free to take more on top.
        for (L, i_in, terms) in self.contain:
            for d in (0, 1):
                sgn = 1.0 if d == 0 else -1.0
                p = 0.0
                for j, v in terms:
                    s = 1.0 if j < 0 else x[j]
                    b, a = self._cols(L, d, v)
                    p -= sgn * s * (b - a)
                drop = prof[(L, d)][i_in] - prof[(L, d)][i_in - 1]
                r.append(W_CONTAIN * min(drop - p / SCALE, 0.0))

        # A line whose riders join it from another and never use a station of
        # its own has nothing to set its entry flow, and the smallness prior
        # then puts it on the floor. The published counts set it: the share of
        # the feeding line's service that turns off at the junction. Paired by
        # the direction a passenger actually travels, since a reversed chain's
        # d == 0 is 상행 and scaling that against the other line's 하행 relates
        # nothing.
        # Off-chain traffic is exempt from the share and added whole. The share
        # describes how the feeding line's *generic* load divides at the
        # junction; lines.OFF_CHAIN's traffic is measured, and all of it turns
        # off -- every KTX out of 용산 is a 호남 or 전라 service and none
        # continues to 부산. Dividing it by the train-count share instead caps
        # the transfer at 28 %, and the measured version of this run showed
        # exactly that: of 23,000 a day injected at 광명, 4,615 reached
        # 호남고속선 and 6,212 rode on to 대전 and 동대구 where none belong,
        # while the fit stopped converging at all -- 400 evaluations against a
        # baseline 26. See README.md, "용산's KTX are drawn".
        for (L, M, seg, share) in self.entry:
            for d in (0, 1):
                dm = d ^ self.rev[L] ^ self.rev[M]
                off = self.offchain.get((M, dm), 0.0)
                want = share * (prof[(M, dm)][seg] - off) + off
                r.append(W_ENTRY * (x[self.idx[("E", L, d)]] - want))

        # Junction conservation, per direction: every line meeting there
        # contributes the through flow it gains, and the total must be nil --
        # what steps off one line steps onto another.
        #
        # A line whose *end* is the junction used to contribute nothing, which
        # left the constraint useless exactly where it was needed. 호남선 starts
        # at 대전조차장, so 경부선's tau there had nothing to balance against and
        # was driven to zero -- the very correction 경부선 needs. A terminal
        # line's through flow is the end segment's load less the passengers who
        # actually use the platform, and at a genuine terminus that is nil of
        # its own accord, which is the consistency check on the whole idea.
        # Lines have to be grouped by the direction a passenger is actually
        # travelling, not by each chain's own order -- a reversed chain's d == 0
        # is 상행, and pairing that against another line's 하행 conserves nothing.
        # 하행 continues as 하행 through a junction because Korean lines are
        # numbered outward from Seoul, and the few that are not (경전선 at
        # 삼랑진, 영동선 at 영주, 경북선 at 김천) still join the radial ones the
        # right way round.
        gain = collections.defaultdict(list)
        # Each line-end's through flow, kept per direction so a handover's two
        # can be held equal below.
        end_through = {}

        def add(nm, L, d, val):
            # Reversing the chain also swaps the sense of a step: forwards along
            # the chain a positive one means joining, backwards it means leaving.
            gain[(nm, d ^ self.rev[L])].append(-val if self.rev[L] else val)

        for L in self.lines:
            stops = self.chains[L]
            for nm, sign, i in ((stops[0][1], 1.0, 0), (stops[-1][1], -1.0, -1)):
                if nm not in self.shared:
                    continue
                v = self._flow(x, L, nm)
                for d in (0, 1):
                    # The platform movement that belongs to the end segment:
                    # boardings where those trains set out, alightings where the
                    # arriving ones empty.
                    plat = self._cols(L, d, v)[d if i == 0 else 1 - d]
                    through = prof[(L, d)][i] - plat / SCALE
                    end_through[(L, nm, d)] = through
                    # A line's through flow at its own end cannot be negative:
                    # at stop 0 it is passengers joining from another line, at
                    # stop n passengers carrying on to one, and neither can be
                    # fewer than nobody. Leaving this out let a junction balance
                    # its books against a line that has no traffic to give --
                    # 경부선 gained 3.6M at 용산 against 경원선 losing the same
                    # there, at the very station 경원선 starts from, and half of
                    # everything it carries.
                    r.append(W_POSITIVE * min(through, 0.0))
                    # And the same smallness prior the taus get. Without it a
                    # line that *ends* at a junction can absorb any amount for
                    # free, which is what licensed 경부선's step at 용산: the
                    # traffic had somewhere to go, so taking it cost nothing.
                    r.append(W_TAU * through)
                    # A handover balances against the receiving line's step,
                    # which is at a different station -- the pool is keyed by
                    # where the traffic lands, not by where it leaves.
                    ho = self.handover.get((L, nm))
                    add(ho[1] if ho else nm, L, d, sign * through)
        for (L, nm, d, i) in self.taus:
            add(nm, L, d, x[self.idx[("T", L, nm, d)]])
        for key, gs in sorted(gain.items()):
            r.append(W_JUNCTION * sum(gs))

        # A handover's two directions are the same trains making both journeys,
        # so its through flow cannot depend on which way it is measured. This is
        # a statement about the railway, not a smoothing prior, which is why it
        # sits at evidence weight rather than beside W_TAUSYM.
        #
        # Without it the handover is an unconstrained ridge and the fit picks an
        # arbitrary point on it. 태백선 is the case that showed it: six trains a
        # day run 태백 → 백산 → 동백산 in each direction, and the fit answered
        # with 하행 16,041/yr against 상행 75,722 — 4.7x, for a service that is
        # symmetric by construction. The arbitrariness then propagated through
        # junction conservation and cost 영동선 and 정선선 their mirrors, which
        # is why that handover had to be reverted rather than kept.
        #
        # Only handover ends. An ordinary junction may genuinely step by
        # different amounts each way, and W_TAUSYM's gentler nudge is the right
        # instrument there.
        for (L, nm) in self.handover:
            a, b = end_through.get((L, nm, 0)), end_through.get((L, nm, 1))
            if a is not None and b is not None:
                r.append(W_HOSYM * (a - b))

        # The two directions' steps at a junction should match: as many people
        # join the outbound trains there as leave the inbound ones. This is the
        # mirror argument applied to the interchange rather than the platform,
        # and without it the taus are the one free way left to drive 하행 and
        # 상행 apart -- which is what put 경전선 and 동해선 behind where the
        # single-line build already had them.
        # A handover's receiving step is held far harder than an ordinary
        # junction's, for the same reason: it is one service, running both ways.
        # This is the half that shows on the map -- `write_geojson` draws the
        # run-on segment from the *receiving* line's tau, so constraining only
        # the handing line's through flow left the drawn 태백-백산 8.7 km as
        # lopsided as before, at an 80 % segment mirror.
        for (L, nm, d, i) in self.taus:
            if d == 0:
                w = W_HOSYM if (L, nm) in self.ho_recv else W_TAUSYM
                r.append(w * (x[self.idx[("T", L, nm, 0)]]
                              - x[self.idx[("T", L, nm, 1)]]))

        # Junction steps, against the published section counts where there are
        # any. A station the train count runs straight through is not a junction
        # at all and its step is held to zero; a station where trains genuinely
        # join or leave keeps the weak prior and lets the 승하차 set the size.
        for (L, nm, d, i) in self.taus:
            k = self.idx[("T", L, nm, d)]
            ch = self.freq.get(L)
            step = FQ.match(nm, ch) if ch is not None else None
            flat = ch is not None and (step is None or step[0] == step[1])
            # Except where the stop is standing in for a junction that is not a
            # station. 경부고속선's count changes at SR분기, which has no 승하차
            # row and no platform, so read at 천안아산 it looks flat -- 177 both
            # sides -- and the rule would forbid the very step the SRT need in
            # order to join at all. The count is not flat; the station it
            # changes at is simply not one this build can draw.
            if (L, nm) in self.ho_recv:
                flat = False
            r.append((W_FREQ if flat else W_TAU) * x[k])
            if not flat and step is not None:
                # The count also gives the direction. Trains leaving cannot put
                # passengers on, and 삼랑진 is 47 trains down to 43 -- 경전선
                # taking services off 경부선 -- so the +2.35M the fit wanted
                # there had the sign of the thing backwards. Both directions
                # take the same rule: forwards a positive step means joining,
                # backwards it means leaving, and either way it is the extra
                # services that carry it.
                grew = step[1] > step[0]
                r.append(W_FREQ * (min(x[k], 0.0) if grew else max(x[k], 0.0)))
        for nm, k, grp in self.groups:
            ks = [self.idx[("S", nm, k, L)] for L in grp]
            r.append(W_SHARESUM * (sum(x[j] for j in ks) - 1.0))
        for nm, k, L in self.shares:
            r.append(W_SHARE * (x[self.idx[("S", nm, k, L)]]
                                - self.share_prior[(nm, k, L)]))
        return np.array(r)

    def bounds(self):
        lo = np.full(self.n, -np.inf)
        hi = np.full(self.n, np.inf)
        for nm, k, L in self.shares:
            lo[self.idx[("S", nm, k, L)]] = 0.0
            hi[self.idx[("S", nm, k, L)]] = 1.0
        for L in self.lines:
            for d in (0, 1):
                lo[self.idx[("E", L, d)]] = 0.0
        return lo, hi


def report(net, x, only=None):
    rows = []
    for L in net.lines:
        down, up = net.profile(x, L, 0) * SCALE, net.profile(x, L, 1) * SCALE
        stops = net.chains[L]
        interior = sum(sum(net._flow(x, L, nm)[0::2]) for _, nm in stops[1:-1])
        users = down[0] + up[-1] + interior
        length = stops[-1][0]
        pkm = sum((down[i] + up[i]) * (stops[i + 1][0] - stops[i][0])
                  for i in range(len(down)))
        mirror = max(abs(down[i] - up[i]) / max(abs(down[i]), abs(up[i]), 1)
                     for i in range(len(down)))
        # The worst single segment is a poor summary: it is dominated by
        # whichever segment carries almost nobody, where a handful of
        # passengers is 100 %. 충북선's 조치원-오송 stub reads 0 against 846 --
        # two people a day -- while its other fifteen segments agree to 3 %.
        # So weight the disagreement by the traffic it applies to.
        gap = np.abs(down - up).sum()
        held = np.maximum(np.abs(down), np.abs(up)).sum()
        weighted = gap / held if held else 0.0
        # d == 0 is the chain's own order; on a reversed chain that is 상행, so
        # swap the pair back before anything is labelled 하행.
        if net.rev[L]:
            down, up = up, down
        rows.append({
            "line": L, "stops": stops, "down": down, "up": up,
            "users": users, "passing": net.table[L]["passing"],
            "mirror": mirror, "weighted": weighted, "length": length,
            "density": pkm / length / 365.0 if length else 0.0,
            "worst": min(down.min(), up.min()),
        })

    print("\n%-11s %5s %7s %11s %11s %7s %7s %9s %10s"
          % ("line", "stops", "km", "통과인원", "yearbook", "mirror", "worst",
             "수송밀도", "min load"))
    print("-" * 90)
    for r in sorted(rows, key=lambda z: -z["density"]):
        print("%-11s %5d %7.1f %11.0f %11.0f %6.1f%% %6.1f%% %9.0f %10.0f%s"
              % (r["line"], len(r["stops"]), r["length"], r["users"],
                 r["passing"], 100 * r["weighted"], 100 * r["mirror"],
                 r["density"], r["worst"],
                 "  *" if r["line"] in net.untyped else ""))
    # These rows are the fit's own answer and it is nil, which is correct: the
    # line runs no train any 승하차 row can be attributed to, so there is
    # nothing to cumulate. It is not what the map draws. 경원선's published
    # 통과인원 is an ordinary 395 a train and `write_geojson` puts it on the
    # sections that run intercity trains, 용산-청량리 and nothing else;
    # 경춘선's is 0.13 a train and stays undrawn. Do not read a zero 수송밀도
    # here as a line the map leaves blank.
    if net.untyped:
        print("* no claimable service on any section, so the fit has nothing "
              "to cumulate.\n  Where the published 통과인원 is a credible "
              "figure the map draws that instead\n  of this row -- see "
              "write_geojson")

    # The junction steps, largest first. These are the part of the answer with
    # the least evidence behind them, so they are where a wrong profile shows
    # up first -- a step of a size the junction cannot plausibly pass is the
    # signature of the fit using one to absorb something else.
    steps = sorted(((abs(x[net.idx[("T", L, nm, d)]]) * SCALE, L, nm, d)
                    for (L, nm, d, i) in net.taus), reverse=True)[:12]
    if steps:
        print("\nbiggest junction steps")
        for v, L, nm, d in steps:
            signed = x[net.idx[("T", L, nm, d)]] * SCALE
            print("   %-11s %-9s %-4s %12.0f   (%.0f%% of the line)"
                  % (L, nm, "하행" if (d ^ net.rev[L]) == 0 else "상행", signed,
                     100 * v / max(net.table[L]["passing"], 1)))

    # An untyped line's profile is not drawn, so a negative in it is not a
    # negative on the map -- `check.py` reads the written file and is the
    # figure to quote. Counted separately rather than dropped, because a fit
    # going deeply negative somewhere is still worth seeing.
    bad = [r for r in rows if r["worst"] < -1000 and r["line"] not in net.untyped]
    hid = [r for r in rows if r["worst"] < -1000 and r["line"] in net.untyped]
    print("\n%d of %d lines still carry a negative segment%s"
          % (len(bad), len(rows),
             (", plus %s in the fit only" % ", ".join(r["line"] for r in hid))
             if hid else ""))
    ratios = [r["users"] / r["passing"] for r in rows if r["passing"] > 0]
    print("통과인원 ratio: median %.3f, worst %.3f / %.3f"
          % (float(np.median(ratios)), min(ratios), max(ratios)))
    print("mirror, weighted by traffic: median %.1f%%, worst %.1f%% (%s)"
          % (100 * float(np.median([r["weighted"] for r in rows])),
             100 * max(r["weighted"] for r in rows),
             max(rows, key=lambda z: z["weighted"])["line"]))

    if only:
        r = [z for z in rows if z["line"] == only][0]
        print("\n%-15s %8s %10s %10s %10s"
              % ("segment", "km", "하행", "상행", "명/일"))
        print("-" * 56)
        for i in range(len(r["down"])):
            a, b = r["stops"][i], r["stops"][i + 1]
            print("%-15s %8.1f %10.0f %10.0f %10.0f"
                  % ((a[1] + "-" + b[1])[:15], b[0] - a[0], r["down"][i],
                     r["up"][i], (r["down"][i] + r["up"][i]) / 365.0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line")
    # More than one session works in this tree, so a run that overwrites
    # data/segments.geojson makes every other session's diff meaningless.
    # A scratch path lets a change be measured against the committed output.
    ap.add_argument("--out", help="write the geojson here instead of data/")
    # Was 400, which was enough until lines.OFF_CHAIN and is not now: the fit
    # took 26 evaluations before that change and takes 482 after it. Both of
    # the first two attempts stopped at the old cap while still descending,
    # and the wreckage looked like a modelling failure -- 호남선 collapsing,
    # 충북선 and 경원선 and 경춘선 all disturbed -- when it was a search cut
    # off partway. Raising the cap alone brought every one of them back.
    #
    # It costs nothing when the fit converges early, since `ftol` stops it
    # either way. **Read `res.message` before believing any number**: a run
    # that ends on the cap is a snapshot of a search, not an answer.
    ap.add_argument("--max-nfev", type=int, default=2000,
                    help="evaluation cap for least_squares (default 2000)")
    args = ap.parse_args()
    if args.out:
        global OUT
        OUT = os.path.abspath(args.out)

    table, _ = LN.resolve()
    serves = M.serves(LN)
    named = M.load_stations(LN)
    print("loading the national track graph ...")
    g = B.load_network()
    corridors = {}
    chains = build_chains(table, serves, g, named, corridors)
    flows = {L: table[L]["flows"] for L in chains}
    net = Network(table, chains, serves, flows)
    print("   %d lines, %d junctions, %d unknowns"
          % (len(net.lines), len(net.shared), net.n))

    x0 = net.x0()
    print("   initial cost %.1f" % (0.5 * float((net.residuals(x0) ** 2).sum())))
    lo, hi = net.bounds()
    res = least_squares(net.residuals, np.clip(x0, lo, hi), bounds=(lo, hi),
                        method="trf", xtol=1e-10, ftol=1e-10,
                        max_nfev=args.max_nfev)
    print("   solved: cost %.1f, %d evaluations, %s"
          % (res.cost, res.nfev, res.message.split(".")[0].lower()))

    rows = report(net, res.x, args.line)
    # The step a handover puts on the receiving line is the through flow that
    # arrives, and it belongs at the junction rather than at the station the
    # model had to hang it on. Pull it out so write_geojson can put it there.
    ho_step = {}
    for (L, end), (Mm, stop) in LN.HANDOVER.items():
        k0 = net.idx.get(("T", Mm, stop, 0))
        k1 = net.idx.get(("T", Mm, stop, 1))
        if k0 is None or k1 is None:
            continue
        ho_step[(Mm, stop)] = (L, end,
                               float(res.x[k0]) * SCALE, float(res.x[k1]) * SCALE)

    # The two directions of each handover, because their *ratio* is the thing
    # that says whether the fit determined the handover or picked a point on a
    # ridge. The same trains make both journeys, so anything far from 1.0 is
    # the fit being unconstrained rather than a finding about the railway --
    # 태백선 read 4.7 before W_HOSYM existed. Cheap to print and it is the only
    # place this number is visible.
    if ho_step:
        print("\nhandovers, both directions")
        for (Mm, stop), (L, end, a, b) in sorted(ho_step.items()):
            lo, hi = sorted((abs(a), abs(b)))
            print("   %-11s %-9s -> %-9s %-9s %11.0f %11.0f   ratio %.2f"
                  % (L, end, Mm, stop, a, b, (hi / lo) if lo > 1e-9 else 0.0))

    write_geojson(rows, corridors, table, ho_step, net.untyped,
                  {L: set(v) for L, v in net.notrain.items()})


def split_at_junction(pts, target):
    """Cut a segment's drawn points where they come nearest `target`.

    Returns (before, after) point lists sharing the cut point, or None if the
    cut would land within a kilometre of either end -- there is nothing to show
    in that case and two near-zero-length features are worse than one.
    """
    if not pts or len(pts) < 3 or target is None:
        return None
    best_i, best_d = None, None
    for i, p in enumerate(pts):
        d = B.haversine(p, target)
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    if best_i in (None, 0, len(pts) - 1):
        return None
    head = sum(B.haversine(a, b) for a, b in zip(pts[:best_i], pts[1:best_i + 1]))
    tail = sum(B.haversine(a, b) for a, b in zip(pts[best_i:], pts[best_i + 1:]))
    if head < 1.0 or tail < 1.0:
        return None
    return pts[:best_i + 1], pts[best_i:]


MERGE_TOL = 0.01        # how far two loads may differ and still be one segment


def merge_unserved(feats, table):
    """Join segments across a station where nothing of this line's gets on.

    A drawn segment claims to be a piece of railway carrying a measured number.
    Where the station between two of them has no 승하차 of the line's own train
    types, there is no measurement at that station and no reason for a break --
    both sides carry the same load because they are the same segment. 경부선's
    수원 to 안양 is eight features all reading 29,471, because the stations
    between are Seoul line 1's and no 무궁화 stops at them; hovering any of the
    eight gives a piece of railway that was never measured as a piece.

    This also disposes of the 부산 도시철도 stops on 경부선 through 부산진 and the
    범일/부산진 pair on 경부고속선's approach, which the map has been splitting at
    stations that line's trains run past. They are the same fault: a break
    without a measurement behind it.

    Four things stop a merge, and each is a real break:
      - a station with flow of the line's types, which is most of them
      - a change in `service` or in `level`, so grey never absorbs drawn track
        and a published figure never absorbs a reconstructed one
      - a `junction` feature, which write_geojson split there on purpose
      - the two loads disagreeing by more than MERGE_TOL, which means the fit
        put a step at that station even though no passengers are recorded --
        it is a junction the model believes in, and hiding it would be a lie
        of a different kind

    Purely a drawing change: no load moves, and a merged segment's load is the
    length-weighted mean of parts that agree to within a per cent anyway.
    """
    out, i = [], 0
    while i < len(feats):
        run = [feats[i]]
        while i + 1 < len(feats):
            a, b = run[-1], feats[i + 1]
            pa, pb = a["properties"], b["properties"]
            if pa["line"] != pb["line"] or pa["to"] != pb["from"]:
                break
            if pa.get("service") != pb.get("service"):
                break
            if pa.get("level") != pb.get("level"):
                break
            if pa.get("junction") or pb.get("junction"):
                break
            if not a["geometry"] or not b["geometry"]:
                break
            flows = (table or {}).get(pa["line"], {}).get("flows") or {}
            if sum(abs(v) for v in flows.get(pa["to"], (0, 0, 0, 0))):
                break
            hi = max(abs(pa["daily"]), abs(pb["daily"]), 1)
            if abs(pa["daily"] - pb["daily"]) / hi > MERGE_TOL:
                break
            run.append(b)
            i += 1
        i += 1
        if len(run) == 1:
            out.append(run[0])
            continue
        km = sum(f["properties"]["km"] for f in run)
        w = km or 1.0
        coords = list(run[0]["geometry"]["coordinates"])
        for f in run[1:]:
            c = f["geometry"]["coordinates"]
            coords.extend(c[1:] if c and c[0] == coords[-1] else c)
        props = dict(run[0]["properties"])
        props["to"] = run[-1]["properties"]["to"]
        props["km"] = round(km, 3)
        for k in ("down", "up"):
            props[k] = round(sum(f["properties"][k] * f["properties"]["km"]
                                 for f in run) / w)
        props["daily"] = round((props["down"] + props["up"]) / 365.0)
        props["density"] = props["daily"]
        # So a reader can tell a long segment from a missing one.
        props["through"] = len(run) - 1
        out.append({"type": "Feature", "properties": props,
                    "geometry": {"type": "LineString", "coordinates": coords}})
    return out


def write_geojson(rows, corridors, table=None, ho_step=None, untyped=(),
                  notrain=None):
    """One LineString per segment, sliced out of the line's own corridor.

    `density` is passengers per km per day over the segment, which is what
    japanriders draws thickness from; `daily` is the headcount on the segment
    itself. Both are written so the map can choose.

    A segment with no intercity trains gets `service`, and the map draws those
    apart rather than as a very thin line. The zero-train rule already holds
    them near nil in the fit, but "held near nil" and "drawn as carrying
    fourteen people" are different claims, and the second one is false. 경원선
    is the case: north of 청량리 the map was drawing 14-120 a day along Seoul's
    line 1, where the metro layer separately and correctly draws 190,000. Two
    reasons, and the label has to say which -- `commuter` where the 광역전철
    runs and only the intercity service is absent, `none` where the sheet shows
    nothing running at all, which for 소요산-백마고지 in 2022 it does.

    There is a third reason a rider figure cannot be drawn, and 경춘선 is the
    only line it applies to: the trains run and their passengers are not in the
    source. The 운전 sheet gives it 26 ITX-청춘 a day each way; the 통과인원 sheet
    gives the whole line 2,399 passengers for the year, which is 춘천's own
    승하차 and nothing else -- no other 경춘선 station has an intercity row at
    all. The fit reproduces the sheet faithfully and draws nine people a day
    over a railway running fifty-two trains, which is a false statement about
    ridership in exactly the way the 경원선 hairline was.

    The test is `published 통과인원 / trains < 1 passenger per train`, both
    figures published and neither modelled. 경춘선 comes to 0.13; the next
    lowest line in the network is 정선선 at 21.9, so this is not a threshold
    dividing a continuum. Under a passenger a train the sheet is not reporting
    a quiet railway, it is not reporting the railway.

    경원선 is the fourth case and it is the one where a number *can* be drawn.
    It is `untyped` -- no station row anywhere can be attributed to any train it
    runs -- so the reconstruction has nothing to work from and the fit now
    correctly puts it at nil. But its 통과인원 is not 경춘선's: 7.50M against 44
    intercity trains a day is 395 a train, a perfectly ordinary figure, and the
    운전 volume says where every one of those passengers is. Intercity trains
    run on 용산-청량리 and on no other section of the line, so all 7.50M rode
    that stretch and nothing else. `level` records that the number came from the
    published line total rather than from the cumulation, because it is a
    different kind of claim and the map should say so.

    Split evenly between the directions, which is an assumption and the only one
    available -- 통과인원 is a single figure with no direction in it. It makes
    the line's mirror 0.0 % by construction, so `check.py` reports it as no
    test rather than as a pass.

    A handover's step is also drawn where it happens rather than where the fit
    had to hang it. The SRT reach 경부고속선 at 평택분기점, which has no platform,
    so the model steps the line up at 천안아산 instead -- and the map then drew
    102,429 a day over the whole 74 km from 광명 when the last 24 km of it carry
    the SRT too. The junction's position is known even though its passenger
    rows are not, so the segment is cut there and the step applied to the far
    side. `ho_step` carries the through flow per direction; note it is not the
    difference between the two segments' loads, because the receiving station's
    own boardings and alightings are mixed into that.
    """
    feats, flat = [], 0
    for r in rows:
        line_at = len(feats)        # where this line's features begin
        spec = (table or {}).get(r["line"], {})
        svc = None
        if spec.get("types"):
            # service_along wants the line's own 기점 -> 종점 order, which is
            # the chain reversed where resolve() swapped the ends.
            order = [s[1] for s in r["stops"]]
            if spec.get("reversed"):
                order = order[::-1]
            svc = FQ.service_along(r["line"], spec["types"], order)
            if svc is not None and spec.get("reversed"):
                svc = svc[::-1]
        # Both counts are 편도, so the trains a segment sees in a day are twice
        # the sheet's, and the published year is spread over 365 of them.
        peak = max((s[0] for s in svc), default=0) if svc else 0
        unrecorded = (peak > 0 and spec.get("passing", 0) > 0
                      and spec["passing"] / 365.0 / (2.0 * peak) < 1.0)
        # An untyped line the sheet *does* report: put the published total on
        # the sections that run intercity trains, since those are the only
        # sections its passengers can have been on.
        # Which sections run an intercity train of *any* kind. This is the
        # right question for placing a published total and for quoting trains a
        # day, and it is not the same as `svc`, which counts only the line's own
        # declared types. 경원선 happened not to care -- 26 of 용산-청량리's 44
        # trains are ITX-청춘, which maps onto one of its types -- but 경의선 in
        # 2023 runs 23 KTX and nothing conventional, so counting its own types
        # gives zero everywhere and the line vanishes off the map rather than
        # being drawn at the 2.09M the yearbook credits it with.
        inter = FQ.intercity_along(r["line"], order) if svc is not None else None
        if inter is not None and spec.get("reversed"):
            inter = inter[::-1]
        ipeak = max(inter) if inter else 0
        published = None
        if (r["line"] in untyped and not unrecorded
                and spec.get("passing", 0) > 0 and ipeak > 0):
            published = spec["passing"] / 2.0
        shape = corridors.get(r["line"]) or {}
        poly, cum, raw = shape.get("poly"), shape.get("cum"), shape.get("raw", {})
        for i in range(len(r["down"])):
            a, b = r["stops"][i], r["stops"][i + 1]
            geom, pts = None, None
            ka, kb = raw.get(a[1]), raw.get(b[1])
            if poly and ka is not None and kb is not None:
                pts = B.slice_corridor(poly, cum, ka, kb)
                if len(pts) >= 2:
                    geom = {"type": "LineString",
                            "coordinates": [[round(p[1], 6), round(p[0], 6)]
                                            for p in pts]}
            if geom is None:
                flat += 1
            km = b[0] - a[0]
            dn, up = float(r["down"][i]), float(r["up"][i])
            live = (published is not None and inter is not None
                    and i < len(inter) and inter[i] > 0)
            if published is not None:
                dn = up = published if live else 0.0

            def emit(g, kmv, d, u, extra=None):
                daily = (d + u) / 365.0
                props = {
                    "line": r["line"], "from": a[1], "to": b[1],
                    "km": round(kmv, 3),
                    "down": round(d), "up": round(u),
                    "daily": round(daily), "density": round(daily),
                }
                # `commuter` and `none` are not equally safe. 전동차 > 0 is a
                # positive count, so the row was filled in and the intercity
                # zero beside it is a real zero. `none` is the whole row blank,
                # and a blank is only a zero where the 승하차 sheet agrees --
                # the same test the fit uses, see `Network.notrain`. Without
                # this the 2023 build drew 동해선's tail at a sensible 414-506 a
                # day and labelled it "no trains ran on this section", having
                # just decided that claim was false.
                trusted = notrain is None or i in notrain.get(r["line"], ())
                # A segment carrying a published figure is not a segment
                # without a rider figure, whatever the section counts say about
                # this line's own train types. 경의선 in 2023 is the case: its
                # 서울-능곡 runs 23 KTX and 18 전동차 and no conventional train,
                # so the commuter test fires on a segment the line's whole
                # 통과인원 was just placed on. The published level wins.
                if live:
                    props["level"] = "passing"
                    props["intercity_trains"] = inter[i]
                elif (svc is not None and i < len(svc) and svc[i][0] == 0
                        and (svc[i][1] > 0 or trusted)):
                    props["service"] = "commuter" if svc[i][1] > 0 else "none"
                    props["commuter_trains"] = svc[i][1]
                elif unrecorded and svc is not None and i < len(svc):
                    props["service"] = "unrecorded"
                    props["intercity_trains"] = svc[i][0]
                if extra:
                    props.update(extra)
                feats.append({"type": "Feature", "properties": props,
                              "geometry": g})

            # Where this segment ends at a handover's receiving stop, the step
            # really happens partway along it, at a junction with no platform.
            # Cut there and give the far side the through flow.
            cut = None
            hs = (ho_step or {}).get((r["line"], b[1]))
            if hs and geom and pts:
                give, gend, tdn, tup = hs
                # The giving line's end stands in for the junction unless
                # lines.HANDOVER_POINT names it outright, which it has to do
                # where the end is nowhere near the metals being cut.
                jpt = LN.HANDOVER_POINT.get((give, gend))
                if jpt is None:
                    jshape = corridors.get(give) or {}
                    jraw = (jshape.get("raw") or {}).get(gend)
                    if jshape.get("poly") is not None and jraw is not None:
                        jpt = B.point_at(jshape["poly"], jshape["cum"], jraw)
                if jpt is not None:
                    cut = split_at_junction(pts, jpt)
                    if cut:
                        head, tail = cut
                        hkm = sum(B.haversine(x, y)
                                  for x, y in zip(head, head[1:]))
                        tkm = sum(B.haversine(x, y)
                                  for x, y in zip(tail, tail[1:]))
                        scale = km / (hkm + tkm) if (hkm + tkm) else 1.0
                        ln = lambda ps: {"type": "LineString",
                                         "coordinates": [[round(q[1], 6),
                                                          round(q[0], 6)]
                                                         for q in ps]}
                        emit(ln(head), hkm * scale, dn, up)
                        # `junction` is the giving line's last stop, which is
                        # where the traffic was last seen and not where it
                        # joins -- 평택지제 is a station 3 km short of the
                        # flying junction. Carry the line's name too, since
                        # what joins is a service and not a station, and the
                        # page has no other way to say so.
                        emit(ln(tail), tkm * scale, dn + tdn, up + tup,
                             {"junction": gend, "join_line": give})
            if not cut:
                emit(geom, km, dn, up)

        # The piece that runs on past the last platform to the junction. Its
        # load is the handover's own tau -- everyone still aboard at 평택지제
        # rides it, and they are exactly the people who step onto 경부고속선 at
        # 평택분기점. Without it the line ends in mid-air pointing at nothing,
        # 7.5 km short of the railway it joins, which is also the whole of
        # 수서고속선's 12.9 % gap against its 영업거리.
        #
        # Drawn in corridor order -- junction first, so its far end meets the
        # next feature's start and the gap check still means something -- and
        # inserted where it belongs geographically rather than appended. Its
        # passengers are not double-counted: the same people ride 경부고속선's
        # own segments, but over different track on the far side of the
        # junction.
        for (L, end), (Mm, stop) in LN.HANDOVER.items():
            if L != r["line"] or end not in (shape.get("ran_on") or {}):
                continue
            hs = (ho_step or {}).get((Mm, stop))
            k_stn = raw.get(end)
            if not hs or poly is None or k_stn is None:
                continue
            # Corridor order, so the stub's far end meets its neighbour: at a
            # line's head that means junction -> station, at its tail the other
            # way round. Getting this backwards puts a 6 km hole in 태백선.
            head = end == r["stops"][0][1]
            k_far = shape["ran_on"][end]
            pts = B.slice_corridor(poly, cum, *((k_far, k_stn) if head
                                                else (k_stn, k_far)))
            if len(pts) < 2:
                continue
            skm = sum(B.haversine(x, y) for x, y in zip(pts, pts[1:]))
            _, _, tdn, tup = hs
            ft = {"type": "Feature",
                  "properties": {
                      "line": L, "from": end, "to": Mm, "stub": Mm,
                      "km": round(skm, 3),
                      "down": round(tdn), "up": round(tup),
                      "daily": round((tdn + tup) / 365.0),
                      "density": round((tdn + tup) / 365.0)},
                  "geometry": {"type": "LineString",
                               "coordinates": [[round(q[1], 6), round(q[0], 6)]
                                               for q in pts]}}
            if head:
                feats.insert(line_at, ft)
            else:
                feats.append(ft)

    raw = len(feats)
    feats = merge_unserved(feats, table)
    with io.open(OUT, "w", encoding="utf-8") as f:
        # The edition is carried in the file rather than written into the
        # page's copy, so switching yearbook does not leave the map calling
        # itself 2022. `index.html` substitutes it into every string that
        # quotes a year.
        json.dump({"type": "FeatureCollection", "year": LN.year(),
                   "features": feats}, f, ensure_ascii=False)
    print("\nwrote %s (%d segments from %d, %d without geometry)"
          % (os.path.relpath(OUT, HERE), len(feats), raw, flat))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
