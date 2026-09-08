"""PART_TYPES plumbing: a type a line runs over only a span of its length.

These read the yearbook but not OSM: about a minute, nearly all of it openpyxl
on the passenger workbook, against the minutes build.py and solve.py need for
the track graph on top. What they cannot check is the fit itself -- that the
contained traffic actually leaves at the boundary -- which only a solve shows.
"""

import unittest

import lines as LN


class PartTypesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table, _ = LN.resolve()

    def test_part_type_reaches_only_its_span(self):
        # 호남선 runs KTX and SRT south of 광주송정 and nowhere else. 목포 is in
        # the span and carries both; 서대전 and 익산 are not, and their
        # high-speed traffic is 호남고속선's.
        fk = self.table["호남선"]["flows_by_kind"]
        for k in ("KTX", "SRT"):
            self.assertIn(k, fk)
            self.assertTrue(any(fk[k].get("목포", ())),
                            "%s missing at 목포" % k)
            for nm in ("서대전", "익산", "정읍", "광주송정"):
                self.assertNotIn(nm, fk[k],
                                 "%s leaked to %s, outside the span" % (k, nm))

    def test_conventional_types_are_untouched_by_the_span(self):
        # The restriction applies to the part types alone: 무궁화 still reaches
        # every station on the line, inside the span and out.
        fk = self.table["호남선"]["flows_by_kind"]
        for nm in ("서대전", "익산", "광주송정", "목포"):
            self.assertTrue(any(fk["무궁화"].get(nm, ())),
                            "무궁화 missing at %s" % nm)

    def test_a_part_type_does_not_make_a_line_fully_typed(self):
        # 호남선's type set is now the complete one, but it does not run every
        # type over its whole length, so 통과인원 stays a ceiling rather than
        # becoming a target. Getting this wrong put 호남선 ahead of 경부선 and
        # collapsed 호남고속선 -- see lines.PART_TYPES.
        self.assertEqual(set(self.table["호남선"]["types"]), set(LN.ALL_TYPES))
        self.assertFalse(self.table["호남선"]["full_types"])
        # A line that really does carry every type keeps the equality.
        self.assertTrue(self.table["전라선"]["full_types"])

    def test_span_stations_are_on_the_line(self):
        # The span is hand-listed because nothing published orders a line's
        # stations, so the one thing worth checking is that it names real ones.
        for canon, (_, span) in LN.PART_TYPES.items():
            roster = self.table[canon]["roster"]
            self.assertTrue(roster, "%s has no roster to check against" % canon)
            self.assertLessEqual(span, roster,
                                 "%s: %s not on the line" % (canon,
                                                             span - roster))

    def test_alias_lands_on_a_station_that_exists(self):
        # A yearbook spelling that matches no chain is traffic that reaches no
        # line, and it fails silently -- a missing key just reads as zero. The
        # alias has to move the row onto a name some line actually stops at.
        fk = self.table["경부고속선"]["flows_by_kind"]
        for k in ("KTX", "SRT"):
            self.assertTrue(any(fk[k].get("김천(구미)", ())),
                            "%s did not reach 김천(구미)" % k)
        # 김천 is a different station 10 km away on 경부선, and keeps its own row.
        flows = LN.station_flows()
        for src, dst in LN.STATION_ALIAS.items():
            self.assertNotIn(src, flows, "%s was not renamed" % src)
            self.assertIn(dst, flows)

    def test_entry_share_names_a_junction_on_the_feeding_line(self):
        # The junction has to be a stop on the line being drawn from, or there
        # is nothing to take a share of.
        for L, (M, J) in LN.ENTRY_SHARE.items():
            self.assertIn(L, self.table)
            self.assertIn(M, self.table)
            self.assertEqual(self.table[L]["first"], J,
                             "%s does not start at %s" % (L, J))

    def test_handover_leaves_from_an_end_and_lands_mid_chain(self):
        # The handing line must really end where the table says, or it has no
        # through flow to give; the receiving stop must be interior, since a
        # step at stop 0 is the entry flow and one at the last stop rides no
        # segment at all.
        for (L, end), (Mm, stop) in LN.HANDOVER.items():
            self.assertIn(L, self.table)
            self.assertIn(Mm, self.table)
            self.assertIn(end, (self.table[L]["first"], self.table[L]["last"]),
                          "%s does not end at %s" % (L, end))
            self.assertNotIn(stop, (self.table[Mm]["first"],
                                    self.table[Mm]["last"]),
                             "%s is an end of %s, not an interior stop"
                             % (stop, Mm))

    def test_through_end_is_the_line_it_names(self):
        # A through end has to be the end the anchor would otherwise read,
        # or suppressing it does nothing.
        for canon, ends in LN.THROUGH_ENDS.items():
            spec = self.table[canon]
            self.assertTrue(spec["through_end"])
            self.assertIn(spec["last"], ends)


if __name__ == "__main__":
    unittest.main()
