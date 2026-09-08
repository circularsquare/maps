import unittest

from build_daegu_busan import compact, daegu_network, expected_weekdays, gimhae_network


class DaeguBusanTest(unittest.TestCase):
    def test_station_spacing_normalization(self):
        self.assertEqual(compact("괘법 르네시떼"), compact("괘법르네시떼"))

    def test_reference_month_weekdays(self):
        self.assertEqual(len(expected_weekdays("2026-06")), 22)
        self.assertEqual(len(expected_weekdays("2025-12")), 23)

    def test_network_rosters_and_daegu_transfers(self):
        nodes, edges, _, groups = daegu_network()
        self.assertEqual((len(nodes), len(groups), len(edges)), (94, 91, 91))
        self.assertEqual({name for name, codes in groups.items() if len(codes) > 1},
                         {"명덕", "반월당", "청라언덕"})
        nodes, edges, _, groups = gimhae_network()
        self.assertEqual((len(nodes), len(groups), len(edges)), (21, 21, 20))


if __name__ == "__main__":
    unittest.main()
