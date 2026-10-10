"""North Korea's rules for build_model.py (build_model.country_rules lists what it reads).

NAMED TRAINS. Every route=train relation in North Korea is one numbered train or train pair
of the Korean State Railway ("Express trains 1/2 Pyongyang - Hyesan", "418 신의주>염주"), or
an international train (K27/28 Beijing - Pyongyang, Pyongyang - Ussuriysk). Each is a named
train (option B): its track counts through the register lines it runs on (kp_register.py),
which are the Ministry's legal lines.

Lines: the Pyongyang Metro (천리마선, 혁신선), Pyongyang's trams T1-T3 and the Kumsusan tram,
Ch'ŏngjin's tram, the Wŏnsan-Kalma resort tram, Hamhŭng's Sŏho line and the Paektu
funicular.

SKIP_ROUTES: the amusement-park monorails at Mangyŏngdae and Taesŏngsan (fairground rides,
not transport), and an empty "Side Route".
"""


def looks_like_service(tags, name, name_en):
    # a route's own tags say route=train, a route_master's route_master=train
    return "train" in (tags.get("route"), tags.get("route_master"))


SERVICE_IF_ALL_ROUTES_ARE = True


SKIP_ROUTES = {13403857, 13403858, 9314076}
