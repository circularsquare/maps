"""Vietnam's rules for build_model.py (build_model.country_rules lists what it reads).

NAMED TRAINS. Every Vietnam Railways train has a number with a letter prefix: SE1-SE22 and
TN/NA on the North-South Railway, SNT1/2 (Sài Gòn - Nha Trang), SQN (Quy Nhơn), SPT1/2 (Phan
Thiết), SP and LC (Lào Cai), HP1/2 and LP2-LP10 (Hải Phòng), QT (Quán Triều), DL (Đà Lạt),
MR1/MR2 (Gia Lâm - Nanning). Each is ONE train (a pair, one each way), mostly once a day, and
OSM maps them one relation per number ("Tàu LP3: Hà Nội => Hải Phòng", ref LP3). So each is a
named train (option B: no percentage of its own, its track counts through the register line it
runs on), as Thailand's SRT trains and Croatia's single B trains are. Even the Hải Phòng and
Đà Lạt trains, several a day: the line a rider uses there is the register line every one of
them runs over, and one OSM line per numbered train would be several copies of it.

OSM also maps each VNR line itself as a route=train relation ("Đường sắt Hà Nội - Lào Cai",
ref ĐSHN-LC, no train number). Those are lines, matched to the register line of that name by
merge_sources: where its stations all lie on the register line it is dropped as the same line
(TWIN_ON_STATIONS: the North-South relation lists only 10 stations over 1,726 km, so the length
and station-share test alone would keep it). The Lào Cai, Đồng Đăng and Quan Triều relations
list their stations as role-less node members, which build_model does not read as stops, so
they make no line at all (2026-10-03); were they to become lines, the Lào Cai and Quan Triều
ones, which start at Hà Nội over the Đồng Đăng line's trunk, would stay as the line as
operated.

Lines: Hà Nội Metro 2A and 3, HCMC Metro 1 (route=subway, never named trains).
"""
import re

# A train number: one to four capital letters (Đ included) and one to three digits, alone in
# the ref or after "Tàu" in the name. Not the line relations' refs (ĐSBN, ĐSHN-LC).
TRAIN_REF = re.compile(r"^[A-ZĐ]{1,4}\d{1,3}(/[A-ZĐ]{0,4}\d{1,3})?$")
TRAIN_NAME = re.compile(r"^Tàu\s+[A-ZĐ]{1,4}\d{1,3}\b")

TWIN_ON_STATIONS = True


def looks_like_service(tags, name, name_en):
    ref = (tags.get("ref") or "").strip()
    return bool(TRAIN_REF.match(ref) or TRAIN_NAME.match(name or ""))
