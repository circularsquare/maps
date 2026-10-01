"""Loaders for the open Korean register data: line order, section km, station names.

    python kr_sources.py                       # summary to data/raw/kr/kr_sources_summary.txt
    python kr_sources.py --raw data/raw/kr

Pure loaders, no OpenStreetMap. kr_register.py joins this onto OSM track. Where the files
came from, how to download each without a login, and what is wrong with them is in
kr_sources.md beside this file; the short version:

  distance_table()  Korail's 각 선구별 거리표 (data.go.kr 15137040, 2024-09-01) plus SR's own
                    수서 distance matrix (15040194): every Korail and SR legal line, stations
                    in order, km from the chain start.
  urban_stations()  KRIC 1294 전국 도시광역철도 역사정보: every metro and light-rail
                    operator, stations in order with lon/lat, English and the km between them.
  names_en()        RAFIS 역 정보 (15132601) for Korail, KRIC 1294 for everyone else.

LINE NAMES ARE OSM'S. kr_register.py names a register line after the OSM track it runs on
(`name` on the railway ways), so every key here is spelled the way OSM spells the track:
경부고속선, 수도권광역급행철도에이선, 서울 지하철 1호선, 부산 도시철도 1호선, 김포 골드라인,
용인경전철, 안심~하양 복선전철 and so on. The mapping from each source's own names is in
DT_LINES and URBAN_LINES below, in one place, so a wrong one is easy to find.
"""
import argparse
import csv
import io
import math
import os
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "kr"

DT_FILE = "dgk_15137040_각 선구별 거리표.xlsx"
SR_FILE = "dgk_15040194_(주)에스알 경부선 역간거리.csv"
RAFIS_FILE = "dgk_15132601_RAFIS_역기본정보.csv"
KRIC_1294 = "kric_1294_station_national.xlsx"


# --------------------------------------------------------------------------------------------
# the distance table

# Station spellings in the distance table that are not how OSM or the rest of the table
# writes the station. Spaces and newlines are stripped before this is applied ("망 우",
# "제 천", "거제\n해맞이", "신해\n운대" all come out right on their own).
DT_NAMES = {
    "대존조차장": "대전조차장",     # typo on sheet 10
    "대전조": "대전조차장",         # abbreviation on sheet 4
    "체천조": "제천조차장",         # typo on sheet 14
    "DMC": "디지털미디어시티",       # sheet 7's 용산선
    "김천구미": "김천(구미)",        # sheet 1 writes it bare; sheet 3 and OSM write 김천(구미)
    "쌍용": "쌍용(나사렛대)",         # KRIC 1294 and OSM's platform signs; the table writes 쌍용
}

# Nodes that are junctions, yards or signal points rather than passenger stops. The table
# names them like stations. Anything containing one of JUNCTION_WORDS, plus the exact names.
JUNCTION_WORDS = ("연결선", "분기", "신호소", "신호장", "조차장", "삼각", "기지", "북단", "남단",
                  "종점", "철송장")
JUNCTION_NAMES = {
    "대구남", "부산북", "신경주분",     # 경부 KTX sheet: the high-speed line's junction points
    "동송정", "청량B", "청량C", "강릉분", "본선",
    "가천",                           # 대구선 starts at 가천 junction; the station closed
    "지천",                           # 대구북연결선's south end; closed as a stop
    "창내", "신대",                   # 평택선 junctions
    "백산",                           # 태백선 ends at 백산 junction
    "모량",                           # 중앙선's legal end at the 동해선 junction
    # the two-node connecting curves are named after the lines they join
    "호남선", "경전선", "중앙선", "영동선", "태백선",
}

# (sheet, first station, last station) as the table writes them -> OSM track name.
# None: a real chain with no OSM track name in probe_kr_ways.py's list (reported as
# unmapped). The KTX *service* chains (sheet 1, sheet 2's 용산~목포 and 계룡~노령, sheet 3's
# 서울~목포 and 서울~광주) are not in this table and are dropped: they run over several
# legal lines.
DT_LINES = {
    ("2-호남KTX운행거리", "오송", "광주송정"): "호남고속선",
    ("3-연결선", "시흥연결선 종점", "부산"): "경부고속선",
    ("4-경부(1)", "서울", "수원"): "경부선",
    ("4-경부(1)", "서울", "대전"): "경부선",           # continuation: 서울 -41.5- 수원 ...
    ("4-경부(1)", "창내", "평택"): "평택선",
    ("4-경부(1)", "신대", "평택"): "평택직결선",
    ("4-경부(1)", "신대", "지제"): "평택삼각선",
    ("4-경부(1)", "병점", "서동탄"): "병점기지선",
    ("4-경부(1)", "두정", "천안"): "천안직결선",
    ("4-경부(1)", "대전", "서대전"): "대전선",
    ("4-경부(1)", "의왕", "오봉"): "남부화물기지선",
    ("4-경부(1)", "부강", "부강화물"): None,           # 부강화물선
    ("4-경부(1)", "서창", "오송"): "오송선",
    ("5-경부(2)", "서울", "동대구"): "경부선",         # continuation: 서울 -166.3- 대전 ...
    ("5-경부(2)", "가천", "영천"): "대구선",
    ("5-경부(2)", "서울", "부산"): "경부선",           # continuation: 서울 -326.3- 동대구 ...
    ("5-경부(2)", "사상", "범일"): "가야선",
    ("5-경부(2)", "미전", "낙동강"): "미전선",
    ("5-경부(2)", "물금", "양산화물"): "양산화물선",
    ("5-경부(2)", "신동", "신동화물"): "신동화물선",
    ("6-경원, 경춘", "용산", "백마고지"): "경원선",
    ("6-경원, 경춘", "망우", "춘천"): "경춘선",
    ("6-경원, 경춘", "평내호평", "평내기지"): None,     # depot spur
    ("7-경의, 안산", "능곡", "의정부"): "교외선",
    ("7-경의, 안산", "서울", "도라산"): "경의선",
    ("7-경의, 안산", "수색", "가좌"): None,             # 수색객차출발선
    ("7-경의, 안산", "금정", "오이도"): "안산선",
    ("7-경의, 안산", "한국항공대", "기지"): None,       # 고양기지선
    ("7-경의, 안산", "문산", "기지"): None,             # 문산기지선
    ("7-경의, 안산", "경의선분기", "인천공항선분기"): "수색직결선",
    ("7-경의, 안산", "오이도", "시흥기지"): "시흥기지선",
    ("7-경의, 안산", "용산", "DMC"): "용산선",
    ("7-경의, 안산", "서울", "청량리"): "서울 지하철 1호선",   # Seoul Metro's, in Korail's table
    ("8-장항, 경인", "구로", "인천"): "경인선",
    ("8-장항, 경인", "대야", "군산항"): "군산항선",
    ("8-장항, 경인", "목천", "동익산"): "익산삼각선",
    ("8-장항, 경인", "천안", "익산"): "장항선",
    ("8-장항, 경인", "개정", "군산화물"): None,         # 군산화물선
    ("8-장항, 경인", "군산옥산", "옥구"): "옥구선",
    ("9-충북, 경북", "조치원", "봉양"): "충북선",
    ("9-충북, 경북", "김천", "영주"): "경북선",
    ("9-충북, 경북", "점촌", "문경"): None,             # 문경선, no passenger service
    ("10-호남", "대전조차장", "목포"): "호남선",       # continuation: 대전조차장 -87.9- 익산 ...
    ("10-호남", "대존조차장", "익산"): "호남선",
    ("10-호남", "안평", "장성화물"): "장성화물선",
    ("10-호남", "강경", "연무대"): None,               # 강경선
    ("10-호남", "호남선", "경전선"): "북송정삼각선",
    ("10-호남", "일로", "대불"): "대불선",
    ("11-전라", "익산", "여수엑스포"): "전라선",
    ("11-전라", "덕양", "적량"): "여천선",
    ("11-전라", "동산", "북전주"): None,               # 북전주선
    ("12-경전", "삼랑진", "보성"): "경전선",
    ("12-경전", "평화", "성산"): "전경삼각선",
    ("12-경전", "광주선분기", "광주"): "광주선",
    ("12-경전", "삼랑진", "광주송정"): "경전선",       # continuation: 삼랑진 -209.3- 보성 ...
    ("12-경전", "용강", "덕산"): None,                 # 덕산선
    ("12-경전", "황길", "광양항"): None,               # 광양항선
    ("12-경전", "창원", "통해"): "진해선",
    ("12-경전", "광양", "태금"): "광양제철선",
    ("12-경전", "진례", "부산신항"): "부산신항선",
    ("12-경전", "부산신항", "북철송장"): "신항북선",
    ("12-경전", "부산신항", "남철송장"): "신항남선",
    ("12-경전", "초남", "신광양항"): "신광양항선",
    ("13-동해", "부산진", "영덕"): "동해선",
    ("13-동해", "부산진", "신선대"): "우암선",
    ("13-동해", "부조", "괴동"): "괴동선",
    ("13-동해", "가야", "부전"): "부전선",
    ("13-동해", "망양", "울산신항"): "울산신항선",
    ("13-동해", "망양", "울산기지"): None,             # depot spur
    ("13-동해", "남창", "온산"): "온산선",
    ("13-동해", "울산", "장생포"): None,               # 장생포선
    ("13-동해", "포항", "영일만항"): "영일만항선",
    ("14-중앙", "청량리", "모량"): "중앙선",           # continuation: 청량리 -148.5- 도담 ...
    ("14-중앙", "광운대", "망 우"): "망우선",
    ("14-중앙", "청량리", "도담"): "중앙선",
    ("14-중앙", "제 천", "체천조"): None,              # 제천조차장선
    ("14-중앙", "중앙선", "영동선"): None,             # 북영주삼각선
    ("14-중앙", "북영천", "북영천분기"): "영천삼각선",
    ("14-중앙", "용문", "용문기지"): None,             # depot spur
    ("15-태백", "제천", "백산"): "태백선",
    ("15-태백", "예미", "조동"): "함백선",
    ("15-태백", "동해", "묵호"): "묵호항선",
    ("15-태백", "태백선", "영동선"): "태백삼각선",
    ("15-태백", "민둥산", "아우라지"): "정선선",
    ("16-영동", "영주", "청량신호소"): "영동선",
    ("16-영동", "동해", "삼화"): "북평선",
    ("16-영동", "동해", "삼척"): "삼척선",
    ("17-과천,분당,일산", "금정", "남태령"): "과천선",
    ("17-과천,분당,일산", "지축", "대화"): "일산선",
    ("17-과천,분당,일산", "왕십리", "수원"): "분당선",
    ("17-과천,분당,일산", "죽전", "기지"): None,       # 분당기지선
    ("18-경강", "판교", "여주"): "경강선",
    ("18-경강", "강릉", "서원주"): "경강선",           # OSM names both halves 경강선
    ("18-경강", "본선", "강릉차량기지"): None,
    ("18-경강", "남강릉", "안인"): "강릉삼각선",
    ("19-수인", "수원", "인천"): "수인선",
    ("20-서해", "대곡", "원시"): "서해선",
    ("20-서해", "시우", "안산"): "안산연결선",
    ("21-중부내륙", "부발", "충주"): "중부내륙선",
}


def _clean(s):
    s = re.sub(r"\s+", "", str(s))
    return DT_NAMES.get(s, s)


def is_junction(name):
    return name in JUNCTION_NAMES or any(w in name for w in JUNCTION_WORDS)


def _read_xlsx(path):
    """{sheet: 2-D list of cell values}, via openpyxl in read-only mode (the file is small)."""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = {}
    for ws in wb.worksheets:
        out[ws.title] = [list(r) for r in ws.iter_rows(values_only=True)]
    wb.close()
    return out


def _dt_chains(path):
    """Every diagonal chain in the distance table: (sheet, names as written, section km).

    Each line's matrix puts its stations on a down-right diagonal; the km to the next station
    is either the cell to the right of a name (upper triangle) or the cell below the next
    name's left neighbour (lower triangle). See probe_kric.distance_table.
    """
    def isnum(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    def istext(v):
        return isinstance(v, str) and v.strip() not in ("", "-") \
            and not v.strip().replace(".", "").isdigit()

    chains = []
    for sheet, g in _read_xlsx(path).items():
        R = len(g)
        C = max((len(r) for r in g), default=0)
        at = lambda r, c: g[r][c] if 0 <= r < R and 0 <= c < len(g[r]) else None
        for r in range(R):
            for c in range(C):
                if not istext(at(r, c)) or istext(at(r - 1, c - 1)):
                    continue
                cells = [(r, c)]
                while istext(at(cells[-1][0] + 1, cells[-1][1] + 1)):
                    cells.append((cells[-1][0] + 1, cells[-1][1] + 1))
                if len(cells) < 2:
                    continue
                if isnum(at(r, c + 1)):
                    seg = [at(a, b + 1) for a, b in cells[:-1]]
                elif isnum(at(r + 1, c)):
                    seg = [at(a, b - 1) for a, b in cells[1:]]
                else:
                    continue
                names = [str(at(a, b)).strip() for a, b in cells]
                chains.append((sheet, names, [float(s) if isnum(s) else None for s in seg]))
    return chains


def distance_table(raw_dir=RAW):
    """{OSM line name: [chain, ...]}, chain = [(station, km from chain start, is_junction)].

    Continuation chains lose their first "section", which is the cumulative distance from
    the line's origin to where that page of the table picks up (경부선 서울 -166.3- 대전), and
    chains that meet end to start are joined, so 경부선 is one chain 서울-부산. A line with
    branches or two separate halves (경강선: 판교-여주 and 강릉-서원주) keeps several.
    Unmapped chains are under the key None as (sheet, first, last) tuples for the summary.
    """
    raw_dir = Path(raw_dir)
    lines, unmapped = {}, []
    raw_chains = _dt_chains(raw_dir / DT_FILE)
    for sheet, names, seg in raw_chains:
        key = (sheet, names[0], names[-1])
        if key not in DT_LINES:
            continue                                    # a KTX service chain: dropped
        osm = DT_LINES[key]
        if osm is None:
            unmapped.append(key)
            continue
        lines.setdefault(osm, []).append(([_clean(n) for n in names], seg))

    out = {}
    for osm, chains in lines.items():
        # a continuation chain: its first two stations are both on another chain of the line
        def elsewhere(name, own):
            return any(name in ns for ns, _ in chains if ns is not own)
        trimmed = []
        for ns, seg in chains:
            if len(ns) > 2 and elsewhere(ns[0], ns) and elsewhere(ns[1], ns):
                ns, seg = ns[1:], seg[1:]
            trimmed.append((ns, seg))
        # join chains that meet end to start (the table pages a long line)
        merged = True
        while merged:
            merged = False
            for i, (a, sa) in enumerate(trimmed):
                for j, (b, sb) in enumerate(trimmed):
                    if i != j and a[-1] == b[0]:
                        trimmed[i] = (a + b[1:], sa + sb)
                        del trimmed[j]
                        merged = True
                        break
                if merged:
                    break
        res = []
        for ns, seg in trimmed:
            km, chain = 0.0, [(ns[0], 0.0, is_junction(ns[0]))]
            for n, s in zip(ns[1:], seg):
                km = km + (s or 0.0)
                chain.append((n, round(km, 2), is_junction(n)))
            res.append(chain)
        out[osm] = res

    # SR's 수서평택고속선: 수서 - 동탄 - 지제 from SR's own matrix. The line legally ends at
    # 평택분기 (61.1 km), a junction on 경부고속선 between 지제 and 천안아산 that this file
    # does not name: SR's 지제-천안아산 25.1 runs partly on 경부고속선.
    sr = _sr_chain(raw_dir / SR_FILE)
    if sr:
        out["수서평택고속선"] = [sr]
    out[None] = unmapped
    return out


def _sr_chain(path):
    if not path.exists():
        return None
    rows = list(csv.reader(io.open(path, encoding="cp949")))
    head = [h.strip() for h in rows[0][1:]]
    first = rows[1]
    km = {head[i]: float(first[i + 1]) for i in range(len(head))}
    stops = [s for s in ("수서", "동탄", "지제") if s in km]
    return [(s, km[s], False) for s in stops]


# --------------------------------------------------------------------------------------------
# KRIC 1294: metros and light rail

# (operator, line as KRIC writes them) -> OSM track name. Korail's rows are left out: they
# are sorted by code string, interleaved and incomplete (SOURCES.md). 서울교통공사's 9호선
# rows (언주-중앙보훈병원) are joined onto 서울시메트로9호선's (개화-신논현); the SR row for
# 동탄 onto GTX-A; 구리도시공사's and 남양주도시공사's 8호선 rows make up 별내선.
URBAN_LINES = {
    ("공항철도", "공항철도선"): "인천국제공항선",
    ("광주교통공사", "1호선"): "광주 도시철도 1호선",
    ("김포골드라인", "김포골드라인"): "김포 골드라인",
    ("남서울경전철", "신림선"): "신림선",
    ("남양주도시공사", "진접선"): "진접선",
    ("남양주도시공사", "8호선"): "별내선",
    ("구리도시공사", "8호선"): "별내선",
    ("대구교통공사", "1호선"): "대구 도시철도 1호선",
    ("대구교통공사", "2호선"): "대구 도시철도 2호선",
    ("대구교통공사", "3호선"): "대구 도시철도 3호선",
    ("대전교통공사", "1호선"): "대전 도시철도 1호선",
    ("부산교통공사", "1호선"): "부산 도시철도 1호선",
    ("부산교통공사", "2호선"): "부산 도시철도 2호선",
    ("부산교통공사", "3호선"): "부산 도시철도 3호선",
    ("부산교통공사", "4호선"): "부산 도시철도 4호선",
    ("부산김해경전철", "부산김해선"): "부산김해경전철",
    ("서울교통공사", "1호선"): "서울 지하철 1호선",
    ("서울교통공사", "2호선"): "2호선",
    ("서울교통공사", "3호선"): "3호선",
    ("서울교통공사", "4호선"): "4호선",
    ("서울교통공사", "5호선"): "5호선",
    ("서울교통공사", "6호선"): "6호선",
    ("서울교통공사", "7호선"): "7호선",
    ("서울교통공사", "8호선"): "8호선",
    ("서울교통공사", "9호선"): "9호선",
    ("서울시메트로9호선㈜", "9호선"): "9호선",
    ("서해철도㈜", "서해선"): "서해선",
    ("신분당선", "신분당선"): "신분당선",
    ("용인경전철", "에버라인"): "용인경전철",
    ("우이신설도시철도㈜", "우이신설선"): "우이신설선",
    ("의정부경전철", "의정부"): "의정부경전철",
    ("인천교통공사", "인천1호선"): "인천 도시철도 1호선",
    ("인천교통공사", "인천2호선"): "인천 도시철도 2호선",
    ("인천교통공사", "7호선"): "7호선",
    ("지티엑스에이운영", "GTX-A"): "수도권광역급행철도에이선",
    ("㈜SR", "GTX-A"): "수도권광역급행철도에이선",
    ("인천공항", "자기부상"): "인천공항 자기부상철도",
}

# How each block writes 상행거리 / 하행거리 (SOURCES.md). The km from row i to row i+1 is:
#   PN  하행[i]  (= 상행[i+1]): 상행 is to the previous row, 하행 to the next
#   NP  상행[i]  (= 하행[i+1]): the reverse
#   PP  상행[i+1]           : both columns are the km from the previous row
#   NN  상행[i]             : both columns are the km to the next row
# Detected per block from which pair agrees; these override where detection cannot tell
# (서울교통공사 writes 상행 = 하행, so only the ends say which way it runs).
CONVENTION = {
    ("서울교통공사", "1호선"): "NN",
    ("서울교통공사", "2호선"): "PP",       # the 성수지선 part is NN, handled in _seoul_2
    ("서울교통공사", "3호선"): "PP",
    ("서울교통공사", "4호선"): "PP",
    ("서울교통공사", "5호선"): "PP",
    ("서울교통공사", "6호선"): "PP",
    ("서울교통공사", "7호선"): "PP",
    ("서울교통공사", "8호선"): "PP",
    ("서울교통공사", "9호선"): "NP",
}

KOREA = (124.5, 132.0, 33.0, 38.7)


def _num(v):
    if v is None:
        return None
    s = str(v).strip().lower().replace("km", "").replace(",", ".")
    try:
        x = float(s)
    except ValueError:
        m = re.match(r"^\s*([0-9.]+)", s)
        x = float(m.group(1)) if m else None
    return None if x is None or math.isnan(x) else round(x, 3)


def _load_1294(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = ws.iter_rows(values_only=True)
    head = [str(h).strip() if h is not None else "" for h in next(rows)]
    out = [dict(zip(head, r)) for r in rows if any(v is not None for v in r)]
    wb.close()
    return out


def _clean_name(s):
    s = re.sub(r"\s+", "", str(s or ""))
    # 대전 writes a trailing 역 on every station, 김포 and 진접선 too; OSM does not
    if s.endswith("역") and len(s) > 2 and s not in ("서울역",):
        s = s[:-1]
    return s


def _base(s):
    """The name without its bracketed sub-name: 양재(서초구청) -> 양재."""
    return re.sub(r"\(.*?\)", "", s).strip()


def _row(r):
    lon, lat = _num(r.get("역 위치(경도)")), _num(r.get("역 위치(위도)"))
    if lon is not None and lat is not None and lon < 90 and lat > 90:
        lon, lat = lat, lon                          # swapped (GTX-A, 자기부상)
    if lon is None or lat is None or not (KOREA[0] <= lon <= KOREA[1] and KOREA[2] <= lat <= KOREA[3]):
        lon = lat = None
    name = _clean_name(r.get("역명(한글)"))
    en = re.sub(r"\s+", " ", str(r.get("역명(영어)") or "")).strip() or None
    return {"name": name, "name_en": en, "lon": lon, "lat": lat,
            "up": _num(r.get("상행거리")), "dn": _num(r.get("하행거리")),
            "code": str(r.get("역 번호") or "").strip()}


def _detect(block):
    if len(block) < 3:
        return "PN"
    same = sum(1 for s in block if s["up"] == s["dn"]) / len(block)
    if same > 0.8:
        return "PP"
    pn = sum(1 for a, b in zip(block, block[1:]) if a["dn"] is not None and a["dn"] == b["up"])
    np_ = sum(1 for a, b in zip(block, block[1:]) if a["up"] is not None and a["up"] == b["dn"])
    return "PN" if pn >= np_ else "NP"


def _km_next(block, conv):
    """km from each row to the next row of the block; the last row's is to whatever follows."""
    out = []
    for i, s in enumerate(block):
        nxt = block[i + 1] if i + 1 < len(block) else None
        if conv == "PN":
            v = s["dn"] if s["dn"] else (nxt["up"] if nxt else None)
        elif conv == "NP":
            v = s["up"] if s["up"] else (nxt["dn"] if nxt else None)
        elif conv == "PP":
            v = nxt["up"] if nxt else None
        else:                                           # NN
            v = s["up"]
        out.append(v if v else None)
    return out


def _as_list(block, km_next, first_km=None):
    res = []
    for i, s in enumerate(block):
        km_prev = first_km if i == 0 else km_next[i - 1]
        res.append(_st(s, km_prev))
    return res


def _st(s, km_prev):
    """The output record. `name` loses the bracketed sub-name KRIC appends (양재(서초구청),
    다대포해수욕장(몰운대), 부산(짐캐리)), which OSM's `name` does not carry; `name_full` keeps
    it. name_en likewise loses its bracket unless that would leave nothing."""
    en = s["name_en"]
    if en:
        en = re.sub(r"\s*\(.*?\)\s*", " ", en).strip() or en
    return {"name": _base(s["name"]) or s["name"], "name_full": s["name"], "name_en": en,
            "lon": s["lon"], "lat": s["lat"], "km_prev": km_prev}


def _hav(a, b):
    if a["lon"] is None or b["lon"] is None:
        return None
    la1, la2 = math.radians(a["lat"]), math.radians(b["lat"])
    dla, dlo = la2 - la1, math.radians(b["lon"] - a["lon"])
    h = math.sin(dla / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def _drop_bad_points(lst):
    """None out the points that do not fit the line, judged against the published km.

    Two stations are consistent when the straight line between them is no longer than the
    track km between them (x1.3 + 1.5 km of slack) and, if they are more than 0.5 km apart by
    track, not at the same point. Neighbour-by-neighbour checks are not enough: Busan 1's
    구서-노포 are wrong together, 20 km off, and agree with each other. So the anchor is the
    station consistent with the most others, and the walk outward from it keeps a station
    only if it is consistent with the last station kept. Unknown section km count as the
    line's mean section.
    """
    n = len(lst)
    known = [s["km_prev"] for s in lst[1:] if s["km_prev"]]
    mean = sum(known) / len(known) if known else 2.0
    pos, p = [], 0.0
    for i, s in enumerate(lst):
        if i:
            p += s["km_prev"] or mean
        pos.append(p)
    has = [s["lon"] is not None for s in lst]

    loop = n > 2 and lst[0]["name"] == lst[-1]["name"]

    def ok(i, j):
        d = _hav(lst[i], lst[j])
        track = abs(pos[i] - pos[j])
        if loop:
            track = min(track, pos[-1] - track)
        if lst[i]["name"] == lst[j]["name"]:
            return d < 1.0                              # a loop's closing station
        # no longer than the track, and not implausibly shorter: 진접선's 별내별가람 is
        # filed at 오남's point, 0.2 km from it across 7.7 km of track
        return track * 0.4 - 1.0 <= d <= track * 1.3 + 1.5 and not (d < 0.05 and track > 0.5)

    idx = [i for i in range(n) if has[i]]
    if len(idx) < 2:
        return []
    score = {i: sum(ok(i, j) for j in idx if j != i) for i in idx}
    anchor = max(idx, key=lambda i: (score[i], -i))
    bad = []
    for step in (1, -1):
        last = anchor
        i = anchor + step
        while 0 <= i < n:
            if has[i]:
                if ok(last, i):
                    last = i
                else:
                    bad.append(i)
            i += step
    for i in bad:
        lst[i]["lon"] = lst[i]["lat"] = None
    return [lst[i]["name"] for i in sorted(bad)]


def urban_stations(raw_dir=RAW):
    """{OSM line name: [[station, ...], ...]} for every non-Korail operator in KRIC 1294.

    station = {name, name_en, lon, lat, km_prev}. km_prev is the km from the previous
    station on the same list, None for the first station and wherever the source gives no
    figure. A branch is its own list and starts with the station it leaves from (with
    km_prev None), so its first km_prev is the branch's first section. lon/lat are None where
    the point was judged wrong; swapped lon/lat are swapped back rather than dropped.
    """
    rows = _load_1294(Path(raw_dir) / KRIC_1294)
    blocks = {}
    for r in rows:
        key = (str(r.get("철도운영기관명") or "").strip(), str(r.get("운영노선") or "").strip())
        if key not in URBAN_LINES:
            continue
        blocks.setdefault(key, []).append(_row(r))

    def conv(key, block):
        return CONVENTION.get(key) or _detect(block)

    def simple(key):
        b = blocks.get(key, [])
        return _as_list(b, _km_next(b, conv(key, b))) if b else []

    out = {}

    for key, osm in URBAN_LINES.items():
        if osm in ("2호선", "5호선", "8호선", "9호선", "별내선", "대구 도시철도 1호선",
                   "서울 지하철 1호선", "수도권광역급행철도에이선", "7호선", "진접선"):
            continue
        if key in blocks:
            out[osm] = [simple(key)]

    # 서울 1호선: the rows run 서울역..청량리 with 동묘앞 (opened 2005) filed last. NN.
    b = blocks[("서울교통공사", "1호선")]
    b = [s for s in b if s["name"] != "동묘앞"]
    i = [s["name"] for s in b].index("동대문")
    b.insert(i + 1, next(s for s in blocks[("서울교통공사", "1호선")] if s["name"] == "동묘앞"))
    out["서울 지하철 1호선"] = [_as_list(b, _km_next(b, "NN"))]

    # 2호선: the loop 시청..충정로 closed back to 시청 (PP), 성수지선 성수-용답-신답-용두-신설동
    # (NN, and the 성수-용답 leg has no figure), 신정지선 신도림-도림천-양천구청-신정네거리-까치산
    # (PP; 까치산 is a stray row at the end of the file under code 234-4).
    b = blocks[("서울교통공사", "2호선")]
    by = {s["name"]: s for s in b}
    loop = [s for s in b if s["code"].isdigit() and 201 <= int(s["code"]) <= 243]
    km = _km_next(loop, "PP")
    km[-1] = loop[0]["up"]                              # 충정로 -> 시청 closes the loop
    main = _as_list(loop + [loop[0]], km + [None])
    names = ["용답", "신답", "용두(동대문구청)", "신설동"]
    sub = [by[n] for n in names if n in by]
    seong = [_st(by["성수"], None)]
    seong += [_st(s, None if j == 0 else sub[j - 1]["up"]) for j, s in enumerate(sub)]
    out["2호선"] = [main]
    out["성수지선"] = [seong]
    names = ["신도림", "도림천", "양천구청", "신정네거리", "까치산"]
    sub = [by[n] for n in names if n in by]
    out["신정지선"] = [[_st(s, None if j == 0 else s["up"]) for j, s in enumerate(sub)]]

    # 5호선: 방화..상일동 main (2511-2554), 마천지선 강동-둔촌동..마천 (2555-2561), 하남선
    # 상일동-강일..하남검단산 (2562-2566). PP throughout.
    b = blocks[("서울교통공사", "5호선")]
    code = lambda s: int(s["code"]) if s["code"].isdigit() else 0
    main = [s for s in b if 2511 <= code(s) <= 2554]
    mac = [s for s in b if 2555 <= code(s) <= 2561]
    han = [s for s in b if 2562 <= code(s) <= 2566]
    by = {s["name"]: s for s in main}
    out["5호선"] = [_as_list(main, _km_next(main, "PP"))]
    out["마천지선"] = [_as_list([by["강동"]] + mac, [mac[0]["up"]] + _km_next(mac, "PP"))]
    out["하남선"] = [_as_list([by["상일동"]] + han, [han[0]["up"]] + _km_next(han, "PP"))]

    # 7호선: 서울교통공사 장암..온수 then 인천교통공사 까치울..석남 (PN; 까치울's 상행 is
    # 온수-까치울).
    s7 = blocks[("서울교통공사", "7호선")]
    i7 = blocks[("인천교통공사", "7호선")]
    km = _km_next(s7, "PP")
    km[-1] = i7[0]["up"]
    out["7호선"] = [_as_list(s7 + i7, km + _km_next(i7, _detect(i7)))]

    # 8호선: 암사..모란 with 남위례 (2021) filed last; it goes between 복정 and 산성. The
    # 암사역사공원 row (2810) is 별내선's.
    b = [s for s in blocks[("서울교통공사", "8호선")] if s["name"] != "암사역사공원"]
    nam = next(s for s in b if s["name"] == "남위례")
    b = [s for s in b if s is not nam]
    b.insert([s["name"] for s in b].index("복정") + 1, nam)
    out["8호선"] = [_as_list(b, _km_next(b, "PP"))]

    # 별내선 암사-암사역사공원-장자호수공원-구리-동구릉-다산-별내, from three operators'
    # rows: 서울교통공사 (암사역사공원; 암사's 상행 1.1 is 암사역사공원-암사), 구리도시공사
    # (동구릉, 구리, 장자호수공원; PN north to south), 남양주도시공사 (별내, 다산; PN).
    # Written north to south, the way the two city corporations list it. 장자호수공원-
    # 암사역사공원 is 3.5 on 서울교통공사's row and 3.8 on 구리's; 3.5 is used. The sum
    # (about 14.5) is over the 12.9 usually quoted for 별내선; nothing on disk says which leg.
    s8 = {s["name"]: s for s in blocks[("서울교통공사", "8호선")]}
    gu = blocks[("구리도시공사", "8호선")]
    nm = blocks[("남양주도시공사", "8호선")]
    seq = nm + gu + [s8["암사역사공원"], s8["암사"]]
    kms = [nm[0]["dn"], nm[1]["dn"], gu[0]["dn"], gu[1]["dn"], s8["암사역사공원"]["up"],
           s8["암사"]["up"]]
    out["별내선"] = [_as_list(seq, kms + [None])]

    # 9호선: 서울시메트로9호선 개화..신논현 (PN) then 서울교통공사 언주..중앙보훈병원 (NP).
    a = blocks[("서울시메트로9호선㈜", "9호선")]
    c = blocks[("서울교통공사", "9호선")]
    km = _km_next(a, _detect(a))
    km[-1] = a[-1]["dn"] or c[0]["dn"]
    out["9호선"] = [_as_list(a + c, km + _km_next(c, "NP"))]

    # 진접선 진접-오남-별내별가람, then on to 불암산 (4호선's terminus): PN, 별내별가람's
    # 하행 4.4 is the last leg.
    b = blocks[("남양주도시공사", "진접선")]
    bul = next(s for s in blocks[("서울교통공사", "4호선")] if s["name"] == "불암산")
    out["진접선"] = [_as_list(b + [bul], _km_next(b, "PN") + [None])]

    # 대구 1호선: 중앙로 is a stray one-row block (code 3140) filed after 3호선; it goes
    # between 반월당 and 대구역. The 2024 extension 안심-대구한의대병원-부호-하양 is its own
    # OSM track, 안심~하양 복선전철.
    b = [dict(s) for s in blocks[("대구교통공사", "1호선")]]
    jung = next((s for s in b if s["code"] == "3140"), None)
    b = [s for s in b if s is not jung]
    if jung:
        b.insert([s["name"] for s in b].index("반월당") + 1, jung)
    kms = _km_next(b, "PN")
    lst = _as_list(b, kms)
    cut = [s["name"] for s in lst].index("안심") if "안심" in [s["name"] for s in lst] else None
    if cut is not None:
        out["대구 도시철도 1호선"] = [lst[:cut + 1]]
        ext = [dict(s) for s in lst[cut:]]
        ext[0]["km_prev"] = None
        out["안심~하양 복선전철"] = [ext]
    else:
        out["대구 도시철도 1호선"] = [lst]

    # GTX-A: 운정중앙-킨텍스-대곡-연신내-서울역 (NP), and 수서-성남-구성-동탄 with 동탄 under
    # SR. 성남-동탄 is given as 22.1 with 구성 (opened 2024-06) between them unmeasured, so
    # 구성 and 동탄 carry no km_prev. 서울역-수서 (opened 2026, 삼성 unserved) is not in the
    # file at all.
    g = blocks[("지티엑스에이운영", "GTX-A")]
    north = [s for s in g if s["code"] <= "X106"]
    south = [s for s in g if s["code"] >= "X108"] + blocks.get(("㈜SR", "GTX-A"), [])
    out["수도권광역급행철도에이선"] = [
        _as_list(north, _km_next(north, "NP")),
        _as_list(south, [south[0]["dn"], None, None, None][:len(south)]),
    ]

    report = {}
    for osm, lists in out.items():
        for lst in lists:
            dropped = _drop_bad_points(lst)
            if dropped:
                report.setdefault(osm, []).extend(dropped)
    out[None] = report
    return out


# --------------------------------------------------------------------------------------------
# English names

def names_en(raw_dir=RAW):
    """{korean station name: english name}. RAFIS for Korail (every intercity station),
    KRIC 1294 for the urban operators; RAFIS wins where both have one. Keys are given with
    and without a bracketed sub-name (양재(서초구청) and 양재) so either spelling finds it."""
    raw_dir = Path(raw_dir)
    out = {}
    for r in _load_1294(raw_dir / KRIC_1294):
        s = _row(r)
        if s["name"] and s["name_en"]:
            out.setdefault(s["name"], s["name_en"])
            out.setdefault(_base(s["name"]), re.sub(r"\s*\(.*?\)\s*", "", s["name_en"]).strip()
                           or s["name_en"])
    path = raw_dir / RAFIS_FILE
    if path.exists():
        for r in csv.DictReader(io.open(path, encoding="cp949")):
            ko = re.sub(r"\s+", "", r.get("역명") or "")
            en = (r.get("영문역명") or "").strip()
            if ko and en and not is_junction(ko):
                out[ko] = en
                out.setdefault(_clean(ko), en)          # 김천구미 -> 김천(구미) too
    return out


# --------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    raw = Path(args.raw)
    if not raw.is_absolute():
        raw = ROOT / raw
    out = Path(args.out) if args.out else raw / "kr_sources_summary.txt"
    w = []

    dt = distance_table(raw)
    unmapped = dt.pop(None)
    w.append("=== distance_table: %d lines" % len(dt))
    for line in sorted(dt):
        for ch in dt[line]:
            stops = [n for n, _, j in ch if not j]
            w.append("  %-14s %3d nodes %3d stops  %7.1f km  %s ~ %s" % (
                line, len(ch), len(stops), ch[-1][1], ch[0][0], ch[-1][0]))
    w.append("\n  chains with no OSM track name:")
    for sheet, a, b in unmapped:
        w.append("    %s  %s ~ %s" % (sheet, a, b))

    us = urban_stations(raw)
    bad = us.pop(None)
    w.append("\n=== urban_stations: %d lines" % len(us))
    for line in sorted(us):
        for lst in us[line]:
            km = sum(s["km_prev"] or 0 for s in lst)
            miss = sum(1 for s in lst[1:] if s["km_prev"] is None)
            nxy = sum(1 for s in lst if s["lon"] is None)
            w.append("  %-22s %3d stations %6.1f km  %s ~ %s%s%s" % (
                line, len(lst), km, lst[0]["name"], lst[-1]["name"],
                "  (%d sections without km)" % miss if miss else "",
                "  (%d without lon/lat)" % nxy if nxy else ""))
    w.append("\n  coordinates judged wrong and set to None:")
    for line, names in sorted(bad.items()):
        w.append("    %s: %s" % (line, ", ".join(names)))

    en = names_en(raw)
    w.append("\n=== names_en: %d names" % len(en))
    for k in ("서울", "김천(구미)", "여수엑스포", "양재", "디지털미디어시티", "대전조차장"):
        w.append("  %s -> %s" % (k, en.get(k)))

    out.write_text("\n".join(w), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
