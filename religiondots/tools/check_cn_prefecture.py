"""Is any Chinese census county drawn in the wrong PREFECTURE? — spec §14.19

THE ONE CLASS OF ERROR NOTHING ELSE HERE CAN SEE. sources/cn.py joins the 2000 census
volumes to DataV adcodes by romanised NAME, and China has a great many counties that
romanise identically — three `WEIXIAN` in Hebei alone (魏县 in Handan, 威县 in Xingtai,
蔚县 in Zhangjiakou). When the resolver picks the wrong one it still returns a REAL adcode
in a REAL province, so every downstream check passes: the county total is right, the
province reconciles, the national figure is exact, `check_mapping` is happy. The only thing
wrong is that half a million people are drawn 400 km from home.

**On 2026-09-08 this found ten of them, about 3.4 million people**, including 郧县's 584,315
residents drawn in 云梦县 and two OVERRIDES entries whose comment named the right county
while the digits named another — which no amount of reading the file would have caught.

HOW IT WORKS, AND IT IS ONE SENTENCE. Both the census file and DataV's index run in GB/T
2260 code order within a province, so a county's NEIGHBOURS IN THE FILE are its neighbours
in code space. A resolution that lands in a prefecture neither neighbour is in is almost
always the wrong same-named county.

WHY THERE IS AN ALLOWLIST RATHER THAN A ZERO TARGET. Twenty-odd counties legitimately trip
the rule, because between 2000 and now they really did change prefecture — 简阳 went to
Chengdu, 无为 to Wuhu, 寿县 to Huainan, 公主岭 to Changchun, 枞阳 to Tongling, 海原 to
Zhongwei — or became provincially administered (济源, 儋州, 石河子, 嘉峪关), or are
prefecture-level cities with no counties at all (东莞, 中山). Those are listed below with
the reason, so that anything NOT on the list is a finding rather than noise.

    python tools/check_cn_prefecture.py          # exit 1 if an unexplained one appears
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent.parent
sys.path[:0] = [str(HERE / "sources"), str(HERE), str(HERE / "taxonomy")]

# census name -> why its adcode is legitimately outside its file neighbours' prefecture.
EXPECTED = {
    "DONGGUAN": "prefecture-level city with no counties; its own 4419xx block",
    "ZHONGZHAN": "中山市, prefecture-level city with no counties (4420xx)",
    "JIANYANG": "简阳市 moved from 资阳 to 成都, 2016",
    "KAI": "开州区; Chongqing has no prefectures, 5001xx districts vs 5002xx counties",
    "ZHONGXIAN": "忠县; Chongqing, as above",
    "WULONG": "武隆区; Chongqing, as above",
    "WUWEI": "无为市 moved from 巢湖 to 芜湖, 2011",
    "SHOUXIAN": "寿县 moved from 六安 to 淮南, 2015",
    "ZONGYANG": "枞阳县 moved from 安庆 to 铜陵, 2015",
    "HUAINING": "怀宁县; Anqing, flagged only by its neighbour's position",
    "GONGZHULING": "公主岭市 moved from 四平 to 长春, 2020",
    "SHUANGLIAO": "双辽市; flagged only because 公主岭 legitimately moved",
    "DANZHOU": "儋州市 became provincially administered, 469003 -> 460400, 2015",
    "QIONGSHAN": "琼山市 -> 海口市琼山区, 2002",
    "SANYA": "三亚市 -> its districts, 2014",
    "JIYUAN": "济源市, provincially administered (419001)",
    "SHIHEZI": "石河子市, XPCC (659001)",
    "JIAYUGUAN": "嘉峪关市, prefecture-level city with no counties",
    "DATONG CHENGQU": "大同市城区 -> 平城区, 2018",
    "GUYUAN": "固原县 -> 原州区 on the prefecture's own upgrade",
    "HAIYUAN": "海原县 moved from 固原 to 中卫, 2004",
    "LINGWU": "灵武市, 银川-administered (640181)",
    "MIQUAN": "米泉市 -> 乌鲁木齐米东区, 2007",
    "LUZHAI": "鹿寨县; 来宾 was carved out of 柳州地区 in 2002, so 2000 order jumps",
    "HESHAN": "合山市; 来宾, as above",
    "PINGXIANG": "凭祥市; 崇左 was carved out of 南宁地区 in 2002",
    "XINGNING": "南宁兴宁区; first row of the province, no left neighbour",
    "SHIZHONG": "枣庄市中区; Shandong has three 市中区 and the file order jumps",
    "JINING SHIZHONG": "济宁市中区 -> 任城区, 2013",
}


def main():
    import cn as cnsrc
    idx_path = HERE / "data" / "raw" / "cn" / "datav" / "county_index.json"
    with open(idx_path, encoding="utf-8") as fh:
        NAME = {r["code"]: (r["name"], r["city"]) for r in json.load(fh)}

    by_prov = cnsrc.read_2000()
    resolve = cnsrc.build_resolver(str(idx_path))[0]
    GB = cnsrc.GB_PROVINCE

    flagged, unexplained = [], []
    for prov, rows in sorted(by_prov.items()):
        names = [n for n, _ in rows]
        codes = [c for c, _ in resolve(prov, names)]
        for i, code in enumerate(codes):
            if not code:
                continue
            prev = next((codes[j] for j in range(i - 1, -1, -1) if codes[j]), None)
            nxt = next((codes[j] for j in range(i + 1, len(codes)) if codes[j]), None)
            near = {c[:4] for c in (prev, nxt) if c}
            if near and code[:4] not in near:
                item = (rows[i][1][0], GB[f"{prov//10000:02d}"][1], names[i], code,
                        prev, nxt)
                flagged.append(item)
                if names[i] not in EXPECTED:
                    unexplained.append(item)

    print(f"counties whose adcode is in a prefecture neither file neighbour is in: "
          f"{len(flagged)}")
    print(f"  explained by a real administrative move: {len(flagged) - len(unexplained)}")
    print(f"  UNEXPLAINED: {len(unexplained)}")
    if not unexplained:
        print("\nok — every cross-prefecture placement is a documented move")
        return 0
    print(f"\n{sum(u[0] for u in unexplained):,} people are drawn outside their prefecture:")
    for pop, pn, nm, code, prev, nxt in sorted(unexplained, reverse=True):
        print(f"  {pop:>10,}  {pn:14s} {nm:22s} -> {code} "
              f"{NAME.get(code, ('?',))[0]:10s}  neighbours {prev}/{nxt}")
    print("\nEach is either a wrong OVERRIDES target or a same-name collision that needs a"
          "\nlist. Place it from its file NEIGHBOURS, never from the name.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
