"""China's rules for build_model.py (build_model.country_rules lists what it reads)."""
import re

CN_TRAIN = re.compile(r"^(?:火车|Train\s*)?[GDCZTKYLSP]?\d{1,5}(?:/[A-Z]?\d{1,5})?(?![0-9号線线])")


def looks_like_service(tags, name, name_en):
    # OSM China maps single trains by number: "D2661西安北-西宁", "K27/28", "Z164/5：上海 ->
    # 拉萨", "6072：宝鸡 -> 平凉", or no name and a ref "C8600". Lines and patterns carry no
    # number up front (北京市郊铁路S2线, 金山铁路, 广清城际).
    return bool(CN_TRAIN.match(tags.get("name") or tags.get("ref") or ""))
