"""South Korea's rules for build_model.py (build_model.country_rules lists what it reads)."""

KR_TRAIN_BRANDS = ("KTX", "SRT", "ITX", "새마을", "무궁화", "누리로", "직통열차", "마음")


def looks_like_service(tags, name, name_en):
    # Korea's lines and operating patterns are named 선 (경부선, 경의·중앙선) and so are
    # its trains ("경부선 KTX: 서울 → 부산"), so the suffix says nothing. The train brand
    # does: every named train carries one, and no line or pattern does.
    return any(b in name for b in KR_TRAIN_BRANDS)
