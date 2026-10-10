# Russia. 2021 census native language by federal subject, urban and rural (sources/ru_census.py),
# on religiondots' 3km Kontur hexes split into urban and rural by density (sources/ru_geo.py), and
# for Crimea and Sevastopol on Kontur UA's 400 m hexes cut out of Ukraine's layer (Anita's ruling,
# 2026-10-05). Inside each unit, languages are placed by the census's nationality per settlement
# (tochno.st; _RuWeighter, 2026-10-07). sources/ru.md is the record.
from _shared import *  # noqa: F401,F403
import numpy as np


def _counts():
    import ru2021
    df = pd.read_csv(NORM / "ru.csv")
    # the 85 subjects the census counted, Crimea (UA-43) and Sevastopol (UA-40) among them;
    # the `country` rows are the same people again
    df = df[(df["geo_level"] == "subject") & (df["area"].isin(["urban", "rural"]))].copy()
    df["node"] = df["source_category"].map(ru2021.resolve)
    df = df[df["node"].notna()]            # "native language not stated" is the gap
    df["unit"] = df["geo_id"] + "/" + df["area"].str[0]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


# ---- placement inside a unit (fix-ru, 2026-10-07; AGENT_BRIEF §4.4: moves people only inside
# the subject x urban/rural unit the census counted them in). Each language's speakers go where
# the people of its nationality live, by the 2021 census's nationality per settlement
# (tochno.st, CC BY; sources/ru_settlements.py). Russian takes the places of everyone whose
# nationality no other language here claims; languages with no nationality follow population.
#
# Census language label -> the tochno nationality columns (short names) whose people it follows.
# A nationality may serve several languages (Mordva: Mordvin, Erzya, Moksha).
NAT = {
    "Татарский": ["Татары", "Астраханские татары", "Кряшены", "Мишари", "Сибирские татары",
                  "Нагайбаки"],
    "Чеченский": ["Чеченцы", "Чеченцы-аккинцы"], "Башкирский": ["Башкиры"],
    "Аварский": ["Аварцы"], "Чувашский": ["Чуваши"],
    "Армянский": ["Армяне", "Черкесогаи", "Хемшилы"],
    "Кабардино-черкесский": ["Кабардинцы", "Черкесы"],
    "Даргинский": ["Даргинцы", "Кайтагцы", "Кубачинцы"], "Кумыкский": ["Кумыки"],
    "Ингушский": ["Ингуши"], "Якутский": ["Якуты"],
    "Осетинский": ["Осетины", "Осетины-дигорцы", "Осетины-иронцы"], "Лезгинский": ["Лезгины"],
    "Казахский": ["Казахи"], "Азербайджанский": ["Азербайджанцы"], "Бурятский": ["Буряты"],
    "Карачаево-балкарский": ["Карачаевцы", "Балкарцы"],
    "Марийский": ["Марийцы", "Горные марийцы", "Лугово-восточные марийцы"],
    "Таджикский": ["Таджики"], "Украинский": ["Украинцы"],
    "Тувинский": ["Тувинцы", "Тувинцы-тоджинцы"],
    "Мордовский": ["Мордва", "Мордва-мокша", "Мордва-эрзя"],
    "Удмуртский": ["Удмурты", "Бесермяне"], "Узбекский": ["Узбеки"],
    "Крымскотатарский": ["Крымские татары"], "Калмыцкий": ["Калмыки"], "Лакский": ["Лакцы"],
    "Табасаранский": ["Табасараны"], "Цыганский": ["Цыгане"],
    "Турецкий": ["Турки", "Турки-месхетинцы"], "Киргизский": ["Киргизы"],
    "Адыгейский": ["Адыгейцы", "Шапсуги"], "Ногайский": ["Ногайцы", "Карагаши"],
    "Коми": ["Коми", "Коми-ижемцы"], "Грузинский": ["Грузины", "Аджарцы", "Ингилойцы"],
    "Алтайский": ["Алтайцы", "Теленгиты"], "Хакасский": ["Хакасы"],
    "Эрзя-мордовский": ["Мордва-эрзя", "Мордва"], "Молдавский": ["Молдаване"],
    "Белорусский": ["Белорусы"], "Курдский": ["Курды", "Курманч", "Езиды"],
    "Коми-пермяцкий": ["Коми-пермяки"], "Абазинский": ["Абазины"], "Ненецкий": ["Ненцы"],
    "Туркменский": ["Туркмены"], "Агульский": ["Агулы"], "Рутульский": ["Рутульцы"],
    "Немецкий": ["Немцы", "Меннониты"], "Мокша-мордовский": ["Мордва-мокша", "Мордва"],
    "Андийский": ["Андийцы"], "Горномарийский": ["Горные марийцы", "Марийцы"],
    "Корейский": ["Корейцы"], "Цезский": ["Дидойцы"], "Китайский": ["Китайцы"],
    "Греческий": ["Греки", "Греки-урумы"], "Арабский": ["Арабы"], "Хантыйский": ["Ханты"],
    "Цахурский": ["Цахуры"], "Каратинский": ["Каратинцы"], "Карельский": ["Карелы"],
    "Эвенкийский": ["Эвенки"], "Чукотский": ["Чукчи"], "Бежтинский": ["Бежтинцы"],
    "Английский": ["Британцы", "Американцы"], "Ахвахский": ["Ахвахцы"],
    "Вьетнамский": ["Вьетнамцы"], "Эвенский": ["Эвены"],
    "Адыгский": ["Адыгейцы", "Черкесы", "Кабардинцы", "Шапсуги"], "Нанайский": ["Нанайцы"],
    "Шорский": ["Шорцы"], "Абхазский": ["Абхазы"], "Долганский": ["Долганы"],
    "Хинди": ["Индийцы"], "Чамалинский": ["Чамалалы"], "Ботлихский": ["Ботлихцы"],
    "Литовский": ["Литовцы"], "Гагаузский": ["Гагаузы"], "Тиндальский": ["Тиндалы"],
    "Болгарский": ["Болгары"], "Корякский": ["Коряки"], "Румынский": ["Румыны"],
    "Иврит": ["Евреи"], "Еврейский": ["Евреи", "Горские евреи"], "Французский": ["Французы"],
    "Гунзибский": ["Гунзибцы"], "Персидский": ["Персы"], "Хваршинский": ["Хваршины"],
    "Польский": ["Поляки"], "Латышский": ["Латыши", "Латгальцы"],
    "Годоберинский": ["Годоберинцы"], "Испанский": ["Испанцы", "Кубинцы"],
    "Эстонский": ["Эстонцы", "Сету"], "Дунганский": ["Дунгане"], "Талышский": ["Талыши"],
    "Ассирийский": ["Ассирийцы"], "Багвалинский": ["Багулалы"], "Мансийский": ["Манси"],
    "Пушту": ["Афганцы"], "Сербскохорватский": ["Сербы", "Хорваты", "Боснийцы", "Черногорцы"],
    "Удинский": ["Удины"], "Арчинский": ["Арчинцы"], "Селькупский": ["Селькупы"],
    "Телеутский": ["Телеуты"], "Нивхский": ["Нивхи"], "Вепсский": ["Вепсы"],
    "Итальянский": ["Итальянцы"], "Финский": ["Финны", "Финны-ингерманландцы"],
    "Монгольский": ["Монголы"], "Тюркский": ["Турки-месхетинцы"], "Тубаларский": ["Тубалары"],
    "Ульчский": ["Ульчи"], "Эскимосский": ["Эскимосы"], "Ительменский": ["Ительмены"],
    "Венгерский": ["Венгры"], "Уйгурский": ["Уйгуры"], "Татский": ["Таты", "Горские евреи"],
    "Удэгейский": ["Удэгейцы"], "Кумандинский": ["Кумандинцы"], "Челканский": ["Челканцы"],
    "Гинухский": ["Гинухцы"], "Идиш": ["Евреи"], "Японский": ["Японцы"],
    "Юкагирский": ["Юкагиры"], "Чешский": ["Чехи"], "Нганасанский": ["Нганасаны"],
    "Каракалпакский": ["Каракалпаки"], "Дари": ["Афганцы"], "Саамский": ["Саамы"],
    "Словенский": ["Словенцы"], "Мегрельский": ["Мегрелы"],
    "Лугово-восточный марийский": ["Лугово-восточные марийцы", "Марийцы"],
    "Алюторский": ["Алюторцы"], "Ногайско-карагашский": ["Карагаши"], "Булгарский": ["Татары"],
    "Кетский": ["Кеты"], "Македонский": ["Македонцы"], "Алеутский": ["Алеуты"],
    "Орочский": ["Орочи"], "Уйльта": ["Уйльта"], "Словацкий": ["Словаки"],
    "Ижорский": ["Ижорцы"], "Энецкий": ["Энцы"], "Тофаларский": ["Тофалары"],
    "Чулымско-тюркский": ["Чулымцы"], "Караимский": ["Караимы"], "Негидальский": ["Негидальцы"],
    "Водский": ["Водь"], "Юртовско-татарский": ["Астраханские татары", "Ногайцы"],
    "Керекский": ["Кереки"], "Югский": ["Юги"], "Юитский": ["Эскимосы"],
}
RESIDUAL = "Русский"   # follows everyone no other language's nationality list claims
SETTLEMENTS = GEO / "ru" / "ru_settlement_nat.parquet"
K_NEAR = 24      # a hex's composition comes from its K_NEAR nearest settlements of its own kind
H_KM = 4.0       # ... weighted exp(-(d - d_nearest) / H_KM) by distance, times their people
FLOOR = 0.05     # every hex keeps this share of the unit's mean seed for every language, so a
                 # language with no people of its nationality near a hex is not ruled out there


def _xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]) * 6371.0


def _ipf(seed, rows, cols, iters=500):
    x = seed.copy()
    for _ in range(iters):
        rs = x.sum(1)
        x *= np.divide(rows, rs, out=np.zeros_like(rows), where=rs > 0)[:, None]
        cs = x.sum(0)
        x *= np.divide(cols, cs, out=np.zeros_like(cols), where=cs > 0)[None, :]
        if np.abs(x.sum(1) - rows).max() <= 1e-6 * rows.sum():
            break
    return x


class _RuWeighter:
    """Where inside its subject x urban/rural unit each language's dots go. A placement weight.

    Each hex takes a nationality mixture from the settlements near it of its own kind (urban
    hexes from towns, rural from villages, same subject). Per unit, a hex x language table is
    raked (IPF) to the hexes' Kontur population (scaled to the unit's drawn total) and the unit's
    census count of each language, from a seed: a language's nationality share in the hex
    (NAT), Russian the share no listed nationality claims, any other language even; each plus
    FLOOR of its unit mean. So the counts are the census's, and Tatar speakers in rural
    Bashkortostan sit where Tatars live."""

    def __init__(self, place):
        import ru2021
        from regroup import move
        from scipy.spatial import cKDTree
        pop = place["pop"].to_numpy(dtype=float)
        unit = place["unit"].astype(str).to_numpy()
        subject = np.array([u.split("/")[0] for u in unit])
        kind = np.array([u.endswith("/u") for u in unit])
        rp = place.geometry.representative_point()
        hx = _xyz(rp.x.to_numpy(), rp.y.to_numpy())

        if not SETTLEMENTS.exists():
            raise SystemExit(f"ru: {SETTLEMENTS} missing; run python sources/ru_settlements.py")
        S = pd.read_parquet(SETTLEMENTS)
        natcols = set(S.columns[6:])
        bad = sorted({c for cs in NAT.values() for c in cs} - natcols)
        if bad:
            raise SystemExit(f"ru: NAT names columns tochno lacks: {bad}")

        counts = _counts()
        counts["node"] = counts["node"].map(move)
        counts = counts.groupby(["unit", "node"])["count"].sum()
        node_nat, self.res_node = {}, move(ru2021.resolve(RESIDUAL))
        for lab, nats in NAT.items():
            node_nat.setdefault(move(ru2021.resolve(lab)), set()).update(nats)
        if self.res_node in node_nat:
            raise SystemExit("ru: a NAT label resolves to Russian's node")
        groups = sorted(node_nat)
        claimed = sorted({c for cs in node_nat.values() for c in cs})
        G = np.column_stack([S[sorted(node_nat[g])].sum(1).to_numpy(float) for g in groups]
                            + [S[claimed].sum(1).to_numpy(float)])
        stated = S["stated"].to_numpy(float)
        sx = _xyz(S["lon"].to_numpy(), S["lat"].to_numpy())
        s_subj, s_urb = S["subject"].to_numpy(), S["urban"].to_numpy()

        # every hex's nationality shares, groups + "claimed by some language" in the last column
        share = np.zeros((len(place), len(groups) + 1), dtype=float)
        has = np.zeros(len(place), dtype=bool)
        self.n_kind_fallback = 0
        for sj in np.unique(subject):
            for k in (True, False):
                hm = np.flatnonzero((subject == sj) & (kind == k))
                if not len(hm):
                    continue
                sm = np.flatnonzero((s_subj == sj) & (s_urb == k) & (stated > 0))
                if len(sm) == 0:
                    sm = np.flatnonzero((s_subj == sj) & (stated > 0))
                    self.n_kind_fallback += 1
                if len(sm) == 0:
                    continue
                kk = min(K_NEAR, len(sm))
                d, j = cKDTree(sx[sm]).query(hx[hm], k=kk)
                d, j = d.reshape(len(hm), kk), j.reshape(len(hm), kk)
                w = np.exp(-(d - d[:, :1]) / H_KM)
                Gs, st = G[sm], stated[sm]
                for a in range(0, len(hm), 2000):
                    b = slice(a, a + 2000)
                    num = np.einsum("hk,hkg->hg", w[b], Gs[j[b]])
                    den = (w[b] * st[j[b]]).sum(1)
                    ok = den > 0
                    share[hm[b][ok]] = num[ok] / den[ok, None]
                    has[hm[b][ok]] = True
        gi = {g: i for i, g in enumerate(groups)}

        self.X, self.pos, self.unit = {}, np.zeros(len(place), dtype=np.int64), unit
        self.n_nat = self.n_res = self.n_even = self.n_nohex = 0
        by = pd.Series(np.arange(len(place))).groupby(unit).apply(lambda s: s.to_numpy())
        for u, c in counts.groupby(level=0):
            rows_i = by[u]
            c = c.droplevel(0)
            nodes = list(c.index)
            cols = c.to_numpy(dtype=float)
            p = pop[rows_i]
            if p.sum() <= 0:
                p = np.ones(len(rows_i))
            rows = p * cols.sum() / p.sum()
            seed = np.ones((len(rows_i), len(nodes)))
            if has[rows_i].any():
                sh = share[rows_i]
                if not has[rows_i].all():   # hexes with no settlement data: the unit's mean
                    sh[~has[rows_i]] = (sh[has[rows_i]] * p[has[rows_i], None]).sum(0) / p[has[rows_i]].sum()
                for j, n in enumerate(nodes):
                    if n in gi:
                        col = sh[:, gi[n]]
                        self.n_nat += 1
                    elif n == self.res_node:
                        col = np.clip(1.0 - sh[:, -1], 0.0, None)
                        self.n_res += 1
                    else:
                        self.n_even += 1
                        continue
                    mean = (col * p).sum() / p.sum()
                    seed[:, j] = col + FLOOR * mean if mean > 0 else 1.0
            else:
                self.n_nohex += 1
            X = _ipf(seed, rows, cols)
            if np.abs(X.sum(0) - cols).max() > max(1.0, 1e-6 * cols.sum()):
                raise SystemExit(f"ru: the placement rake does not meet unit {u}'s counts")
            self.X[u] = {n: X[:, j] for j, n in enumerate(nodes)}
            self.pos[rows_i] = np.arange(len(rows_i))

    def weights(self, node, idx, count, plain=False):
        cols = self.X.get(self.unit[idx[0]])
        if cols is None or node not in cols:
            return None
        w = cols[node][self.pos[idx]]
        return w if w.sum() > 0 else None

    def summary(self):
        return (f"placed inside units by settlement nationality: {self.n_nat} (unit, language) "
                f"pairs follow their nationality, {self.n_res} Russian on the unclaimed rest, "
                f"{self.n_even} with no nationality even; {self.n_kind_fallback} unit(s) with no "
                f"settlement of their own kind used all the subject's, {self.n_nohex} with none "
                "placed by population")


ENTRY = dict(
    name="Russia",
    source=("All-Russian Population Census 2020 (held in 2021), Volume 5, Table 6, population by "
            "native language (Rosstat)"),
    how="census, 2021, native language",
    parts=[dict(covers="Everyone", source="2021 census, native language", rest=True)],
    grain=("85 federal subjects (Crimea and Sevastopol included), each split urban and rural: "
           "168 units, 880,000 people on average; inside each, placement follows nationality "
           "by settlement"),
    gap="16.6 million people (11.3%) with no native language recorded, most counted from records",
    # religiondots' frame: it stops at the antimeridian, leaving Chukotka's sliver beyond it to a pan
    view=[19.0, 41.0, 180.0, 78.0],
    counts=_counts,
    mappings=["ru2021"],
    place=GEO / "ru" / "ru_grid_3km.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=lambda place: _RuWeighter(place),
    note_public=(
        "The 2021 census asked each person's native language, which in the countries of the "
        "former Soviet Union leans towards identity rather than everyday use: people often name "
        "the language of their nationality while speaking Russian at home. Rosstat publishes the "
        "answers only for each federal subject, split into urban and rural; the densest grid "
        "cells stand in for the towns. Inside each of those, a language's dots are placed where "
        "people of its nationality live, from the same census's nationality count for every "
        "settlement (compiled by To Be Precise, tochno.st), and Russian fills the rest. The "
        "count for each subject is the census's; only the placement inside it is estimated. Crimea and Sevastopol are "
        "drawn from this census, which has counted them since 2014; there 206,000 people named "
        "Crimean Tatar and 59,000 Tatar. 275,000 people said only \"Mordvin\" and 318,000 only "
        "\"Mari\", and are drawn that way. 11.3% have no native language recorded, from 27% in "
        "Khanty-Mansi and 24% in Moscow to under 3% in Tatarstan and Chechnya; they are not "
        "drawn."),
)
