"""The 65 units Tajikistan's statistics agency publishes population for, below region level.

One row per unit, in the order the annual bulletin lists them. Columns:
  code      our code: ISO 3166-2 region + two-digit sequence
  name      English name (current official name, Latin transliteration)
  name_tg   Tajik name as written in OSM
  osm       how the polygon is made (see prep_boundaries.py):
              "r<id>"  an OSM boundary relation used as is
              "c:<key>" a city carved out of the districts around it from an OSM
                        place outline (CITY_OUTLINES in prep_boundaries.py)
  bul       regex matched against the Tajik row label in the annual bulletin
            (legacy Tajik font: Њ=Ҳ, Љ=Ҷ, Ѓ=Ғ, ќ=қ, ї=ӣ, ў=ӯ). Rows are matched in
            order, each search starting after the previous unit's row, so the first
            hit is the unit row and not a settlement row of the same name.
  cen       regex matched against the Russian label in census 2020 table 1
"""

REGIONS = [
    # code, name, name_tg, bulletin header regex, census region regex
    ("TJ-GB", "Gorno-Badakhshan", "Вилояти Мухтори Кӯҳистони Бадахшон", r"^ВМКБ", r"Горно - Бадахшанская"),
    ("TJ-SU", "Sughd", "Вилояти Суғд", r"^Вилояти Суѓд", r"Согдийская область"),
    ("TJ-KT", "Khatlon", "Вилояти Хатлон", r"^Вилояти Хатлон", r"Хатлонская область"),
    ("TJ-DU", "Dushanbe", "Шаҳри Душанбе", r"^Шањри Душанбе", r"город Душанбе"),
    ("TJ-RA", "Districts of Republican Subordination", "Ноҳияҳои тобеи ҷумҳурӣ",
     r"^Шањру ноњияњои тобеи", r"Города и районы республиканского подчинения"),
]

UNITS = [
    # Gorno-Badakhshan
    ("TJ-GB-01", "Khorugh", "Шаҳри Хоруғ", "c:khorugh", r"Хоруѓ", r"город Хорог"),
    ("TJ-GB-02", "Vanj", "Ноҳияи Ванҷ", "r3281956", r"Ванљ", r"Ванчский район"),
    ("TJ-GB-03", "Ishkoshim", "Ноҳияи Ишкошим", "r3281971", r"Ишкошим", r"Ишкашимский район"),
    ("TJ-GB-04", "Darvoz", "Ноҳияи Дарвоз", "r3281969", r"Дарвоз", r"Дарвозский район"),
    ("TJ-GB-05", "Murghob", "Ноҳияи Мурғоб", "r3281933", r"Ноњияи Мурѓоб", r"Мургабский район"),
    ("TJ-GB-06", "Roshtqal'a", "Ноҳияи Роштқалъа", "r3281930", r"Роштќалъа", r"Рошткалинский район"),
    ("TJ-GB-07", "Rushon", "Ноҳияи Рӯшон", "r3281942", r"Рўшон", r"Рушанский район"),
    ("TJ-GB-08", "Shughnon", "Ноҳияи Шуғнон", "r3281967", r"Шуѓнон", r"Шугнанский район"),
    # Sughd
    ("TJ-SU-01", "Khujand", "Шаҳри Хуҷанд", "r15520796", r"Хуљанд", r"город Худжанд"),
    ("TJ-SU-02", "Isfara", "Шаҳри Исфара", "r3280683", r"шањри Исфара", r"город Исфара - всего"),
    ("TJ-SU-03", "Guliston", "Шаҳри Гулистон", "r15520864", r"^Гулистон$", r"Гулистон - всего"),
    ("TJ-SU-04", "Konibodom", "Шаҳри Конибодом", "r3280684", r"шањри Конибодом", r"город Канибадам-всего"),
    ("TJ-SU-05", "Panjakent", "Шаҳри Панҷакент", "r3280687", r"шањри Панљакент", r"город Пенджикент - всего"),
    ("TJ-SU-06", "Istaravshan", "Шаҳри Истаравшан", "r3281013", r"шањри Истаравшан", r"город Истаравшан - всего"),
    ("TJ-SU-07", "Istiqlol", "Шаҳри Истиқлол", "c:istiqlol", r"Истиќлол", r"город Истиклол"),
    ("TJ-SU-08", "Buston", "Шаҳри Бӯстон", "r15520839", r"^Бўстон$", r"город Бустон - всего"),
    ("TJ-SU-09", "Ayni", "Ноҳияи Айнӣ", "r3280692", r"Айнї", r"Айнинский район"),
    ("TJ-SU-10", "Asht", "Ноҳияи Ашт", "r3280682", r"^Ноњияи Ашт", r"Аштский район"),
    ("TJ-SU-11", "Devashtich", "Ноҳияи Деваштич", "r3280690", r"Деваштич", r"район Деваштич"),
    ("TJ-SU-12", "Zafarobod", "Ноҳияи Зафаробод", "r3280689", r"^Ноњияи Зафаробод", r"Зафарободский район"),
    ("TJ-SU-13", "Mastchoh", "Ноҳияи Мастчоҳ", "r3280691", r"^Ноњияи\s*Мастчоњ", r"Матчинский район"),
    ("TJ-SU-14", "Spitamen", "Ноҳияи Спитамен", "r3280688", r"Спитамен", r"район Спитамен"),
    ("TJ-SU-15", "Jabbor Rasulov", "Ноҳияи Ҷаббор Расулов", "r3281014", r"Расулов", r"Дж\.Расулова"),
    ("TJ-SU-16", "Bobojon Ghafurov", "Ноҳияи Бобоҷон Ғафуров", "r3280693", r"Ноњияи Б\.\s*Ѓафуров", r"район Б\. Гафурова"),
    ("TJ-SU-17", "Shahriston", "Ноҳияи Шаҳристон", "r3280685", r"Шањристон", r"Шахристанский район"),
    ("TJ-SU-18", "Kuhistoni Mastchoh", "Ноҳияи Кӯҳистони Мастчоҳ", "r3280686", r"Кўњистони", r"Кухистони"),
    # Khatlon
    ("TJ-KT-01", "Bokhtar", "Шаҳри Бохтар", "c:bokhtar", r"шањри Бохтар", r"город Бохтар"),
    ("TJ-KT-02", "Kulob", "Шаҳри Кӯлоб", "r3281947", r"шањри Кўлоб", r"Куляб -\s*всего"),
    ("TJ-KT-03", "Norak", "Шаҳри Норак", "r3281959", r"шањри Норак", r"Нурек - всего"),
    ("TJ-KT-04", "Levakant", "Шаҳри Левакант", "r3281940", r"Левакант", r"Левакант- всего"),
    ("TJ-KT-05", "Baljuvon", "Ноҳияи Балҷувон", "r3281963", r"Балљувон", r"Бальджувонский"),
    ("TJ-KT-06", "Kushoniyon", "Ноҳияи Кӯшониён", "r3281954", r"Кушониѐн", r"Кушониёнский"),
    ("TJ-KT-07", "Vakhsh", "Ноҳияи Вахш", "r3281943", r"^Ноњияи Вахш", r"Вахшский район"),
    ("TJ-KT-08", "Vose", "Ноҳияи Восеъ", "r3281955", r"Восеъ", r"Восейский район"),
    ("TJ-KT-09", "Danghara", "Ноҳияи Данғара", "r3281965", r"^Ноњияи Данѓара", r"Дангаринский район"),
    ("TJ-KT-10", "Yovon", "Ноҳияи Ёвон", "r3281960", r"^Ноњияи Ёвон", r"Яванский район"),
    ("TJ-KT-11", "Jaloliddini Balkhi", "Ноҳияи Ҷалолиддини Балхӣ", "r3281953", r"Балхї", r"Дж\.Балхи"),
    ("TJ-KT-12", "Muminobod", "Ноҳияи Мӯъминобод", "r3281938", r"^Ноњияи Мўминобод", r"Муминободский район"),
    ("TJ-KT-13", "Hamadoni", "Ноҳияи Ҳамадонӣ", "r3281951", r"Њамадонї", r"М\.С\.А\.Хамадони"),
    ("TJ-KT-14", "Nosiri Khusrav", "Ноҳияи Носири Хусрав", "r3281932", r"Хусрав", r"район Н\. Хусрав"),
    ("TJ-KT-15", "Panj", "Ноҳияи Панҷ", "r3281958", r"^Ноњияи Панљ", r"Пянджский район"),
    ("TJ-KT-16", "Temurmalik", "Ноҳияи Темурмалик", "r3281950", r"^Ноњияи Темурмалик", r"район Темурмалик"),
    ("TJ-KT-17", "Khovaling", "Ноҳияи Ховалинг", "r3281962", r"^Ноњияи Ховалинг", r"Ховалингский район"),
    ("TJ-KT-18", "Farkhor", "Ноҳияи Фархор", "r3281966", r"^Ноњияи Фархор", r"Фархорский район"),
    ("TJ-KT-19", "Khuroson", "Ноҳияи Хуросон", "r3281945", r"^Ноњияи Хуросон", r"район Хуросон"),
    ("TJ-KT-20", "Dusti", "Ноҳияи Дӯстӣ", "r3281964", r"^Ноњияи Дўстї", r"район Дусти"),
    ("TJ-KT-21", "Qubodiyon", "Ноҳияи Қубодиён", "r3281961", r"^Ноњияи Ќубодиѐн", r"Кубодиёнский"),
    ("TJ-KT-22", "Abdurahmoni Jomi", "Ноҳияи Абдураҳмони Ҷомӣ", "r3281941", r"Љомї", r"район А\. Джоми"),
    ("TJ-KT-23", "Jayhun", "Ноҳияи Ҷайҳун", "r3281946", r"Љайњун", r"район Джайхун"),
    ("TJ-KT-24", "Shahritus", "Ноҳияи Шаҳритус", "r3281934", r"Шањри-?\s*тус", r"Шаартузский район"),
    ("TJ-KT-25", "Shamsiddin Shohin", "Ноҳияи Шамсиддин Шоҳин", "r6870523", r"Шоњин", r"район Ш\. Шохин"),
    # Dushanbe: one unit; the bulletin has no district split (see README)
    ("TJ-DU-01", "Dushanbe", "Шаҳри Душанбе", "r7328360", r"^Шањри Душанбе", r"город Душанбе"),
    # Districts of Republican Subordination
    ("TJ-RA-01", "Vahdat", "Шаҳри Ваҳдат", "r3281970", r"Шањри Вањдат", r"Вахдат - всего"),
    ("TJ-RA-02", "Tursunzoda", "Шаҳри Турсунзода", "r3281944", r"шањри Турсунзода", r"Турсунзаде - всего"),
    ("TJ-RA-03", "Varzob", "Ноҳияи Варзоб", "r3281939", r"Варзоб", r"Варзобский район"),
    ("TJ-RA-04", "Hisor", "Шаҳри Ҳисор", "r3281935", r"Шањри\s+Њисор", r"Гиссар - всего"),
    ("TJ-RA-05", "Lakhsh", "Ноҳияи Лахш", "r3281957", r"^Ноњияи Лахш", r"Лахшский район"),
    ("TJ-RA-06", "Nurobod", "Ноҳияи Нуробод", "r3281949", r"^Ноњияи Нуробод", r"Нурободский район"),
    ("TJ-RA-07", "Rasht", "Ноҳияи Рашт", "r3281968", r"^Ноњияи Рашт", r"Раштский район"),
    ("TJ-RA-08", "Sangvor", "Ноҳияи Сангвор", "r3281972", r"Сангвор", r"Сангворский район"),
    ("TJ-RA-09", "Tojikobod", "Ноҳияи Тоҷикобод", "r3281931", r"^Ноњияи Тољикобод", r"Таджикобадский район"),
    ("TJ-RA-10", "Faizobod", "Ноҳияи Файзобод", "r3281936", r"^Ноњияи Файзобод", r"Файзабадский район"),
    ("TJ-RA-11", "Shahrinav", "Ноҳияи Шаҳринав", "r3281948", r"^Ноњияи Шањринав", r"Шахринавский район"),
    ("TJ-RA-12", "Roghun", "Шаҳри Роғун", "r3281937", r"Шањри\s+Роѓун", r"Рогун - всего"),
    ("TJ-RA-13", "Rudaki", "Ноҳияи Рӯдакӣ", "r3281973", r"Рўдакї", r"район Рудаки"),
]

# Units whose census 2010/2020 figures are on a different territory from the
# bulletin's: Dushanbe took in part of Rudaki between the October 2020 census
# and 1 January 2021 (census Dushanbe 948,251 on 126.6 km2; bulletin 1 Jan 2021
# 1,185.4 thousand on ~203 km2). Their census years are left out.
CENSUS_OFF_BASIS = {"TJ-DU-01", "TJ-RA-13"}
