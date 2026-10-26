"""The T-047 sample: 1 x 1 degree tiles by south-west corner, a label and the kind picked for."""
TILES = [
    # cities
    (40, -74, "New York", "city"), (46, 23, "Cluj", "city"), (51, -1, "London", "city"),
    (35, 139, "Tokyo", "city"), (6, 3, "Lagos", "city"), (19, 72, "Mumbai", "city"),
    (-24, -47, "Sao Paulo", "city"), (30, 31, "Cairo", "city"), (41, 28, "Istanbul", "city"),
    # the other built tiles
    (40, -75, "Trenton", "built"), (41, -74, "Hudson Valley", "built"), (41, -75, "Poconos", "built"),
    (46, 24, "Mures", "built"), (47, 23, "Maramures", "built"), (47, 24, "Bistrita", "built"),
    # farmland
    (41, -94, "Iowa", "farm"), (49, 33, "Ukraine", "farm"), (30, 75, "Punjab", "farm"),
    (-35, -62, "Pampa", "farm"), (48, 1, "Beauce", "farm"), (13, 2, "Sahel", "farm"),
    # desert
    (24, 10, "Sahara", "desert"), (22, 47, "Rub al Khali", "desert"), (38, 83, "Taklamakan", "desert"),
    (-25, 130, "Central Australia", "desert"),
    # ice, tundra, boreal
    (70, -45, "Greenland", "ice"), (68, 70, "Yamal", "tundra"), (64, -100, "Nunavut", "tundra"),
    (60, 100, "Siberian taiga", "forest"), (56, -72, "Quebec lakes", "lakes"),
    # rainforest, wetlands, deltas
    (-4, -62, "Amazon", "forest"), (0, 21, "Congo", "forest"), (23, 90, "Bangladesh", "delta"),
    (10, 105, "Mekong delta", "delta"), (8, 30, "Sudd", "wetland"),
    # mountains
    (46, 7, "Alps", "mountain"), (27, 86, "Everest", "mountain"), (-14, -72, "Cusco", "mountain"),
    (39, -107, "Rockies", "mountain"), (61, 6, "Norway fjords", "mountain"),
    # coasts and islands
    (59, 18, "Stockholm archipelago", "coast"), (37, 25, "Cyclades", "coast"),
]
