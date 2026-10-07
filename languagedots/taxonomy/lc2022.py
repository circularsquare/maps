"""Saint Lucia, Census 2022: born in Saint Lucia or abroad, and region of birth -> node. No
language question; every row `derived` (sources/lc.md). Built as Dominica (dm2011.py).

  born in Saint Lucia      Kweyol, on dm.txt's Antillean Creole node (Glottolog lists Saint
                           Lucian Creole French, sain1246, separately; the task brief and dm.md
                           keep one Kweyol node for the Lesser Antilles). The public REDATAM
                           base folds ethnicity into "African Descent/Black" and "Other", so
                           white or Indo-Saint Lucians cannot be set apart.
  born in North America    English (the United States and Canada)
  born elsewhere           CSO's region of birth only ("Latin America and the Caribbean",
                           "Europe", "Other"): pooled, unnamed, on `other`.
"""
KWEYOL = "creole.french_based.antillean"
EN = "indoeuropean.germanic.english"
CODES = {"Born St Lucia": KWEYOL, "North America": EN,
         "Latin America and the Caribbean": "other", "Europe": "other", "Other": "other"}
