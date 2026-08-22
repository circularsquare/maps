// Köppen-Geiger class labels. A lookup table, nothing else.
//
// The codes come from GHS's CL_KOP_* columns, which ship integers with no
// legend; match_ghs.py maps them to these strings using Beck et al. 2018's
// class order and records how that was verified.
//
// Deliberately NOT a colour palette. The card already carries four ramps plus
// the language palette, and colours.js owns colour — a fifth scheme here would
// both crowd the card and put colour in two places.

export const KOPPEN = {
  Af:  'Tropical rainforest',
  Am:  'Tropical monsoon',
  Aw:  'Tropical savanna',
  BWh: 'Hot desert',
  BWk: 'Cold desert',
  BSh: 'Hot semi-arid',
  BSk: 'Cold semi-arid',
  Csa: 'Mediterranean, hot summer',
  Csb: 'Mediterranean, warm summer',
  Csc: 'Mediterranean, cold summer',
  Cwa: 'Humid subtropical, dry winter',
  Cwb: 'Subtropical highland',
  Cwc: 'Subpolar highland',
  Cfa: 'Humid subtropical',
  Cfb: 'Temperate oceanic',
  Cfc: 'Subpolar oceanic',
  Dsa: 'Continental, hot dry summer',
  Dsb: 'Continental, warm dry summer',
  Dsc: 'Subarctic, dry summer',
  Dsd: 'Subarctic, dry summer, severe winter',
  Dwa: 'Continental, hot summer, dry winter',
  Dwb: 'Continental, warm summer, dry winter',
  Dwc: 'Subarctic, dry winter',
  Dwd: 'Subarctic, dry winter, severe winter',
  Dfa: 'Continental, hot summer',
  Dfb: 'Continental, warm summer',
  Dfc: 'Subarctic',
  Dfd: 'Subarctic, severe winter',
  ET:  'Tundra',
  EF:  'Ice cap',
};

export const koppenLabel = code => KOPPEN[code] || null;
