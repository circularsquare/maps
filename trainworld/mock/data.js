// Fake New York network for the T-009 mock. Hand-placed stations, invented lines and numbers.

const STATIONS = {
  inwood:   ["Inwood 207 St", -73.9187, 40.8673],
  w168:     ["168 St", -73.9398, 40.8404],
  w125:     ["125 St", -73.9585, 40.8156],
  w96:      ["96 St", -73.9724, 40.7939],
  w72:      ["72 St", -73.9819, 40.7784],
  colc:     ["Columbus Circle", -73.9819, 40.7681],
  tsq:      ["Times Sq", -73.9866, 40.7559],
  usq:      ["Union Sq", -73.9903, 40.7359],
  hou:      ["Houston St", -73.9970, 40.7252],
  cityhall: ["City Hall", -74.0068, 40.7134],
  bowl:     ["Bowling Green", -74.0140, 40.7048],

  gct:      ["Grand Central", -73.9772, 40.7527],
  lic:      ["Long Island City", -73.9480, 40.7470],
  sunny:    ["Sunnyside", -73.9245, 40.7440],
  wood:     ["Woodside", -73.9030, 40.7456],
  jh:       ["Jackson Heights", -73.8913, 40.7466],
  cor:      ["Corona", -73.8620, 40.7498],
  flu:      ["Flushing", -73.8303, 40.7596],

  gp:       ["Greenpoint", -73.9541, 40.7313],
  wb:       ["Williamsburg", -73.9571, 40.7172],
  bs:       ["Bed-Stuy", -73.9534, 40.6896],
  fr:       ["Franklin Av", -73.9558, 40.6800],
  pp:       ["Prospect Park", -73.9620, 40.6615],

  bh:       ["Borough Hall", -73.9903, 40.6930],
  atl:      ["Atlantic Av", -73.9776, 40.6842],
  eny:      ["East New York", -73.9030, 40.6787],
  wdh:      ["Woodhaven", -73.8550, 40.6893],
  jam:      ["Jamaica", -73.8080, 40.7000],

  nwk:      ["Newark Penn", -74.1645, 40.7342],
  har:      ["Harrison", -74.1561, 40.7390],
  jsq:      ["Journal Sq", -74.0631, 40.7327],
  grove:    ["Grove St", -74.0431, 40.7195],
  exch:     ["Exchange Pl", -74.0331, 40.7163],
};

// colour = index into the line palette (1-8). tph = trains an hour [peak, shoulder, off-peak].
// rtt = round trip in minutes. riders = boardings a day. board = boardings a day per stop, in k.
const LINES = [
  { id: "B", name: "Broadway", colour: 1, tph: [20, 12, 6], rtt: 72, km: 21.4, riders: 312000,
    fare: 2.90, fullest: 86,
    stops: ["inwood", "w168", "w125", "w96", "w72", "colc", "tsq", "usq", "hou", "cityhall", "bowl"],
    board: [14, 22, 31, 27, 30, 33, 52, 38, 19, 29, 17] },
  { id: "F", name: "Flushing", colour: 2, tph: [16, 10, 6], rtt: 56, km: 15.1, riders: 188000,
    fare: 2.90, fullest: 91,
    stops: ["tsq", "gct", "lic", "sunny", "wood", "jh", "cor", "flu"],
    board: [34, 29, 21, 12, 19, 26, 15, 32] },
  { id: "C", name: "Crosstown", colour: 3, tph: [10, 8, 5], rtt: 44, km: 9.6, riders: 96000,
    fare: 2.90, fullest: 64,
    stops: ["lic", "gp", "wb", "bs", "fr", "pp"],
    board: [18, 14, 22, 15, 16, 11] },
  { id: "A", name: "Atlantic", colour: 4, tph: [12, 8, 4], rtt: 70, km: 22.8, riders: 141000,
    fare: 2.90, fullest: 78,
    stops: ["cityhall", "bh", "atl", "fr", "eny", "wdh", "jam"],
    board: [26, 21, 30, 17, 14, 9, 24] },
  { id: "H", name: "Hudson", colour: 5, tph: [12, 8, 4], rtt: 52, km: 18.3, riders: 74000,
    fare: 2.90, fullest: 71,
    stops: ["nwk", "har", "jsq", "grove", "exch", "cityhall"],
    board: [19, 4, 17, 11, 9, 14] },
];

const MONEY = {
  cash: 1.84e9,
  rows: [
    ["Fares", 2.35e6],
    ["Running trains", -1.06e6],
    ["Track and station upkeep", -0.31e6],
    ["Loan interest", -0.12e6],
  ],
};

const CITY = {
  population: "21.7M",
  jobs: "10.2M",
  periods: [            // train share by period, %
    ["Morning peak", 13.1],
    ["Midday", 6.4],
    ["Evening peak", 12.2],
    ["Evening", 5.8],
    ["Night", 2.9],
  ],
};

// Cost per km by level, $M (mock numbers).
const LEVEL_COST = { "-3": 310, "-2": 240, "-1": 180, "0": 38, "1": 95, "2": 120, "3": 150 };
const LEVEL_NAME = { "-3": "Deep tunnel", "-2": "Tunnel", "-1": "Tunnel", "0": "Ground",
                     "1": "Viaduct", "2": "Viaduct", "3": "High viaduct" };
