// A new city in the real app (T-364, from T-288's London check): a new game, a jump to the city's
// centre: its pack opens, it becomes the active city with its boundary layer, the top bar names it
// and gives its no-rail split; ground-level track across a water crossing is refused and the same
// route in a tunnel accepted; a test line through the centre (deep tunnel, constructed, 10 trains an
// hour at the peak) reaches a final solve and carries riders of every purpose, purposes adding up;
// no page errors. Cities are the table below, or a row file in ROW (checks/pack_gate.py --browser
// makes one for any pack, T-538).
//
//     node trainworld/checks/city-browser.mjs <id>        (from the maps root)
//
// Needs a Vite dev server (BASE, default http://localhost:6125) and Playwright (PLAYWRIGHT_MODULE,
// else `playwright`, else the npx cache). Screenshots artifacts/trainworld/<task>-<id>-homes.png and
// -line.png; results <task>-<id>-browser.json (OUT and JSON_OUT override).
import { writeFile, readFile, mkdtemp, readdir, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

// name: as the top bar shows it; water: two points either side of a river or bay; line: [station, lon, lat]
const CITIES = {
  lon: {
    task: 'T-288', name: 'London', centre: [-0.1246, 51.5073], water: [[-0.1100, 51.5125], [-0.1080, 51.5040]],
    lineName: 'Central test', colour: '#dc241f', view: [-0.14, 51.514],
    line: [['Notting Hill Gate', -0.1967, 51.5090], ['Queensway', -0.1874, 51.5107], ['Lancaster Gate', -0.1756, 51.5119],
      ['Marble Arch', -0.1584, 51.5136], ['Bond Street', -0.1494, 51.5142], ['Oxford Circus', -0.1415, 51.5152],
      ['Tottenham Court Road', -0.1308, 51.5165], ['Holborn', -0.1200, 51.5174], ['Chancery Lane', -0.1111, 51.5185],
      ["St Paul's", -0.0976, 51.5146], ['Bank', -0.0886, 51.5133], ['Liverpool Street', -0.0823, 51.5178]],
  },
  par: {
    task: 'T-289', name: 'Paris', centre: [2.3488, 48.8534], water: [[2.3190, 48.8652], [2.3190, 48.8610]],
    lineName: 'Ligne 1 test', colour: '#ffcd00', view: [2.345, 48.861],
    line: [['Charles de Gaulle Etoile', 2.2950, 48.8738], ['George V', 2.3006, 48.8720], ['Franklin D. Roosevelt', 2.3095, 48.8689],
      ['Champs-Elysees Clemenceau', 2.3145, 48.8676], ['Concorde', 2.3229, 48.8656], ['Tuileries', 2.3297, 48.8645],
      ['Palais Royal', 2.3364, 48.8625], ['Louvre Rivoli', 2.3409, 48.8609], ['Chatelet', 2.3475, 48.8585],
      ['Hotel de Ville', 2.3522, 48.8573], ['Saint-Paul', 2.3610, 48.8551], ['Bastille', 2.3691, 48.8531],
      ['Gare de Lyon', 2.3733, 48.8448], ['Reuilly Diderot', 2.3872, 48.8472], ['Nation', 2.3959, 48.8484]],
  },
  sha: {
    // T-291: Line 2's stations at their OSM station nodes; the water pair across the Huangpu at the Bund
    task: 'T-291', name: 'Shanghai', centre: [121.4737, 31.2304], water: [[121.4855, 31.2380], [121.5050, 31.2400]],
    lineName: 'Line 2 test', colour: '#8cc220', view: [121.48, 31.232],
    line: [['Zhongshan Park', 121.4116, 31.2201], ['Jiangsu Road', 121.4263, 31.2215], ["Jing'an Temple", 121.4421, 31.2253],
      ['West Nanjing Road', 121.4560, 31.2318], ["People's Square", 121.4711, 31.2342], ['East Nanjing Road', 121.4792, 31.2391],
      ['Lujiazui', 121.4981, 31.2401], ['Century Avenue', 121.5228, 31.2310], ['Science and Technology Museum', 121.5387, 31.2219],
      ['Century Park', 121.5467, 31.2108], ['Longyang Road', 121.5535, 31.2052]],
  },
  mum: {
    // T-292: the Western line from Churchgate to Andheri at its OSM station nodes (Santacruz placed
    // by hand); the water pair across Thane Creek, Mankhurd to Vashi, checked on the water cache
    task: 'T-292', name: 'Mumbai', centre: [72.8428, 19.0192], water: [[72.9400, 19.0500], [73.0000, 19.0700]],
    lineName: 'Western line test', colour: '#e4002b', view: [72.84, 19.03],
    line: [['Churchgate', 72.8272, 18.9355], ['Marine Lines', 72.8237, 18.9458], ['Charni Road', 72.8182, 18.9518],
      ['Grant Road', 72.8158, 18.9634], ['Mumbai Central', 72.8193, 18.9696], ['Mahalaxmi', 72.8242, 18.9825],
      ['Lower Parel', 72.8303, 18.9957], ['Prabhadevi', 72.8359, 19.0075], ['Dadar', 72.8428, 19.0192],
      ['Matunga Road', 72.8467, 19.0278], ['Mahim', 72.8466, 19.0407], ['Bandra', 72.8402, 19.0550],
      ['Khar Road', 72.8400, 19.0682], ['Santacruz', 72.8410, 19.0810], ['Vile Parle', 72.8440, 19.0999],
      ['Andheri', 72.8464, 19.1197]],
  },
  lag: {
    // T-459: the Red Line, Agbado to Oyingbo, on to the Blue Line's Marina, at their OSM station
    // nodes; the water pair across the Lagos Lagoon, Ebute Metta to Ikoyi
    task: 'T-459', name: 'Lagos', centre: [3.3786, 6.5158], water: [[3.3920, 6.4990], [3.4300, 6.4600]],
    lineName: 'Red line test', colour: '#d7262b', view: [3.36, 6.53],
    line: [['Agbado', 3.3031, 6.6871], ['Iju', 3.3193, 6.6621], ['Agege', 3.3262, 6.6195], ['Ikeja', 3.3373, 6.5919],
      ['Oshodi', 3.3514, 6.5573], ['Mushin', 3.3614, 6.5325], ['Yaba', 3.3700, 6.5115],
      ['Mobolaji Johnson', 3.3737, 6.5018], ['Oyingbo', 3.3815, 6.4834], ['Marina', 3.3950, 6.4462]],
  },
  sao: {
    // T-458: Metro Line 1, Tucuruvi to Jabaquara, at its OSM station nodes (Luz at the Line 4 node);
    // the water pair across the Guarapiranga reservoir, checked on the water cache
    task: 'T-458', name: 'São Paulo', centre: [-46.6339, -23.5503], water: [[-46.7600, -23.7000], [-46.7000, -23.7000]],
    lineName: 'Line 1 test', colour: '#0455a1', view: [-46.63, -23.56],
    line: [['Tucuruvi', -46.6033, -23.4801], ['Parada Inglesa', -46.6089, -23.4870],
      ['Jardim São Paulo', -46.6170, -23.4924], ['Santana', -46.6251, -23.5024], ['Carandiru', -46.6252, -23.5091],
      ['Portuguesa-Tietê', -46.6254, -23.5162], ['Armênia', -46.6289, -23.5251], ['Tiradentes', -46.6325, -23.5309],
      ['Luz', -46.6344, -23.5354], ['São Bento', -46.6343, -23.5440], ['Sé', -46.6334, -23.5504],
      ['Japão-Liberdade', -46.6359, -23.5552], ['São Joaquim', -46.6387, -23.5618], ['Vergueiro', -46.6399, -23.5688],
      ['Paraíso', -46.6410, -23.5754], ['Ana Rosa', -46.6384, -23.5814], ['Vila Mariana', -46.6347, -23.5892],
      ['Santa Cruz', -46.6367, -23.5990], ['Praça da Árvore', -46.6379, -23.6106], ['Saúde', -46.6392, -23.6184],
      ['São Judas', -46.6408, -23.6256], ['Conceição', -46.6413, -23.6350], ['Jabaquara', -46.6410, -23.6463]],
  },
  tyo: {
    // T-420: the Ginza line, Shibuya to Asakusa (positions to about 100 m); the water pair across
    // the Sumida by the Azuma bridge
    task: 'T-420', name: 'Tokyo', centre: [139.7671, 35.6812], water: [[139.7960, 35.7105], [139.8045, 35.7085]],
    lineName: 'Ginza line test', colour: '#ff9500', view: [139.75, 35.685],
    line: [['Shibuya', 139.7016, 35.6590], ['Omotesando', 139.7123, 35.6652], ['Gaiemmae', 139.7178, 35.6705],
      ['Aoyama-itchome', 139.7238, 35.6727], ['Akasaka-mitsuke', 139.7370, 35.6770], ['Tameike-sanno', 139.7413, 35.6736],
      ['Toranomon', 139.7497, 35.6703], ['Shimbashi', 139.7583, 35.6667], ['Ginza', 139.7640, 35.6717],
      ['Kyobashi', 139.7700, 35.6766], ['Nihombashi', 139.7740, 35.6820], ['Mitsukoshimae', 139.7735, 35.6871],
      ['Kanda', 139.7709, 35.6939], ['Suehirocho', 139.7717, 35.7027], ['Ueno-hirokoji', 139.7727, 35.7075],
      ['Ueno', 139.7770, 35.7114], ['Inaricho', 139.7826, 35.7114], ['Tawaramachi', 139.7905, 35.7110],
      ['Asakusa', 139.7976, 35.7107]],
  },
  ber: {
    // T-290: the Stadtbahn from Westkreuz to Ostkreuz; the water pair across the Spree by the
    // Oberbaumbruecke, checked on the water cache
    task: 'T-290', name: 'Berlin', centre: [13.3777, 52.5163], water: [[13.4430, 52.5005], [13.4468, 52.5032]],
    lineName: 'Stadtbahn test', colour: '#008d4f', view: [13.38, 52.515],
    line: [['Westkreuz', 13.2836, 52.5011], ['Charlottenburg', 13.3044, 52.5049], ['Savignyplatz', 13.3195, 52.5052],
      ['Zoologischer Garten', 13.3326, 52.5067], ['Tiergarten', 13.3365, 52.5139], ['Bellevue', 13.3474, 52.5199],
      ['Hauptbahnhof', 13.3695, 52.5251], ['Friedrichstrasse', 13.3870, 52.5203], ['Hackescher Markt', 13.4022, 52.5225],
      ['Alexanderplatz', 13.4114, 52.5215], ['Jannowitzbruecke', 13.4180, 52.5150], ['Ostbahnhof', 13.4344, 52.5104],
      ['Warschauer Strasse', 13.4491, 52.5059], ['Ostkreuz', 13.4694, 52.5031]],
  },
  seo: {
    // T-457: Line 2 from City Hall east and south round to Sadang, 26 of the loop's 43 stations
    // (from the OSM extract; the whole loop costs more than a new game's money);
    // the water pair across the Han between Ichon and Banpo
    task: 'T-457', name: 'Seoul', centre: [126.9780, 37.5665], water: [[126.9960, 37.5250], [126.9960, 37.5050]],
    lineName: 'Line 2 test', colour: '#00a84d', view: [126.98, 37.53],
    line: [['City Hall', 126.9771, 37.5655], ['Euljiro 1-ga', 126.9826, 37.5659], ['Euljiro 3-ga', 126.9925, 37.5663],
      ['Euljiro 4-ga', 126.9982, 37.5666], ['Dongdaemun History and Culture Park', 127.0081, 37.5653], ['Sindang', 127.0163, 37.5654],
      ['Sangwangsimni', 127.0293, 37.5644], ['Wangsimni', 127.0372, 37.5612], ['Hanyang Univ.', 127.0437, 37.5557],
      ['Ttukseom', 127.0474, 37.5472], ['Seongsu', 127.0561, 37.5446], ['Konkuk Univ.', 127.0709, 37.5404],
      ['Guui', 127.0862, 37.5372], ['Gangbyeon', 127.0947, 37.5351], ['Jamsillaru', 127.1038, 37.5207],
      ['Jamsil', 127.1001, 37.5132], ['Jamsilsaenae', 127.0856, 37.5116], ['Sports Complex', 127.0737, 37.5111],
      ['Samseong', 127.0631, 37.5088], ['Seolleung', 127.0490, 37.5045], ['Yeoksam', 127.0370, 37.5008],
      ['Gangnam', 127.0276, 37.4979], ['Seoul Nat. Univ. of Education', 127.0137, 37.4937], ['Seocho', 127.0077, 37.4918],
      ['Bangbae', 126.9977, 37.4815], ['Sadang', 126.9817, 37.4766]],
  },
  fuk: {
    // T-432: the Kuko line, Meinohama to Fukuoka Airport (positions to about 150 m); the water pair
    // across Dokai Bay by the Wakato bridge, checked on the mask
    task: 'T-432', name: 'Fukuoka', centre: [130.4207, 33.5897], water: [[130.8125, 33.9035], [130.8185, 33.8955]],
    lineName: 'Kuko line test', colour: '#f08300', view: [130.40, 33.59],
    line: [['Meinohama', 130.3243, 33.5838], ['Muromi', 130.3367, 33.5833], ['Fujisaki', 130.3487, 33.5826],
      ['Nishijin', 130.3590, 33.5832], ['Tojinmachi', 130.3692, 33.5864], ['Ohorikoen', 130.3778, 33.5885],
      ['Akasaka', 130.3885, 33.5898], ['Tenjin', 130.3992, 33.5913], ['Nakasu-Kawabata', 130.4073, 33.5946],
      ['Gion', 130.4134, 33.5934], ['Hakata', 130.4207, 33.5897], ['Higashi-Hie', 130.4325, 33.5859],
      ['Fukuokakuko', 130.4507, 33.5866]],
  },
  spk: {
    // T-432: the Namboku line, Asabu to Makomanai (positions to about 150 m); the water pair across
    // the Toyohira by Nakajima Park, checked on the mask
    task: 'T-432', name: 'Sapporo', centre: [141.3508, 43.0687], water: [[141.3570, 43.0450], [141.3700, 43.0450]],
    lineName: 'Namboku line test', colour: '#00a650', view: [141.35, 43.06],
    line: [['Asabu', 141.3404, 43.1078], ['Kita-34-jo', 141.3407, 43.1003], ['Kita-24-jo', 141.3408, 43.0906],
      ['Kita-18-jo', 141.3408, 43.0838], ['Kita-12-jo', 141.3408, 43.0757], ['Sapporo', 141.3507, 43.0683],
      ['Odori', 141.3542, 43.0606], ['Susukino', 141.3531, 43.0556], ['Nakajima-koen', 141.3537, 43.0489],
      ['Horohira-bashi', 141.3540, 43.0427], ['Nakanoshima', 141.3557, 43.0385], ['Hiragishi', 141.3610, 43.0322],
      ['Minami-Hiragishi', 141.3633, 43.0217], ['Sumikawa', 141.3556, 43.0093], ['Jieitai-mae', 141.3474, 42.9998],
      ['Makomanai', 141.3443, 42.9928]],
  },
  sdj: {
    // T-432: the Namboku line, Izumi-Chuo to Tomizawa, at its OSM station nodes (Asahigaoka placed
    // by hand); the water pair across the Hirose at Nagamachi, checked on the mask
    task: 'T-432', name: 'Sendai', centre: [140.8824, 38.2601], water: [[140.8868, 38.2420], [140.8868, 38.2345]],
    lineName: 'Namboku line test', colour: '#1a9e4b', view: [140.88, 38.26],
    line: [['Izumi-Chuo', 140.8807, 38.3232], ['Yaotome', 140.8838, 38.3132], ['Kuromatsu', 140.8864, 38.3031],
      ['Asahigaoka', 140.8830, 38.2955], ['Dainohara', 140.8784, 38.2883], ['Kita-Sendai', 140.8697, 38.2818],
      ['Kita-Yobancho', 140.8691, 38.2720], ['Kotodai-Koen', 140.8712, 38.2668], ['Hirose-dori', 140.8755, 38.2629],
      ['Sendai', 140.8800, 38.2598], ['Itsutsubashi', 140.8811, 38.2519], ['Atagobashi', 140.8817, 38.2474],
      ['Kawaramachi', 140.8879, 38.2407], ['Nagamachi-Itchome', 140.8876, 38.2343], ['Nagamachi', 140.8856, 38.2268],
      ['Nagamachi-Minami', 140.8771, 38.2247], ['Tomizawa', 140.8707, 38.2145]],
  },
  hij: {
    // T-432: the Astram line, Hondori to Omachi, at its OSM station nodes (Gion-shinbashi-kita left
    // out); the water pair across the Kyobashi river, checked on the mask
    task: 'T-432', name: 'Hiroshima', centre: [132.4757, 34.3975], water: [[132.4570, 34.3756], [132.4660, 34.3756]],
    lineName: 'Astram line test', colour: '#e4007f', view: [132.46, 34.40],
    line: [['Hondori', 132.4570, 34.3932], ['Kencho-mae', 132.4582, 34.3967], ['Johoku', 132.4588, 34.4054],
      ['Shin-Hakushima', 132.4603, 34.4076], ['Hakushima', 132.4627, 34.4109], ['Ushita', 132.4648, 34.4175],
      ['Fudoin-mae', 132.4689, 34.4266], ['Nishihara', 132.4748, 34.4434], ['Nakasuji', 132.4770, 34.4517],
      ['Furuichi', 132.4752, 34.4577], ['Omachi', 132.4705, 34.4617]],
  },
  ngo: {
    // T-431: the Higashiyama line, Nagoya to Motoyama, at its OSM station nodes; the water pair
    // across the Shonai river west of Nagoya Station, checked on the mask
    task: 'T-431', name: 'Nagoya', centre: [136.8815, 35.1709], water: [[136.8330, 35.1560], [136.8420, 35.1560]],
    lineName: 'Higashiyama line test', colour: '#f6ab00', view: [136.92, 35.17],
    line: [['Nagoya', 136.8839, 35.1712], ['Fushimi', 136.8974, 35.1692], ['Sakae', 136.9088, 35.1700],
      ['Shinsakae-machi', 136.9202, 35.1708], ['Chikusa', 136.9299, 35.1707], ['Imaike', 136.9383, 35.1698],
      ['Ikeshita', 136.9460, 35.1678], ['Kakuozan', 136.9529, 35.1666], ['Motoyama', 136.9637, 35.1639]],
  },
  osa: {
    // T-431: the Midosuji line, Shin-Osaka to Tennoji, at its OSM station nodes; the water pair
    // across the Yodo (Shin-Yodogawa) by Juso, checked on the mask
    task: 'T-431', name: 'Osaka', centre: [135.4959, 34.7025], water: [[135.4850, 34.7080], [135.4850, 34.7190]],
    lineName: 'Midosuji line test', colour: '#e5171f', view: [135.50, 34.69],
    line: [['Shin-Osaka', 135.4985, 34.7329], ['Nishinakajima-Minamigata', 135.4986, 34.7266], ['Nakatsu', 135.4968, 34.7110],
      ['Umeda', 135.4976, 34.7034], ['Yodoyabashi', 135.5010, 34.6916], ['Hommachi', 135.5007, 34.6835],
      ['Shinsaibashi', 135.5004, 34.6745], ['Namba', 135.5003, 34.6662], ['Daikokucho', 135.4979, 34.6562],
      ['Dobutsuen-mae', 135.5045, 34.6488], ['Tennoji', 135.5153, 34.6461]],
  },
  ttr: {
    // T-420 stage 3: a row made from the pack: the water pair across the mask's
    // nearest water to the origin, a 9-stop test line 1.2 km apart on the most populous heading
    task: 'T-420s3', name: 'Tottori', centre: [134.1719, 35.5187], water: [[134.1664, 35.5156], [134.1664, 35.5218]],
    lineName: 'Tottori test', colour: '#2a7ab0', view: [134.1719, 35.5187],
    line: [['Tottori 1', 134.2231, 35.5075], ['Tottori 2', 134.2103, 35.5103], ['Tottori 3', 134.1975, 35.5131], ['Tottori 4', 134.1847, 35.5159], ['Tottori 5', 134.1719, 35.5187], ['Tottori 6', 134.1591, 35.5215], ['Tottori 7', 134.1463, 35.5243], ['Tottori 8', 134.1335, 35.5271]],
  },
  kmk: {
    // T-420 stage 3: a row made from the pack: the water pair across the mask's
    // nearest water to the origin, a 9-stop test line 1.2 km apart on the most populous heading
    task: 'T-420s3', name: 'Kamikawa', centre: [142.3844, 43.7521], water: [[142.3604, 43.777], [142.3604, 43.7797]],
    lineName: 'Kamikawa test', colour: '#2a7ab0', view: [142.3844, 43.7521],
    line: [['Kamikawa 1', 142.4143, 43.7147], ['Kamikawa 2', 142.4068, 43.7241], ['Kamikawa 3', 142.3993, 43.7334], ['Kamikawa 4', 142.3919, 43.7428], ['Kamikawa 5', 142.3844, 43.7521], ['Kamikawa 6', 142.3769, 43.7614], ['Kamikawa 7', 142.3695, 43.7708], ['Kamikawa 8', 142.362, 43.7801], ['Kamikawa 9', 142.3545, 43.7895]],
  },
  oka: {
    // T-420 stage 3: a row made from the pack: the water pair across the mask's
    // nearest water to the origin, a 9-stop test line 1.2 km apart on the most populous heading
    task: 'T-420s3', name: 'Okinawa', centre: [127.7031, 26.2021], water: [[127.6805, 26.1965], [127.6875, 26.1965]],
    lineName: 'Okinawa test', colour: '#2a7ab0', view: [127.7031, 26.2021],
    line: [['Okinawa 1', 127.7031, 26.1589], ['Okinawa 2', 127.7031, 26.1697], ['Okinawa 3', 127.7031, 26.1805], ['Okinawa 4', 127.7031, 26.1913], ['Okinawa 5', 127.7031, 26.2021], ['Okinawa 6', 127.7031, 26.2129], ['Okinawa 7', 127.7031, 26.2237], ['Okinawa 8', 127.7031, 26.2345], ['Okinawa 9', 127.7031, 26.2453]],
  },
  szo: {
    // T-420 stage 3: a row made from the pack: the water pair across the mask's
    // nearest water to the origin, a 9-stop test line 1.2 km apart on the most populous heading
    task: 'T-420s3', name: 'Shizuoka', centre: [138.3781, 34.9729], water: [[138.3565, 34.9758], [138.362, 34.9758]],
    lineName: 'Shizuoka test', colour: '#2a7ab0', view: [138.3781, 34.9729],
    line: [['Shizuoka 1', 138.3645, 34.9312], ['Shizuoka 2', 138.3679, 34.9416], ['Shizuoka 3', 138.3713, 34.9521], ['Shizuoka 4', 138.3747, 34.9625], ['Shizuoka 5', 138.3781, 34.9729], ['Shizuoka 6', 138.3815, 34.9833], ['Shizuoka 7', 138.3849, 34.9937], ['Shizuoka 8', 138.3883, 35.0042], ['Shizuoka 9', 138.3917, 35.0146]],
  },
};

async function playwright() {
  if (process.env.PLAYWRIGHT_MODULE) return import(process.env.PLAYWRIGHT_MODULE);
  try { return await import('playwright'); } catch { /* not installed here */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright: set PLAYWRIGHT_MODULE to a file URL of playwright/index.mjs');
}

const id = process.argv[2];
// T-538: checks/pack_gate.py writes a row (OSM stations, a water pair from the mask) for any pack
const C = process.env.ROW ? JSON.parse(await readFile(process.env.ROW, 'utf8')) : CITIES[id];
if (!C) { console.error(`usage: city-browser.mjs <${Object.keys(CITIES).join('|')}>`); process.exit(2); }
const { chromium } = await playwright();
const BASE = process.env.BASE ?? 'http://localhost:6125';
const WORKERS = process.env.WORKERS ?? '3';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), `cb-${id}-`));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--enable-unsafe-swiftshader'] });
const out = process.env.OUT ?? `artifacts/trainworld/${C.task}-${id}-`;
const IGNORE = /favicon|openfreemap|AJAXError/;
const result = { city: id, checks: [] };
const check = (name, ok, detail) => { result.checks.push({ name, ok: !!ok, detail }); console.log(ok ? 'ok  ' : 'FAIL', name, JSON.stringify(detail ?? '')); };
const topBar = (page) => page.evaluate(() => ({ city: document.querySelector('.topbar .city, .city')?.textContent ?? null, split: document.querySelector('.split-row')?.getAttribute('data-tip') ?? null }));
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error' && !IGNORE.test(m.text())) errors.push(m.text()); });
  await page.goto(`${BASE}/?debug=1&new=1&paused=1&demandWorkers=${WORKERS}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client && window.twDemand.view()?.round >= 0, undefined, { timeout: 240000 });

  // 1. jump to the city: its pack opens, it becomes the active city, its boundary layer exists
  const t0 = await page.evaluate(ll => { window.tw.map.jumpTo({ center: ll, zoom: 11.2 }); return performance.now(); }, C.centre);
  await page.waitForFunction(id => window.twDemand.client.start.loads.find(l => l.item === id)?.openedAt, id, { timeout: 300000 });
  result.opening = await page.evaluate(([id, t0]) => { const l = window.twDemand.client.start.loads.find(l => l.item === id); return { download: Math.round(l.downloadedAt - l.requestedAt), open: Math.round(l.openedAt - l.downloadedAt), fromPan: Math.round(l.openedAt - t0) }; }, [id, t0]);
  check(`${C.name} opens at runtime`, true, result.opening);
  await page.waitForFunction(id => window.twDemand.activeCity.value === id, id, { timeout: 60000 });
  // T-400: one outline layer for every zone; the focused zone's own outline is the focus source
  const layer = await page.evaluate(async () => {
    for (let k = 0; k < 120; k++) {
      const d = await window.tw.map.getSource('tw-city-focus')?.getData?.();
      if (window.tw.map.getLayer('tw-city-boundaries') && (d?.coordinates?.length ?? 0) > 0) return true;
      await new Promise((r) => setTimeout(r, 500));
    }
    return false;
  });
  check(`${C.name} is the active city and its boundary layer exists`, layer, { layer });
  await page.evaluate(() => { const d = document.querySelector('#debug'); if (d) d.style.display = 'none'; window.tw.display.value = { ...window.tw.display.value, commuters: true, commuterEnd: 'home' }; });
  // the no-rail split: wait until the top bar has the city's own figures
  await page.waitForFunction(name => /Train .*Walking .*Driving/.test(document.querySelector('.split-row')?.getAttribute('data-tip') ?? '') && (document.querySelector('.split-row')?.getAttribute('data-tip') ?? '').includes(name) && !/–/.test(document.querySelector('.split-row .pct')?.textContent ?? '–'), C.name, { timeout: 300000 });
  await page.waitForTimeout(6000);
  result.noRail = { topBar: await topBar(page), cities: await page.evaluate(id => window.twDemand.cities().get(id) ?? null, id) };
  check(`the top bar names ${C.name} and gives its split`, (result.noRail.topBar.city ?? '').includes(C.name) && /Walking/.test(result.noRail.topBar.split ?? ''), result.noRail);
  await page.screenshot({ path: out + 'homes.png' });

  // 2. the water mask: ground track across the crossing refused, a tunnel accepted
  const water = await page.evaluate(async ([p, q]) => {
    const t = window.tw, a = t.lngLatToPack(p[0], p[1]), b = t.lngLatToPack(q[0], q[1]);
    const g = await t.client.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: 0 }, to: { kind: 'free', x: b.x, y: b.y, level: 0 }, pis: [], single: false });
    const u = await t.client.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: -2 }, to: { kind: 'free', x: b.x, y: b.y, level: -2 }, pis: [], single: false });
    if (u.ok) await t.client.edit({ op: 'undo' });
    return { ground: g.ok ? 'accepted' : JSON.stringify(g).slice(0, 200), tunnel: u.ok };
  }, C.water);
  check('ground track across the water refused, a tunnel accepted', /GroundOverWater/.test(water.ground) && water.tunnel, water);

  // 3. a test line through the centre, constructed
  const before = await page.evaluate(() => window.twDemand.view()?.version ?? 0);
  await page.evaluate(async ([stops, name, colour]) => {
    const t = window.tw, run = async op => { const r = await t.client.edit(op); if (!r.ok) throw Error(JSON.stringify({ op, r })); return r; };
    // a 300 m tail beyond each end, so every platform sits on a through node
    const pts = stops.map(s => t.lngLatToPack(s[1], s[2]));
    const tail = (a, b) => { const dx = a.x - b.x, dy = a.y - b.y, l = Math.hypot(dx, dy); return { x: a.x + 300 * dx / l, y: a.y + 300 * dy / l }; };
    const path = [tail(pts[0], pts[1]), ...pts, tail(pts.at(-1), pts.at(-2))];
    const nodes = [];
    for (let i = 1; i < path.length; i++) {
      const from = i === 1 ? { kind: 'free', x: path[0].x, y: path[0].y, level: -2 } : { kind: 'node', node: nodes.at(-1) };
      await run({ op: 'route', from, to: { kind: 'free', x: path[i].x, y: path[i].y, level: -2 }, pis: [], single: false });
      const e = t.world.value.edges.at(-1);
      if (i === 1) nodes.push(e.a);
      nodes.push(e.b);
    }
    const at = nodes.slice(1, -1);
    for (let i = 0; i < stops.length; i++) await run({ op: 'addStation', at: { kind: 'node', node: at[i] }, platform: 160, name: stops[i][0] });
    const line = (await run({ op: 'addLine', stops: at, name, colour })).line;
    await run({ op: 'setTrainCount', line, trains: [10, 10, 5] });
    await run({ op: 'constructAll' });
  }, [C.line, C.lineName, C.colour]);
  await page.waitForFunction(v => window.twDemand.view()?.version > v && window.twDemand.view()?.round === 3 && window.twDemand.view()?.final, before, { timeout: 900000 });
  const sums = await page.evaluate(name => {
    const dv = window.twDemand.view(), keys = [...dv.purposes.keys()], w = window.tw.world.value;
    const rel = (a, b) => Math.abs(a - b) / Math.max(1, Math.abs(b));
    const l = w.save.lines.find(l => l.name === name), d = dv.lines.get(l.id);
    const by = Object.fromEntries(keys.map(k => [k, Math.round(dv.purposes.get(k).lines.get(l.id)?.ridersPerDay ?? 0)]));
    const worst = rel(keys.reduce((s, k) => s + (dv.purposes.get(k).lines.get(l.id)?.ridersPerDay ?? 0), 0), d.ridersPerDay);
    const stations = Object.fromEntries(w.save.stations.map(s => [s.name, Math.round((dv.stations.get(s.id)?.boardings ?? 0))]));
    return { final: !!dv.final, riders: Math.round(d.ridersPerDay), by, worst, stations };
  }, C.lineName);
  result.line = sums;
  check(`the ${C.lineName} reaches a final solve with riders`, sums.final && sums.riders > 0, { final: sums.final, riders: sums.riders });
  check(`the ${C.lineName} carries every purpose, purposes add up`, ['work', 'shopping', 'social', 'school', 'medical'].every(k => sums.by[k] > 0) && sums.worst < 1e-4, sums.by);
  await page.waitForTimeout(4000);
  result.withLine = { topBar: await topBar(page), cities: await page.evaluate(id => window.twDemand.cities().get(id) ?? null, id) };
  check(`${C.name} has a mode split with the line`, result.withLine.cities, result.withLine);
  await page.evaluate(ll => window.tw.map.jumpTo({ center: ll, zoom: 12.6 }), C.view);
  await page.waitForTimeout(8000);
  await page.screenshot({ path: out + 'line.png' });
  result.errors = errors;
  check('no page errors', errors.length === 0, errors);
} finally {
  await writeFile(process.env.JSON_OUT ?? `artifacts/trainworld/${C.task}-${id}-browser.json`, JSON.stringify(result, null, 2));
  await ctx.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
const pass = result.checks.length === 8 && result.checks.every(c => c.ok);
console.log(pass ? `PASS city-browser ${id}` : `FAIL city-browser ${id}`);
process.exit(pass ? 0 : 1);
