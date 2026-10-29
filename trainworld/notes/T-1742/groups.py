# -*- coding: utf-8 -*-
import json, re, sys, collections
d = 'C:/Users/anita/projects/maps/artifacts/trainworld/t1742/'
S = lambda x: sum(x) if isinstance(x, list) else (x or 0)
def load(c): return json.load(open(d + c + '.json', encoding='utf-8'))
def grp(c, f):
    j = load(c); g = collections.OrderedDict()
    for l in j['lines']:
        k = f(l['name'] or '')
        t = g.setdefault(k, [0, 0.0, 0.0, 0.0]); t[0] += 1; t[1] += l['riders']['0']; t[2] += S(l['pkm']['0']); t[3] += l['km']
    tot = [sum(t[i] for t in g.values()) for i in range(4)]
    print('==', c, 'rail trips %.0f' % j['days'][0]['railTripsPerDay'], 'line riders %.0f pkm %.0f  pkm/rider %.2f km' % (tot[1], tot[2], tot[2] / tot[1]))
    for k, t in g.items():
        print('   %-36s lines %3d riders %10.0f pkm %12.0f pkm/rider %5.2f' % (k, t[0], t[1], t[2], t[2] / t[1] if t[1] else 0))
def nyc(n):
    if re.fullmatch(r'[1-7]|[A-Z]|[67]X|FX', n): return 'subway'
    if n in ('PATH',): return 'PATH'
    if n in ('SIR',): return 'SIR'
    if n in ('HBLR', 'NLR', 'RVLN', 'PRIN'): return 'NJT light rail'
    if n in ('NEC', 'NJCL', 'NJCLL', 'MNE', 'MNEG', 'MNBN', 'MNBNP', 'BNTN', 'BNTNM', 'PASC', 'RARV'): return 'NJT rail'
    if n in ('Hudson', 'Harlem', 'New Haven', 'New Canaan', 'Danbury', 'Waterbury'): return 'Metro-North'
    return 'LIRR'
def lon(n):
    if n in ('Northern', 'Central', 'Victoria', 'Jubilee', 'Piccadilly', 'District', 'Metropolitan', 'Bakerloo', 'Circle', 'Waterloo & City', 'Hammersmith & City'): return 'Tube'
    if n == 'DLR': return 'DLR'
    if n == 'Elizabeth': return 'Elizabeth'
    if n in ('windrush', 'weaver', 'mildmay', 'lioness', 'suffragette', 'liberty'): return 'Overground'
    return 'National Rail'
def par(n):
    if re.fullmatch(r'\d+B?', n): return 'Metro'
    if n in ('A', 'B', 'C', 'D', 'E'): return 'RER'
    if re.fullmatch(r'T\d+[ab]?', n): return 'Tram'
    if n in ('ORLYVAL', 'CDG VAL'): return 'VAL'
    return 'Transilien'
def tor(n):
    if n in ('1', '2', '4'): return 'TTC subway ' + n
    if n in ('5', '6'): return 'TTC light rail ' + n
    if re.fullmatch(r'5\d\d', n): return 'TTC streetcar'
    if n == 'UP': return 'UP Express'
    return 'GO rail'
def seo(n):
    m = re.fullmatch(r'Line (\d)', n)
    if m: return 'Seoul Lines 1-9'
    return 'other lines'
def sdj(n):
    if '\u5357\u5317\u7dda' in n or '\u6771\u897f\u7dda' in n: return 'Sendai subway'
    if n.startswith('JR'): return 'JR lines'
    return 'other (SAR, Abukuma Exp.)'
METRO = ['銀座線', '丸ノ内線（１）', '丸ノ内線（２）', '日比谷線', '東西線', '千代田線', '有楽町線', '半蔵門線', '南北線', '副都心線']
TOEI = ['大江戸線', '浅草線', '三田線']
JR = ['山手線', '京浜東北・根岸線', '総武線各駅停車', '埼京線', '南武線', '中央本線', '武蔵野線', '総武本線', '上野東京ライン（1）', '上野東京ライン（2）', '横浜線', '京葉線（１）', '京葉線（２）', '京葉線（３）', '湘南新宿ライン', '東海道本線', '横須賀線', '常磐線各駅停車', '常磐線快速', '高崎線', '東北本線', '青梅線', '相模線', '川越線', '内房線', '外房線', '南武支線', '八高線', '国分寺線', '千葉線', '五日市線', '鶴見線（１）', '鶴見線（２）', '鶴見線（３）', '成田線（１）', '成田支線（１）', '成田支線（２）', '東金線', '水戸線', '御殿場線', '久留里線', '豊島線', '西武園線', '大師線']
def tyo(n):
    if n in METRO: return 'Tokyo Metro'
    if n in TOEI: return 'Toei subway (Oedo, Asakusa, Mita)'
    if n in JR: return 'JR lines'
    return 'private and other'
for c, f in [('nyc', nyc), ('lon', lon), ('par', par), ('seo', seo), ('tor', tor), ('sdj', sdj), ('tyo', tyo)]:
    grp(c, f)
