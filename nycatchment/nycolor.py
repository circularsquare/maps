import pandas as pd
import matplotlib.pyplot as plt 
import geopandas as gpd
from shapely.geometry import Point
import colorsys
import random
from PIL import ImageColor



def getStationColor(routes):
    rtot, gtot, btot, stot = (0, 0, 0, 0)
    for route in routes:
        r, g, b = ImageColor.getcolor(colorDict[route], "RGB")
        rtot += r/255;
        gtot += g/255;
        btot += b/255;
        stot += colorsys.rgb_to_hsv(r/255, g/255, b/255)[1]
    r, g, b, s = (rtot / len(routes), gtot / len(routes), btot / len(routes), stot / len(routes))
    h, sMerged, v = colorsys.rgb_to_hsv(r, g, b)
    
    noiseFactor = 0.06
    if len(routes) > 4:
        noiseFactor = 0.01
    elif routes[0] in ('B', 'D', 'F', 'M', 'N', 'Q', 'R', 'W'):
        noiseFactor = 0.045
    elif routes[0] in ('A', 'C', 'E', '1', '2', '3', '7', 'G', 'SIR'):
        noiseFactor = 0.07
    elif routes[0] in ('4', '5', '6'):
        noiseFactor = 0.08
    elif routes[0] in ('L'):
        noiseFactor = 0.12
    vnoiseFactor = 0.02
    if routes[0] == 'L':
        vnoiseFactor = 0.05
    hnew = h + ((random.random()-0.5)*noiseFactor)%1
    snew = min(max((s + sMerged)/2 + (random.random()-0.5)*0.1 - 0, 0), 1)
    vnew = min(max(v + (random.random()-0.5)*vnoiseFactor - 0, 0), 1)
    r, g, b = colorsys.hsv_to_rgb(hnew, snew, vnew)
    hex = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
    return hex

colorDict = {
    # NYC Subway (your existing colors - keep as is)
    'A': '#005ba6', 'C': '#0039A6', 'E': '#0042a6', 
    'B': '#ff6c26', 'D': '#ff8426', 'F': '#FF6319', 'M': '#ff7519', 
    'G': '#6CBE45', 'L': '#A7A9AC', 
    'J': '#996633', 'Z': '#945e28',   
    'N': '#FCCC0A', 'Q': '#fcd40a', 'R': '#fcdc0a', 'W': '#fcf40a', 
    '1': '#e32b24', '2': '#EE352E', '3': '#ee2e4b', 
    '4': '#00a336', '5': '#00a357', '6': '#00933C', 
    '7': '#B933AD', 'S': '#808183', 'SIR': '#0039A6',
    
    # NJ Transit Rail Lines - UPDATED TO MATCH CONVENTION
    'Northeast Corridor': '#CE1126',        # RED (official NEC red)
    'NEC': '#CE1126',
    'NEC / NJCL': '#8B2F5F',
    'NEC / NJCL / ME / Main/Bergen': '#942D5E',
    'NEC / NJCL / RVL': '#9B3068',
    
    'Raritan Valley Line': '#FAA634',       # ORANGE (official RVL orange)
    'RVL': '#FAA634',
    
    'Morris and Essex Line': '#00A94F',     # GREEN (official M&E green) - Morristown
    'ME': '#00A94F',
    'Morristown Line / ME': '#00A94F',
    'Montclair-Boonton Line / ME': '#1A8A42',
    
    'Gladstone Branch / ME': '#8BC540',     # PALE GREEN (official Gladstone)
    
    'Main/Bergen Line': '#FFD100',          # YELLOW (official Main Line)
    'Main Line': '#FFD100',
    'ML': '#FFD100',
    
    'Bergen Line': '#808080',               # GRAY (Bergen Line conventional)
    'Bergen': '#808080',
    
    'Pascack Valley Line': '#8B4789',
    'Pascack Valley': '#8B4789',
    'PVL': '#8B4789',
    
    'North Jersey Coast Line': '#0076BF',
    'NJCL': '#0076BF',
    
    'Atlantic City Line': '#00A5E3',
    'Meadowlands Rail Line': '#003D79',
    'West Trenton Line': '#8E3A80',
    
    # NJ Light Rail
    'Hudson-Bergen Light Rail': '#009FDA',
    'HBLR': '#009FDA',
    'Newark Light Rail': '#FDDA24',
    'River Line Light Rail': '#7C3F8D',
    
    # PATH
    'PATH': '#D93A30',
    
    # PATCO
    'PATCO': '#EE3E34',
    
    # Metro-North Lines (official colors)
    'Hudson': '#009B3A',
    'Harlem': '#0039A6',
    'New Haven': '#EE3124',
    'Wassaic': '#0039A6',
    'Waterbury': '#EE3124',
    'Danbury': '#EE3124',
    'New Canaan': '#EE3124',
    'MNR': '#0039A6',
    
    # LIRR Branches - UPDATED TO MATCH CONVENTION
    'Babylon': '#00985F',
    'Belmont': '#0039A6',
    'City Terminal Zone': '#4D5357',
    'Far Rockaway': '#6E3219',
    'Hempstead': '#CE8E00',
    'Long Beach': '#FF6319',
    'Montauk': '#006983',
    
    'Oyster Bay': '#00A94F',                # GREEN (official)
    
    'Port Jefferson': '#0062A0',            # BLUE (official)
    
    'Port Washington': '#C60C30',
    'Ronkonkoma': '#A626AA',
    'West Hempstead': '#00AF3F',
    'Greenport': '#006983',
    'LIRR': '#0039A6',
}

subway = pd.read_csv('data/nyc/subwaystations.csv')
subway = subway[['Daytime Routes', 'Latitude', 'Longitude', 'population']]

subway['routes'] = subway['Daytime Routes'].apply(lambda x: x.split(' '))
subway['color'] = subway.routes.apply(getStationColor)

nyrail = pd.read_csv('data/nyc/MTA_Rail_Stations.csv')
nyrail = nyrail.drop(['Parking Map URL', 'Accessibility', 'Railroad', 'Zone', 'Station URL', 'Outbound Title', 'Inbound Title'], axis=1)
print(nyrail.columns)
nyrail['route'] = nyrail.Branch;
nyrail = nyrail[['route', 'Latitude', 'Longitude']]
{''}

njt = gpd.read_file('data/nyc/NJ_Passenger_Rail_Stations.csv')
geometry = [Point(xy) for xy in zip(njt['X'], njt['Y'])]
gdf = gpd.GeoDataFrame(njt, geometry=geometry, crs='EPSG:3424')
gdf = gdf.to_crs('EPSG:4326')
njt['Longitude'] = gdf.geometry.x
njt['Latitude'] = gdf.geometry.y
njt['route'] = njt['RAIL_LINE']
njt = njt[['route', 'Longitude', 'Latitude']]

rail = pd.concat([njt, nyrail])
print(rail.iloc[0])
print(rail.route.unique())

rail['population'] = 0
rail['routes'] = rail.route.apply(lambda x: [x])
rail['color'] = rail.routes.apply(getStationColor);

rail.to_csv('data/nyc/rail.csv')
allrail = pd.concat([subway, rail])
allrail.to_csv('data/nyc/allrail.csv')