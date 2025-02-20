import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import geopandas
from cartopy import crs as ccrs 
import cartopy.feature as cfeature
import cartopy
import os
from geodatasets import get_path
import colorsys
import random
from PIL import ImageColor
import re
import cartopy.io.shapereader as shpreader


df = pd.read_csv('faostat/FAOSTAT_data_en_11-21-2024.csv')
print(df.sort_values('Value').groupby('Area').last()[['Item', 'Value']])

print(df.sort_values('Value').groupby('Item').last()[['Area', 'Value']])