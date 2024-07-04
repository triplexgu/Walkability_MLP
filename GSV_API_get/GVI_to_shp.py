# combine all the poi parts df into one
globals().clear()

import pandas as pd
# SEG的结果
P1 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\calculated_GVI_SEG\P1_SEG.csv')
P2 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\calculated_GVI_SEG\P2_SEG.csv')
P3 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\calculated_GVI_SEG\P3_SEG.csv')
P4 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\calculated_GVI_SEG\P4_SEG.csv')
P5 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\calculated_GVI_SEG\P5_SEG.csv')
P6 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\calculated_GVI_SEG\P6_SEG.csv')
P7 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\calculated_GVI_SEG\P7_SEG.csv')

# pymeanshift的结果
P1 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\calculated_totGVI_pymeanshift\P1.csv')
P2 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\calculated_totGVI_pymeanshift\P2_P3B1.csv')
P3 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\calculated_totGVI_pymeanshift\P3B2.csv')
P4 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\calculated_totGVI_pymeanshift\P4.csv')
P5 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\calculated_totGVI_pymeanshift\P5.csv')
P6 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\calculated_totGVI_pymeanshift\P6B2.csv')
P7 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Helsinki_GreenView-master\calculated_totGVI_pymeanshift\P7B2.csv')

coords = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\Sampling_coords.csv')

# generate shp based on combined df
import geopandas as gpd
import glob
import os
from geopandas import GeoDataFrame
from shapely.geometry import Point
import numpy as np
import pycrs
import matplotlib.pyplot as plt

# combine two dfs
POIs = pd.DataFrame()
dfs = [POIs,P1,P2,P3,P4,P5,P6,P7]
POIs = pd.concat(dfs,axis=0).drop(columns=['GVI','part_id','batch_id'])
combined = pd.merge(POIs,coords[['OBJECTID','X','Y']],left_on='object_id',right_on='OBJECTID')
# X:lon Y:lat

#wanted objects to float
for col in combined.columns:
    if col != 'batch_id':
        combined[col] = combined[col].astype(float)

# set the wanted crs
crs = {'init': 'epsg:4326'} # RD-NEW
#create coordinate object for the geodataframe
geometry = [Point(xy) for xy in zip(combined.X, combined.Y)]

#create a geodataframe
totGVI = gpd.GeoDataFrame(combined, geometry = geometry)

#SEG
#totGVI.to_file(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\GVI_to_SHP\totGVI.geojson',drive='GeoJson',encoding='utf-8')

#PYMEANSHIFT
totGVI.to_file(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\GVI_to_SHP\totGVI_PYMEANSHIFT.geojson',drive='GeoJson',encoding='utf-8')
#check how it looks
print(totGVI.head())

#Geodata.plot()
#plt.show()