# combine all the poi parts df into one
globals().clear()

import pandas as pd
# SEG的结果
P1 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\P1_GVI_DeepLabV3.csv')
P2 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\P2_GVI_DeepLabV3.csv')
P3 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\P3_GVI_DeepLabV3.csv')
P4 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\P4_GVI_DeepLabV3.csv')
P5 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\P5_GVI_DeepLabV3.csv')
P6 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\P6_GVI_DeepLabV3.csv')
P7 = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\P7_GVI_DeepLabV3.csv')

print(P1['top'].max())
print(P2['top'].max())
print(P3['top'].max())
print(P4['top'].max())
print(P5['top'].max())
print(P6['top'].max())
print(P7['top'].max())

# pymeanshift的结果
P1_pm = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\Pymeanshift\P1_GVI_pymeanshift.csv')
P2_pm = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\Pymeanshift\P2_GVI_pymeanshift.csv')
P3_pm = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\Pymeanshift\P3_GVI_pymeanshift.csv')
P4_pm = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\Pymeanshift\P4_GVI_pymeanshift.csv')
P5_pm = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\Pymeanshift\P5_GVI_pymeanshift.csv')
P6_pm = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\Pymeanshift\P6_GVI_pymeanshift.csv')
P7_pm = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\Pymeanshift\P7_GVI_pymeanshift.csv')

coords = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\POI_25m_withAngles.csv')

# generate shp based on combined df
import geopandas as gpd
from shapely.geometry import Point

# combine two dfs
POIs = pd.DataFrame()
dfs = [POIs,P1,P2,P3,P4,P5,P6,P7]
POIs = pd.concat(dfs,axis=0).drop(columns=['Unnamed: 0'])
combined = pd.merge(POIs,coords[['TARGET_FID','X','Y']],left_on='object_id',right_on='TARGET_FID')
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
VS = gpd.GeoDataFrame(combined, geometry = geometry)

#SEG
VS.to_file(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\VS_DeepLabV3_NEW.geojson',drive='GeoJson',encoding='utf-8')
a = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\VS_DeepLabV3_NEW.geojson')
print(a['top'].max())

#PYMEANSHIFT
totGVI.to_file(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\GVI_DeepLabV3.geojson',drive='GeoJson',encoding='utf-8')
totGVI.to_file(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\GVI_Pymeanshift.geojson',drive='GeoJson',encoding='utf-8')
#check how it looks
print(totGVI.head())

#Geodata.plot()
#plt.show()