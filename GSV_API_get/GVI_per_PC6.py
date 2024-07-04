globals().clear()

import geopandas as gpd
import math
import numpy as np
import pandas as pd

totGVI_DeepLab = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\GVI_to_SHP\totGVI_SEG.geojson')
totGVI_Pymean = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\GVI_to_SHP\totGVI_PYMEANSHIFT.geojson')

PC6 = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\Tree_species\PC6_只有两个.geojson')\
    .drop(columns=['COUNT_OBJECTID','COUNT_OBJECTID_1','COUNT_OBJECTID_12'])
PC6_10m = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\Tree_species\PC6_两个_10mBuffer.geojson')\
    .drop(columns=['COUNT_OBJECTID','COUNT_OBJECTID_1','COUNT_OBJECTID_12'])

"""
# DeepLab
"""
# 10m buffer - 469个PC6
inter_DeepLab = gpd.sjoin(PC6_10m,totGVI_DeepLab ,how='inner',predicate = 'intersects').drop(columns=['index_right','OBJECTID_right'])
# 对每一个PC6，取所有点的GVI的mean
csv_DeepLab_10m = inter_DeepLab.groupby(by='Postcode',as_index=False).agg({'totGVI':'mean'})
# join with original PC6 - 恢复原先的大小！！！！再intersect一次
totGVI_DeepLab_PC6 = PC6[['Postcode','geometry']].merge(csv_DeepLab_10m,how='left',on = 'Postcode')
# fill na with 0
totGVI_DeepLab_PC6['totGVI'] = totGVI_DeepLab_PC6['totGVI'].fillna(0)

totGVI_DeepLab_PC6.to_file(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\处理好的\totGVI_DeepLab_PC6.geojson', driver="GeoJSON")

"""
# Pymeanshift
"""
# 10m buffer - 469个PC6
inter_Pymean = gpd.sjoin(PC6_10m,totGVI_Pymean,how='inner',predicate = 'intersects').drop(columns=['index_right','OBJECTID_right'])
# 对每一个PC6，取所有点的GVI的mean
csv_Pymean_10m = inter_Pymean.groupby(by='Postcode',as_index=False).agg({'totGVI':'mean'})
# join with original PC6 - 恢复原先的大小！！！！再intersect一次
totGVI_Pymean_PC6 = PC6[['Postcode','geometry']].merge(csv_Pymean_10m,how='left',on = 'Postcode')
# fill na with 0
totGVI_Pymean_PC6['totGVI'] = totGVI_Pymean_PC6['totGVI'].fillna(0)

totGVI_Pymean_PC6.to_file(r'D:\WUR\master\MLP\master thesis\data\GSV_API_GET\处理好的\totGVI_Pymean_PC6.geojson', driver="GeoJSON")