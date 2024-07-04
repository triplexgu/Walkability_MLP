globals().clear()

import geopandas as gpd

splitGVI_DeepLab = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\VS_DeepLabV3_NEW.geojson')
splitGVI_Pymean = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\GVI_Pymeanshift.geojson')

PC6 = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\数据汇总\PC6_只有两个.geojson')\
    .drop(columns=['COUNT_OBJECTID','COUNT_OBJECTID_1','COUNT_OBJECTID_12'])
PC6_10m = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\数据汇总\PC6_只有两个_10m.geojson')\
    .drop(columns=['COUNT_OBJECTID','COUNT_OBJECTID_1','COUNT_OBJECTID_12'])

"""
# DeepLab
"""
# 10m buffer - 469个PC6
inter_DeepLab = gpd.sjoin(PC6_10m,splitGVI_DeepLab ,how='inner',predicate = 'intersects').drop(columns=['index_right'])
#inter_DeepLab['top'] = inter_DeepLab['top'].apply(lambda x: x/5)
#inter_DeepLab['low'] = inter_DeepLab['low'].apply(lambda x: x/5)
# 对每一个PC6，取所有点的GVI的mean
csv_DeepLab_10m = inter_DeepLab.groupby(by='Postcode',as_index=False).agg({'top_right': 'mean',
                                                                           'top_middle': 'mean',
                                                                           'top_left': 'mean',
                                                                           'mid_up_right': 'mean', 'mid_up_middle': 'mean',
                                                                          'mid_up_left': 'mean',
                                                                          'mid_down_right': 'mean', 'mid_down_middle': 'mean',
                                                                          'mid_down_left': 'mean',
                                                                           'low_right': 'mean',
                                                                           'low_middle': 'mean',
                                                                           'low_left': 'mean',
                                                                            'top': 'mean',
                                                                           'low': 'mean'})
# join with original PC6 - 恢复原先的大小！！！！再intersect一次
totGVI_DeepLab_PC6 = PC6[['Postcode','geometry']].merge(csv_DeepLab_10m,how='left',on = 'Postcode')
# fill na with 0
totGVI_DeepLab_PC6 = totGVI_DeepLab_PC6.fillna(0)

totGVI_DeepLab_PC6.to_file(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\处理好的\VS_DeepLab_PC6.geojson', driver="GeoJSON")
totGVI_DeepLab_PC6.to_file(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\处理好的\NEW_VS_DeepLab_PC6.geojson', driver="GeoJSON")

a = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\处理好的\VS_DeepLab_PC6.geojson')
print(a['top'].max())

"""
# Pymeanshift
"""
# 10m buffer - 469个PC6
inter_Pymean = gpd.sjoin(PC6_10m,splitGVI_Pymean,how='inner',predicate = 'intersects').drop(columns=['index_right'])
inter_Pymean['top'] = inter_Pymean['top'].apply(lambda x: x/5)
inter_Pymean['low'] = inter_Pymean['low'].apply(lambda x: x/5)
# 对每一个PC6，取所有点的GVI的mean
csv_Pymean_10m = inter_Pymean.groupby(by='Postcode',as_index=False).agg({'top_right': 'mean',
                                                                           'top_middle': 'mean',
                                                                           'top_left': 'mean',
                                                                           'middle_right': 'mean',
                                                                            'middle_middle': 'mean',
                                                                           'middle_left': 'mean',
                                                                           'low_right': 'mean',
                                                                           'low_middle': 'mean',
                                                                           'low_left': 'mean',
                                                                            'top': 'mean',
                                                                           'middle': 'mean',
                                                                           'low': 'mean'})
# join with original PC6 - 恢复原先的大小！！！！再intersect一次
totGVI_Pymean_PC6 = PC6[['Postcode','geometry']].merge(csv_Pymean_10m,how='left',on = 'Postcode')
# fill na with 0
totGVI_Pymean_PC6= totGVI_Pymean_PC6.fillna(0)

totGVI_Pymean_PC6.to_file(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\处理好的\VS_Pymean_PC6.geojson', driver="GeoJSON")
totGVI_Pymean_PC6.to_file(r'D:\WUR\master\MLP\master thesis\data\数据汇总\VegetationStructure\VS_Pymean_PC6.geojson', driver="GeoJSON")