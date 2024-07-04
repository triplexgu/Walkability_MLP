globals().clear()

import geopandas as gpd
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PC6 = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\Tree_species\PC6_只有两个.geojson')\
    .drop(columns=['COUNT_OBJECTID','COUNT_OBJECTID_1','COUNT_OBJECTID_12'])
# 526个PC6 parcel！！！
PC6_5m = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\Tree_species\PC6_两个_5mBuffer.geojson')\
    .drop(columns=['COUNT_OBJECTID','COUNT_OBJECTID_1','COUNT_OBJECTID_12'])
PC6_10m = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\Tree_species\PC6_两个_10mBuffer.geojson')\
    .drop(columns=['COUNT_OBJECTID','COUNT_OBJECTID_1','COUNT_OBJECTID_12'])

Bomen = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\Tree_species\Bomen_只有两个.geojson')[['OBJECTID','Categorie', 'geometry']]

"""
先算intersection
"""
# 10m buffer - 460个
inter = gpd.sjoin(PC6_10m,Bomen,how='inner',predicate = 'intersects').drop(columns=['index_right','OBJECTID_right'])
#test_10m = inter.dissolve(by='Postcode',as_index=False)

# 对每一个PC6 parcel - 计算frequency，总物种数量
PCs = inter['Postcode'].unique().tolist() #460个
SDI_calcu = inter.copy()
SDI_calcu['frequency'] = 0

frequency_full = pd.DataFrame()

for parcel in PCs:
    parcel_df = SDI_calcu[SDI_calcu['Postcode']==parcel]
    tot_num_indi = len(parcel_df)
    # 先变成dataframe，之后再合并
    frequency = parcel_df.groupby(by=['Postcode','Categorie'],as_index=False).agg({'frequency':'count'})
    frequency_full = pd.concat([frequency_full,frequency])
    SDI_calcu.loc[SDI_calcu['Postcode']==parcel,'tot_num_indi'] = tot_num_indi

SDI_calcu = SDI_calcu.dissolve(by=['Postcode', 'naam', 'code', 'MAX_name','Categorie'],
                               as_index=False,
                               aggfunc={'frequency':'first','tot_num_indi':'first'})
SDI_calcu = SDI_calcu.merge(frequency_full[['Postcode','Categorie','frequency']],
                            left_on=['Postcode','Categorie'],right_on=['Postcode','Categorie'],how='left').\
    drop(columns=['frequency_x'])

# 对单个物种的shannon
def shannon_calcu(target_frequency,tot_num_indi):
    # STEP 1: 计算各个物种在每个block内的proportion
    prop = target_frequency/tot_num_indi # 单个物种的出现数/总个数
    # STEP 2: Calculate the Natural Log of the Proportions
    log_calcu = math.log(prop)
    # STEP 3: Multiply the Proportions by the Natural Log of the Proportions
    muti_prop = prop * log_calcu
    # STEP 4: Take the reverse
    SDI_single = muti_prop * (-1)
    return SDI_single

for idx, row in SDI_calcu.iterrows():
    target_frequency = row['frequency_y']
    tot_num_indi = row['tot_num_indi']
    SDI_calcu.at[idx,'SDI_indi'] = shannon_calcu(target_frequency,tot_num_indi)

# 把所有Shannon diversity index加起来，计算Shannon diveristy index
SDI_tot = SDI_calcu.copy()

# 注意：evenness和diveristy是两个东西！evenness需要在0，1之间，但diveristy完全可以超过1！！！！
SDI_tot = SDI_tot.dissolve(by=['Postcode'],as_index=False,aggfunc={'SDI_indi':'sum','tot_num_indi':'first'})
# 恢复原先的大小！！！！再intersect一次
SDI_tot_back = PC6[['Postcode','geometry']].merge(SDI_tot[['Postcode','SDI_indi']],how='left',on = 'Postcode')
# fill 没有树的地方用0
SDI_tot_back['SDI_indi'] = SDI_tot_back['SDI_indi'].fillna(0)

SDI_tot_back.to_file(r'D:\WUR\master\MLP\master thesis\data\Tree_species\处理好的\PC6_species_diversity.geojson', driver="GeoJSON")