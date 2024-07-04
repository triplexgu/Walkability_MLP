globals().clear()

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Load the GeoDataFrame
cluster_df = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\数据汇总\PC6_corr_df.geojson')[['totGVI', 'VSD', 'top', 'low',
       'vitality_level', 'noise_level','building_height', 'density', 'width_index','geometry','Postcode','avg_WI']]
# Fill missing values with 0 for 'noise_level' and 'vitality_level'
cluster_df['noise_level'] = cluster_df['noise_level'].fillna(0)
cluster_df['vitality_level'] = cluster_df['vitality_level'].fillna(0)

## ------------提取每个cluster对应的PC6名字
cluster = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\Comparison_sites\result_cluster.geojson')

## ------------根据PC6merge cluster
cluster_df = cluster_df.merge(cluster[['Postcode','Cluster']],left_on='Postcode',right_on='Postcode',how='left').drop(columns=['geometry','Postcode'])

# for each cluster, calculate each columns' mean
mean_values = cluster_df.groupby('Cluster').mean()
mean_values.to_csv(r'D:\WUR\master\MLP\master thesis\data\数据汇总\comparison_analysis\cluster_结果_原始数据.csv')