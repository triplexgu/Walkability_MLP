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

# Extract numeric columns
numeric_columns = cluster_df.select_dtypes(include=['float64', 'int64']).columns

# Calculate quantiles
quantiles = cluster_df[numeric_columns].quantile([0.25, 0.5, 0.75])

# Save quantiles to CSV
quantiles.to_csv(r'D:\WUR\master\MLP\master thesis\data\数据汇总\quantiles_summary.csv')