globals().clear()

import geopandas as gpd

"""
# independent variables
"""
# 加载PC6信息
"""
PC6 = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\数据汇总\PC6_只有两个.geojson') \
    .drop(columns=['COUNT_OBJECTID', 'COUNT_OBJECTID_1', 'COUNT_OBJECTID_12'])
PC6_10m = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\数据汇总\PC6_只有两个_10m.geojson') \
    .drop(columns=['COUNT_OBJECTID', 'COUNT_OBJECTID_1', 'COUNT_OBJECTID_12', 'BUFF_DIST', 'ORIG_FID'])
"""
PC6 = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\数据汇总\PC6_只有两个.geojson') \
    .drop(columns=['COUNT_OBJECTID', 'COUNT_OBJECTID_1', 'COUNT_OBJECTID_12'])
PC6_10m = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\数据汇总\PC6_只有两个_10m.geojson') \
    .drop(columns=['COUNT_OBJECTID', 'COUNT_OBJECTID_1', 'COUNT_OBJECTID_12', 'BUFF_DIST', 'ORIG_FID'])

# -------------------NDVI (496,2)
PC6_NDVI = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\数据汇总\NDVI\meanNDVI_perPC6.geojson')
PC6_NDVI = PC6[['Postcode', 'geometry']]. \
    merge(PC6_NDVI[['Postcode', 'mean_NDVI']], how='left', left_on='Postcode', right_on='Postcode')
PC6_NDVI = PC6_NDVI.fillna(0)[['Postcode', 'mean_NDVI']]

# -------------------totGVI
# DeepLabV3+ (496,2)
PC6_totGVI_DeepLab = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\数据汇总\totGVI\totGVI_DeepLab_PC6.geojson'). \
    dissolve(by='Postcode', as_index=False, aggfunc='first')[['totGVI', 'Postcode']]
PC6_totGVI_DeepLab = PC6_totGVI_DeepLab.drop(PC6_totGVI_DeepLab[PC6_totGVI_DeepLab['Postcode'] == '1017TP'].index)
# Pymeanshift

# -------------------VSD （496,2）
PC6_VSD = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\数据汇总\VSD\PC6_species_diversity.geojson'). \
    dissolve(by='Postcode', as_index=False, aggfunc='first')[['Postcode', 'SDI_indi']]
PC6_VSD = PC6_VSD.drop(PC6_VSD[PC6_VSD['Postcode'] == '1017TP'].index)
PC6_VSD = PC6_VSD.rename(columns={'SDI_indi': 'VSD'})

# -------------------Vegetation Structure
# DeepLabV3+ (496,4)
PC6_VS_DeepLab = \
gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\Vegetation_Structure\GSV_split\处理好的\VS_DeepLab_PC6.geojson'). \
    dissolve(by='Postcode', as_index=False, aggfunc='first')[['Postcode', 'top', 'low']]
PC6_VS_DeepLab = PC6_VS_DeepLab.drop(PC6_VS_DeepLab[PC6_VS_DeepLab['Postcode'] == '1017TP'].index)

# 记得这边要改一下！pymeanshift的算法还有问题没有修改完！
# Pymeanshift

"""
# dependent variable
"""
# -------------------PC6 Walkability Index (496,3) - 有geometry
PC6_walkabilityindex = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\数据汇总\PC6_AMS_WI.geojson')
PC6_walkabilityindex = PC6_walkabilityindex.rename(columns={'PC6':'Postcode'})
# 把WI和PC6合起来
PC6_WI_SITE = PC6[['Postcode', 'geometry']].merge(PC6_walkabilityindex[['Postcode', 'avg_WI']].drop_duplicates(subset='Postcode'),
                                                   how='left', on='Postcode')
# 去掉不太对的那个PC
PC6_walkabilityindex = PC6_walkabilityindex.drop(PC6_walkabilityindex[PC6_walkabilityindex['Postcode'] == '1017TP'].index)

# -------------------Pearson correlation between greenery variables
covariates = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\数据汇总\covariates\PC6_covariates.geojson').\
    drop(columns={'geometry',
       'INWONER', 'MAN', 'VROUW', 'M_INKHH', 'WOZWONING', 'INW_1524',
       'INW_2544', 'INW_4564', 'INW_65PL', 'P_NL_ACHTG', 'P_WE_MIG_A',
       'P_NW_MIG_A'})
print(covariates.columns)
dfs = [PC6_WI_SITE, PC6_NDVI, PC6_totGVI_DeepLab, PC6_VSD, PC6_VS_DeepLab, covariates]
from functools import reduce
import pandas as pd
gam_df = reduce(lambda left, right: pd.merge(left, right, on=['Postcode'],how='left'), dfs)
gam_df.to_file(r'D:\WUR\master\MLP\master thesis\data\数据汇总\PC6_corr_df.geojson', driver='GeoJSON')

"""
# new df for correlation
"""
globals().clear()

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pygam import LogisticGAM
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

cor_df = gpd.read_file(r'D:\WUR\master\MLP\master thesis\data\数据汇总\PC6_corr_df.geojson').rename(columns={'avg_WI':'WalkabilityIndex'}).\
    drop(columns='geometry')
cor_df['noise_level'] = cor_df['noise_level'].fillna(0)
cor_df['vitality_level'] = cor_df['vitality_level'].fillna(0)
# Function to identify outliers
def identify_outliers(column):
    Q1 = np.percentile(column, 25)
    Q3 = np.percentile(column, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (column < lower_bound) | (column > upper_bound)

a = ['WalkabilityIndex', 'mean_NDVI','totGVI', 'VSD', 'top', 'low',
       'noise_level', 'building_height', 'density', 'width_index','net_parking_pressure']
## 不要对vitality level和noise level去除异常值！！！！！！！
# Replace outliers with mean
for col in a:
    outliers = identify_outliers(cor_df[col])
    cor_df.loc[outliers, col] = cor_df[col].mean()

class_mean = cor_df['class_index'].mean()
width_mean = cor_df['width_index'].mean()
cor_df['class_index'] = cor_df['class_index'].fillna(class_mean)
cor_df['width_index'] = cor_df['width_index'].fillna(width_mean)

# STEP 1
# 1. CHECK CORRELATION MATRIX
correlation_matrix = cor_df.iloc[:,2:].corr()
plt.figure(figsize=(10, 8))  # Set the figure size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
#plt.show()
plt.savefig(r"D:\WUR\master\MLP\master thesis\data\数据汇总\Corr_analysis\correlationMatrix.png")

# 2. CHECK VIF for collinearity
# Select the predictor variables for your GAM
X = cor_df.iloc[:, 2:]
y = cor_df.iloc[:,0]
# Create a DataFrame to store the VIF results
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# Print the VIF results
print(vif_data)

cor_df.to_csv(r'D:\WUR\master\MLP\master thesis\data\数据汇总\PC6_corr_df_orig.csv')

# STEP 2 ------------------prepare normalized table
## prepare a new df after column deletation
cor_df = pd.read_csv(r'D:\WUR\master\MLP\master thesis\data\数据汇总\PC6_corr_df_orig.csv', index_col=0)
cor_df = cor_df.drop(columns=['mean_NDVI','net_parking_pressure','class_index','Postcode'])
# 对数据进行normalization
scaler = preprocessing.MinMaxScaler()
gam_df_scaled = scaler.fit_transform(cor_df)
gam_df_scaled = pd.DataFrame(gam_df_scaled)
gam_df_scaled.columns = cor_df.columns
X_scaled = gam_df_scaled.iloc[:, 1:]
y_scaled = gam_df_scaled.iloc[:, 0]

gam_df_scaled.to_csv(r'D:\WUR\master\MLP\master thesis\data\数据汇总\PC6_corr_df_scaled.csv')

# STEP 3 ------------------check spatial autocorrelation
import esda
from esda.moran import Moran
import libpysal

weights = libpysal.weights.Queen.from_dataframe(cor_df,use_index=True)  # generate spatial weights (Queen in this case)
moran_df = pd.DataFrame(index=X.columns,columns=['Moran'])

for idx,col in moran_df.iterrows():
    for col in X.columns[:-1]:
        if idx == col:
            spatial_auto = esda.Moran(cor_df[[col]], weights)  # calculate Moran's I
            moran_df.at[idx,'Moran'] = spatial_auto.I

print(moran_df)
# 都显示clustering，但是除了mean_NDVI,其他都还好，所以可以继续用OLS

# STEP 4 ------------------测试OSL model
X_0 = X_scaled
vars = X_0.columns.tolist()
formula = 'WalkabilityIndex ~ ' + '+'.join(vars)
new_df_0 = gam_df_scaled
import statsmodels.formula.api as smf
lm_vital_0 = smf.ols(formula=formula, data=new_df_0).fit()
summary_table = lm_vital_0.summary()
print(lm_vital_0.summary())

# Get the R-squared value
r_squared = lm_vital_0.rsquared
print(f"R-squared (Deviance Explained): {r_squared:.2%}")

# Save the summary table as a CSV file
summary_table_as_csv = summary_table.tables[1].as_html()
summary_table_df = pd.read_html(summary_table_as_csv, header=0, index_col=0)[0]
summary_table_df.to_csv("D:\WUR\master\MLP\master thesis\data\数据汇总\Corr_analysis\ols_summary_table.csv")

# STEP 5 ------------------测试Random forest
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn import metrics
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

param_dist = {'n_estimators': randint(50,100),
              'max_depth': randint(1,20)}
# Create a random forest classifier
rf = RandomForestRegressor()
# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf,
                                 param_distributions = param_dist,
                                 n_iter=5,
                                 cv=5)
# Fit the random search object to the data
rand_search.fit(X_scaled, y_scaled)
# Create a variable for the best model
best_rf = rand_search.best_estimator_
# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)
#Best hyperparameters: {'max_depth': 1, 'n_estimators': 80}

rf = RandomForestRegressor(max_depth=5,n_estimators = 85, oob_score=True,random_state=42)
rf.fit(X_scaled, y_scaled)
feature_importances = rf.feature_importances_
y_pred = rf.oob_prediction_
print("R2", rf.oob_score_.round(3)) # 0.186
print("MSE",metrics.mean_squared_error(y_scaled,y_pred))

from sklearn.metrics import mean_squared_error
# Calculate MSE for the model
mse_model = mean_squared_error(y_scaled, y_pred)
# Calculate MSE for the baseline model (e.g., mean of y)
mean_y = y_scaled.mean()
mse_baseline = mean_squared_error(y_scaled, [mean_y] * len(y_scaled))
# Calculate Deviance Explained
deviance_explained = 1 - (mse_model / mse_baseline)
print(f"Deviance Explained: {deviance_explained:.2%}")

# Get feature importances
feature_importances = rf.feature_importances_
# Create a DataFrame to store feature names and their importances
importance_df = pd.DataFrame({'Feature': X_scaled.columns, 'Importance': feature_importances})
# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Set the figure size
fig, ax = plt.subplots(figsize=(8, 25))
# Plot feature importances using a bar chart
importance_df.plot(x='Feature', y='Importance', kind='barh', legend=False, color='skyblue', ax=ax)
# Add labels and title with larger font size
ax.set_xlabel('Variables', fontsize=20)
ax.set_ylabel('Value', fontsize=20)
ax.set_title('Bar Plot Example', fontsize=23)
# Adjust y-axis tick label properties
ax.tick_params(axis='y', which='major', labelsize=20)
# Ensure y-tick labels are not cut off
plt.tight_layout()
# Show the plot
plt.show()
# Save the plot to a file
plt.savefig(r'D:\WUR\master\MLP\master thesis\data\数据汇总\Corr_analysis\feature_importance.png')

import shap
# Create Tree Explainer object that can calculate shap values
# Calculate SHAP values
# Get feature names
feature_names = X_scaled.columns
# Create a dictionary to store feature names and importance scores
feature_importance_dict = dict(zip(feature_names, feature_importances))
# Sort features based on importance
sorted_feature_names = sorted(feature_names, key=lambda x: feature_importance_dict[x], reverse=True)
# Calculate SHAP values
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_scaled)
# Create a summary plot with SHAP values, ordered by the sorted feature names
shap.summary_plot(shap_values, X_scaled, feature_names=X_scaled.columns)
plt.savefig(r'D:\WUR\master\MLP\master thesis\data\数据汇总\Corr_analysis\shap.png')

importance_df.to_csv("D:\WUR\master\MLP\master thesis\data\数据汇总\Corr_analysis\\rf_summary_table.csv",encoding='utf-8')