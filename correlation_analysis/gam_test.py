import numpy as np
import pygam
import pandas as pd


# Create your GAM model
gam_model = pygam.LinearGAM()
df = pd.read_csv(r'D:/WUR/master/MLP/master thesis/data/数据汇总/PC6_corr_df.csv')
X = df.iloc[:,2:]
y = df.iloc[:,1]
# Fit the model to your data (X, y)
gam_model.fit(X, y)
predicted_values = gam_model.predict(X)
import matplotlib.pyplot as plt
gam_model.plot()
plt.show()
plt.savefig(r'D:/WUR/master/MLP/master thesis/data/数据汇总/gam_python_test.csv')

# Generate predictions over a range of input values
input_values = np.linspace(min_input, max_input, num_points)
predicted_values = gam_model.predict(input_values)
