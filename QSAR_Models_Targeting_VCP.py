# Import libraries:
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle


# Retrieve dataset:
dataset_url = 'VCP_00_bioactivity_data_3class_pIC50_pubchem_fp.csv'
dataset = pd.read_csv(dataset_url)
print(dataset)

X = dataset.drop(['pIC50'], axis=1)
print(X)

Y = dataset.iloc[:,-1]
print(Y)

# Remove low variance feature:
def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

X = remove_low_variance(X, threshold=0.1)
print(X)
X.to_csv('descriptor_list.csv', index=False)

# Create machine learning model:
model = DecisionTreeRegressor()
model.fit(X,Y)
r2 = model.score(X, Y)
print(r2)

# Model prediction:
Y_pred = model.predict(X)
print(Y_pred)

# Model performance:
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y, Y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y, Y_pred))

# Data visualization:
plt.figure(figsize=(5,5))
plt.scatter(x=Y, y=Y_pred, c="#7CAE00", alpha=0.3)

# Add trendline
# https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
z = np.polyfit(Y, Y_pred, 1)
p = np.poly1d(z)

plt.plot(Y,p(Y),"#F8766D")
plt.ylabel('Predicted pIC50')
plt.xlabel('Experimental pIC50')
plt.savefig('Experimental_vs_Predicted_pIC50.pdf')

pickle.dump(model, open('VCP_targeting_drug_model.pkl', 'wb'))
