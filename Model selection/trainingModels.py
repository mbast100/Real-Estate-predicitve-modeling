from sklearn.linear_model   import LinearRegression
from StratifiedTest import housing, housingLabels
from dataPipeline import pipeline
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

data = housing
dataLables = housingLabels
linReg = LinearRegression()
treeReg = DecisionTreeRegressor()

prepared_data = pipeline.fit_transform(data)

linReg.fit(prepared_data, dataLables)
treeReg.fit(prepared_data, dataLables)

test = data.iloc[:5]
testLabels = dataLables.iloc[:5]
testPrepared = pipeline.transform(test)

print("Predicitions using linear regressing : ", linReg.predict(testPrepared))
print("labels: ", list(testLabels))

lin_error = mean_squared_error(testLabels, linReg.predict(testPrepared))

print("error in US dollars: ",np.sqrt(lin_error))
print("\n")
print("Prediction using decision tree: ", treeReg.predict(testPrepared))
print("labels: ", list(testLabels))

tree_error = mean_squared_error(testLabels,treeReg.predict(testPrepared))
print("error using decision tree: ", tree_error)