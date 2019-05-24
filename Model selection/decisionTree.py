from sklearn.tree import DecisionTreeRegressor
from dataPipeline import preparedData
from StratifiedTest import housingLabels
from sklearn.metrics import mean_squared_error

treeReg = DecisionTreeRegressor()
treeReg.fit(preparedData,housingLabels)

dataPredictions = treeReg.predict(preparedData)
error = mean_squared_error(dataPredictions,housingLabels)
print(list(dataPredictions))
print(error)