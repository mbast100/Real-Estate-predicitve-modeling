import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("housing.csv")

#returns description of the data set, a summary of the numerical attributes
# null values are ignored 
print(data.describe())

#to get a quick feel of the data set 
# shows number of instances on the vertical avaergae
#data.hist(bins = 50, figsize = (20,15) )
#plt.show
data.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1, s = data["population"]/100, label = "population",figsize = (10,7), c ="median_house_value", cmap = plt.get_cmap("jet"), colorbar = True)
plt.show()


#to have discrete categories

data["income_cat"] = np.ceil(data["median_income"]/1.5)
data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace = True)

data["income_cat"].hist()

plt.show()


