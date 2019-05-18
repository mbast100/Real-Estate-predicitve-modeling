from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

#based on dataset
rooms_ix = 3
bedrooms_ix = 4
population_ix = 5
households_ix = 6
housing = pd.read_csv("housing.csv")
# hyperparamter -> bedrooms_per_room

class CombinedAttributes(BaseEstimator, TransformerMixin):

    def __init__(self, bedrooms_per_room = True): #no args or kargs

        self.bedtooms_per_room = bedrooms_per_room
    
    def fit(self, X, y=None):

        return self 
    def transform(self, X, y=None):

        rooms_per_household = X[:,3] / X[:,6]
        population_per_household = X[:,5] / X[:,6]
        if self.bedtooms_per_room:
            bedrooms_per_room = X[:,4] / X[:,3]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household,population_per_household]

'''
test = CombinedAttributes(bedrooms_per_room = True)
print(test)
housing_extra_attributes = test.transform(housing.values)

print(pd.DataFrame(housing_extra_attributes))
'''