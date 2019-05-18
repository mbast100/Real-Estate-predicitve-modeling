from StratifiedTest import housing, stratTrainSet
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# getting stratified data 

data = housing
data_labels = stratTrainSet['median_house_value'].copy()

# to take care of missing values

imputer = SimpleImputer(strategy = "median")

numerical_data = data.drop("ocean_proximity", axis =1)
#print(numerical_data.head(5))
imputer.fit(numerical_data)
#print("statistics : ",imputer.statistics_)

# categorizing none nuerical data
def encode_text(attribute):

    categorizing_data = data[attribute]
    encoded , categories = categorizing_data.factorize()
    return encoded , categories
#only one attribute will be equal to one, while the others will be 0
#fit.transform needs a 2D array, thats why we reshape

def encode_text_oneHot(attribute):
    
    encoded , categories = encode_text(attribute)
    encoder = OneHotEncoder()
    hotEncoder = encoder.fit_transform(encoded.reshape(-1,1))

    return hotEncoder, categories

'''
if __name__ == "__main__":

    a, b = encode_text_oneHot("ocean_proximity")
    
    print(a.toarray())
'''
    


