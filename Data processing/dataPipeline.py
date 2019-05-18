from cleaningData import numerical_data
from DfSelector import DfSelector
from CombinedAttributes import CombinedAttributes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from cleaningData import encode_text_oneHot
from sklearn.pipeline import FeatureUnion
from cleaningData import oneShotEncoder
from StratifiedTest import housing
from sklearn.preprocessing import OneHotEncoder

data = housing
numerical_attributes = list(numerical_data)
categoical_attributes = ["ocean_proximity"]

numerical_pipeline = Pipeline([('Selector',DfSelector(numerical_attributes)),('imputer', SimpleImputer(strategy ="median")),('attribs_adder', CombinedAttributes()),('standarized_saler',StandardScaler())])

categoical_pipeline = Pipeline([('selector',DfSelector(categoical_attributes)),("encoder", OneHotEncoder())])

pipeline = FeatureUnion(transformer_list=[("numerical_pipeline", numerical_pipeline),("catergorical_pipeline",categoical_pipeline)])

final = pipeline.fit_transform(data)

print(final.shape)
