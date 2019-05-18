from sklearn.base import BaseEstimator, TransformerMixin

class DfSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute):
         self.attribute = attribute
    
    def fit(self, X, y=None):

        return self
    def transform(self, X):

        return X[self.attribute].values