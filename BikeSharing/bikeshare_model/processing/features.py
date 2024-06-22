from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = pd.Categorical(X[feature]).codes
        return X

class NumericalScaler(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.variables])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.variables] = self.scaler.transform(X[self.variables])
        return X