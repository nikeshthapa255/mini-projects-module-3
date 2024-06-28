from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class EnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        self.linear_model = LinearRegression()
        self.random_forest_model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state
        )

    def fit(self, X, y):
        self.linear_model.fit(X, y)
        self.random_forest_model.fit(X, y)
        return self

    def predict(self, X):
        linear_pred = self.linear_model.predict(X)
        rf_pred = self.random_forest_model.predict(X)
        return np.mean([linear_pred, rf_pred], axis=0)
