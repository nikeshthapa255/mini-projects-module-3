from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from bikeshare_model.ensemble_refressor import EnsembleRegressor
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import CategoricalEncoder, NumericalScaler

# Create the preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ("categorical_encoder", CategoricalEncoder(variables=config.model.categorical_features)),
    ("numerical_scaler", NumericalScaler(variables=config.model.numerical_features))
])


bikeshare_pipe = Pipeline([
    ("categorical_encoder", CategoricalEncoder(variables=config.model.categorical_features)),
    ("numerical_scaler", NumericalScaler(variables=config.model.numerical_features)),
    ("model_rf", EnsembleRegressor(n_estimators=config.model.n_estimators, 
                                       max_depth=config.model.max_depth,
                                       random_state=config.model.random_state))
])