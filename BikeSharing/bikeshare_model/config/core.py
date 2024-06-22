import sys
from pathlib import Path
from typing import Dict, List
from pydantic import BaseModel
from strictyaml import YAML, load

import bikeshare_model

# Path setup
PACKAGE_ROOT = Path(bikeshare_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

class AppConfig(BaseModel):
    """
    Application-level config.
    """
    package_name: str
    training_data_file: str
    pipeline_name: str
    pipeline_save_file: str

class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """
    target: str
    features: List[str]
    unused_fields: List[str]
    categorical_features: List[str]
    numerical_features: List[str]
    test_size: float
    random_state: int
    n_estimators: int
    max_depth: int

class Config(BaseModel):
    """Master config object."""
    app: AppConfig
    model: ModelConfig

def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""
    if not cfg_path:
        cfg_path = find_config_file()

    with open(cfg_path, "r") as conf_file:
        parsed_config = load(conf_file.read())
        return parsed_config

def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    _config = Config(
        app=AppConfig(**parsed_config.data),
        model=ModelConfig(**parsed_config.data),
    )
    return _config

config = create_and_validate_config()
