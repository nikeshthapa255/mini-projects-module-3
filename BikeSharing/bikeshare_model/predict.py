import sys
from pathlib import Path

# Set the path to the root of your project
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t

import pandas as pd

from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import config
from bikeshare_model.processing.data_manager import load_pipeline
from bikeshare_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app.pipeline_save_file}{_version}.pkl"
_bikeshare_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(*, input_data: t.Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _bikeshare_pipe.predict(validated_data[config.model.features])
        results = {
            "predictions": [pred for pred in predictions],
            "version": _version,
            "errors": errors,
        }

    return results

if __name__ == "__main__":
    data_in = {
        "season": [1], "yr": [0], "mnth": [0], "hr": [0], "holiday": [0], 
        "weekday": [0], "workingday": [1], "weathersit": [2], "temp": [0.24], 
        "atemp": [0.2879], "hum": [0.81], "windspeed": [0.0]
    }
    
    print(make_prediction(input_data=data_in))
