from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from bikeshare_model.config.core import config

def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    validated_data = input_data.copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors

class DataInputSchema(BaseModel):
    season: Optional[int]
    yr: Optional[int]
    mnth: Optional[int]
    hr: Optional[int]
    holiday: Optional[int]
    weekday: Optional[int]
    workingday: Optional[int]
    weathersit: Optional[int]
    temp: Optional[float]
    atemp: Optional[float]
    hum: Optional[float]
    windspeed: Optional[float]

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]