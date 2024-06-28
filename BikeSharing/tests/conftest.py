import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
from sklearn.model_selection import train_test_split

from bikeshare_model.config.core import config
from bikeshare_model.processing.data_manager import load_dataset



@pytest.fixture
def sample_input_data():
    data = load_dataset(file_name=config.app.training_data_file)

    X = data.drop(config.model.target, axis=1)       # predictors
    y = data[config.model.target]                    # target

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,  # predictors
        y,  # target
        test_size=config.model.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model.random_state,
    )

    return X_test, y_test