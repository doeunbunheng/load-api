"""MLflow pyfunc wrapper that always returns the default probability."""

import pandas as pd
from mlflow.pyfunc import PythonModel


class LoanDefaultProbabilityModel(PythonModel):
    """Wrap a fitted sklearn pipeline so pyfunc predicts one probability column."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        probabilities = self.pipeline.predict_proba(model_input)[:, 1]
        return pd.DataFrame({"default_probability": probabilities})
