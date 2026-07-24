"""
Model Training & Evaluation Tests
"""

import pytest
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


def make_sample_dataset(n_samples=500):
    """Create a small dataset for testing."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    df["label"] = y
    return df


class TestDataGeneration:
    def test_sample_dataset_shape(self):
        df = make_sample_dataset()
        assert df.shape == (500, 11)

    def test_no_nulls(self):
        df = make_sample_dataset()
        assert df.isnull().sum().sum() == 0

    def test_binary_target(self):
        df = make_sample_dataset()
        assert set(df["label"].unique()).issubset({0, 1})


class TestModelQuality:
    @pytest.fixture(autouse=True)
    def train_small_model(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        df = make_sample_dataset(300)
        X = df.drop("label", axis=1).values
        y = df["label"].values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=20, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def test_model_accuracy_above_threshold(self):
        from sklearn.metrics import accuracy_score
        preds = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        assert acc > 0.70, f"Accuracy too low: {acc}"

    def test_predictions_are_binary(self):
        preds = self.model.predict(self.X_test)
        assert set(preds).issubset({0, 1})

    def test_probabilities_sum_to_one(self):
        probas = self.model.predict_proba(self.X_test)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_model_serialization(self, tmp_path):
        model_path = tmp_path / "model.joblib"
        joblib.dump(self.model, model_path)
        loaded = joblib.load(model_path)
        preds_original = self.model.predict(self.X_test)
        preds_loaded   = loaded.predict(self.X_test)
        np.testing.assert_array_equal(preds_original, preds_loaded)
