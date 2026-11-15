"""
Machine Learning Demand & Price Multiplier Forecaster (scikit-learn).

Generates synthetic aviation booking records and trains RandomForestRegressor &
GradientBoostingRegressor models to predict optimal fare multipliers.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS: List[str] = [
    "days_to_departure",
    "load_factor",
    "booking_velocity",
    "competitor_ratio",
    "is_weekend",
    "is_peak_season",
    "elasticity",
]


def generate_synthetic_aviation_dataset(n_samples: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic historical flight booking dataset."""
    rng = np.random.default_rng(seed)

    days_to_departure = rng.integers(0, 91, size=n_samples)
    load_factor = np.clip(
        0.92 - 0.008 * days_to_departure + rng.normal(0.0, 0.14, size=n_samples),
        0.05,
        0.99,
    )
    booking_velocity = np.clip(
        1.2 + 3.5 * np.exp(-0.08 * days_to_departure) + rng.exponential(0.9, size=n_samples),
        0.2,
        12.0,
    )
    competitor_ratio = np.clip(rng.normal(1.02, 0.10, size=n_samples), 0.75, 1.32)
    is_weekend = rng.integers(0, 2, size=n_samples)
    is_peak_season = rng.choice([0, 1], size=n_samples, p=[0.72, 0.28])
    elasticity = np.clip(rng.normal(1.22, 0.24, size=n_samples), 0.65, 2.10)

    # Ground-truth structural willingness-to-pay multiplier with stochastic market noise
    time_term = 1.0 + 0.56 * np.exp(-0.11 * days_to_departure)
    load_term = 1.0 + 0.70 * (load_factor ** 2.3)
    vel_term = 1.0 + np.minimum(0.32, 0.05 * np.maximum(0.0, booking_velocity - 2.0))
    season_term = 1.0 + 0.10 * is_weekend + 0.22 * is_peak_season
    comp_term = 1.0 + 0.28 * (competitor_ratio - 1.0)
    elast_damp = 1.0 / (1.0 + np.maximum(0.0, elasticity - 1.15) * 0.18)

    raw_mult = time_term * load_term * vel_term * season_term * comp_term
    target_multiplier = np.clip(
        1.0 + (raw_mult - 1.0) * elast_damp + rng.normal(0.0, 0.035, size=n_samples),
        0.72,
        3.20,
    )

    return pd.DataFrame(
        {
            "days_to_departure": days_to_departure,
            "load_factor": np.round(load_factor, 4),
            "booking_velocity": np.round(booking_velocity, 3),
            "competitor_ratio": np.round(competitor_ratio, 3),
            "is_weekend": is_weekend,
            "is_peak_season": is_peak_season,
            "elasticity": np.round(elasticity, 3),
            "optimal_multiplier": np.round(target_multiplier, 4),
        }
    )


class DemandPricingMLModel:
    """Trains and serves scikit-learn ensemble models for fare multiplier prediction."""

    def __init__(self) -> None:
        self.rf_model = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=140, learning_rate=0.07, max_depth=4, random_state=42)
        self.is_trained = False
        self.metrics: Dict[str, Any] = {}

    def train(self, n_samples: int = 2500) -> Dict[str, Any]:
        """Train both ensemble models and return benchmark metrics + feature importances."""
        df = generate_synthetic_aviation_dataset(n_samples=n_samples)
        X = df[FEATURE_COLUMNS]
        y = df["optimal_multiplier"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.rf_model.fit(X_train, y_train)
        self.gb_model.fit(X_train, y_train)

        rf_pred = self.rf_model.predict(X_test)
        gb_pred = self.gb_model.predict(X_test)

        self.is_trained = True
        self.metrics = {
            "dataset_rows": n_samples,
            "random_forest": {
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, rf_pred))), 4),
                "mae": round(float(mean_absolute_error(y_test, rf_pred)), 4),
                "r2": round(float(r2_score(y_test, rf_pred)), 4),
            },
            "gradient_boosting": {
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, gb_pred))), 4),
                "mae": round(float(mean_absolute_error(y_test, gb_pred)), 4),
                "r2": round(float(r2_score(y_test, gb_pred)), 4),
            },
            "feature_importance": {
                col: round(float(imp), 4)
                for col, imp in sorted(
                    zip(FEATURE_COLUMNS, self.gb_model.feature_importances_),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
        }
        return self.metrics

    def predict_multiplier(
        self,
        days_to_departure: int,
        load_factor: float,
        booking_velocity: float,
        competitor_ratio: float,
        is_weekend: int,
        is_peak_season: int,
        elasticity: float,
    ) -> float:
        """Predict optimal fare multiplier using Gradient Boosting model."""
        if not self.is_trained:
            self.train()
        row = pd.DataFrame(
            [
                {
                    "days_to_departure": days_to_departure,
                    "load_factor": load_factor,
                    "booking_velocity": booking_velocity,
                    "competitor_ratio": competitor_ratio,
                    "is_weekend": is_weekend,
                    "is_peak_season": is_peak_season,
                    "elasticity": elasticity,
                }
            ]
        )
        return round(float(self.gb_model.predict(row)[0]), 4)
