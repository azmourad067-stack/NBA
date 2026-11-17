"""Pre-game models: win probability, point margin and game total."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import BASE_FEATURES, CORE_FEATURES, MARKET_FEATURES, TOTAL_FEATURES

PROBABILITY_MODELS = ("elo", "logit", "gbm", "margin", "ensemble")
ENSEMBLE_MEMBERS = ("logit", "gbm", "margin")
DEFAULT_MARGIN_SIGMA = 12.5
MIN_GBM_ROWS = 400


def matrix(rows: list[dict[str, Any]], columns: list[str]) -> np.ndarray:
    return np.asarray([[np.nan if r.get(c) is None else float(r[c]) for c in columns] for r in rows], dtype=float)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _linear(estimator: Any) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


@dataclass
class PredictionEngine:
    """Bundle of models trained on the same chronological history.

    * ``logit``  — regularised logistic regression on compact core features.
    * ``gbm``    — gradient-boosted trees on the full feature set (NaN-native).
    * ``margin`` — ridge regression on the point margin, converted to a win
      probability through a normal distribution fitted on residuals.
    * ``total``  — ridge regression on the combined score.
    * ``ensemble`` — average of logit, gbm and margin probabilities.
    """

    use_market: bool = False
    random_state: int = 42
    fitted: bool = False
    train_games: int = 0
    margin_sigma: float = DEFAULT_MARGIN_SIGMA
    total_sigma: float = 18.0
    home_win_rate: float = 0.58
    models: dict[str, Any] = field(default_factory=dict)
    ensemble_members: tuple[str, ...] = ENSEMBLE_MEMBERS
    gbm_columns: list[str] = field(default_factory=list)

    @property
    def core_columns(self) -> list[str]:
        return CORE_FEATURES + (MARKET_FEATURES if self.use_market else [])

    @property
    def full_columns(self) -> list[str]:
        return BASE_FEATURES + (MARKET_FEATURES if self.use_market else [])

    def fit(self, rows: list[dict[str, Any]]) -> "PredictionEngine":
        train = [r for r in rows if r.get("target_home_win") in (0, 1)]
        if len(train) < 20:
            raise ValueError("At least 20 completed games are required to fit the engine.")
        y = np.asarray([r["target_home_win"] for r in train], dtype=int)
        if len(set(y.tolist())) < 2:
            raise ValueError("Training labels contain a single class.")
        margins = np.asarray([r["target_margin"] for r in train], dtype=float)
        totals = np.asarray([r["target_total"] for r in train], dtype=float)
        x_core, x_full, x_total = matrix(train, self.core_columns), matrix(train, self.full_columns), matrix(train, TOTAL_FEATURES)

        self.models = {}
        self.models["logit"] = _linear(LogisticRegression(C=0.05, max_iter=3000)).fit(x_core, y)
        self.models["margin"] = _linear(Ridge(alpha=10.0)).fit(x_core, margins)
        residuals = margins - self.models["margin"].predict(x_core)
        self.margin_sigma = float(max(8.0, np.std(residuals)))
        self.models["total"] = _linear(Ridge(alpha=10.0)).fit(x_total, totals)
        self.total_sigma = float(max(8.0, np.std(totals - self.models["total"].predict(x_total))))
        members = ["logit", "margin"]
        # Trees cannot bin a column that is entirely missing (e.g. no box scores).
        observed = ~np.all(np.isnan(x_full), axis=0)
        self.gbm_columns = [c for c, keep in zip(self.full_columns, observed) if keep]
        if len(train) >= MIN_GBM_ROWS and self.gbm_columns:
            self.models["gbm"] = HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.04, max_iter=250, min_samples_leaf=40,
                l2_regularization=1.0, random_state=self.random_state,
            ).fit(x_full[:, observed], y)
            members.append("gbm")
        self.ensemble_members = tuple(members)
        self.home_win_rate = float(y.mean())
        self.train_games = len(train)
        self.fitted = True
        return self

    def predict(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.fitted:
            raise RuntimeError("PredictionEngine.fit must be called first.")
        if not rows:
            return []
        x_core, x_total = matrix(rows, self.core_columns), matrix(rows, TOTAL_FEATURES)
        probs: dict[str, np.ndarray] = {
            "logit": self.models["logit"].predict_proba(x_core)[:, 1],
        }
        margin = self.models["margin"].predict(x_core)
        probs["margin"] = np.asarray([normal_cdf(m / self.margin_sigma) for m in margin])
        if "gbm" in self.models:
            probs["gbm"] = self.models["gbm"].predict_proba(matrix(rows, self.gbm_columns))[:, 1]
        probs["ensemble"] = np.mean([probs[m] for m in self.ensemble_members], axis=0)
        total = self.models["total"].predict(x_total)

        out = []
        for i, row in enumerate(rows):
            p = float(np.clip(probs["ensemble"][i], 0.01, 0.99))
            item = {
                "game_provider_id": row["game_provider_id"],
                "played_at": row["played_at"],
                "home_team_provider_id": row.get("home_team_provider_id"),
                "away_team_provider_id": row.get("away_team_provider_id"),
                "home_win_probability": p,
                "away_win_probability": 1.0 - p,
                "predicted_margin": float(margin[i]),
                "predicted_spread": float(-margin[i]),
                "predicted_total": float(total[i]),
                "fair_home_odds": 1.0 / p,
                "fair_away_odds": 1.0 / (1.0 - p),
                "prob_elo": float(row["home_elo_prob"]),
            }
            for name, values in probs.items():
                if name != "ensemble":
                    item[f"prob_{name}"] = float(values[i])
            for key in ("market_home_prob", "market_home_odds", "market_away_odds", "market_home_spread", "market_total"):
                item[key] = row.get(key)
            out.append(item)
        return out

    def feature_importance(self, top: int = 15) -> list[dict[str, Any]]:
        """Standardised logistic coefficients (sign = effect on home win)."""
        if "logit" not in self.models:
            return []
        pipe = self.models["logit"]
        coefs = pipe.named_steps["model"].coef_[0]
        names = list(self.core_columns)
        names += [f"missing:{self.core_columns[i]}" for i in pipe.named_steps["imputer"].indicator_.features_] if pipe.named_steps["imputer"].indicator_ is not None else []
        pairs = sorted(zip(names, coefs), key=lambda kv: abs(kv[1]), reverse=True)[:top]
        return [{"feature": n, "coefficient": float(c)} for n, c in pairs]
