"""High-level orchestration used by the CLI and the Streamlit dashboard."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib

from . import __version__
from .betting import simulate_bankroll, value_bets
from .dataset import BASE_FEATURES, FEATURE_COLUMNS, aggregate_verified_market, build_feature_rows
from .evaluation import probability_metrics, summarize_predictions, walk_forward
from .models import PredictionEngine
from .sources import GameData, load_from_database

MIN_TRAIN_GAMES = 100
MIN_TEST_GAMES = 25
MARKET_FEATURE_MIN_COVERAGE = 0.6


def build_rows(data: GameData) -> list[dict[str, Any]]:
    games, stats, odds = data
    tipoffs = {str(g["game_provider_id"]): g.get("scheduled_tipoff_at") or g.get("game_date") for g in games}
    quotes = aggregate_verified_market(odds, tipoffs)
    return build_feature_rows(games, stats, include_upcoming=True, market_quotes=quotes)


def load_engine_dataset(db: Any) -> list[dict[str, Any]]:
    return build_rows(load_from_database(db))


def _completed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("target_home_win") in (0, 1)]


def _upcoming(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("target_home_win") is None]


def should_use_market(rows: list[dict[str, Any]]) -> bool:
    completed = _completed(rows)
    if not completed:
        return False
    coverage = sum(r.get("market_home_prob") is not None for r in completed) / len(completed)
    return coverage >= MARKET_FEATURE_MIN_COVERAGE


def evaluate_dataset(
    rows: list[dict[str, Any]],
    min_train_games: int = MIN_TRAIN_GAMES,
    retrain_every: int | None = None,
    use_market: bool | None = None,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    usable = _completed(rows)
    n = len(usable)
    if n < min_train_games + MIN_TEST_GAMES:
        return {
            "status": "insufficient_history", "games": n, "minimum_required": min_train_games + MIN_TEST_GAMES,
            "message": f"Historique insuffisant : {n} matchs terminés ; au moins {min_train_games + MIN_TEST_GAMES} requis.",
            "metrics": {}, "predictions": [],
        }
    if len({r["target_home_win"] for r in usable[:min_train_games]}) < 2:
        return {"status": "insufficient_classes", "games": n, "metrics": {}, "predictions": []}
    if use_market is None:
        use_market = should_use_market(rows)
    step = retrain_every or max(50, min(250, (n - min_train_games) // 8 or 50))
    predictions = walk_forward(usable, min_train_games=min_train_games, retrain_every=step,
                               use_market=use_market, progress=progress)
    summary = summarize_predictions(predictions)
    ens = summary["models"].get("ensemble", {})
    metrics = {
        "accuracy": ens.get("accuracy"), "brier_score": ens.get("brier_score"),
        "log_loss": ens.get("log_loss"), "roc_auc": ens.get("roc_auc"),
        "elo_brier_score": summary["models"].get("elo", {}).get("brier_score"),
        "elo_accuracy": summary["models"].get("elo", {}).get("accuracy"),
        **summary["regression"],
    }
    if "market" in summary["models"]:
        metrics["market_brier_score"] = summary["models"]["market"]["brier_score"]
        metrics["market_test_games"] = summary["models"]["market"]["games"]
    betting = simulate_bankroll(predictions)
    return {
        "status": "evaluated", "games": n, "train_games": min_train_games, "test_games": len(predictions),
        "split": f"walk_forward_expanding(retrain_every={step})", "use_market_feature": use_market,
        "metrics": metrics, "models": summary["models"], "market_comparison": summary["market_comparison"],
        "calibration": summary["calibration"], "by_season": summary["by_season"],
        "betting_backtest": {k: v for k, v in betting.items() if k != "history"},
        "predictions": predictions, "feature_columns": FEATURE_COLUMNS if use_market else BASE_FEATURES,
    }


def train_engine(rows: list[dict[str, Any]], use_market: bool | None = None) -> PredictionEngine:
    if use_market is None:
        use_market = should_use_market(rows)
    return PredictionEngine(use_market=use_market).fit(_completed(rows))


def predict_upcoming_games(
    rows: list[dict[str, Any]],
    min_train_games: int = MIN_TRAIN_GAMES,
    engine: PredictionEngine | None = None,
    min_edge: float = 0.03,
) -> dict[str, Any]:
    completed, upcoming = _completed(rows), _upcoming(rows)
    if engine is None:
        if len(completed) < min_train_games:
            return {"status": "insufficient_history", "games": len(completed), "predictions": []}
        if len({r["target_home_win"] for r in completed}) < 2:
            return {"status": "insufficient_classes", "games": len(completed), "predictions": []}
    if not upcoming:
        return {"status": "no_upcoming_games", "games": 0, "predictions": []}
    engine = engine or train_engine(rows)
    predictions = engine.predict(upcoming)
    for pred in predictions:
        pred["elo_home_prob"] = pred["prob_elo"]
        picks = value_bets(pred, min_edge=min_edge)
        pred["value_bet"] = picks[0] if picks else None
    return {"status": "predicted", "games": len(predictions), "train_games": engine.train_games,
            "predictions": predictions, "importance": engine.feature_importance()}


def build_report_from_rows(
    rows: list[dict[str, Any]],
    min_train_games: int = MIN_TRAIN_GAMES,
    retrain_every: int | None = None,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    report = evaluate_dataset(rows, min_train_games=min_train_games, retrain_every=retrain_every, progress=progress)
    report["upcoming"] = predict_upcoming_games(rows, min_train_games=min_train_games)
    completed_rows, upcoming_rows = _completed(rows), _upcoming(rows)
    report["engine_version"] = __version__
    report["dataset"] = {
        "games": len(completed_rows),
        "upcoming_games": len(upcoming_rows),
        "market_feature_coverage": sum(r.get("market_home_prob") is not None for r in completed_rows),
        "box_score_coverage": sum(r.get("home_ewm_net_rating") is not None for r in completed_rows),
        "seasons": sorted({r["season"] for r in rows if r.get("season") is not None}),
        "feature_count": len(report.get("feature_columns") or BASE_FEATURES),
        "time_leakage_policy": "verified_point_in_time AND source_captured_at < scheduled_tipoff_at; "
                               "features frozen before each game; walk-forward split by calendar day",
    }
    return report


def build_engine_report(db: Any, min_train_games: int = MIN_TRAIN_GAMES, **kwargs: Any) -> dict[str, Any]:
    return build_report_from_rows(load_engine_dataset(db), min_train_games=min_train_games, **kwargs)


def save_engine(engine: PredictionEngine, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"engine": engine, "version": __version__, "saved_at": datetime.now(timezone.utc).isoformat()}, path)
    return path


def load_engine(path: str | Path) -> PredictionEngine:
    bundle = joblib.load(path)
    return bundle["engine"]


__all__ = [
    "build_rows", "load_engine_dataset", "evaluate_dataset", "train_engine", "predict_upcoming_games",
    "build_report_from_rows", "build_engine_report", "save_engine", "load_engine", "probability_metrics",
]
