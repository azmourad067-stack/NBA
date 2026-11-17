"""Walk-forward (expanding window) out-of-sample evaluation."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .models import PredictionEngine


def probability_metrics(y: list[int] | np.ndarray, p: list[float] | np.ndarray) -> dict[str, float | None]:
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if len(y_arr) == 0:
        return {"games": 0}
    return {
        "games": int(len(y_arr)),
        "accuracy": float(accuracy_score(y_arr, p_arr >= 0.5)),
        "brier_score": float(brier_score_loss(y_arr, p_arr)),
        "log_loss": float(log_loss(y_arr, p_arr, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_arr, p_arr)) if len(set(y_arr.tolist())) == 2 else None,
    }


def calibration_table(y: list[int], p: list[float], bins: int = 10) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    y_arr, p_arr = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p_arr >= lo) & ((p_arr < hi) if hi < 1.0 else (p_arr <= hi))
        if mask.sum() == 0:
            continue
        table.append({
            "bin": f"{lo:.1f}-{hi:.1f}", "games": int(mask.sum()),
            "mean_predicted": float(p_arr[mask].mean()), "observed_rate": float(y_arr[mask].mean()),
        })
    return table


def _date_aligned_cut(rows: list[dict[str, Any]], index: int) -> int:
    """Move a split index forward so a calendar day is never split between train and test."""
    if index <= 0 or index >= len(rows):
        return index
    day = rows[index - 1]["played_at"][:10]
    while index < len(rows) and rows[index]["played_at"][:10] == day:
        index += 1
    return index


def walk_forward(
    rows: list[dict[str, Any]],
    min_train_games: int = 300,
    retrain_every: int = 150,
    use_market: bool = False,
    max_train_games: int | None = None,
    progress: Callable[[float], None] | None = None,
) -> list[dict[str, Any]]:
    """Expanding-window retraining; every test game is predicted by a model
    trained only on games played on strictly earlier days."""
    completed = [r for r in rows if r.get("target_home_win") in (0, 1)]
    completed.sort(key=lambda r: (r["played_at"], r["game_provider_id"]))
    n = len(completed)
    start = _date_aligned_cut(completed, min_train_games)
    predictions: list[dict[str, Any]] = []
    while start < n:
        end = _date_aligned_cut(completed, min(n, start + retrain_every))
        train = completed[:start]
        if max_train_games:
            train = train[-max_train_games:]
        engine = PredictionEngine(use_market=use_market).fit(train)
        test = completed[start:end]
        for row, pred in zip(test, engine.predict(test)):
            pred["actual_home_win"] = int(row["target_home_win"])
            pred["actual_margin"] = row["target_margin"]
            pred["actual_total"] = row["target_total"]
            pred["season"] = row.get("season")
            pred["train_games"] = len(train)
            predictions.append(pred)
        start = end
        if progress:
            progress(start / n)
    return predictions


def summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {"models": {}, "calibration": [], "by_season": [], "regression": {}}
    y = [p["actual_home_win"] for p in predictions]
    models: dict[str, Any] = {}
    models["ensemble"] = probability_metrics(y, [p["home_win_probability"] for p in predictions])
    for name in ("logit", "gbm", "margin", "elo"):
        key = f"prob_{name}"
        subset = [p for p in predictions if p.get(key) is not None]
        if subset:
            models[name] = probability_metrics([p["actual_home_win"] for p in subset], [p[key] for p in subset])
    home_rate = float(np.mean(y))
    models["always_home"] = probability_metrics(y, [home_rate] * len(y))

    market = [p for p in predictions if p.get("market_home_prob") is not None]
    comparison = None
    if market:
        ym = [p["actual_home_win"] for p in market]
        models["market"] = probability_metrics(ym, [p["market_home_prob"] for p in market])
        comparison = {
            "games": len(market),
            "model": probability_metrics(ym, [p["home_win_probability"] for p in market]),
            "market": models["market"],
        }

    margin_err = np.asarray([p["predicted_margin"] - p["actual_margin"] for p in predictions])
    total_err = np.asarray([p["predicted_total"] - p["actual_total"] for p in predictions])
    regression = {
        "margin_mae": float(np.mean(np.abs(margin_err))),
        "margin_rmse": float(math.sqrt(np.mean(margin_err ** 2))),
        "total_mae": float(np.mean(np.abs(total_err))),
        "total_rmse": float(math.sqrt(np.mean(total_err ** 2))),
    }
    spread_rows = [p for p in predictions if p.get("market_home_spread") is not None]
    if spread_rows:
        # Home covers when margin + home_spread > 0 (pushes ignored).
        decided = [p for p in spread_rows if p["actual_margin"] + p["market_home_spread"] != 0]
        hits = [((p["predicted_margin"] + p["market_home_spread"]) > 0) == ((p["actual_margin"] + p["market_home_spread"]) > 0) for p in decided]
        regression["ats_games"] = len(decided)
        regression["ats_hit_rate"] = float(np.mean(hits)) if hits else None
    total_rows = [p for p in predictions if p.get("market_total") is not None and p["actual_total"] != p["market_total"]]
    if total_rows:
        hits = [(p["predicted_total"] > p["market_total"]) == (p["actual_total"] > p["market_total"]) for p in total_rows]
        regression["ou_games"] = len(total_rows)
        regression["ou_hit_rate"] = float(np.mean(hits))

    by_season_rows: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for p in predictions:
        by_season_rows[p.get("season")].append(p)
    by_season = []
    for season, items in sorted(by_season_rows.items(), key=lambda kv: (kv[0] is None, kv[0])):
        ys = [p["actual_home_win"] for p in items]
        ens = probability_metrics(ys, [p["home_win_probability"] for p in items])
        elo = probability_metrics(ys, [p["prob_elo"] for p in items])
        by_season.append({"season": season, "games": len(items),
                          "accuracy": ens["accuracy"], "brier_score": ens["brier_score"],
                          "elo_accuracy": elo["accuracy"], "elo_brier_score": elo["brier_score"]})
    return {
        "models": models,
        "market_comparison": comparison,
        "calibration": calibration_table(y, [p["home_win_probability"] for p in predictions]),
        "by_season": by_season,
        "regression": regression,
    }
