"""Command line entry point.

Examples (from nba_quant_engine_collector/nba_quant_engine):

    python -m engine.cli evaluate --source db
    python -m engine.cli predict --source db --save-model artifacts/engine.joblib
    python -m engine.cli evaluate --source demo
    python -m engine.cli evaluate --source fivethirtyeight --path nbaallelo.csv --first-season 2005
    python -m engine.cli evaluate --source csv --games games.csv --team-stats stats.csv --odds odds.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv

from .service import build_report_from_rows, build_rows, predict_upcoming_games, save_engine, train_engine
from .sources import GameData, generate_synthetic_league, load_fivethirtyeight, load_from_csv, load_from_database


def _load(args: argparse.Namespace) -> GameData:
    if args.source == "demo":
        return generate_synthetic_league(seasons=args.demo_seasons, seed=args.seed)
    if args.source == "csv":
        if not args.games:
            raise SystemExit("--games is required with --source csv")
        return load_from_csv(args.games, args.team_stats, args.odds)
    if args.source == "fivethirtyeight":
        if not args.path:
            raise SystemExit("--path is required with --source fivethirtyeight")
        return load_fivethirtyeight(args.path, first_season=args.first_season, last_season=args.last_season)
    from collector.db import SupabaseDatabase

    load_dotenv()
    dsn = os.getenv("SUPABASE_DB_URL", "").strip()
    if not dsn:
        raise SystemExit("SUPABASE_DB_URL is required. No DunkAPI key or collection is needed.")
    db = SupabaseDatabase(dsn, schema=os.getenv("NBA_DB_SCHEMA", "nba").strip() or "nba")
    try:
        db.verify_schema()
        return load_from_database(db)
    finally:
        db.close()


def _print_summary(report: dict[str, Any]) -> None:
    out = sys.stderr
    print(f"\n== NBA Quant Engine v{report.get('engine_version')} ==", file=out)
    ds = report.get("dataset", {})
    print(f"Matchs terminés: {ds.get('games')} | à venir: {ds.get('upcoming_games')} | "
          f"box-scores: {ds.get('box_score_coverage')} | cotes PIT: {ds.get('market_feature_coverage')}", file=out)
    if report.get("status") != "evaluated":
        print(f"Statut: {report.get('status')} {report.get('message', '')}", file=out)
        return
    print(f"Évaluation walk-forward: {report['test_games']} matchs hors échantillon ({report['split']})", file=out)
    print(f"{'modèle':<12}{'matchs':>8}{'accuracy':>10}{'brier':>9}{'logloss':>9}{'auc':>8}", file=out)
    for name, m in report.get("models", {}).items():
        auc = f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "  -  "
        print(f"{name:<12}{m['games']:>8}{m['accuracy']:>10.3f}{m['brier_score']:>9.4f}{m['log_loss']:>9.4f}{auc:>8}", file=out)
    met = report["metrics"]
    print(f"Marge: MAE {met['margin_mae']:.2f} pts | Total: MAE {met['total_mae']:.2f} pts", file=out)
    if met.get("ats_hit_rate") is not None:
        print(f"Contre le spread: {met['ats_hit_rate']:.3f} sur {met['ats_games']} | Over/Under: {met.get('ou_hit_rate', 0):.3f}", file=out)
    bt = report.get("betting_backtest", {})
    if bt.get("bets"):
        print(f"Backtest value-bet: {bt['bets']} paris, ROI {bt['roi']:+.3%}, bankroll {bt['bankroll_growth']:+.1%}, "
              f"drawdown max {bt['max_drawdown']:.1%}", file=out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="NBA pre-game prediction engine (read-only).")
    parser.add_argument("command", nargs="?", default="evaluate", choices=["evaluate", "predict"])
    parser.add_argument("--source", default="db", choices=["db", "demo", "csv", "fivethirtyeight"])
    parser.add_argument("--min-train-games", type=int, default=100)
    parser.add_argument("--retrain-every", type=int, default=None)
    parser.add_argument("--path", help="nbaallelo.csv for --source fivethirtyeight")
    parser.add_argument("--first-season", type=int, default=2005)
    parser.add_argument("--last-season", type=int, default=None)
    parser.add_argument("--games")
    parser.add_argument("--team-stats")
    parser.add_argument("--odds")
    parser.add_argument("--demo-seasons", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-model", help="Persist the model trained on all completed games (joblib).")
    parser.add_argument("--include-predictions", action="store_true", help="Include every out-of-sample prediction in the JSON.")
    parser.add_argument("--output", help="Write the JSON report to this file instead of stdout.")
    args = parser.parse_args(argv)

    rows = build_rows(_load(args))
    if args.command == "predict":
        report: dict[str, Any] = predict_upcoming_games(rows, min_train_games=args.min_train_games)
        if args.save_model and report.get("status") in ("predicted", "no_upcoming_games"):
            save_engine(train_engine(rows), args.save_model)
    else:
        report = build_report_from_rows(rows, min_train_games=args.min_train_games, retrain_every=args.retrain_every)
        _print_summary(report)
        if not args.include_predictions:
            report["predictions"] = report.get("predictions", [])[-25:]
        if args.save_model:
            save_engine(train_engine(rows), args.save_model)

    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
