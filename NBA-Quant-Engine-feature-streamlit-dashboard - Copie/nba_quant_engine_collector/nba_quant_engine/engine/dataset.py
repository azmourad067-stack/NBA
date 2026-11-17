"""Chronological, leakage-safe pre-game feature construction.

Every feature row is frozen *before* the game's own result or box score is
folded into team state. Upcoming games are emitted as prediction targets but
never update any state.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from statistics import median
from typing import Any, Iterable

from .elo import EloRatings, elo_to_spread

LEGACY_FEATURES = [
    "home_elo_prob", "home_rest_days", "away_rest_days", "rest_days_diff",
    "home_win_rate_last5", "away_win_rate_last5",
    "home_points_for_last5", "home_points_against_last5",
    "away_points_for_last5", "away_points_against_last5",
    "home_point_diff_last5", "away_point_diff_last5",
    "home_fg_pct_last5", "away_fg_pct_last5",
    "home_three_made_last5", "away_three_made_last5",
    "home_rebounds_last5", "away_rebounds_last5",
    "home_assists_last5", "away_assists_last5",
    "home_turnovers_last5", "away_turnovers_last5",
]

_SIDE_FEATURES = [
    "b2b", "games_last7", "win_rate_last10", "margin_last10",
    "season_games", "season_win_pct", "season_margin",
    "ewm_margin", "ewm_points_for", "ewm_points_against",
    "ewm_off_rating", "ewm_def_rating", "ewm_net_rating", "ewm_pace",
    "ewm_efg", "ewm_tov_pct", "ewm_orb_pct", "ewm_ftr",
    "ewm_opp_efg", "ewm_opp_tov_pct", "ewm_drb_pct", "ewm_opp_ftr",
    "ewm_three_rate",
]

DIFF_FEATURES = [
    "elo_diff", "elo_spread",
    "diff_margin_last10", "diff_season_margin", "diff_season_win_pct",
    "diff_ewm_margin", "diff_ewm_net_rating", "diff_efg_matchup",
    "diff_tov_matchup", "diff_orb_matchup", "diff_ftr_matchup",
    "expected_pace", "expected_total_proxy",
]

BASE_FEATURES = (
    LEGACY_FEATURES
    + [f"{side}_{name}" for side in ("home", "away") for name in _SIDE_FEATURES]
    + DIFF_FEATURES
)
MARKET_FEATURES = ["market_home_prob"]
FEATURE_COLUMNS = BASE_FEATURES + MARKET_FEATURES

# Compact, robust subset used by the linear models (fewer collinear inputs).
CORE_FEATURES = [
    "elo_diff", "rest_days_diff",
    "home_b2b", "away_b2b", "home_games_last7", "away_games_last7",
    "diff_margin_last10", "diff_season_margin", "diff_ewm_margin",
    "diff_ewm_net_rating", "diff_efg_matchup", "diff_tov_matchup",
    "diff_orb_matchup", "diff_ftr_matchup",
    "home_season_games", "away_season_games",
]

TOTAL_FEATURES = [
    "expected_pace", "expected_total_proxy",
    "home_ewm_points_for", "home_ewm_points_against",
    "away_ewm_points_for", "away_ewm_points_against",
    "home_ewm_off_rating", "away_ewm_off_rating",
    "home_ewm_def_rating", "away_ewm_def_rating",
    "home_ewm_pace", "away_ewm_pace", "home_b2b", "away_b2b",
]


def _as_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _mean(values: Iterable[float | int | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def season_from_date(dt: datetime) -> int:
    """NBA season end year: games from August onward belong to next year's season."""
    return dt.year + 1 if dt.month >= 8 else dt.year


# ---------------------------------------------------------------------------
# Box-score derived metrics
# ---------------------------------------------------------------------------

def possessions(own: dict[str, Any], opp: dict[str, Any]) -> float | None:
    """Two-sided possession estimate (Basketball-Reference style)."""
    def one(team: dict[str, Any], other: dict[str, Any]) -> float | None:
        fga, fta, tov = _num(team.get("field_goals_attempted")), _num(team.get("free_throws_attempted")), _num(team.get("turnovers"))
        orb, fgm = _num(team.get("offensive_rebounds")), _num(team.get("field_goals_made"))
        odrb = _num(other.get("defensive_rebounds"))
        if None in (fga, fta, tov, orb, fgm):
            return None
        orb_rate = orb / (orb + odrb) if odrb is not None and (orb + odrb) > 0 else 0.3
        return fga + 0.4 * fta - 1.07 * orb_rate * (fga - fgm) + tov

    a, b = one(own, opp), one(opp, own)
    if a is None and b is None:
        return None
    if a is None or b is None:
        return a if a is not None else b
    return 0.5 * (a + b)


def four_factors(own: dict[str, Any], opp: dict[str, Any]) -> dict[str, float | None]:
    fgm, fga = _num(own.get("field_goals_made")), _num(own.get("field_goals_attempted"))
    tpm, tpa = _num(own.get("three_pointers_made")), _num(own.get("three_pointers_attempted"))
    fta, tov = _num(own.get("free_throws_attempted")), _num(own.get("turnovers"))
    ftm = _num(own.get("free_throws_made"))
    orb, opp_drb = _num(own.get("offensive_rebounds")), _num(opp.get("defensive_rebounds"))
    out: dict[str, float | None] = {"efg": None, "tov_pct": None, "orb_pct": None, "ftr": None, "three_rate": None}
    if fgm is not None and fga:
        out["efg"] = (fgm + 0.5 * (tpm or 0.0)) / fga
        out["ftr"] = (ftm if ftm is not None else 0.0) / fga if ftm is not None else None
        out["three_rate"] = tpa / fga if tpa is not None else None
        if fta is not None and tov is not None:
            denom = fga + 0.44 * fta + tov
            out["tov_pct"] = tov / denom if denom else None
    if orb is not None and opp_drb is not None and (orb + opp_drb) > 0:
        out["orb_pct"] = orb / (orb + opp_drb)
    return out


# ---------------------------------------------------------------------------
# Team state
# ---------------------------------------------------------------------------

@dataclass
class _Ewm:
    span: float
    value: float | None = None

    def update(self, x: float | None) -> None:
        if x is None:
            return
        alpha = 2.0 / (self.span + 1.0)
        self.value = x if self.value is None else alpha * x + (1 - alpha) * self.value


_EWM_KEYS = (
    "margin", "points_for", "points_against", "off_rating", "def_rating", "net_rating", "pace",
    "efg", "tov_pct", "orb_pct", "ftr", "opp_efg", "opp_tov_pct", "drb_pct", "opp_ftr", "three_rate",
)


@dataclass
class TeamState:
    window: int = 5
    ewm_span: float = 10.0
    recent: deque = field(default_factory=lambda: deque(maxlen=10))
    short: deque | None = None
    dates: deque = field(default_factory=lambda: deque(maxlen=10))
    season: int | None = None
    season_games: int = 0
    season_wins: int = 0
    season_margin: float = 0.0
    ewm: dict[str, _Ewm] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.short = deque(maxlen=self.window)
        self.ewm = {key: _Ewm(self.ewm_span) for key in _EWM_KEYS}

    def roll_season(self, season: int) -> None:
        if self.season is not None and season != self.season:
            self.season_games, self.season_wins, self.season_margin = 0, 0, 0.0
            # Shrink form toward neutral between seasons instead of wiping it.
            for key in ("margin", "net_rating"):
                if self.ewm[key].value is not None:
                    self.ewm[key].value *= 0.5
        self.season = season

    def summary(self, now: datetime) -> dict[str, float | None]:
        short = list(self.short)
        recent = list(self.recent)
        out: dict[str, float | None] = {}
        if self.dates:
            rest = (now - self.dates[-1]).total_seconds() / 86400.0
            out["rest_days"] = max(0.0, min(rest, 10.0))
            out["b2b"] = 1.0 if rest < 1.6 else 0.0
            out["games_last7"] = float(sum(1 for d in self.dates if (now - d) <= timedelta(days=7)))
        else:
            out["rest_days"] = None
            out["b2b"] = 0.0
            out["games_last7"] = 0.0
        if short:
            fg_m = sum(r["fgm"] for r in short if r["fgm"] is not None and r["fga"] is not None)
            fg_a = sum(r["fga"] for r in short if r["fgm"] is not None and r["fga"] is not None)
            out.update({
                "win_rate": _mean(r["win"] for r in short),
                "points_for": _mean(r["points_for"] for r in short),
                "points_against": _mean(r["points_against"] for r in short),
                "point_diff": _mean(r["points_for"] - r["points_against"] for r in short),
                "fg_pct": fg_m / fg_a if fg_a else None,
                "three_made": _mean(r["three_made"] for r in short),
                "rebounds": _mean(r["rebounds"] for r in short),
                "assists": _mean(r["assists"] for r in short),
                "turnovers": _mean(r["turnovers"] for r in short),
            })
        else:
            out.update({k: None for k in ("win_rate", "points_for", "points_against", "point_diff",
                                           "fg_pct", "three_made", "rebounds", "assists", "turnovers")})
        out["win_rate_last10"] = _mean(r["win"] for r in recent) if recent else None
        out["margin_last10"] = _mean(r["points_for"] - r["points_against"] for r in recent) if recent else None
        out["season_games"] = float(self.season_games)
        out["season_win_pct"] = self.season_wins / self.season_games if self.season_games else None
        out["season_margin"] = self.season_margin / self.season_games if self.season_games else None
        for key in _EWM_KEYS:
            out[f"ewm_{key}"] = self.ewm[key].value
        return out

    def record(self, played_at: datetime, own_pts: float, opp_pts: float,
               own: dict[str, Any], opp: dict[str, Any]) -> None:
        win = 1 if own_pts > opp_pts else 0
        off_reb, def_reb = _num(own.get("offensive_rebounds")), _num(own.get("defensive_rebounds"))
        rebounds = None if off_reb is None and def_reb is None else (off_reb or 0) + (def_reb or 0)
        entry = {
            "win": win, "points_for": own_pts, "points_against": opp_pts,
            "fgm": _num(own.get("field_goals_made")), "fga": _num(own.get("field_goals_attempted")),
            "three_made": _num(own.get("three_pointers_made")), "rebounds": rebounds,
            "assists": _num(own.get("assists")), "turnovers": _num(own.get("turnovers")),
        }
        self.short.append(entry)
        self.recent.append(entry)
        self.dates.append(played_at)
        self.season_games += 1
        self.season_wins += win
        self.season_margin += own_pts - opp_pts

        poss = possessions(own, opp) if own and opp else None
        ff, opp_ff = four_factors(own, opp), four_factors(opp, own)
        updates: dict[str, float | None] = {
            "margin": own_pts - opp_pts, "points_for": own_pts, "points_against": opp_pts,
            "efg": ff["efg"], "tov_pct": ff["tov_pct"], "orb_pct": ff["orb_pct"], "ftr": ff["ftr"],
            "three_rate": ff["three_rate"], "opp_efg": opp_ff["efg"], "opp_tov_pct": opp_ff["tov_pct"],
            "drb_pct": (1.0 - opp_ff["orb_pct"]) if opp_ff["orb_pct"] is not None else None,
            "opp_ftr": opp_ff["ftr"],
        }
        if poss and poss > 50:
            off_rtg, def_rtg = 100.0 * own_pts / poss, 100.0 * opp_pts / poss
            updates.update({"pace": poss, "off_rating": off_rtg, "def_rating": def_rtg, "net_rating": off_rtg - def_rtg})
        for key, value in updates.items():
            self.ewm[key].update(value)


def _diff(a: float | None, b: float | None) -> float | None:
    return a - b if a is not None and b is not None else None


def _add(a: float | None, b: float | None) -> float | None:
    return a + b if a is not None and b is not None else None


def build_feature_rows(
    games: list[dict[str, Any]],
    team_stats: list[dict[str, Any]],
    market_home_prob: dict[str, float] | None = None,
    history_window: int = 5,
    elo_k: float = 20.0,
    home_advantage: float = 100.0,
    include_upcoming: bool = False,
    market_quotes: dict[str, dict[str, Any]] | None = None,
    ewm_span: float = 10.0,
) -> list[dict[str, Any]]:
    """Build chronological pre-game rows; current-game stats are added only afterwards."""
    if history_window < 1:
        raise ValueError("history_window must be >= 1")
    stats_by_game: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for stat in team_stats:
        stats_by_game[str(stat.get("game_provider_id"))][str(stat.get("team_provider_id"))] = stat

    ordered = []
    for game in games:
        status = str(game.get("game_status", "")).lower()
        is_final = status == "final" and game.get("home_score") is not None and game.get("away_score") is not None
        is_upcoming = include_upcoming and status == "scheduled"
        if not (is_final or is_upcoming):
            continue
        played_at = _as_datetime(game.get("scheduled_tipoff_at") or game.get("game_date"))
        if played_at is not None:
            ordered.append((played_at, game))
    ordered.sort(key=lambda item: (item[0], str(item[1].get("game_provider_id"))))

    states: dict[str, TeamState] = defaultdict(lambda: TeamState(window=history_window, ewm_span=ewm_span))
    elo = EloRatings(k=elo_k, home_advantage=home_advantage)
    quotes = market_quotes or {}
    rows: list[dict[str, Any]] = []

    for played_at, game in ordered:
        game_id = str(game["game_provider_id"])
        home_id = str(game["home_team_provider_id"])
        away_id = str(game["away_team_provider_id"])
        season = int(game.get("season_end_year") or season_from_date(played_at))
        states[home_id].roll_season(season)
        states[away_id].roll_season(season)
        h = states[home_id].summary(played_at)
        a = states[away_id].summary(played_at)
        elo_prob = elo.expected_home(home_id, away_id, season)
        elo_diff = elo.elo_diff(home_id, away_id, season)
        has_score = game.get("home_score") is not None and game.get("away_score") is not None
        quote = quotes.get(game_id, {})
        market_prob = (market_home_prob or {}).get(game_id, quote.get("home_prob"))

        row: dict[str, Any] = {
            "game_provider_id": game_id, "played_at": played_at.isoformat(), "season": season,
            "season_type": game.get("season_type"),
            "home_team_provider_id": home_id, "away_team_provider_id": away_id,
            "home_score": int(game["home_score"]) if has_score else None,
            "away_score": int(game["away_score"]) if has_score else None,
            "target_home_win": int(game["home_score"] > game["away_score"]) if has_score else None,
            "target_margin": float(game["home_score"]) - float(game["away_score"]) if has_score else None,
            "target_total": float(game["home_score"]) + float(game["away_score"]) if has_score else None,
            "home_elo_prob": elo_prob, "elo_diff": elo_diff, "elo_spread": elo_to_spread(elo_diff),
            "home_rest_days": h["rest_days"], "away_rest_days": a["rest_days"],
            "rest_days_diff": _diff(h["rest_days"], a["rest_days"]),
            "market_home_prob": market_prob,
            "market_home_odds": quote.get("home_odds"), "market_away_odds": quote.get("away_odds"),
            "market_home_spread": quote.get("home_spread"), "market_total": quote.get("total"),
        }
        for side, s in (("home", h), ("away", a)):
            row[f"{side}_win_rate_last5"] = s["win_rate"]
            row[f"{side}_points_for_last5"] = s["points_for"]
            row[f"{side}_points_against_last5"] = s["points_against"]
            row[f"{side}_point_diff_last5"] = s["point_diff"]
            row[f"{side}_fg_pct_last5"] = s["fg_pct"]
            row[f"{side}_three_made_last5"] = s["three_made"]
            row[f"{side}_rebounds_last5"] = s["rebounds"]
            row[f"{side}_assists_last5"] = s["assists"]
            row[f"{side}_turnovers_last5"] = s["turnovers"]
            for name in _SIDE_FEATURES:
                row[f"{side}_{name}"] = s.get(name)

        row["diff_margin_last10"] = _diff(h["margin_last10"], a["margin_last10"])
        row["diff_season_margin"] = _diff(h["season_margin"], a["season_margin"])
        row["diff_season_win_pct"] = _diff(h["season_win_pct"], a["season_win_pct"])
        row["diff_ewm_margin"] = _diff(h["ewm_margin"], a["ewm_margin"])
        row["diff_ewm_net_rating"] = _diff(h["ewm_net_rating"], a["ewm_net_rating"])
        # Matchup factors: own offence vs opponent defence.
        row["diff_efg_matchup"] = _diff(_diff(h["ewm_efg"], a["ewm_opp_efg"]), _diff(a["ewm_efg"], h["ewm_opp_efg"]))
        row["diff_tov_matchup"] = _diff(_diff(h["ewm_tov_pct"], a["ewm_opp_tov_pct"]), _diff(a["ewm_tov_pct"], h["ewm_opp_tov_pct"]))
        row["diff_orb_matchup"] = _diff(_diff(h["ewm_orb_pct"], a["ewm_drb_pct"]), _diff(a["ewm_orb_pct"], h["ewm_drb_pct"]))
        row["diff_ftr_matchup"] = _diff(_diff(h["ewm_ftr"], a["ewm_opp_ftr"]), _diff(a["ewm_ftr"], h["ewm_opp_ftr"]))
        row["expected_pace"] = 0.5 * h["ewm_pace"] + 0.5 * a["ewm_pace"] if h["ewm_pace"] and a["ewm_pace"] else None
        row["expected_total_proxy"] = (
            0.5 * (_add(h["ewm_points_for"], a["ewm_points_against"]) or 0) + 0.5 * (_add(a["ewm_points_for"], h["ewm_points_against"]) or 0)
            if None not in (h["ewm_points_for"], a["ewm_points_against"], a["ewm_points_for"], h["ewm_points_against"]) else None
        )
        rows.append(row)

        # Upcoming games are prediction targets, never history updates.
        if row["target_home_win"] is None:
            continue

        # Historical labels/stats update state only after this game's feature row is frozen.
        home_stats = stats_by_game.get(game_id, {}).get(home_id, {})
        away_stats = stats_by_game.get(game_id, {}).get(away_id, {})
        hs, as_ = float(game["home_score"]), float(game["away_score"])
        states[home_id].record(played_at, hs, as_, home_stats, away_stats)
        states[away_id].record(played_at, as_, hs, away_stats, home_stats)
        elo.update(home_id, away_id, hs, as_, season)
    return rows


# ---------------------------------------------------------------------------
# Market aggregation (strict point-in-time)
# ---------------------------------------------------------------------------

def _verified_before_tipoff(row: dict[str, Any], tipoffs: dict[str, datetime | str]) -> bool:
    game_id = str(row.get("game_provider_id"))
    tipoff, captured = _as_datetime(tipoffs.get(game_id)), _as_datetime(row.get("source_captured_at"))
    if not tipoff or not captured or captured >= tipoff:
        return False
    return row.get("snapshot_time_quality") == "verified_point_in_time"


def aggregate_verified_market(
    odds_rows: list[dict[str, Any]], tipoffs: dict[str, datetime | str]
) -> dict[str, dict[str, Any]]:
    """Latest verified pre-tipoff quote per book, then median across books.

    Returns per game: home_prob (no-vig), home_odds, away_odds (median decimal
    prices actually offered), home_spread and total (median closing lines).
    """
    latest: dict[str, dict[str, dict[tuple[str, str], tuple[datetime, float, float | None]]]] = defaultdict(lambda: defaultdict(dict))
    for row in odds_rows:
        if not _verified_before_tipoff(row, tipoffs):
            continue
        market = str(row.get("market_type", "")).lower()
        selection = str(row.get("selection", "")).lower()
        odds = _num(row.get("decimal_odds"))
        if odds is None or odds <= 1.0:
            continue
        if market == "moneyline" and selection not in {"home", "away"}:
            continue
        if market not in {"moneyline", "spread", "total"}:
            continue
        captured = _as_datetime(row.get("source_captured_at"))
        book = str(row.get("sportsbook") or "unknown")
        key = (market, selection)
        previous = latest[str(row.get("game_provider_id"))][book].get(key)
        if previous is None or captured > previous[0]:
            latest[str(row.get("game_provider_id"))][book][key] = (captured, odds, _num(row.get("line_value")))

    result: dict[str, dict[str, Any]] = {}
    for game_id, books in latest.items():
        probs, home_odds, away_odds, spreads, totals = [], [], [], [], []
        for sides in books.values():
            if ("moneyline", "home") in sides and ("moneyline", "away") in sides:
                h, a = sides[("moneyline", "home")][1], sides[("moneyline", "away")][1]
                probs.append((1 / h) / (1 / h + 1 / a))
                home_odds.append(h)
                away_odds.append(a)
            if ("spread", "home") in sides and sides[("spread", "home")][2] is not None:
                spreads.append(sides[("spread", "home")][2])
            for sel in ("over", "under"):
                if ("total", sel) in sides and sides[("total", sel)][2] is not None:
                    totals.append(sides[("total", sel)][2])
                    break
        quote: dict[str, Any] = {}
        if probs:
            quote.update(home_prob=float(median(probs)), home_odds=float(median(home_odds)), away_odds=float(median(away_odds)))
        if spreads:
            quote["home_spread"] = float(median(spreads))
        if totals:
            quote["total"] = float(median(totals))
        if quote:
            result[game_id] = quote
    return result


def aggregate_verified_market_probabilities(
    odds_rows: list[dict[str, Any]], tipoffs: dict[str, datetime | str]
) -> dict[str, float]:
    """Use paired no-vig moneyline prices from verified snapshots strictly before tipoff."""
    return {
        game_id: quote["home_prob"]
        for game_id, quote in aggregate_verified_market(odds_rows, tipoffs).items()
        if "home_prob" in quote
    }
