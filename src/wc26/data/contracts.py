"""Schema and distribution checks on ingest.

A contract violation fails the load -- loudly, never with a warning.
"""

import pandas as pd

MATCH_COLUMNS = [
    "match_id",
    "match_date",
    "home_id",
    "away_id",
    "home_score",
    "away_score",
    "tournament",
    "comp_tier",
    "neutral",
]

FIXTURE_COLUMNS = ["match_id", "match_date", "home_id", "away_id", "tournament", "neutral"]

VALID_TIERS = {"worldcup", "continental", "qualifier", "friendly", "other"}


class ContractViolation(ValueError):
    """Raised when ingested data violates an ingestion contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractViolation(message)


def require_score_range(scores: pd.DataFrame) -> None:
    """Scores present and in [0, 31] -- catches parsing disasters."""
    _require(not scores.isna().any().any(), "completed matches contain NA scores")
    _require(
        bool(((scores >= 0) & (scores <= 31)).all().all()),
        "scores outside [0, 31] -- parsing disaster suspected",
    )


def validate_matches(matches: pd.DataFrame) -> None:
    """Fail the load if the completed-matches table violates any contract."""
    _require(
        list(matches.columns) == MATCH_COLUMNS,
        f"matches columns {list(matches.columns)} != contract {MATCH_COLUMNS}",
    )
    _require(len(matches) > 0, "matches table is empty")
    _require(
        pd.api.types.is_datetime64_any_dtype(matches["match_date"]),
        f"match_date must be a parsed date, got {matches['match_date'].dtype}",
    )
    _require(
        matches["neutral"].dtype == bool,
        f"neutral must be bool, got {matches['neutral'].dtype}",
    )

    require_score_range(matches[["home_score", "away_score"]])

    # No duplicate match_id with conflicting scores (re-ingest of the same match is fine).
    per_id = matches.groupby("match_id")[["home_score", "away_score"]].nunique()
    conflicting = per_id[(per_id > 1).any(axis=1)]
    _require(
        conflicting.empty,
        f"{len(conflicting)} match_ids carry conflicting scores: {conflicting.index[:5].tolist()}",
    )

    years = matches["match_date"].dt.year
    _require(
        bool(years.between(1870, 2027).all()),
        f"match_date outside sane range: [{years.min()}, {years.max()}]",
    )
    _require(
        bool(matches["comp_tier"].isin(VALID_TIERS).all()),
        f"unknown comp_tier values: {set(matches['comp_tier'].unique()) - VALID_TIERS}",
    )
    _require(
        bool((matches["home_id"].str.len() > 0).all() and (matches["away_id"].str.len() > 0).all()),
        "empty team id after canonicalisation",
    )

    # Distribution check: mean goals/match per decade catches parsing disasters.
    # A [2.0, 3.5] band describes the modern game, but measured reality is
    # ~4.0-5.6 goals/match before 1960 (1900s: 4.16, 1930s: 4.32, 1950s: 4.00).
    # DECISION: enforce [2.0, 3.5] from 1960 onward and a loose parsing-disaster
    # bound [1.0, 7.0] for earlier decades, rather than silently widening the
    # modern band. Decades with < 100 matches are too noisy to judge either way.
    by_decade = matches.assign(
        decade=(years // 10) * 10,
        total_goals=matches["home_score"] + matches["away_score"],
    ).groupby("decade")["total_goals"]
    means = by_decade.mean()[by_decade.count() >= 100]
    modern = means[means.index >= 1960]
    bad_modern = modern[(modern < 2.0) | (modern > 3.5)]
    _require(
        bad_modern.empty,
        f"mean goals/match out of [2.0, 3.5] for modern decades: {bad_modern.to_dict()}",
    )
    bad_early = means[(means < 1.0) | (means > 7.0)]
    _require(
        bad_early.empty,
        f"mean goals/match out of [1.0, 7.0] for decades: {bad_early.to_dict()}",
    )


def validate_fixtures(fixtures: pd.DataFrame) -> None:
    """Fail the load if the upcoming-fixtures table violates any contract."""
    _require(
        list(fixtures.columns) == FIXTURE_COLUMNS,
        f"fixtures columns {list(fixtures.columns)} != contract {FIXTURE_COLUMNS}",
    )
    _require(
        bool(fixtures["match_id"].is_unique),
        "duplicate match_id in fixtures",
    )
    _require(
        fixtures["neutral"].dtype == bool,
        f"neutral must be bool, got {fixtures['neutral'].dtype}",
    )
