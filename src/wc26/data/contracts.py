"""Schema and distribution checks on ingest.

A contract violation fails the load -- loudly, never with a warning.
"""

import pandas as pd

MATCH_COLUMNS = [
    "match_id",
    "event_time",
    "knowledge_time",
    "home_id",
    "away_id",
    "home_score",
    "away_score",
    "tournament",
    "comp_tier",
    "neutral",
    "host_home",
]

FIXTURE_COLUMNS = ["match_id", "event_time", "home_id", "away_id", "tournament", "neutral"]

VALID_TIERS = {"worldcup", "continental", "qualifier", "nationsleague", "friendly", "other"}


class ContractViolation(ValueError):
    """Raised when ingested data violates an ingestion contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractViolation(message)


def require_score_range(scores: pd.DataFrame) -> None:
    """Scores present and in [0, 31].

    Must run on the values BEFORE any narrowing cast: int8 wraps silently
    (287 becomes 31), which would let a parsing disaster sail past this check.
    """
    _require(not scores.isna().any().any(), "completed matches contain NA scores")
    _require(
        bool(((scores >= 0) & (scores <= 31)).all().all()),
        "scores outside [0, 31] -- parsing disaster suspected",
    )


def _require_tz_aware(table: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        is_datetime = pd.api.types.is_datetime64_any_dtype(table[column])
        _require(
            is_datetime and table[column].dt.tz is not None,
            f"{column} must be a tz-aware timestamp, got {table[column].dtype}",
        )


def validate_matches(matches: pd.DataFrame) -> None:
    """Fail the load if the completed-matches table violates any contract."""
    _require(
        list(matches.columns) == MATCH_COLUMNS,
        f"matches columns {list(matches.columns)} != contract {MATCH_COLUMNS}",
    )
    _require(len(matches) > 0, "matches table is empty")

    require_score_range(matches[["home_score", "away_score"]])

    # The schema is enforced here, not trusted from the producer.
    _require_tz_aware(matches, ("event_time", "knowledge_time"))
    for column in ("home_score", "away_score"):
        _require(
            str(matches[column].dtype) == "int8",
            f"{column} must be int8, got {matches[column].dtype}",
        )
    for column in ("tournament", "comp_tier"):
        _require(
            isinstance(matches[column].dtype, pd.CategoricalDtype),
            f"{column} must be categorical, got {matches[column].dtype}",
        )
    for column in ("neutral", "host_home"):
        _require(
            matches[column].dtype == bool,
            f"{column} must be bool, got {matches[column].dtype}",
        )

    # No duplicate match_id with conflicting scores (re-ingest of the same match is fine).
    per_id = matches.groupby("match_id")[["home_score", "away_score"]].nunique()
    conflicting = per_id[(per_id > 1).any(axis=1)]
    _require(
        conflicting.empty,
        f"{len(conflicting)} match_ids carry conflicting scores: {conflicting.index[:5].tolist()}",
    )

    years = matches["event_time"].dt.year
    _require(
        bool(years.between(1870, 2027).all()),
        f"event_time outside sane range: [{years.min()}, {years.max()}]",
    )
    _require(
        bool((matches["knowledge_time"] >= matches["event_time"]).all()),
        "knowledge_time earlier than event_time",
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
        total_goals=matches["home_score"].astype(int) + matches["away_score"].astype(int),
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
    _require_tz_aware(fixtures, ("event_time",))
    _require(
        fixtures["neutral"].dtype == bool,
        f"neutral must be bool, got {fixtures['neutral'].dtype}",
    )
