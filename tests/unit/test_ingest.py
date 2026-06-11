"""Ingest: raw CSV -> typed tables -> content-addressed snapshot."""

from pathlib import Path

import pandas as pd
import pytest

from wc26.data.contracts import ContractViolation
from wc26.data.ingest import (
    KNOWLEDGE_LAG,
    build_snapshot,
    build_tables,
    latest_snapshot_id,
    load_raw,
    load_snapshot,
    match_id,
)

HEADER = "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral"

ROWS = [
    "2022-12-18,Argentina,France,3,3,FIFA World Cup,Lusail,Qatar,TRUE",
    "2022-11-20,Qatar,Ecuador,0,2,FIFA World Cup,Al Khor,Qatar,FALSE",
    "2023-03-24,Germany,Peru,2,0,Friendly,Mainz,Germany,FALSE",
    "2026-06-11,Mexico,South Africa,NA,NA,FIFA World Cup,Mexico City,Mexico,FALSE",
    "2026-06-13,Haiti,Scotland,NA,NA,FIFA World Cup,Boston,United States,TRUE",
]


def _write_csv(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join([HEADER, *rows]) + "\n")
    return path


@pytest.fixture
def raw_frame(tmp_path: Path) -> pd.DataFrame:
    return load_raw(_write_csv(tmp_path / "results.csv", ROWS))


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError, match="martj42"):
        load_raw(Path("/nonexistent/results.csv"))


def test_match_id_excludes_scores() -> None:
    # Same fixture before and after the result is known -> same id.
    assert match_id("2026-06-11", "Mexico", "South Africa", "FIFA World Cup") == match_id(
        "2026-06-11", "Mexico", "South Africa", "FIFA World Cup"
    )
    assert match_id("2026-06-11", "Mexico", "South Africa", "FIFA World Cup") != match_id(
        "2026-06-11", "South Africa", "Mexico", "FIFA World Cup"
    )


def test_completed_and_scheduled_rows_split(raw_frame: pd.DataFrame) -> None:
    matches, fixtures = build_tables(raw_frame)
    assert len(matches) == 3
    assert len(fixtures) == 2
    assert matches["home_score"].dtype == "int8"


def test_knowledge_time_is_event_plus_three_hours(raw_frame: pd.DataFrame) -> None:
    matches, _ = build_tables(raw_frame)
    assert (matches["knowledge_time"] - matches["event_time"] == KNOWLEDGE_LAG).all()


def test_host_home_only_in_final_tournaments(raw_frame: pd.DataFrame) -> None:
    matches, _ = build_tables(raw_frame)
    by_home = matches.set_index("home_id")["host_home"]
    assert bool(by_home["Qatar"])  # host playing at home in a World Cup
    assert not bool(by_home["Argentina"])  # neutral-venue final
    assert not bool(by_home["Germany"])  # non-neutral friendly is not hosting


def test_canonicalisation_applied(tmp_path: Path) -> None:
    raw = load_raw(
        _write_csv(
            tmp_path / "alias.csv",
            ["2023-03-24,USA,Korea Republic,1,0,Friendly,Austin,United States,FALSE"],
        )
    )
    matches, _ = build_tables(raw)
    assert matches.loc[0, "home_id"] == "United States"
    assert matches.loc[0, "away_id"] == "South Korea"


def test_known_conflicting_rows_are_dropped(tmp_path: Path) -> None:
    raw = load_raw(
        _write_csv(
            tmp_path / "conflict.csv",
            [
                "1974-02-17,Tahiti,New Caledonia,2,1,Friendly,Papeete,Tahiti,FALSE",
                "1974-02-17,Tahiti,New Caledonia,1,2,Friendly,Papeete,Tahiti,FALSE",
                *ROWS,
            ],
        )
    )
    matches, _ = build_tables(raw)
    assert "Tahiti" not in set(matches["home_id"])


def test_exact_duplicate_rows_collapse_to_one(tmp_path: Path) -> None:
    raw = load_raw(_write_csv(tmp_path / "dup.csv", [ROWS[0], ROWS[0], *ROWS[1:]]))
    matches, _ = build_tables(raw)
    assert matches["match_id"].is_unique


def test_new_conflicting_scores_fail_the_load(tmp_path: Path) -> None:
    # A conflict NOT on the known-exclusion list must still fail, end to end.
    raw_path = _write_csv(
        tmp_path / "conflict.csv",
        [
            "2023-03-24,Germany,Peru,2,0,Friendly,Mainz,Germany,FALSE",
            "2023-03-24,Germany,Peru,0,2,Friendly,Mainz,Germany,FALSE",
        ],
    )
    with pytest.raises(ContractViolation, match="conflicting scores"):
        build_snapshot(raw_path, tmp_path / "snapshots")


def test_corrupt_score_fails_before_int8_cast(tmp_path: Path) -> None:
    # int8 wraps 287 -> 31 silently; the range contract must fire on raw values.
    raw = load_raw(
        _write_csv(
            tmp_path / "corrupt.csv",
            ["2023-03-24,Germany,Peru,287,0,Friendly,Mainz,Germany,FALSE"],
        )
    )
    with pytest.raises(ContractViolation, match=r"\[0, 31\]"):
        build_tables(raw)


def test_missing_neutral_fails(tmp_path: Path) -> None:
    raw = load_raw(
        _write_csv(
            tmp_path / "noneutral.csv",
            ["2023-03-24,Germany,Peru,2,0,Friendly,Mainz,Germany,"],
        )
    )
    with pytest.raises(ContractViolation, match="neutral"):
        build_tables(raw)


def test_completed_result_supersedes_stale_schedule_row(tmp_path: Path) -> None:
    # Daily refresh case: the played result lands while the appended schedule
    # row (NA scores) is still present. The result wins; no id in both tables.
    played = "2026-06-11,Mexico,South Africa,2,1,FIFA World Cup,Mexico City,Mexico,FALSE"
    stale = "2026-06-11,Mexico,South Africa,NA,NA,FIFA World Cup,Mexico City,Mexico,FALSE"
    raw = load_raw(_write_csv(tmp_path / "stale.csv", [played, stale]))
    matches, fixtures = build_tables(raw)
    assert len(matches) == 1
    assert fixtures.empty


def test_tables_sorted_by_event_time_then_match_id(raw_frame: pd.DataFrame) -> None:
    # ROWS is deliberately out of chronological order; deterministic row order
    # feeds downstream seeded rng consumption.
    matches, fixtures = build_tables(raw_frame)
    for table in (matches, fixtures):
        sort_keys = list(zip(table["event_time"], table["match_id"]))
        assert sort_keys == sorted(sort_keys)


def test_snapshot_is_content_addressed(tmp_path: Path) -> None:
    raw_path = _write_csv(tmp_path / "results.csv", ROWS)
    snapshots = tmp_path / "snapshots"
    first = build_snapshot(raw_path, snapshots)
    second = build_snapshot(raw_path, snapshots)  # identical bytes -> identical id
    assert first.snapshot_id == second.snapshot_id

    changed = _write_csv(tmp_path / "results2.csv", ROWS[:-1])
    third = build_snapshot(changed, snapshots)
    assert third.snapshot_id != first.snapshot_id


def test_snapshot_roundtrip_and_manifest(tmp_path: Path) -> None:
    raw_path = _write_csv(tmp_path / "results.csv", ROWS)
    snapshots = tmp_path / "snapshots"
    snapshot = build_snapshot(raw_path, snapshots)

    matches, fixtures = load_snapshot(snapshots, snapshot.snapshot_id)
    assert len(matches) == snapshot.n_matches == 3
    assert len(fixtures) == snapshot.n_fixtures == 2
    assert latest_snapshot_id(snapshots) == snapshot.snapshot_id

    # Dtypes must survive the parquet round-trip, not just the row counts.
    assert matches["home_score"].dtype == "int8"
    assert isinstance(matches["tournament"].dtype, pd.CategoricalDtype)
    assert isinstance(matches["comp_tier"].dtype, pd.CategoricalDtype)
    assert matches["event_time"].dt.tz is not None
    assert matches["host_home"].dtype == bool
    assert fixtures["neutral"].dtype == bool

    manifest = (snapshots / "manifest.json").read_text()
    assert snapshot.snapshot_id in manifest
    assert "raw_sha256" in manifest


def test_latest_snapshot_id_picks_newest_deterministically(tmp_path: Path) -> None:
    import json

    manifest = {
        "aaa": {"created": "2026-06-10T00:00:00.000000+00:00"},
        "zzz": {"created": "2026-06-12T00:00:00.000000+00:00"},
        "mmm": {"created": "2026-06-12T00:00:00.000000+00:00"},  # tie with zzz
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert latest_snapshot_id(tmp_path) == "zzz"  # newest; tie broken by id


REAL_CSV = Path(__file__).resolve().parents[2] / "data" / "raw" / "results.csv"


@pytest.mark.skipif(not REAL_CSV.exists(), reason="raw dataset not present (CI)")
def test_real_dataset_passes_all_contracts(tmp_path: Path) -> None:
    """Integration: the full 49k-row table ingests cleanly end to end."""
    snapshot = build_snapshot(REAL_CSV, tmp_path / "snapshots")
    matches, fixtures = load_snapshot(tmp_path / "snapshots", snapshot.snapshot_id)

    assert snapshot.n_matches > 49_000
    assert snapshot.n_fixtures == 72  # the 2026 group stage
    teams_2026 = set(fixtures["home_id"]) | set(fixtures["away_id"])
    assert len(teams_2026) == 48
    assert matches["match_id"].is_unique
