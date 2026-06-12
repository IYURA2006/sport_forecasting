"""Ingest: raw CSV -> cleaned tables -> CSV snapshot + manifest."""

import json
from pathlib import Path

import pandas as pd
import pytest

from wc26.data.contracts import ContractViolation
from wc26.data.ingest import build_snapshot, build_tables, load_raw, match_id
from wc26.data.loader import load_fixtures, load_matches

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
    assert pd.api.types.is_integer_dtype(matches["home_score"])


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


def test_corrupt_score_fails_the_load(tmp_path: Path) -> None:
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


def test_tables_sorted_by_date_then_match_id(raw_frame: pd.DataFrame) -> None:
    # ROWS is deliberately out of chronological order; deterministic row order
    # feeds downstream seeded rng consumption.
    matches, fixtures = build_tables(raw_frame)
    for table in (matches, fixtures):
        sort_keys = list(zip(table["match_date"], table["match_id"]))
        assert sort_keys == sorted(sort_keys)


def test_snapshot_roundtrip_and_manifest(tmp_path: Path) -> None:
    raw_path = _write_csv(tmp_path / "results.csv", ROWS)
    snapshots = tmp_path / "snapshots"
    snapshot = build_snapshot(raw_path, snapshots)

    matches = load_matches(snapshots_dir=snapshots)
    fixtures = load_fixtures(snapshots_dir=snapshots)
    assert len(matches) == snapshot.n_matches == 3
    assert len(fixtures) == snapshot.n_fixtures == 2
    # Dates must survive the CSV round-trip as parsed timestamps.
    assert pd.api.types.is_datetime64_any_dtype(matches["match_date"])
    assert matches["neutral"].dtype == bool

    manifest = json.loads((snapshots / "manifest.json").read_text())
    assert manifest["raw_sha256"] == snapshot.raw_sha256
    assert manifest["n_matches"] == 3
    assert manifest["date_range"] == ["2022-11-20", "2023-03-24"]


def test_rebuild_overwrites_snapshot_in_place(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots"
    build_snapshot(_write_csv(tmp_path / "results.csv", ROWS), snapshots)
    build_snapshot(_write_csv(tmp_path / "results2.csv", ROWS[:3]), snapshots)
    # One snapshot, no accumulation: the manifest and tables reflect the last build.
    manifest = json.loads((snapshots / "manifest.json").read_text())
    assert manifest["n_fixtures"] == 0
    assert load_fixtures(snapshots_dir=snapshots).empty


REAL_CSV = Path(__file__).resolve().parents[2] / "data" / "raw" / "results.csv"


@pytest.mark.skipif(not REAL_CSV.exists(), reason="raw dataset not present (CI)")
def test_real_dataset_passes_all_contracts(tmp_path: Path) -> None:
    """Integration: the full 49k-row table ingests cleanly end to end."""
    snapshot = build_snapshot(REAL_CSV, tmp_path / "snapshots")
    matches = load_matches(snapshots_dir=tmp_path / "snapshots")
    fixtures = load_fixtures(snapshots_dir=tmp_path / "snapshots")

    assert snapshot.n_matches == len(matches) > 49_000
    assert snapshot.n_fixtures == len(fixtures) == 72  # the 2026 group stage
    teams_2026 = set(fixtures["home_id"]) | set(fixtures["away_id"])
    assert len(teams_2026) == 48
    assert matches["match_id"].is_unique
