"""Entity resolution: canonical team ids and competition-tier mapping."""

import pytest

from wc26.data.entities import canonical_team_id, comp_tier


def test_alias_maps_to_canonical() -> None:
    assert canonical_team_id("USA") == "United States"
    assert canonical_team_id("Korea Republic") == "South Korea"
    assert canonical_team_id("Türkiye") == "Turkey"


def test_canonical_name_passes_through() -> None:
    assert canonical_team_id("Brazil") == "Brazil"


def test_whitespace_is_stripped() -> None:
    assert canonical_team_id("  USA ") == "United States"


def test_successor_states_not_merged() -> None:
    # Yugoslavia and Serbia are deliberately separate entities.
    assert canonical_team_id("Yugoslavia") == "Yugoslavia"
    assert canonical_team_id("Serbia") == "Serbia"


@pytest.mark.parametrize(
    ("label", "tier"),
    [
        ("FIFA World Cup", "worldcup"),
        ("FIFA World Cup qualification", "qualifier"),
        ("UEFA Euro qualification", "qualifier"),
        ("UEFA Nations League", "nationsleague"),
        ("Friendly", "friendly"),
        ("Copa América", "continental"),
        ("AFC Asian Cup", "continental"),
    ],
)
def test_comp_tier_mapping(label: str, tier: str) -> None:
    assert comp_tier(label) == tier


def test_unknown_tournament_is_other() -> None:
    assert comp_tier("King's Cup") == "other"
