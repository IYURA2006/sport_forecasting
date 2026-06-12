"""Team-name canonicalisation and competition-tier mapping.

Rule: mapping may use date/venue/team fields, never score fields, so identity
resolution cannot leak results into anything downstream. Successor states
(e.g. Yugoslavia -> Serbia) are NOT merged -- they stay separate entities,
documented here by their absence from the alias table.
"""

# Canonical IDs are the martj42 dataset's own names (the largest source wins).
# Aliases cover variants used by FIFA, odds feeds, and common English forms.
TEAM_ALIASES: dict[str, str] = {
    # The results data never says bare "Ireland" (it has "Republic of Ireland" and
    # "Northern Ireland"); odds feeds conventionally mean the Republic.
    "Ireland": "Republic of Ireland",
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "China PR": "China",
    "Congo DR": "DR Congo",
    "Cabo Verde": "Cape Verde",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Macedonia": "North Macedonia",
    "Swaziland": "Eswatini",
}

# Continental-championship FINAL tournaments (not qualifiers — those are caught
# by the "qualification" substring rule below).
_CONTINENTAL_FINALS = {
    "Copa América",
    "African Cup of Nations",
    "Africa Cup of Nations",
    "AFC Asian Cup",
    "UEFA Euro",
    "Gold Cup",
    "CONCACAF Championship",
    "Oceania Nations Cup",
    "OFC Nations Cup",
    "Confederations Cup",
    "FIFA Confederations Cup",
}


def canonical_team_id(name: str) -> str:
    """Map a raw team name to its canonical ID (identity if already canonical)."""
    cleaned = name.strip()
    return TEAM_ALIASES.get(cleaned, cleaned)


def comp_tier(tournament: str) -> str:
    """Map a raw tournament label to a coarse competition tier.

    Tiers: {worldcup, continental, qualifier, friendly, other}. Tiers only drive
    the friendly down-weight and fold selection, so every other competitive
    label (Nations League, minor cups) lands in "other" with full weight.
    """
    label = tournament.strip()
    if label == "FIFA World Cup":
        return "worldcup"
    if "qualification" in label.lower():
        return "qualifier"
    if label == "Friendly":
        return "friendly"
    if label in _CONTINENTAL_FINALS:
        return "continental"
    return "other"
