"""Canonical mapping from ISO-3166 "inverted" country names to natural phrasing.

The reverse-geocoder emits names like "Korea, Republic of"; templated queries and
training captions read better (and must match each other) as "Republic of Korea".
Keep this in sync with the query-generation notebook.
"""

COUNTRY_DISPLAY: dict[str, str] = {
    "Bolivia, Plurinational State of": "Plurinational State of Bolivia",
    "Congo, The Democratic Republic of the": "Democratic Republic of the Congo",
    "Iran, Islamic Republic of": "Islamic Republic of Iran",
    "Korea, Democratic People's Republic of": "Democratic People's Republic of Korea",
    "Korea, Republic of": "Republic of Korea",
    "Micronesia, Federated States of": "Federated States of Micronesia",
    "Moldova, Republic of": "Republic of Moldova",
    "Palestine, State of": "State of Palestine",
    "Saint Helena, Ascension and Tristan da Cunha": "Saint Helena",
    "Taiwan, Province of China": "Taiwan",
    "Tanzania, United Republic of": "United Republic of Tanzania",
    "Venezuela, Bolivarian Republic of": "Bolivarian Republic of Venezuela",
    "Virgin Islands, British": "British Virgin Islands",
    "Virgin Islands, U.S.": "U.S. Virgin Islands",
    # kept as-is: genuine multi-island name, not an inversion artifact
    "Bonaire, Sint Eustatius and Saba": "Bonaire, Sint Eustatius and Saba",
}


def display_country(name: str) -> str:
    """Natural phrasing for a (possibly ISO-inverted) country name."""
    return COUNTRY_DISPLAY.get(name, name)
