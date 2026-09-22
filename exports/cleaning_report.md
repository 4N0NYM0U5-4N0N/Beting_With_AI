# Cleaning Report

- Source rows read: 5701
- Valid historical matches exported: 5700

## Decisions

- Loaded authoritative source /home/runner/workspace/data/raw/epl_historical_clean.csv.
- Blank fields are not imputed. Rows missing core match identifiers are excluded.
- Historical dates are parsed with explicit day-first formats; no locale-inferred parsing is used.
- Suspicious statistical records are retained and reported, not silently corrected.
- Features will use only information from dates strictly before the fixture date.
- Excluded 1 malformed row(s) with missing core fields; the original values remain in the source file.
- No duplicate match keys were found.
- Retained 5700 valid historical matches after structural cleaning.
- Flagged 1 suspicious conditions without modifying their source values.

## Flagged anomalies

- Source row 3810 (15/08/2021 Newcastle vs West Ham): **impossible_statistical_relationship** — away_shots_on_target (9) exceeds away_shots (8).

Suspicious records are reported rather than corrected. The impossible shots-on-target relationship is not silently changed; it remains visible in this audit.
