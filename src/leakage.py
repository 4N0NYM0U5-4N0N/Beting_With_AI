from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import SAME_MATCH_STAT_COLUMNS


def audit_features(
    feature_frame: pd.DataFrame,
    feature_dictionary: list[dict[str, str]],
    predictor_columns: list[str],
) -> pd.DataFrame:
    metadata = {item["feature_name"]: item for item in feature_dictionary}
    rows: list[dict[str, str]] = []
    for name in predictor_columns:
        item = dict(metadata.get(name, {}))
        item.setdefault("feature_name", name)
        item.setdefault("source_columns", "")
        item.setdefault("calculation_method", "")
        item.setdefault("information_cutoff", "")
        item.setdefault("same_match_information_can_enter", "UNKNOWN")
        item.setdefault("future_matches_can_enter", "UNKNOWN")
        classification = "SAFE"
        reason = "Metadata states a strict pre-match cutoff."
        lowered = name.lower()
        if any(stat.lower() in lowered for stat in SAME_MATCH_STAT_COLUMNS):
            classification = "LEAKAGE"
            reason = "Feature name exposes same-match target/statistic information."
        elif not item["source_columns"] or item["information_cutoff"] != "strictly before current fixture date":
            classification = "UNKNOWN"
            reason = "Missing verifiable source or information cutoff."
        elif (
            item["same_match_information_can_enter"].upper() != "NO"
            or item["future_matches_can_enter"].upper() != "NO"
        ):
            classification = "LEAKAGE"
            reason = "Metadata permits same-match or future information."
        item["classification"] = classification
        item["audit_reason"] = reason
        rows.append(item)
    return pd.DataFrame(rows).sort_values("feature_name").reset_index(drop=True)


def write_leakage_report(audit: pd.DataFrame, path: str | Path) -> None:
    counts = audit["classification"].value_counts().to_dict() if not audit.empty else {}
    lines = [
        "# Leakage Audit",
        "",
        "The audit is conservative: only predictors with explicit source metadata and a "
        "strictly-before-fixture cutoff are eligible for modeling.",
        "",
        f"- SAFE features: {counts.get('SAFE', 0)}",
        f"- LEAKAGE features: {counts.get('LEAKAGE', 0)}",
        f"- UNKNOWN features: {counts.get('UNKNOWN', 0)}",
        "",
        "## Classification rules",
        "",
        "- SAFE: source is historical pre-match information, cutoff is strictly before the fixture date, and same-match/future entry is explicitly disallowed.",
        "- LEAKAGE: same-match statistics or target information can enter the predictor, or metadata permits it.",
        "- UNKNOWN: provenance or cutoff cannot be verified. UNKNOWN predictors are excluded from modeling.",
        "",
        "## Feature-level findings",
        "",
        "| Feature | Classification | Source | Method | Reason |",
        "|---|---|---|---|---|",
    ]
    for _, row in audit.iterrows():
        lines.append(
            f"| `{row['feature_name']}` | **{row['classification']}** | "
            f"{row['source_columns']} | {row['calculation_method']} | {row['audit_reason']} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
