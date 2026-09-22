---
name: Same-date cutoff
description: Leakage-control decision for EPL historical features when kickoff times are unavailable.
---

Pre-match feature generation must use only matches whose dates are strictly earlier than the current fixture date. All fixtures on the same date are evaluated before any results from that date are added to team histories, league tables, or head-to-head state.

**Why:** The source contains dates but no kickoff times, so using another fixture from the same date could introduce information that was not available before the target match.

**How to apply:** Preserve the date-batch processing rule in feature rebuilds, current-fixture inference, validation, and any future feature additions.