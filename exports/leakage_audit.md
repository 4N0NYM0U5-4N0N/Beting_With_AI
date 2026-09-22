# Leakage Audit

The audit is conservative: only predictors with explicit source metadata and a strictly-before-fixture cutoff are eligible for modeling.

- SAFE features: 80
- LEAKAGE features: 0
- UNKNOWN features: 0

## Classification rules

- SAFE: source is historical pre-match information, cutoff is strictly before the fixture date, and same-match/future entry is explicitly disallowed.
- LEAKAGE: same-match statistics or target information can enter the predictor, or metadata permits it.
- UNKNOWN: provenance or cutoff cannot be verified. UNKNOWN predictors are excluded from modeling.

## Feature-level findings

| Feature | Classification | Source | Method | Reason |
|---|---|---|---|---|
| `away_league_position_before_match` | **SAFE** | prior results in current season | table sorted by points, goal difference, goals scored | Metadata states a strict pre-match cutoff. |
| `away_odds` | **SAFE** | historical bookmaker odds | decimal odds converted to implied and normalized probabilities | Metadata states a strict pre-match cutoff. |
| `away_team_away_goal_difference` | **SAFE** | prior away-only matches for team | venue-restricted cumulative mean | Metadata states a strict pre-match cutoff. |
| `away_team_away_goals_conceded` | **SAFE** | prior away-only matches for team | venue-restricted cumulative mean | Metadata states a strict pre-match cutoff. |
| `away_team_away_goals_scored` | **SAFE** | prior away-only matches for team | venue-restricted cumulative mean | Metadata states a strict pre-match cutoff. |
| `away_team_away_matches_played` | **SAFE** | prior away-only matches for team | venue-restricted cumulative mean | Metadata states a strict pre-match cutoff. |
| `away_team_away_points_per_match` | **SAFE** | prior away-only matches for team | venue-restricted cumulative mean | Metadata states a strict pre-match cutoff. |
| `away_team_overall_draws` | **SAFE** | historical match results and statistics | cumulative aggregate | Metadata states a strict pre-match cutoff. |
| `away_team_overall_goal_difference` | **SAFE** | historical match results and statistics | cumulative aggregate | Metadata states a strict pre-match cutoff. |
| `away_team_overall_goals_conceded` | **SAFE** | historical match results and statistics | cumulative aggregate | Metadata states a strict pre-match cutoff. |
| `away_team_overall_goals_scored` | **SAFE** | historical match results and statistics | cumulative aggregate | Metadata states a strict pre-match cutoff. |
| `away_team_overall_losses` | **SAFE** | historical match results and statistics | cumulative aggregate | Metadata states a strict pre-match cutoff. |
| `away_team_overall_matches_played` | **SAFE** | historical match results and statistics | cumulative aggregate | Metadata states a strict pre-match cutoff. |
| `away_team_overall_points_per_match` | **SAFE** | historical match results and statistics | cumulative aggregate | Metadata states a strict pre-match cutoff. |
| `away_team_overall_wins` | **SAFE** | historical match results and statistics | cumulative aggregate | Metadata states a strict pre-match cutoff. |
| `away_team_rolling10_corners` | **SAFE** | historical match statistics | mean of last 10 available matches | Metadata states a strict pre-match cutoff. |
| `away_team_rolling10_goal_difference` | **SAFE** | historical match statistics | mean of last 10 available matches | Metadata states a strict pre-match cutoff. |
| `away_team_rolling10_goals_conceded` | **SAFE** | historical match statistics | mean of last 10 available matches | Metadata states a strict pre-match cutoff. |
| `away_team_rolling10_goals_scored` | **SAFE** | historical match statistics | mean of last 10 available matches | Metadata states a strict pre-match cutoff. |
| `away_team_rolling10_matches_available` | **SAFE** | historical matches before fixture | count of last 10 available matches | Metadata states a strict pre-match cutoff. |
| `away_team_rolling10_points` | **SAFE** | historical match statistics | mean of last 10 available matches | Metadata states a strict pre-match cutoff. |
| `away_team_rolling10_shots` | **SAFE** | historical match statistics | mean of last 10 available matches | Metadata states a strict pre-match cutoff. |
| `away_team_rolling10_shots_on_target` | **SAFE** | historical match statistics | mean of last 10 available matches | Metadata states a strict pre-match cutoff. |
| `away_team_rolling10_yellow_cards` | **SAFE** | historical match statistics | mean of last 10 available matches | Metadata states a strict pre-match cutoff. |
| `away_team_rolling5_corners` | **SAFE** | historical match statistics | mean of last 5 available matches | Metadata states a strict pre-match cutoff. |
| `away_team_rolling5_goal_difference` | **SAFE** | historical match statistics | mean of last 5 available matches | Metadata states a strict pre-match cutoff. |
| `away_team_rolling5_goals_conceded` | **SAFE** | historical match statistics | mean of last 5 available matches | Metadata states a strict pre-match cutoff. |
| `away_team_rolling5_goals_scored` | **SAFE** | historical match statistics | mean of last 5 available matches | Metadata states a strict pre-match cutoff. |
| `away_team_rolling5_matches_available` | **SAFE** | historical matches before fixture | count of last 5 available matches | Metadata states a strict pre-match cutoff. |
| `away_team_rolling5_points` | **SAFE** | historical match statistics | mean of last 5 available matches | Metadata states a strict pre-match cutoff. |
| `away_team_rolling5_shots` | **SAFE** | historical match statistics | mean of last 5 available matches | Metadata states a strict pre-match cutoff. |
| `away_team_rolling5_shots_on_target` | **SAFE** | historical match statistics | mean of last 5 available matches | Metadata states a strict pre-match cutoff. |
| `away_team_rolling5_yellow_cards` | **SAFE** | historical match statistics | mean of last 5 available matches | Metadata states a strict pre-match cutoff. |
| `bookmaker_implied_away` | **SAFE** | historical bookmaker odds | decimal odds converted to implied and normalized probabilities | Metadata states a strict pre-match cutoff. |
| `bookmaker_implied_draw` | **SAFE** | historical bookmaker odds | decimal odds converted to implied and normalized probabilities | Metadata states a strict pre-match cutoff. |
| `bookmaker_implied_home` | **SAFE** | historical bookmaker odds | decimal odds converted to implied and normalized probabilities | Metadata states a strict pre-match cutoff. |
| `bookmaker_overround` | **SAFE** | historical bookmaker odds | decimal odds converted to implied and normalized probabilities | Metadata states a strict pre-match cutoff. |
| `bookmaker_prob_away` | **SAFE** | historical bookmaker odds | decimal odds converted to implied and normalized probabilities | Metadata states a strict pre-match cutoff. |
| `bookmaker_prob_draw` | **SAFE** | historical bookmaker odds | decimal odds converted to implied and normalized probabilities | Metadata states a strict pre-match cutoff. |
| `bookmaker_prob_home` | **SAFE** | historical bookmaker odds | decimal odds converted to implied and normalized probabilities | Metadata states a strict pre-match cutoff. |
| `draw_odds` | **SAFE** | historical bookmaker odds | decimal odds converted to implied and normalized probabilities | Metadata states a strict pre-match cutoff. |
| `h2h_away_win_rate` | **SAFE** | prior head-to-head results | mean/rate over prior same-team fixtures | Metadata states a strict pre-match cutoff. |
| `h2h_draw_rate` | **SAFE** | prior head-to-head results | mean/rate over prior same-team fixtures | Metadata states a strict pre-match cutoff. |
| `h2h_home_perspective_goal_difference` | **SAFE** | prior head-to-head results | mean/rate over prior same-team fixtures | Metadata states a strict pre-match cutoff. |
| `h2h_home_win_rate` | **SAFE** | prior head-to-head results | mean/rate over prior same-team fixtures | Metadata states a strict pre-match cutoff. |
| `h2h_matches_played` | **SAFE** | prior head-to-head results | mean/rate over prior same-team fixtures | Metadata states a strict pre-match cutoff. |
| `home_league_position_before_match` | **SAFE** | prior results in current season | table sorted by points, goal difference, goals scored | Metadata states a strict pre-match cutoff. |
| `home_odds` | **SAFE** | historical bookmaker odds | decimal odds converted to implied and normalized probabilities | Metadata states a strict pre-match cutoff. |
| `home_team_home_goal_difference` | **SAFE** | prior home-only matches for team | venue-restricted cumulative mean | Metadata states a strict pre-match cutoff. |
| `home_team_home_goals_conceded` | **SAFE** | prior home-only matches for team | venue-restricted cumulative mean | Metadata states a strict pre-match cutoff. |
| `home_team_home_goals_scored` | **SAFE** | prior home-only matches for team | venue-restricted cumulative mean | Metadata states a strict pre-match cutoff. |
| `home_team_home_matches_played` | **SAFE** | prior home-only matches for team | venue-restricted cumulative mean | Metadata states a strict pre-match cutoff. |
| `home_team_home_points_per_match` | **SAFE** | prior home-only matches for team | venue-restricted cumulative mean | Metadata states a strict pre-match cutoff. |
| `home_team_overall_draws` | **SAFE** | historical match results and statistics | cumulative aggregate | Metadata states a strict pre-match cutoff. |
| `home_team_overall_goal_difference` | **SAFE** | historical match results and statistics | cumulative aggregate | Metadata states a strict pre-match cutoff. |
| `home_team_overall_goals_conceded` | **SAFE** | historical match results and statistics | cumulative aggregate | Metadata states a strict pre-match cutoff. |
| `home_team_overall_goals_scored` | **SAFE** | historical match results and statistics | cumulative aggregate | Metadata states a strict pre-match cutoff. |
| `home_team_overall_losses` | **SAFE** | historical match results and statistics | cumulative aggregate | Metadata states a strict pre-match cutoff. |
| `home_team_overall_matches_played` | **SAFE** | historical match results and statistics | cumulative aggregate | Metadata states a strict pre-match cutoff. |
| `home_team_overall_points_per_match` | **SAFE** | historical match results and statistics | cumulative aggregate | Metadata states a strict pre-match cutoff. |
| `home_team_overall_wins` | **SAFE** | historical match results and statistics | cumulative aggregate | Metadata states a strict pre-match cutoff. |
| `home_team_rolling10_corners` | **SAFE** | historical match statistics | mean of last 10 available matches | Metadata states a strict pre-match cutoff. |
| `home_team_rolling10_goal_difference` | **SAFE** | historical match statistics | mean of last 10 available matches | Metadata states a strict pre-match cutoff. |
| `home_team_rolling10_goals_conceded` | **SAFE** | historical match statistics | mean of last 10 available matches | Metadata states a strict pre-match cutoff. |
| `home_team_rolling10_goals_scored` | **SAFE** | historical match statistics | mean of last 10 available matches | Metadata states a strict pre-match cutoff. |
| `home_team_rolling10_matches_available` | **SAFE** | historical matches before fixture | count of last 10 available matches | Metadata states a strict pre-match cutoff. |
| `home_team_rolling10_points` | **SAFE** | historical match statistics | mean of last 10 available matches | Metadata states a strict pre-match cutoff. |
| `home_team_rolling10_shots` | **SAFE** | historical match statistics | mean of last 10 available matches | Metadata states a strict pre-match cutoff. |
| `home_team_rolling10_shots_on_target` | **SAFE** | historical match statistics | mean of last 10 available matches | Metadata states a strict pre-match cutoff. |
| `home_team_rolling10_yellow_cards` | **SAFE** | historical match statistics | mean of last 10 available matches | Metadata states a strict pre-match cutoff. |
| `home_team_rolling5_corners` | **SAFE** | historical match statistics | mean of last 5 available matches | Metadata states a strict pre-match cutoff. |
| `home_team_rolling5_goal_difference` | **SAFE** | historical match statistics | mean of last 5 available matches | Metadata states a strict pre-match cutoff. |
| `home_team_rolling5_goals_conceded` | **SAFE** | historical match statistics | mean of last 5 available matches | Metadata states a strict pre-match cutoff. |
| `home_team_rolling5_goals_scored` | **SAFE** | historical match statistics | mean of last 5 available matches | Metadata states a strict pre-match cutoff. |
| `home_team_rolling5_matches_available` | **SAFE** | historical matches before fixture | count of last 5 available matches | Metadata states a strict pre-match cutoff. |
| `home_team_rolling5_points` | **SAFE** | historical match statistics | mean of last 5 available matches | Metadata states a strict pre-match cutoff. |
| `home_team_rolling5_shots` | **SAFE** | historical match statistics | mean of last 5 available matches | Metadata states a strict pre-match cutoff. |
| `home_team_rolling5_shots_on_target` | **SAFE** | historical match statistics | mean of last 5 available matches | Metadata states a strict pre-match cutoff. |
| `home_team_rolling5_yellow_cards` | **SAFE** | historical match statistics | mean of last 5 available matches | Metadata states a strict pre-match cutoff. |
| `league_position_difference_before_match` | **SAFE** | prior results in current season | table sorted by points, goal difference, goals scored | Metadata states a strict pre-match cutoff. |
