SELECT elapsed/60.0 AS minute,
 CASE WHEN home='TOR' THEN
 home_win_probability(goal_diff,remaining,score_clock,pregame_logit,shot_diff,skater_diff,home_empty,away_empty,situation_missing)
 ELSE 1.0-home_win_probability(goal_diff,remaining,score_clock,pregame_logit,shot_diff,skater_diff,home_empty,away_empty,situation_missing)
 END AS toronto_probability, home_goals, away_goals
 FROM game_states ORDER BY elapsed;
