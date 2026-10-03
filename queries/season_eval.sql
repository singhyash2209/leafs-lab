SELECT CAST(season AS TEXT) AS season,
 AVG(ABS(expected_points-actual_points)) AS points_mae,
 AVG((p_playoffs-actual_playoffs)*(p_playoffs-actual_playoffs)) AS playoff_brier,
 0.25 AS playoff_baseline,
 AVG((p_cup-actual_cup)*(p_cup-actual_cup)) AS cup_brier,
 31.0/1024.0 AS cup_baseline
 FROM season_predictions GROUP BY season ORDER BY season;
