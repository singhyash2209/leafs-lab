SELECT 'selected_calibrated' AS model,
 SUM(weight*(prediction-home_win)*(prediction-home_win))/SUM(weight) AS brier,
 SUM(weight*CASE WHEN (prediction>=0.5)=home_win THEN 1.0 ELSE 0.0 END)/SUM(weight) AS accuracy
 FROM test_predictions
 UNION ALL SELECT 'baseline',SUM(weight*(baseline-home_win)*(baseline-home_win))/SUM(weight),
 SUM(weight*CASE WHEN (baseline>=0.5)=home_win THEN 1.0 ELSE 0.0 END)/SUM(weight) FROM test_predictions
 UNION ALL SELECT 'selected_raw',SUM(weight*(raw_prediction-home_win)*(raw_prediction-home_win))/SUM(weight),
 SUM(weight*CASE WHEN (raw_prediction>=0.5)=home_win THEN 1.0 ELSE 0.0 END)/SUM(weight) FROM test_predictions
 UNION ALL SELECT 'pregame_only',SUM(weight*(pregame_probability-home_win)*(pregame_probability-home_win))/SUM(weight),
 SUM(weight*CASE WHEN (pregame_probability>=0.5)=home_win THEN 1.0 ELSE 0.0 END)/SUM(weight) FROM test_predictions
 UNION ALL SELECT 'constant_50_percent',SUM(weight*(0.5-home_win)*(0.5-home_win))/SUM(weight),
 SUM(weight*home_win)/SUM(weight) FROM test_predictions;

SELECT MIN(9,CAST(prediction*10 AS INTEGER)) AS bin,
 SUM(weight*prediction)/SUM(weight) AS mean_prediction,
 SUM(weight*home_win)/SUM(weight) AS observed_rate,
 COUNT(DISTINCT game_id) AS games, COUNT(*) AS states
 FROM test_predictions GROUP BY bin ORDER BY bin;
