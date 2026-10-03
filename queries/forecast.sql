SELECT team,elo,points_sum*1.0/simulations AS mean_points,
 points_p10,points_p90,playoffs_count*1.0/simulations AS playoffs,
 round_2_count*1.0/simulations AS round_2,
 conference_final_count*1.0/simulations AS conference_final,
 final_count*1.0/simulations AS final,
 cup_count*1.0/simulations AS cup, cup_count AS cup_simulation_wins
 FROM forecast_counts ORDER BY cup DESC,team;
