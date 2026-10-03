from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import joblib
from .data import NHLClient, SourceUnavailable, FINAL, add_elo, snapshots, validate_schedules
from .model import train, predict, explain_state
from .season import fit_goal_rates, simulate, standings_from_api, check_schedule

def save_json(path,data):
    Path(path).parent.mkdir(exist_ok=True,parents=True)
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)))

def fetch(config,offline=False):
    client=NHLClient(offline=offline); all_games=[]
    seasons=config['train_seasons']+[config['validation_season'],config['test_season'],config['forecast_season']]
    for season in seasons:
        print(f'Loading schedule {season}',flush=True)
        all_games.append(client.schedule(season,refresh=season==config['forecast_season']))
    games=pd.concat(all_games,ignore_index=True)
    games.to_csv('data/games.csv',index=False)
    historical=config['train_seasons']+[config['validation_season'],config['test_season']]
    counts=validate_schedules(games,historical,config['minimum_games_per_season'])
    completed=games[(games.game_type==2)&games.status.isin(FINAL)&(games.date<config['as_of'])]
    print(f'Loading PBP for {len(completed)} completed games (cached downloads reused)',flush=True)
    _,errors=client.pbps(completed.game_id)
    save_json('outputs/source_errors.json',errors)
    coverage=1-len(errors)/max(len(completed),1)
    save_json('outputs/download_status.json',{'completed_games':len(completed),'download_coverage':coverage,
                                             'season_counts':counts,'errors':len(errors)})
    if coverage<config['minimum_pbp_coverage']: raise SourceUnavailable('PBP coverage below configured minimum; see source_errors.json')

def prepare(config):
    games=pd.read_csv('data/games.csv')
    eligible=games[(games.game_type==2)&games.status.isin(FINAL)&(games.date<config['as_of'])].copy()
    rated,ratings=add_elo(eligible,config); rated.to_csv('data/rated_games.csv',index=False)
    frames=[]; errors=[]; client=NHLClient(offline=True)
    for game in rated.to_dict('records'):
        try: frames.append(snapshots(client.get(f"gamecenter/{game['game_id']}/play-by-play"),game))
        except Exception as e: errors.append({'game_id':game['game_id'],'season':game['season'],'error':str(e)})
    save_json('outputs/parser_errors.json',errors)
    if not frames: raise SourceUnavailable('No verified play-by-play observations')
    states=pd.concat(frames,ignore_index=True)
    coverage={}
    for season in config['train_seasons']+[config['validation_season'],config['test_season']]:
        denominator=int((rated.season==season).sum()); numerator=states[states.season==season].game_id.nunique()
        coverage[season]={'eligible_games':denominator,'parsed_games':int(numerator),'coverage':numerator/max(denominator,1)}
        if coverage[season]['coverage']<config['minimum_pbp_coverage']:
            raise SourceUnavailable(f'Parsed PBP coverage below minimum for {season}: {coverage[season]}')
    if states[['goal_diff','remaining','score_clock','pregame_logit']].isna().any().any(): raise ValueError('Missing required features')
    if not states.groupby('game_id').weight.sum().between(.999999,1.000001).all(): raise ValueError('Game weights do not sum to one')
    states.to_csv('data/states.csv',index=False)
    save_json('outputs/data_quality.json',{'coverage_by_season':coverage,'states':len(states),'games':int(states.game_id.nunique()),
                                         'situation_missing_fraction':float(states.situation_missing.mean()),
                                         'missing_features':states.isna().sum().to_dict(),
                                         'excluded_games':len(errors),'last_game_date':rated.date.max()})
    summary=states.groupby('season').agg(games=('game_id','nunique'),states=('game_id','size'),
                                       mean_goal_diff=('goal_diff','mean'),situation_missing=('situation_missing','mean')).reset_index()
    summary.to_csv('outputs/eda_by_season.csv',index=False)
    states.groupby(['goal_diff']).agg(states=('game_id','size'),games=('game_id','nunique')).reset_index().to_csv('outputs/eda_score_states.csv',index=False)
    return states

def forecast(config,offline=False,season=None,as_of=None,expected=None,out='outputs/forecast.json',sensitivity=True):
    season=season or config['forecast_season']; as_of=as_of or config['as_of']; expected=expected or config['expected_games_per_team']
    cutoff=(datetime.fromisoformat(as_of)-timedelta(days=1)).date().isoformat()
    client=NHLClient(offline=offline)
    games=pd.read_csv('data/games.csv'); current=games[(games.season==season)&(games.game_type==2)]
    if current.empty: current=client.schedule(season,refresh=True).query('game_type == 2')
    last_regular_date=datetime.fromisoformat(str(current.date.max())).date()
    if datetime.fromisoformat(as_of).date()>last_regular_date+timedelta(days=1):
        raise SourceUnavailable('Postseason live updates need a conditional playoff engine; this version supports forecasts through the regular-season finish.')
    played=current[current.date<as_of]
    if not played.status.isin(FINAL).all(): raise ValueError('Incomplete/past postponed games require a reviewed schedule before forecasting')
    source=client.get(f'standings/{cutoff}',refresh=True)
    metadata_only_date=None
    if not source.get('standings') and played.empty:
        # Before opening day the NHL endpoint can return no rows. Read only
        # league/division membership from opening-day standings, zero every
        # outcome field, and disclose the metadata date. No later results enter.
        metadata_only_date=str(current.date.min())
        metadata_source=client.get(f'standings/{metadata_only_date}')
        metadata=standings_from_api(metadata_source)[['team','conference','division']]
        standings=metadata.assign(points=0,rw=0,row=0,wins=0,gf=0,ga=0,gp=0)
    else:
        standings=standings_from_api(source)
        dates={x.get('date') for x in source.get('standings',[]) if x.get('date')}
        if dates and dates!={cutoff}: raise ValueError(f'Standings dates {dates} disagree with cutoff {cutoff}')
    appearances=pd.concat([played.home,played.away]).value_counts()
    for row in standings.to_dict('records'):
        if int(appearances.get(row['team'],0))!=row['gp']:
            raise ValueError('Standings games played do not reconcile with cutoff schedule')
    remaining=current[current.date>=as_of].copy()
    check_schedule(remaining,standings,expected)
    history=games[(games.game_type==2)&games.status.isin(FINAL)&(games.date<as_of)]
    rated,ratings=add_elo(history,config)
    if len(rated) and rated.season.max()<season:
        ratings={t:1500+(r-1500)*config['elo_season_retention'] for t,r in ratings.items()}
    ratings={t:ratings.get(t,1500) for t in standings.team}
    rates=fit_goal_rates(rated[rated.season.isin(config['train_seasons'])])
    result=simulate(remaining,standings,ratings,rates,n=config['simulations'],seed=config['seed'],home_advantage=config['elo_home_advantage'])
    result.update(as_of=as_of,cutoff_date=cutoff,season=season,goal_model=rates,expected_games_per_team=expected,
                  source='NHL schedules, dated standings, and historical results',status='real_data_simulated',
                  membership_only_source_date=metadata_only_date)
    if sensitivity and 'TOR' in ratings:
        scenarios=[]
        for delta in [-50,50]:
            alternative=dict(ratings); alternative['TOR']+=delta
            sim=simulate(remaining,standings,alternative,rates,n=min(2000,config['simulations']),seed=config['seed'],home_advantage=config['elo_home_advantage'])
            tor=next(t for t in sim['teams'] if t['team']=='TOR')
            scenarios.append({'toronto_elo_change':delta,'playoffs':tor['playoffs'],'cup':tor['cup'],'simulations':sim['simulations']})
        result['sensitivity']=scenarios
    save_json(out,result)
    pd.DataFrame(result['teams']).drop(columns='cup_mc_95_interval').to_csv(Path(out).with_suffix('.csv'),index=False)
    return result

def game_report(config,date='2026-09-30'):
    games=pd.read_csv('data/rated_games.csv'); selected=games[(games.date==date)&((games.home=='TOR')|(games.away=='TOR'))]
    if len(selected)!=1: raise SourceUnavailable('Exactly one verified Toronto game required for selected date')
    game=selected.iloc[0].to_dict(); client=NHLClient(offline=True)
    states=snapshots(client.get(f"gamecenter/{game['game_id']}/play-by-play"),game)
    bundle=joblib.load('outputs/model.joblib'); states['home_probability']=predict(bundle,states)
    states['toronto_probability']=states.home_probability if game['home']=='TOR' else 1-states.home_probability
    states.to_csv('outputs/game_curve.csv',index=False)
    # Report the first post-goal state at 2-1 (not cherry-picked maximum).
    key=states[(states.home_goals==2)&(states.away_goals==1)] if game['home']=='TOR' else states[(states.away_goals==2)&(states.home_goals==1)]
    explanation=explain_state(bundle,key.iloc[0].to_dict()) if len(key) else None
    save_json('outputs/game_explanation.json',{'game_id':game['game_id'],'date':date,'home':game['home'],'away':game['away'],
                                               'first_2_1_state':key.iloc[0].to_dict() if len(key) else None,
                                               'explanation':explanation,
                                               'note':'Contemporaneous post-event prediction. Final outcome was not a feature.'})
    return states

def season_backtest(config,offline=False,season=None):
    """Independent preseason forecast test, not parameter selection.
    Only one season: useful diagnostic, insufficient to establish Cup calibration.
    """
    season=season or config['test_season']; year=int(str(season)[:4]); as_of=f'{year}-10-01'
    result=forecast(config,offline,season,as_of,82,f'outputs/backtest_{season}_forecast.json',False)
    client=NHLClient(offline=offline); full=client.schedule(season)
    reg=full[(full.game_type==2)&full.status.isin(FINAL)]
    playoffs=full[(full.game_type==3)&full.status.isin(FINAL)]
    if len(reg)<1100 or playoffs.empty: raise SourceUnavailable('Complete held-out season results unavailable')
    win_counts={}
    for game in playoffs.to_dict('records'):
        winner=game['home'] if game['home_score']>game['away_score'] else game['away']
        win_counts[winner]=win_counts.get(winner,0)+1
    # 16 wins identify champion only when every postseason game is available.
    champions=[t for t,wins in win_counts.items() if wins==16]
    if len(champions)!=1: raise ValueError('Incomplete playoff outcomes; cannot score Cup forecast')
    entrants=set(playoffs.home)|set(playoffs.away); scores={}
    for name,threshold in [('playoffs',0),('round_2',4),('conference_final',8),('final',12),('cup',16)]:
        p=np.array([t[name] for t in result['teams']]); y=np.array([int(t['team'] in entrants) if name=='playoffs' else int(win_counts.get(t['team'],0)>=threshold) for t in result['teams']])
        expected_count={'playoffs':16,'round_2':8,'conference_final':4,'final':2,'cup':1}[name]
        if int(y.sum())!=expected_count: raise ValueError('Incomplete postseason labels for '+name)
        scores[name]={'brier':float(np.mean((p-y)**2)),'clubs':32,'positives':int(y.sum())}
    actual_pts={}
    for g in reg.to_dict('records'):
        hw=g['home_score']>g['away_score']; extra=g['last_period'] in ('OT','SO')
        actual_pts[g['home']]=actual_pts.get(g['home'],0)+(2 if hw else int(extra))
        actual_pts[g['away']]=actual_pts.get(g['away'],0)+(int(extra) if hw else 2)
    errors=[t['mean_points']-actual_pts[t['team']] for t in result['teams']]
    baselines={name:{'constant_rate':count/32,'brier':(count/32)*(1-count/32)} for name,count in
               [('playoffs',16),('round_2',8),('conference_final',4),('final',2),('cup',1)]}
    result_metrics={'season':season,'as_of':as_of,'probability_scores':scores,'naive_baselines':baselines,
                 'points_mae':float(np.mean(np.abs(errors))),'points_rmse':float(np.sqrt(np.mean(np.array(errors)**2))),
                 'warning':'Rolling-origin retrospective season diagnostic; club outcomes are dependent. A few seasons cannot establish Cup calibration.'}
    save_json(f'outputs/backtest_{season}_metrics.json',result_metrics)
    return result_metrics

def update(config,offline=False):
    """Fast daily forecast using frozen trained model and measured evaluation.
    No repeated historic PBP downloads, no retrospective reselection on test.
    """
    for required in ['data/games.csv','outputs/model.joblib','outputs/metrics.json','outputs/data_quality.json','outputs/game_curve.csv']:
        if not Path(required).exists(): raise SourceUnavailable('Initial full run required: '+required)
    client=NHLClient(offline=offline)
    old=pd.read_csv('data/games.csv')
    latest=client.schedule(config['forecast_season'],refresh=True)
    games=pd.concat([old[old.season!=config['forecast_season']],latest],ignore_index=True)
    games.to_csv('data/games.csv',index=False)
    history=games[(games.game_type==2)&games.status.isin(FINAL)&(games.date<config['as_of'])]
    rated,_=add_elo(history,config); rated.to_csv('data/rated_games.csv',index=False)
    forecast(config,offline)
    from .report import build
    build(config)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Leafs Lab: no forecasts without verified inputs')
    parser.add_argument('command',choices=['fetch','prepare','train','forecast','game','report','backtest','run','update'])
    parser.add_argument('--config',default='config.json'); parser.add_argument('--offline',action='store_true')
    parser.add_argument('--as-of'); parser.add_argument('--date',default='2026-09-30')
    parser.add_argument('--simulations',type=int); parser.add_argument('--season',type=int); parser.add_argument('--expected-games',type=int)
    args=parser.parse_args(argv); config=json.loads(Path(args.config).read_text())
    if args.as_of: config['as_of']=args.as_of
    if args.simulations: config['simulations']=args.simulations
    try:
        if args.command=='update': update(config,args.offline)
        if args.command in ('fetch','run'): fetch(config,args.offline)
        if args.command in ('prepare','run'): prepare(config)
        if args.command in ('train','run'): train(pd.read_csv('data/states.csv'),config)
        if args.command in ('forecast','run'): forecast(config,args.offline,args.season,expected=args.expected_games)
        if args.command in ('game','run'): game_report(config,args.date)
        if args.command in ('backtest','run'):
            backtests=[]
            for historical_season in [config['train_seasons'][-1],config['validation_season'],config['test_season']]:
                print(f'Rolling-origin season backtest: {historical_season}',flush=True)
                backtests.append(season_backtest(config,args.offline,historical_season))
            save_json('outputs/season_backtest_metrics.json',{'seasons':backtests,
                      'mean_points_mae':float(np.mean([b['points_mae'] for b in backtests])),
                      'latest_playoffs_beats_naive_baseline':backtests[-1]['probability_scores']['playoffs']['brier']<.25,
                      'model_review_note':'Strength-uncertainty variant added after inspecting initial fixed-strength baseline performance. Retrospective season scores are exploratory, not a pristine untouched holdout.',
                      'warning':'Three seasons provide a diagnostic, not established Cup calibration; outcomes within each season are dependent.'})
        if args.command in ('report','run'):
            from .report import build
            build(config)
        if Path('outputs/run_status.json').exists(): Path('outputs/run_status.json').unlink()
        save_json('outputs/last_command.json',{'command':args.command,'status':'completed','at':datetime.now(timezone.utc).isoformat()})
        return 0
    except (SourceUnavailable,ValueError,FileNotFoundError,KeyError) as e:
        save_json('outputs/run_status.json',{'status':'blocked','command':args.command,'reason':str(e),
                  'at':datetime.now(timezone.utc).isoformat(),'forecasts_publishable':False})
        print(f'BLOCKED: {e}',file=sys.stderr); return 2

if __name__=='__main__': sys.exit(main())
