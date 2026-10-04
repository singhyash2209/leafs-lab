"""GitHub runner entry point. Refresh only forecasts; preserve frozen evaluation.
No ChatGPT, OpenAI API, model.joblib, PBP downloads or credentials required.
An invalid API response stops publication, preserving the last successful site.
"""
from pathlib import Path
import argparse,copy,json,sqlite3,subprocess,sys
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from leafslab.data import NHLClient,validate_schedules
from leafslab.cli import forecast,save_json
from readable_dashboard import readable
from archive_dashboard import publish

def refresh(as_of=None,offline=False):
    config=json.loads(Path('config.json').read_text())
    config['as_of']=as_of or datetime.now(ZoneInfo('America/Toronto')).date().isoformat()
    client=NHLClient(offline=offline)
    seasons=list(dict.fromkeys(config['train_seasons']+[config['validation_season'],config['test_season'],config['forecast_season']]))
    frames=[]
    for season in seasons:
        print(f'Checking NHL schedule {season}',flush=True)
        frames.append(client.schedule(season,refresh=(season==config['forecast_season'] and not offline)))
    games=pd.concat(frames,ignore_index=True)
    validate_schedules(games,seasons[:-1],config['minimum_games_per_season'])
    games.to_csv('data/games.csv',index=False)
    # The existing engine checks cutoff standings, completed scores, season
    # membership, games-per-team and playoff mass before emitting a forecast.
    result=forecast(config,offline=offline,sensitivity=False)
    artifact=json.loads(Path('outputs/artifact.json').read_text())
    now=datetime.now(timezone.utc).isoformat()
    connection=sqlite3.connect(':memory:')
    counts=[]
    for team in result['teams']:
        row={k:team[k] for k in ['team','elo','points_p10','points_p90']}
        row.update(simulations=result['simulations'],points_sum=team['points_sum'])
        for stage in ['playoffs','round_2','conference_final','final','cup']:
            row[stage+'_count']=team[stage+'_count']
        counts.append(row)
    pd.DataFrame(counts).to_sql('forecast_counts',connection,index=False)
    sql=Path('queries/forecast.sql').read_text()
    teams=pd.read_sql_query(sql,connection).to_dict('records');connection.close()
    by_team={t['team']:t for t in result['teams']}
    for row in teams:
        for k in ['mean_points','playoffs','round_2','conference_final','final','cup']:
            if abs(row[k]-by_team[row['team']][k])>1e-10: raise ValueError('Forecast SQL reconciliation failed')
    datasets=artifact['snapshot']['datasets']
    frozen={k:copy.deepcopy(v) for k,v in datasets.items() if k not in ['teams','toronto','cup_comparison']}
    datasets['teams']=teams;datasets['toronto']=[next(t for t in teams if t['team']=='TOR')]
    datasets['cup_comparison']=teams[:10]+([datasets['toronto'][0]] if 'TOR' not in [t['team'] for t in teams[:10]] else [])
    artifact['snapshot']['generatedAt']=now;artifact['manifest']['generatedAt']=now
    intro=next(b for b in artifact['manifest']['blocks'] if b['id']=='intro')
    intro['body']=f"# {artifact['manifest']['title']}\n\nFrom the September 30 game to a testable forecasting project. Season snapshot: {config['as_of']}."
    block={'id':'automation_status','type':'markdown','body':f"### Daily API refresh\n\nThe season forecast is refreshed by Python on GitHub Actions, scheduled daily at 6:17 AM Toronto time. Last successful run: {now}. NHL results cutoff: {result['cutoff_date']}. The historical game analysis and model evaluation remain frozen.\n\nThe current engine supports the configured regular season ({str(result['season'])[:4]}-{str(result['season'])[4:]}). During playoffs or before an unconfigured season, it keeps the last verified snapshot. New seasons require configuration and validation. API outages stop publication rather than substituting invented data. GitHub may delay runs or disable an inactive public repository's schedule; the dated dashboard and archive remain readable."}
    artifact['manifest']['blocks']=[b for b in artifact['manifest']['blocks'] if b['id']!='automation_status']
    artifact['manifest']['blocks'].insert(1,block)
    for source in artifact['sources']:
        if source['id']=='forecast':
            source['query'].update(executed_at=now,description='SQL recomputation from current Monte Carlo counts in an in-memory SQLite database; validated against outputs/forecast.json. Python fetches NHL schedules and dated standings before simulation.')
            source['query']['filters']={'as_of':config['as_of'],'cutoff_date':result['cutoff_date'],'season':result['season'],'simulations':result['simulations']}
    artifact['manifest']['sources']=copy.deepcopy(artifact['sources'])
    artifact=readable(artifact,result)
    for k,v in frozen.items():
        if artifact['snapshot']['datasets'][k]!=v: raise ValueError('Frozen evaluation changed')
    save_json('outputs/artifact.json',artifact)
    subprocess.run(['node','scripts/render_dashboard.mjs'],check=True)
    publish(Path('outputs/dashboard.html'),Path('outputs/forecast.json'),Path('docs'))
    save_json('outputs/refresh_status.json',{'status':'success','generated_at':now,'as_of':config['as_of'],'season':result['season'],'api':'https://api-web.nhle.com/v1','scheduler':'GitHub Actions','simulations':result['simulations'],'frozen_evaluation_unchanged':True})
    print('Validated forecast and archive ready for publication',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--as-of');parser.add_argument('--offline',action='store_true');args=parser.parse_args()
    refresh(args.as_of,args.offline)
