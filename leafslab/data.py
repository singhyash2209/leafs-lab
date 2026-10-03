"""NHL JSON adapters. Network responses must pass validation before modeling."""
from __future__ import annotations
import hashlib, json, time, os, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
BASE = 'https://api-web.nhle.com/v1'
# Historical codes remain necessary for earlier seasons. Nonparticipating codes
# may return no games/404; every completed game is independently deduplicated.
CODES = 'ANA ARI BOS BUF CAR CBJ CGY CHI COL DAL DET EDM FLA LAK MIN MTL NJD NSH NYI NYR OTT PHI PIT SEA SJS STL TBL TOR UTA VAN VGK WPG WSH'.split()
FINAL = {'OFF', 'FINAL'}

class SourceUnavailable(RuntimeError): pass

class NHLClient:
    def __init__(self, root='data/raw', offline=False):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.offline=offline
    def get(self, endpoint, refresh=False):
        path=self.root/(endpoint.replace('/','__')+'.json')
        if path.exists() and (self.offline or not refresh):
            meta_path=path.with_suffix('.meta.json')
            if not meta_path.exists(): raise SourceUnavailable(f'Missing cached-source provenance: {meta_path}')
            payload=path.read_bytes(); meta=json.loads(meta_path.read_text())
            if hashlib.sha256(payload).hexdigest()!=meta.get('sha256'): raise ValueError(f'Cached-source checksum mismatch: {path}')
            return json.loads(payload)
        if self.offline: raise SourceUnavailable(f'Missing cached source: {path}')
        url=BASE+'/'+endpoint
        for attempt in range(3):
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'LeafsLab/0.1 (research)'})
                with urllib.request.urlopen(req,timeout=25) as response: payload=response.read()
                data=json.loads(payload)
                tmp=path.with_suffix('.tmp'); tmp.write_bytes(payload); tmp.replace(path)
                meta={'url':url,'retrieved_at':datetime.now(timezone.utc).isoformat(),
                      'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}
                path.with_suffix('.meta.json').write_text(json.dumps(meta,indent=2))
                return data
            except urllib.error.HTTPError as e:
                if e.code in (403,404):
                    raise SourceUnavailable(f'{url}: HTTP {e.code}; no source data substituted') from e
                error=e
            except (urllib.error.URLError,TimeoutError,json.JSONDecodeError) as e: error=e
            time.sleep(0.5*(2**attempt))
        raise SourceUnavailable(f'{url}: {error}')
    def schedule(self, season, refresh=False):
        out={}; errors=[]
        def load_schedule(code):
            try: return code,self.get(f'club-schedule-season/{code}/{season}',refresh),None
            except SourceUnavailable as e: return code,None,str(e)
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses=list(pool.map(load_schedule,CODES))
        for code,source,error in responses:
            if error:
                if 'HTTP 404' in error: continue
                errors.append(error); continue
            if not isinstance(source.get('games'),list): raise ValueError(f'Schedule schema changed: {code}')
            for game in source['games']:
                if int(game.get('gameType',0)) not in (2,3): continue
                gid=int(game['id']); h=game['homeTeam']; a=game['awayTeam']
                row={'game_id':gid,'season':int(game['season']),'game_type':int(game['gameType']),
                     'date':game['gameDate'],'start_utc':game['startTimeUTC'],
                     'home':h['abbrev'],'away':a['abbrev'],'home_id':int(h['id']),'away_id':int(a['id']),
                     'status':game['gameState'],'home_score':h.get('score'), 'away_score':a.get('score'),
                     'last_period':game.get('gameOutcome',{}).get('lastPeriodType')}
                if gid in out and out[gid]!=row: raise ValueError(f'Conflicting schedules for {gid}')
                out[gid]=row
        if errors: raise SourceUnavailable('Incomplete league schedule. First error: '+errors[0])
        if not out: raise SourceUnavailable(f'No schedule returned for {season}')
        df=pd.DataFrame(out.values()).sort_values(['start_utc','game_id']).reset_index(drop=True)
        if (df.season!=int(season)).any(): raise ValueError('Cross-season schedule response')
        return df
    def pbps(self, ids, refresh_ids=()):
        refresh_ids=set(refresh_ids); errors=[]; out={}
        def load(gid):
            try: return gid,self.get(f'gamecenter/{gid}/play-by-play',gid in refresh_ids),None
            except Exception as e: return gid,None,str(e)
        with ThreadPoolExecutor(max_workers=int(os.environ.get('LEAFSLAB_DOWNLOAD_WORKERS','8'))) as pool:
            for number,(gid,source,error) in enumerate(pool.map(load, map(int,ids)),1):
                if number%100==0: print(f'PBP progress: {number}/{len(ids)} ({len(errors)} retrieval errors)',flush=True)
                if error: errors.append({'game_id':gid,'error':error})
                else: out[gid]=source
        return out,errors

def elo_probability(home_rating,away_rating,home_advantage=45):
    return 1/(1+10**(-(np.asarray(home_rating)-np.asarray(away_rating)+home_advantage)/400))

def add_elo(games, config):
    """All pregame ratings use only prior calendar dates, then update as a batch.
    Mean regression at each new season. New franchises default to 1500.
    """
    games=games.sort_values(['date','start_utc','game_id']).copy()
    ratings={}; rows=[]; previous=None
    for (season,date), group in games.groupby(['season','date'],sort=True):
        if season!=previous:
            ratings={k:1500+(v-1500)*config['elo_season_retention'] for k,v in ratings.items()}
            previous=season
        changes={}
        for row in group.to_dict('records'):
            h=ratings.get(row['home'],1500); a=ratings.get(row['away'],1500)
            p=float(elo_probability(h,a,config['elo_home_advantage']))
            row.update(home_elo=h,away_elo=a,pregame_p=p); rows.append(row)
            if row['status'] in FINAL and pd.notna(row['home_score']) and pd.notna(row['away_score']):
                y=float(row['home_score']>row['away_score'])
                if row['home_score']==row['away_score']: raise ValueError('Final game has tied score')
                delta=config['elo_k']*(y-p)
                changes[row['home']]=changes.get(row['home'],0)+delta
                changes[row['away']]=changes.get(row['away'],0)-delta
        for code,delta in changes.items(): ratings[code]=ratings.get(code,1500)+delta
    return pd.DataFrame(rows),ratings

def seconds(clock):
    m,s=map(int,clock.split(':'))
    if m<0 or not 0<=s<60: raise ValueError(f'Invalid clock {clock}')
    return 60*m+s

def snapshots(source, game):
    """Post-event regulation states only, excluding terminal observations.
    situationCode digits: away goalie, away skaters, home skaters, home goalie.
    Shot feature is cumulative known shots on goal, not final boxscore totals.
    Unknown situation is explicit; no penalty-state forward-fill.
    """
    if int(source['id'])!=int(game['game_id']): raise ValueError('PBP ID mismatch')
    if source.get('gameState') not in FINAL: raise ValueError('Nonfinal PBP cannot enter labeled dataset')
    if int(source['homeTeam']['score'])!=int(game['home_score']) or int(source['awayTeam']['score'])!=int(game['away_score']):
        raise ValueError('Final scores disagree between schedule and PBP')
    y=int(game['home_score']>game['away_score']); h=a=hs=aws=0; records=[]; last=-1; seen=set()
    plays=sorted(source.get('plays',[]),key=lambda p:int(p.get('sortOrder',p.get('eventId',0))))
    if not plays: raise ValueError('Empty PBP')
    for event in plays:
        eid=event['eventId']
        if eid in seen: raise ValueError('Duplicate event ID')
        seen.add(eid)
        period=event['periodDescriptor']['number']
        if period>3: continue
        t=(period-1)*1200+seconds(event['timeInPeriod'])
        if t<last: raise ValueError('Nonmonotone game clock')
        last=t; typ=event['typeDescKey']; d=event.get('details',{})
        owner=d.get('eventOwnerTeamId')
        if typ=='goal':
            if owner==game['home_id']: h+=1
            elif owner==game['away_id']: a+=1
            else: raise ValueError('Goal missing valid owner')
            if 'homeScore' in d and (int(d['homeScore'])!=h or int(d['awayScore'])!=a):
                raise ValueError('Running score disagrees with goal fields')
        if typ in ('goal','shot-on-goal'):
            if owner==game['home_id']: hs+=1
            elif owner==game['away_id']: aws+=1
        if t>=3600 or typ in ('game-end','period-end','period-start'): continue
        code=str(event.get('situationCode','')).zfill(4)
        known=len(code)==4 and code.isdigit() and code!='0000'
        if known:
            ag,ask,hsk,hg=map(int,code)
            known=ag in (0,1) and hg in (0,1) and 3<=ask<=6 and 3<=hsk<=6
        if not known: ag=hg=1; ask=hsk=5
        record={'game_id':game['game_id'],'season':game['season'],'date':game['date'],
                'home':game['home'],'away':game['away'],'elapsed':t,'event':typ,
                'home_goals':h,'away_goals':a,'goal_diff':h-a,'remaining':(3600-t)/3600,
                'shot_diff':hs-aws,'skater_diff':hsk-ask,'home_empty':1-hg,'away_empty':1-ag,
                'situation_missing':int(not known),'pregame_logit':np.log(game['pregame_p']/(1-game['pregame_p'])),
                'home_win':y,'goal_at_second':int(typ=='goal')}
        record['score_clock']=record['goal_diff']/np.sqrt(record['remaining']+0.025)
        records.append(record)
    outcome=game.get('last_period')
    if outcome=='REG' and (h!=int(game['home_score']) or a!=int(game['away_score'])):
        raise ValueError('Regulation goal events do not reconcile with final score')
    if outcome in ('OT','SO'):
        expected_home=int(game['home_score'])-int(y==1)
        expected_away=int(game['away_score'])-int(y==0)
        if h!=expected_home or a!=expected_away or h!=a:
            raise ValueError('Regulation goal events do not reconcile with overtime outcome')
    # One observation per game-second avoids duplicated stoppage records dominating.
    df=pd.DataFrame(records)
    if df.empty: raise ValueError('No eligible regulation states')
    df['goal_at_second']=df.groupby(['game_id','elapsed']).goal_at_second.transform('max')
    df=df.drop_duplicates(['game_id','elapsed'],keep='last').reset_index(drop=True)
    df['weight']=1/len(df)
    return df

def validate_schedules(games, seasons, minimum):
    reg=games[(games.game_type==2)&games.status.isin(FINAL)]
    counts=reg.groupby('season').game_id.nunique().to_dict()
    for season in seasons:
        if counts.get(season,0)<minimum: raise ValueError(f'Insufficient games for {season}: {counts.get(season,0)}')
    if games.game_id.duplicated().any(): raise ValueError('Duplicate game IDs')
    return counts
