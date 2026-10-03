"""Schedule-aware Monte Carlo, points rules and division/wildcard bracket.
Forecast intervals measure Monte Carlo error only, not model uncertainty.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .data import elo_probability, FINAL

def fit_goal_rates(training):
    """Poisson MLE on regulation goals; strip deciding OT/SO tally from winner."""
    f=training.copy()
    if f.last_period.isna().any(): raise ValueError('Missing outcome period; cannot infer regulation goals safely')
    h=f.home_score.to_numpy(float); a=f.away_score.to_numpy(float)
    extra=f.last_period.isin(['OT','SO']).to_numpy()
    home_wins=h>a; h=h-(extra&home_wins); a=a-(extra&~home_wins)
    if (h<0).any() or (a<0).any() or np.any(extra&(h!=a)): raise ValueError('Invalid regulation-goal reconstruction')
    d=(f.home_elo-f.away_elo).to_numpy()/400
    def objective(theta):
        base,adv,beta=theta
        lh=np.exp(base+adv+beta*d); la=np.exp(base-beta*d)
        return np.sum(lh-h*np.log(lh)+la-a*np.log(la))
    fit=minimize(objective,[np.log(2.8),.05,.4],bounds=[(.2,2),(-.3,.3),(0,2)])
    if not fit.success: raise ValueError(f'Goal model optimization failed: {fit.message}')
    long=pd.concat([f[['season','date','home','home_elo']].rename(columns={'home':'team','home_elo':'rating'}),
                    f[['season','date','away','away_elo']].rename(columns={'away':'team','away_elo':'rating'})])
    drift=long.sort_values('date').groupby(['season','team']).rating.agg(['first','last'])
    rating_sd=float((drift['last']-drift['first']).std(ddof=1))
    # Historical Elo drift is a proxy for season-strength uncertainty, not an
    # independently calibrated posterior. No test-season outcomes fit this SD.
    return {'rating_uncertainty_sd':rating_sd,'log_base':float(fit.x[0]),'home_effect':float(fit.x[1]),'strength_effect':float(fit.x[2]),
            'training_games':len(f),'method':'Poisson MLE; independent regulation goals; fixed ratings'}

def monte_carlo_interval(hits,n):
    # Wilson binomial interval for numerical simulation uncertainty.
    z=1.96; p=hits/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,float(mid-half)),min(1,float(mid+half))]

def series_winner(a,b,ratings,rng,home_advantage=45):
    # a has home ice: games 1,2,5,7. No reseeding inside conference bracket.
    wins={a:0,b:0}
    for home_a in [True,True,False,False,True,False,True]:
        p=float(elo_probability(ratings[a],ratings[b],home_advantage if home_a else -home_advantage))
        winner=a if rng.random()<p else b; wins[winner]+=1
        if wins[winner]==4: return winner
    raise AssertionError('Series must end')

def postseason_bracket(rank,metadata):
    """Conference: top three per division, then best two remaining wildcards.
    Better division winner draws lower wildcard. Returns two division paths.
    """
    paths=[]
    for conf in sorted({m['conference'] for m in metadata.values()}):
        ordered=[x for x in rank if metadata[x]['conference']==conf]
        divisions=sorted({metadata[x]['division'] for x in ordered})
        if len(divisions)!=2: raise ValueError('Expected two divisions per conference')
        seeds=[[x for x in ordered if metadata[x]['division']==div][:3] for div in divisions]
        if any(len(x)!=3 for x in seeds): raise ValueError('Insufficient division teams')
        chosen=sum(seeds,[]); wild=[x for x in ordered if x not in chosen][:2]
        seeds.sort(key=lambda xs:rank.index(xs[0]))
        paths.append([[(seeds[0][0],wild[1]),(seeds[0][1],seeds[0][2])],
                      [(seeds[1][0],wild[0]),(seeds[1][1],seeds[1][2])]])
    if len(paths)!=2: raise ValueError('Expected two conferences')
    return paths

def simulate(schedule,standings,ratings,params,n=10000,seed=504,home_advantage=45):
    """standings rows: team, conference, division, points, rw, row, wins, gf, ga, gp.
    schedule must contain only games not completed at the forecast cutoff.
    """
    rng=np.random.default_rng(seed); teams=sorted(standings.team.tolist()); index={t:i for i,t in enumerate(teams)}
    if len(teams)!=32 or standings.team.duplicated().any(): raise ValueError('Exactly 32 unique active teams required')
    if set(ratings)!=set(teams): raise ValueError('Ratings must cover active teams exactly')
    if n<100: raise ValueError('Use at least 100 simulations')
    s=standings.set_index('team').loc[teams]; metadata=s[['conference','division']].to_dict('index')
    rating_sd=params.get('rating_uncertainty_sd',0)
    latent=np.tile(np.array([ratings[t] for t in teams]),(n,1))
    if rating_sd>0: latent+=rng.normal(0,rating_sd,(n,32))
    pts=np.tile(s.points.to_numpy(float),(n,1)); rw=np.tile(s.rw.to_numpy(float),(n,1))
    row=np.tile(s.row.to_numpy(float),(n,1)); wins=np.tile(s.wins.to_numpy(float),(n,1))
    gf=np.tile(s.gf.to_numpy(float),(n,1)); ga=np.tile(s.ga.to_numpy(float),(n,1))
    for g in schedule.sort_values(['date','game_id']).to_dict('records'):
        hi=index[g['home']]; ai=index[g['away']]; d=(latent[:,hi]-latent[:,ai])/400
        lh=np.exp(params['log_base']+params['home_effect']+params['strength_effect']*d)
        la=np.exp(params['log_base']-params['strength_effect']*d)
        hg=rng.poisson(lh,n); ag=rng.poisson(la,n); tie=hg==ag
        win=(hg>ag)|(tie&(rng.random(n)<elo_probability(latent[:,hi],latent[:,ai],home_advantage)))
        # Approximation: all simulated tied regulation games resolve in OT;
        # ROW tie-break will overcount real shootouts, explicitly disclosed.
        pts[:,hi]+=2*win+(tie&~win); pts[:,ai]+=2*~win+(tie&win)
        rw[:,hi]+=win&~tie; rw[:,ai]+=~win&~tie
        row[:,hi]+=win; row[:,ai]+=~win; wins[:,hi]+=win; wins[:,ai]+=~win
        hg=hg+(tie&win); ag=ag+(tie&~win)
        gf[:,hi]+=hg; ga[:,hi]+=ag; gf[:,ai]+=ag; ga[:,ai]+=hg
    stages={k:np.zeros(32,dtype=int) for k in ['playoffs','round_2','conference_final','final','cup']}
    def higher(a,b,rank): return (a,b) if rank.index(a)<rank.index(b) else (b,a)
    for simulation in range(n):
        # Official head-to-head tie-break is not implemented; GD/GF then seeded
        # random order resolve residual ties. Never claim official exact seeding.
        order=np.lexsort((rng.random(32),-gf[simulation],-(gf-ga)[simulation],-wins[simulation],-row[simulation],-rw[simulation],-pts[simulation]))
        sim_ratings={t:float(latent[simulation,i]) for t,i in index.items()}
        rank=[teams[i] for i in order]; bracket=postseason_bracket(rank,metadata); finalists=[]
        for conference in bracket:
            division_winners=[]
            for division in conference:
                round2=[]
                for a,b in division:
                    stages['playoffs'][index[a]]+=1; stages['playoffs'][index[b]]+=1
                    a,b=higher(a,b,rank); winner=series_winner(a,b,sim_ratings,rng,home_advantage)
                    stages['round_2'][index[winner]]+=1; round2.append(winner)
                a,b=higher(*round2,rank); winner=series_winner(a,b,sim_ratings,rng,home_advantage)
                stages['conference_final'][index[winner]]+=1; division_winners.append(winner)
            a,b=higher(*division_winners,rank); winner=series_winner(a,b,sim_ratings,rng,home_advantage)
            stages['final'][index[winner]]+=1; finalists.append(winner)
        a,b=higher(*finalists,rank); winner=series_winner(a,b,sim_ratings,rng,home_advantage)
        stages['cup'][index[winner]]+=1
    for name,total in [('playoffs',16),('round_2',8),('conference_final',4),('final',2),('cup',1)]:
        if stages[name].sum()!=n*total: raise AssertionError('Bracket stage mass does not reconcile')
    records=[]
    for t,i in index.items():
        records.append({'team':t,'elo':float(ratings[t]),'mean_points':float(pts[:,i].mean()),'points_sum':int(pts[:,i].sum()),
                        'points_p10':float(np.quantile(pts[:,i],.1)),'points_p90':float(np.quantile(pts[:,i],.9)),
                        **{k:float(v[i]/n) for k,v in stages.items()},**{k+'_count':int(v[i]) for k,v in stages.items()},
                        'cup_simulation_wins':int(stages['cup'][i]),'cup_mc_95_interval':monte_carlo_interval(int(stages['cup'][i]),n)})
    return {'simulations':n,'seed':seed,'remaining_games':len(schedule),'rating_uncertainty_sd':rating_sd,'teams':records,
            'assumptions':['One latent team-strength draw per simulated season; SD estimated from training-season Elo drift, not a calibrated posterior. No explicit injury, lineup or trade adjustments.',
                           'Independent Poisson regulation goals; all simulated ties resolve in overtime.',
                           'Standings ties approximate official rules: no head-to-head tie-break.',
                           'Playoff game probabilities use regular-season Elo; no separate playoff fit.',
                           'Cup intervals show simulation error only, not forecasting/model uncertainty.']}

def standings_from_api(source):
    rows=[]
    for x in source.get('standings',[]):
        code=x['teamAbbrev']; code=code.get('default') if isinstance(code,dict) else code
        rows.append({'team':code,'conference':x['conferenceAbbrev'],'division':x['divisionAbbrev'],
                     'points':x['points'],'rw':x['regulationWins'],'row':x['regulationPlusOtWins'],
                     'wins':x['wins'],'gf':x['goalFor'],'ga':x['goalAgainst'],'gp':x['gamesPlayed']})
    if len(rows)!=32: raise ValueError('Standings source must contain 32 active clubs')
    return pd.DataFrame(rows)

def check_schedule(remaining,standings,expected):
    appearances=pd.concat([remaining.home,remaining.away]).value_counts()
    for row in standings.to_dict('records'):
        count=int(appearances.get(row['team'],0))+int(row['gp'])
        if count!=expected: raise ValueError(f"Schedule incomplete for {row['team']}: {count}, expected {expected}")
    if set(appearances.index)-set(standings.team): raise ValueError('Unknown schedule teams')
