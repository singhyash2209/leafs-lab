"""Synthetic fixtures test software only. No hockey performance claims."""
import tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
from leafslab.data import snapshots, add_elo, elo_probability, seconds
from leafslab.model import train, predict, FEATURES
from leafslab.season import simulate, postseason_bracket, series_winner, check_schedule, fit_goal_rates
CONFIG={'train_seasons':[20222023,20232024],'validation_season':20242025,'test_season':20252026,
        'elo_k':20,'elo_home_advantage':45,'elo_season_retention':.75,'seed':504}

def game_fixture():
    game={'game_id':2023020001,'season':20232024,'date':'2023-10-01','home':'TOR','away':'NYI',
          'home_id':10,'away_id':2,'home_score':2,'away_score':1,'pregame_p':.55,'last_period':'REG'}
    def event(eid,period,clock,kind,owner=None,score=None,code='1551'):
        details={}
        if owner: details['eventOwnerTeamId']=owner
        if score: details.update(homeScore=score[0],awayScore=score[1])
        return {'eventId':eid,'sortOrder':eid,'periodDescriptor':{'number':period},
                'timeInPeriod':clock,'typeDescKey':kind,'situationCode':code,'details':details}
    source={'id':game['game_id'],'gameState':'OFF','homeTeam':{'score':2},'awayTeam':{'score':1},
            'plays':[event(1,1,'00:01','faceoff'),event(2,1,'07:14','goal',10,(1,0)),
                     event(3,1,'11:55','goal',10,(2,0)),event(4,3,'14:09','goal',2,(2,1)),
                     event(5,3,'19:00','shot-on-goal',2,code='0651'),event(6,3,'20:00','game-end')]}
    return game,source

def fake_states():
    rng=np.random.default_rng(12); rows=[]
    for season in [20222023,20232024,20242025,20252026]:
        year=int(str(season)[:4])
        for gid in range(40):
            # Paired chronological labels ensure both classes in each split.
            y=gid%2
            for t in range(20):
                remaining=1-t/20; diff=int(rng.integers(-2,3))
                if rng.random()<.65: diff=1 if y else -1
                rows.append({'game_id':season*100+gid,'season':season,
                             'date':(pd.Timestamp(f'{year}-10-01')+pd.Timedelta(days=gid*3)).date().isoformat(),
                             'goal_diff':diff,'remaining':remaining,'score_clock':diff/np.sqrt(remaining+.025),
                             'pregame_logit':float(rng.normal(0,.2)),'shot_diff':int(rng.integers(-8,9)),
                             'skater_diff':0,'home_empty':0,'away_empty':0,'situation_missing':0,
                             'elapsed':t*180,'weight':1/20,'home_win':y})
    return pd.DataFrame(rows)

def league_fixture():
    rows=[]
    for conf in ['E','W']:
        for div in ['A','B']:
            for i in range(8): rows.append({'team':f'{conf}{div}{i}','conference':conf,'division':conf+div,
                'points':0,'rw':0,'row':0,'wins':0,'gf':0,'ga':0,'gp':0})
    return pd.DataFrame(rows)

class Tests(unittest.TestCase):
    def test_clock_and_elo(self):
        self.assertEqual(seconds('14:09'),849)
        self.assertAlmostEqual(float(elo_probability(1500,1500,0)),.5)
        self.assertGreater(float(elo_probability(1500,1500,45)),.5)
        with self.assertRaises(ValueError): seconds('01:99')
    def test_post_goal_state_and_empty_net(self):
        g,p=game_fixture(); f=snapshots(p,g)
        state=f[f.elapsed==3249].iloc[0]
        self.assertEqual(state.goal_diff,1); self.assertAlmostEqual(state.remaining,351/3600)
        self.assertEqual(f.iloc[-1].away_empty,1); self.assertEqual(f.iloc[-1].skater_diff,-1)
        self.assertLess(f.elapsed.max(),3600); self.assertAlmostEqual(f.weight.sum(),1)
    def test_missing_situation_is_visible(self):
        g,p=game_fixture(); p['plays'][0].pop('situationCode')
        self.assertEqual(snapshots(p,g).iloc[0].situation_missing,1)
    def test_schema_and_duplicate_rejection(self):
        g,p=game_fixture(); p['plays'].insert(1,p['plays'][0])
        with self.assertRaises(ValueError): snapshots(p,g)
        g,p=game_fixture(); p['homeTeam']['score']=3
        with self.assertRaises(ValueError): snapshots(p,g)
    def test_elo_no_same_day_outcome_leakage(self):
        rows=[]
        for gid,date in [(1,'2023-10-01'),(2,'2023-10-01'),(3,'2023-10-02')]:
            rows.append({'game_id':gid,'season':20232024,'date':date,'start_utc':date+'T19:00:00Z',
                         'home':'TOR','away':'NYI','status':'OFF','home_score':2,'away_score':1})
        f,r=add_elo(pd.DataFrame(rows),CONFIG)
        self.assertEqual(f.iloc[0].home_elo,f.iloc[1].home_elo)
        self.assertGreater(f.iloc[2].home_elo,f.iloc[0].home_elo)
        changed=pd.DataFrame(rows); changed.loc[2,'home_score']=0
        f2,_=add_elo(changed,CONFIG)
        np.testing.assert_array_equal(f.home_elo,f2.home_elo)
    def test_training_temporal_pipeline(self):
        f=fake_states()
        with tempfile.TemporaryDirectory() as tmp:
            model,report=train(f,CONFIG,tmp,origin="fixture")
            p=predict(model,f.tail(20))
            self.assertTrue(np.isfinite(p).all()); self.assertTrue(((p>=0)&(p<=1)).all())
            self.assertEqual(report['partitions']['test']['games'],40)
            self.assertLess(report['partitions']['training']['last'],report['partitions']['tuning']['first'])
            self.assertTrue((Path(tmp)/'model.joblib').exists())
    def test_bracket_stage_mass_and_repeatability(self):
        s=league_fixture(); empty=pd.DataFrame(columns=['date','game_id','home','away'])
        ratings={t:1500 for t in s.team}; params={'log_base':1.03,'home_effect':.05,'strength_effect':.4}
        r=simulate(empty,s,ratings,params,n=100,seed=504)
        repeat=simulate(empty,s,ratings,params,n=100,seed=504)
        self.assertEqual(r,repeat)
        for stage,count in [('playoffs',16),('round_2',8),('conference_final',4),('final',2),('cup',1)]:
            self.assertAlmostEqual(sum(t[stage] for t in r['teams']),count)
        for t in r['teams']:
            self.assertGreaterEqual(t['playoffs'],t['round_2']); self.assertGreaterEqual(t['final'],t['cup'])
    def test_schedule_completeness_gate(self):
        s=league_fixture(); empty=pd.DataFrame(columns=['home','away'])
        check_schedule(empty,s,0)
        with self.assertRaises(ValueError): check_schedule(empty,s,84)
    def test_nonempty_complete_schedule(self):
        s=league_fixture(); teams=s.team.tolist()
        schedule=pd.DataFrame([{'home':teams[i],'away':teams[i+1]} for i in range(0,32,2)])
        check_schedule(schedule,s,1)

    def test_regular_season_points_mass(self):
        s=league_fixture(); schedule=pd.DataFrame([{'date':'2026-10-02','game_id':1,'home':'EA0','away':'EA1'}])
        ratings={t:1500 for t in s.team}
        r=simulate(schedule,s,ratings,{'log_base':-2,'home_effect':0,'strength_effect':0},n=100,seed=7)
        points=sum(t['mean_points'] for t in r['teams'])
        self.assertGreaterEqual(points,2); self.assertLessEqual(points,3)
    def test_goal_fit_rejects_invalid_overtime(self):
        df=pd.DataFrame([{'home_score':4,'away_score':1,'last_period':'OT','home_elo':1500,'away_elo':1500}])
        with self.assertRaises(ValueError): fit_goal_rates(df)

if __name__=='__main__': unittest.main()
