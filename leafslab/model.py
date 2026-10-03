"""Chronological model selection, separate calibration, untouched season test."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score
FEATURES=['goal_diff','remaining','score_clock','pregame_logit','shot_diff',
          'skater_diff','home_empty','away_empty','situation_missing']
BASE_FEATURES=['goal_diff','remaining','score_clock']

def metrics(y,p,w):
    y=np.asarray(y); p=np.clip(np.asarray(p),1e-6,1-1e-6); w=np.asarray(w)
    return {'brier':float(np.average((p-y)**2,weights=w)),
            'log_loss':float(log_loss(y,p,sample_weight=w,labels=[0,1])),
            'auc':float(roc_auc_score(y,p,sample_weight=w)) if len(np.unique(y))==2 else None,
            'accuracy':float(accuracy_score(y,p>=.5,sample_weight=w))}

def fit_model(frame,features,kind='logistic',**params):
    if kind=='logistic':
        model=Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(max_iter=1000,C=params.get('C',1)))])
        model.fit(frame[features],frame.home_win,classifier__sample_weight=frame.weight)
    else:
        model=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=params.get('leaves',15),
                                             min_samples_leaf=100,l2_regularization=2,random_state=504)
        model.fit(frame[features],frame.home_win,sample_weight=frame.weight)
    return model

def calibration_bins(frame,p,bins=10):
    tmp=frame[['game_id','home_win','weight']].copy(); tmp['p']=p
    tmp['bin']=np.minimum((tmp.p*bins).astype(int),bins-1)
    out=[]
    for b,g in tmp.groupby('bin'):
        out.append({'bin':int(b),'mean_prediction':float(np.average(g.p,weights=g.weight)),
                    'observed_rate':float(np.average(g.home_win,weights=g.weight)),
                    'games':int(g.game_id.nunique()),'states':len(g)})
    return out

def bootstrap_metric_difference(frame,p,baseline,seed=504,n=400):
    tmp=frame[['game_id','home_win','weight']].copy()
    tmp['diff']=((np.asarray(p)-tmp.home_win)**2-(np.asarray(baseline)-tmp.home_win)**2)*tmp.weight
    bygame=tmp.groupby('game_id')['diff'].sum().to_numpy()
    rng=np.random.default_rng(seed)
    draws=np.array([rng.choice(bygame,len(bygame),replace=True).mean() for _ in range(n)])
    return {'brier_difference':float(bygame.mean()),'95_percent_game_bootstrap_interval':np.quantile(draws,[.025,.975]).tolist(),
            'bootstrap_draws':n,'interpretation':'Negative favors selected model; interval crossing zero is inconclusive.'}

def cohort_check(frame,p):
    """Home-leading 80-90% states, one equal contribution per eligible game.
    This is a selected-cohort calibration diagnostic, not a proof about one game.
    """
    f=frame.copy(); f['p']=p
    f=f[(f.p>=.8)&(f.p<.9)&(f.goal_diff>0)]
    if f.empty: return {'games':0,'observed_rate':None}
    g=f.groupby('game_id').agg(home_win=('home_win','first'),p=('p','mean'))
    rng=np.random.default_rng(504); values=g.home_win.to_numpy()
    boots=[rng.choice(values,len(values),replace=True).mean() for _ in range(1000)]
    return {'games':len(g),'mean_prediction':float(g.p.mean()),'observed_rate':float(g.home_win.mean()),
            '95_percent_game_bootstrap_interval':np.quantile(boots,[.025,.975]).tolist()}

def train(states,config,out='outputs',origin='verified_nhl'):
    out=Path(out); out.mkdir(exist_ok=True,parents=True)
    training=states[states.season.isin(config['train_seasons'])].copy()
    validation=states[states.season==config['validation_season']].copy()
    testing=states[states.season==config['test_season']].copy()
    dates=sorted(validation.date.unique()); split=dates[len(dates)//2]
    tuning=validation[validation.date<split].copy(); calibration=validation[validation.date>=split].copy()
    groups=[training,tuning,calibration,testing]
    if any(len(g)==0 or g.home_win.nunique()!=2 for g in groups): raise ValueError('Empty/one-class temporal partition')
    for left,right in zip(groups,groups[1:]):
        if left.date.max()>=right.date.min(): raise ValueError('Temporal split overlap')
        if set(left.game_id)&set(right.game_id): raise ValueError('Game leakage')
    baseline=fit_model(training,BASE_FEATURES)
    candidates={
        'logistic_C0.1':fit_model(training,FEATURES,C=.1),
        'logistic_C1':fit_model(training,FEATURES,C=1),
        'boosted_7leaves':fit_model(training,FEATURES,'boosted',leaves=7),
        'boosted_15leaves':fit_model(training,FEATURES,'boosted',leaves=15),
        'score_time_baseline':baseline}
    candidate_features={name:BASE_FEATURES if name=='score_time_baseline' else FEATURES for name in candidates}
    selection={name:metrics(tuning.home_win,m.predict_proba(tuning[candidate_features[name]])[:,1],tuning.weight) for name,m in candidates.items()}
    best=min(selection,key=lambda name:selection[name]['brier']); model=candidates[best]; selected_features=candidate_features[best]
    raw=np.clip(model.predict_proba(calibration[selected_features])[:,1],1e-6,1-1e-6)
    calibrator=LogisticRegression(C=1e6,max_iter=1000)
    calibrator.fit(np.log(raw/(1-raw)).reshape(-1,1),calibration.home_win,sample_weight=calibration.weight)
    bundle={'model':model,'calibrator':calibrator,'baseline':baseline,'features':selected_features,
            'selected':best,'config':config,'version':'0.1'}
    joblib.dump(bundle,out/'model.joblib')
    raw_test=model.predict_proba(testing[selected_features])[:,1]
    p=predict(bundle,testing); basep=baseline.predict_proba(testing[BASE_FEATURES])[:,1]
    # PFI only describes a frozen model. Never use these test diagnostics for selection.
    rng=np.random.default_rng(config['seed']); importance=[]
    original=metrics(testing.home_win,p,testing.weight)['brier']
    for feature in FEATURES:
        perm=testing.copy(); perm[feature]=rng.permutation(perm[feature].to_numpy())
        importance.append({'feature':feature,'brier_increase':metrics(perm.home_win,predict(bundle,perm),perm.weight)['brier']-original})
    report={'status':'real_data_evaluated' if origin=='verified_nhl' else 'fixture','selected_model':best,'tuning_results':selection,
            'test':{'baseline':metrics(testing.home_win,basep,testing.weight),
                    'selected_raw':metrics(testing.home_win,raw_test,testing.weight),
                    'selected_calibrated':metrics(testing.home_win,p,testing.weight),
                    'pregame_only':metrics(testing.home_win,1/(1+np.exp(-testing.pregame_logit)),testing.weight),
                    'constant_50_percent':metrics(testing.home_win,np.full(len(testing),.5),testing.weight)},
            'paired_bootstrap':bootstrap_metric_difference(testing,p,basep),
            'calibration':calibration_bins(testing,p),'instinct_cohort':cohort_check(testing,p),
            'feature_importance':importance,
            'partitions':{name:{'games':int(g.game_id.nunique()),'states':len(g),'first':str(g.date.min()),'last':str(g.date.max())}
                          for name,g in zip(['training','tuning','calibration','test'],groups)}}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    export=testing[['game_id','date','elapsed','home_win','weight']].assign(prediction=p,baseline=basep,raw_prediction=raw_test,pregame_probability=1/(1+np.exp(-testing.pregame_logit)))
    temporary=out/'test_predictions.tmp.csv'; export.to_csv(temporary,index=False)
    readback=pd.read_csv(temporary)
    if len(readback)!=len(testing) or readback.game_id.nunique()!=testing.game_id.nunique(): raise ValueError('Incomplete prediction export')
    temporary.replace(out/'test_predictions.csv')
    return bundle,report

def predict(bundle,frame):
    raw=np.clip(bundle['model'].predict_proba(frame[bundle['features']])[:,1],1e-6,1-1e-6)
    return bundle['calibrator'].predict_proba(np.log(raw/(1-raw)).reshape(-1,1))[:,1]

def explain_state(bundle,state):
    """Deterministic probability contrasts, explicitly noncausal and nonadditive."""
    frame=pd.DataFrame([state]); p=float(predict(bundle,frame)[0]); contrasts=[]
    for feature,value in [('goal_diff',0),('pregame_logit',0),('skater_diff',0),('home_empty',0),('away_empty',0)]:
        variant=frame.copy(); variant[feature]=value
        variant['score_clock']=variant.goal_diff/np.sqrt(variant.remaining+.025)
        q=float(predict(bundle,variant)[0])
        contrasts.append({'feature':feature,'alternative_value':value,'alternative_probability':q,'difference_percentage_points':100*(p-q)})
    return {'home_win_probability':p,'contrasts':contrasts,
            'warning':'One-feature comparisons are not causal or additive; hypothetical states may be rare.'}
