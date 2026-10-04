"""Improve presentation without changing forecast or evaluation numbers."""
import argparse, json, re
from pathlib import Path

TEAM_NAMES = dict(zip(
 'ANA BOS BUF CAR CBJ CGY CHI COL DAL DET EDM FLA LAK MIN MTL NJD NSH NYI NYR OTT PHI PIT SEA SJS STL TBL TOR UTA VAN VGK WPG WSH'.split(),
 ['Anaheim Ducks','Boston Bruins','Buffalo Sabres','Carolina Hurricanes','Columbus Blue Jackets','Calgary Flames','Chicago Blackhawks','Colorado Avalanche','Dallas Stars','Detroit Red Wings','Edmonton Oilers','Florida Panthers','Los Angeles Kings','Minnesota Wild','Montreal Canadiens','New Jersey Devils','Nashville Predators','New York Islanders','New York Rangers','Ottawa Senators','Philadelphia Flyers','Pittsburgh Penguins','Seattle Kraken','San Jose Sharks','St. Louis Blues','Tampa Bay Lightning','Toronto Maple Leafs','Utah Mammoth','Vancouver Canucks','Vegas Golden Knights','Winnipeg Jets','Washington Capitals']))
MODEL_NAMES = {'selected_calibrated':'Calibrated Model','baseline':'Score/Time Baseline','selected_raw':'Uncalibrated Model','pregame_only':'Pregame Strength Only','constant_50_percent':'Always 50/50'}

def readable(artifact, forecast):
    def clean(value):
        if isinstance(value,str):
            value=value.replace(' — ', ' - ').replace('—','-').replace('–','-')
            return re.sub(r'(?<!\d)(20\d{2})(20\d{2})(?!\d)',r'\1-\2',value)
        if isinstance(value,list): return [clean(v) for v in value]
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
        return value
    artifact=clean(artifact)
    m=artifact['manifest']; d=artifact['snapshot']['datasets']
    m['title']='Leafs Lab - From a 2-1 Lead to Cup Chances'
    intro=next(b for b in m['blocks'] if b['id']=='intro')
    intro['body']=re.sub(r'^# [^\n]+', '# '+m['title'], intro['body'])
    for name in ['teams','toronto','cup_comparison']:
        for row in d[name]: row['team_name']=TEAM_NAMES[row['team']]
    for row in d['models']: row['model_label']=MODEL_NAMES[row['model']]
    for chart in m['charts']:
        # Explicit encoding format fixes tooltip rounding of small probabilities.
        if chart.get('valueFormat')=='percent': chart['encodings']['y']['format']='percent'
        if chart['id']=='brier':
            chart['encodings']['x'].update(field='model_label',label='Prediction Approach')
            chart['encodings']['y']['label']='Brier score (lower is better)'
        if chart['id']=='cup':
            chart['title']='Stanley Cup win probability: top 10 teams and Toronto'
            chart['type']='horizontalBar'
            chart['encodings']['x'].update(field='team_name',label='NHL Team')
            chart['encodings']['y']['label']='Chance of winning the Stanley Cup'
    table=next(t for t in m['tables'] if t['id']=='teams')
    table['title']='NHL league forecast: projected points and postseason chances'
    labels={'team_name':'Full team name','team':'Full team name','elo':'Team strength (Elo rating)','mean_points':'Projected season points','playoffs':'Make playoffs','final':'Reach Stanley Cup Final','cup':'Win Stanley Cup'}
    for c in table['columns']:
        c['label']=labels[c['field']]
        if c['field']=='team': c['field']='team_name'
    model_guide={'id':'model_guide','type':'markdown','body':'### What the prediction approaches mean\n\n- **Calibrated Model:** the selected game model after probability calibration.\n- **Score/Time Baseline:** a simpler prediction using the score and time remaining.\n- **Uncalibrated Model:** the selected game model before calibration.\n- **Pregame Strength Only:** team-strength information before the game.\n- **Always 50/50:** assigns each side a 50% chance.\n\nThe x-axis compares prediction approaches. Brier score measures probability error; lower is better.'}
    tor=next(t for t in forecast['teams'] if t['team']=='TOR'); n=forecast['simulations']; lo,hi=tor['cup_mc_95_interval']
    frequency = f'That is approximately 1 in {1/tor["cup"]:.0f} simulated seasons. The bar is small because all teams share one axis; it is not zero.' if tor['cup'] else 'No Cup wins occurred in this simulation run; a zero simulated count does not establish that a Cup win is impossible.'
    cup_guide={'id':'cup_guide','type':'markdown','sourceId':'forecast','body':f"### Toronto's small bar explained\n\n**Toronto Maple Leafs: {tor['cup']:.2%} chance of winning the Stanley Cup**, with {tor['cup_simulation_wins']:,} Cup wins in {n:,} simulated seasons. {frequency}\n\nToronto's projected points: {tor['mean_points']:.1f}, with an 80% simulation range of {tor['points_p10']:.0f}-{tor['points_p90']:.0f}. Chance of making the playoffs: {tor['playoffs']:.2%}; reaching the Stanley Cup Final: {tor['final']:.2%}.\n\nThe 95% Monte Carlo interval for the Cup estimate is {lo:.2%}-{hi:.2%}. This measures simulation sampling error only. The experimental model can still be wrong; it does not include explicit injury, lineup or trade adjustments."}
    m['blocks']=[b for b in m['blocks'] if b['id'] not in ['model_guide','cup_guide']]
    for target,block in [('block_brier',model_guide),('block_cup',cup_guide)]:
        idx=next(i for i,b in enumerate(m['blocks']) if b['id']==target);m['blocks'].insert(idx+1,block)
    # Keep operational details in GitHub; the public page leads with hockey.
    season=str(forecast['season']); label=season[:4]+'-'+season[4:]
    intro['body']='# '+m['title']+'\n\nA game-night question: how likely is Toronto to win from a 2-1 lead?'
    m['blocks']=[b for b in m['blocks'] if b['id']!='automation_status']
    next(b for b in m['blocks'] if b['id']=='season_guardrail')['body']='## '+label+' Stanley Cup Forecast\n\nExperimental model estimates. Cup intervals describe simulation error, not full model uncertainty.'
    return artifact

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--forecast',type=Path,required=True);a=p.parse_args()
    original=json.loads(a.artifact.read_text()); result=readable(original,json.loads(a.forecast.read_text()))
    for name,rows in original['snapshot']['datasets'].items():
        for old,new in zip(rows,result['snapshot']['datasets'][name]):
            for k,v in old.items():
                if isinstance(v,(int,float)): assert new[k]==v,(name,k)
    a.artifact.write_text(json.dumps(result,indent=2)+'\n')
