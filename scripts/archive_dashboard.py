"""Publish a verified portable dashboard into a season archive; preserve older seasons."""
import argparse, html, json, re, shutil
from pathlib import Path


def publish(portable, forecast_path, docs):
    forecast=json.loads(forecast_path.read_text())
    if forecast.get('status')!='real_data_simulated':
        raise ValueError('Only verified real-data forecasts may be archived')
    season=str(forecast['season'])
    if not re.fullmatch(r'\d{8}',season) or int(season[4:])!=int(season[:4])+1:
        raise ValueError('Invalid season identity')
    date=forecast['as_of']
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',date): raise ValueError('Invalid snapshot date')
    text=portable.read_text()
    if date not in text or 'data-analytics' not in text: raise ValueError('Portable dashboard date/runtime mismatch')
    slug=season[:4]+'-'+season[6:]
    docs.mkdir(exist_ok=True,parents=True)
    registry_path=docs/'seasons.json'
    registry=json.loads(registry_path.read_text()) if registry_path.exists() else []
    entry={'season':season,'label':season[:4]+'-'+season[4:],'as_of':date,'path':'seasons/'+slug+'/index.html','status':'Experimental forecast snapshot'}
    old=next((r for r in registry if r['season']==season),None)
    if old and old['as_of']>date: raise ValueError('Refusing to replace a newer archive with an older snapshot')
    registry=[r for r in registry if r['season']!=season]+[entry]
    registry.sort(key=lambda r:r['season'],reverse=True)
    target=docs/'seasons'/slug
    target.mkdir(parents=True,exist_ok=True)
    archive_nav='<nav style="padding:16px 24px;font:16px/1.5 system-ui;background:#003e7e;color:white"><a style="color:white" href="../../index.html">Leafs Lab home</a> · <a style="color:white" href="../../archive.html">Season archive</a> · '+html.escape(entry['label'])+' Stanley Cup Forecast</nav>'
    (target/'index.html').write_text(text.replace('<body>','<body>'+archive_nav,1))
    registry_path.write_text(json.dumps(registry,indent=2)+'\n')
    newest=registry[0]
    if newest['season']==season:
        nav='<nav style="padding:16px 24px;font:16px/1.5 system-ui;background:#003e7e;color:white"><strong>Leafs Lab</strong> · <a style="color:white" href="archive.html">Season archive</a> · <a style="color:white" href="https://github.com/singhyash2209/leafs-lab">Python code and methodology</a> · '+html.escape(entry['label'])+' Stanley Cup Forecast</nav>'
        (docs/'index.html').write_text(text.replace('<body>','<body>'+nav,1))
    rows=''.join('<tr><td>'+html.escape(r['label'])+'</td><td>'+html.escape(r['as_of'])+'</td><td>'+html.escape(r['status'])+'</td><td><a href="'+html.escape(r['path'])+'">Open dashboard</a></td></tr>' for r in registry)
    archive='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Leafs Lab - Season archive</title><style>body{font:16px/1.6 system-ui;color:#183047;max-width:1000px;margin:40px auto;padding:0 20px}a{color:#003e7e}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #ccd5de;padding:14px}header{border-top:8px solid #003e7e} .scroll{overflow:auto}</style></head><body><header><h1>Leafs Lab season archive</h1></header><p><a href="index.html">Latest published dashboard</a> · <a href="https://github.com/singhyash2209/leafs-lab">Code, notebook and methodology</a></p><p>Each season keeps its own dashboard and date. These are experimental statistical forecasts, with game-model and season-model validation reported separately.</p><div class="scroll"><table><thead><tr><th>Season</th><th>Snapshot date</th><th>Status</th><th>Explore</th></tr></thead><tbody>'''+rows+'''</tbody></table></div><p>Future seasons appear after their data, configuration and model checks are completed and published. This archive does not imply automatic forecasting for unconfigured seasons. Completed seasons remain available; historical revisions are retained in GitHub commit history.</p></body></html>'''
    (docs/'archive.html').write_text(archive)
    (docs/'.nojekyll').touch()
    print(json.dumps({'season':season,'as_of':date,'archives':len(registry),'path':str(target/'index.html')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--html',type=Path,required=True)
    parser.add_argument('--forecast',type=Path,required=True)
    parser.add_argument('--docs',type=Path,default=Path('docs'))
    args=parser.parse_args()
    publish(args.html,args.forecast,args.docs)
