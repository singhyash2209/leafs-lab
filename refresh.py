"""Daily batch update. Scheduler activation is external and not implied."""
from datetime import datetime
from zoneinfo import ZoneInfo
import os, sys
from pathlib import Path
from leafslab.cli import main
if __name__=='__main__':
    today=datetime.now(ZoneInfo('America/Toronto')).date().isoformat()
    # A failed refresh must not leave an old artifact presented as current.
    for name in ['artifact.json','dashboard.html','report.md']:
        path=Path('outputs')/name
        if path.exists(): path.unlink()
    command='update' if Path('outputs/model.joblib').exists() else 'run'
    raise SystemExit(main([command,'--as-of',today]))
