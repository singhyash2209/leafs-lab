"""Build the frozen one-game dashboard from the saved analysis."""
import copy
import json
from pathlib import Path

artifact = copy.deepcopy(json.loads(Path("outputs/artifact.json").read_text()))
manifest = artifact["manifest"]
manifest["title"] = "Leafs Lab - Game Night Win Probability"
manifest["description"] = "Toronto Maple Leafs vs New York Islanders, September 30, 2026."
manifest["cards"] = [card for card in manifest["cards"] if card["id"] == "game_p"]
manifest["charts"] = [chart for chart in manifest["charts"] if chart["id"] == "game_curve"]
chart = manifest["charts"][0]
chart["title"] = "Toronto Maple Leafs Win Probability During the Game"
chart["encodings"]["x"]["label"] = "Game Time Elapsed (Minutes)"
chart["encodings"]["y"]["label"] = "Toronto Maple Leafs Win Probability"
manifest["tables"] = []
manifest["sources"] = [source for source in manifest["sources"] if source["id"] in {"game", "model"}]
manifest["blocks"] = [
    {"id": "intro", "type": "markdown", "body": "# Leafs Lab - Game Night Win Probability\n\nToronto Maple Leafs vs New York Islanders - September 30, 2026.\n\nOne game, one question: how likely was Toronto to win while leading 2-1 with 5:51 remaining?"},
    {"id": "game_headline", "type": "metric-strip", "cardIds": ["game_p"]},
    {"id": "interpretation", "type": "markdown", "body": "The model estimated an **85.9% Toronto win probability** at that moment. The chart below follows the estimate through the game as the score and time remaining changed."},
    {"id": "block_game_curve", "type": "chart", "chartId": "game_curve"},
    {"id": "method", "type": "markdown", "sourceId": "model", "body": "### How to Read This\n\nThis is a saved, retrospective analysis of one game. The model uses score, time remaining, prior team strength, shots, skaters and goalie-pulled indicators. This game was excluded from training.\n\nAn estimate is not a guaranteed result. On held-out games, the model did not beat a simpler score/time baseline. [View the code and methodology](https://github.com/singhyash2209/leafs-lab)."},
]
artifact["snapshot"]["datasets"] = {key: artifact["snapshot"]["datasets"][key] for key in ("game_summary", "game")}
Path("outputs/game_night_artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
print("Built frozen one-game artifact")
