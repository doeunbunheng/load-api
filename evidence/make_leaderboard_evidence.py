import json
from pathlib import Path

import pandas as pd

base = Path(__file__).resolve().parent.parent
leaderboard = pd.read_json(base / "artifacts" / "leaderboard.json")
links = json.loads((base / "artifacts" / "selected_versions.json").read_text(encoding="utf-8"))["run_links"]
leaderboard["run_link"] = leaderboard["run_name"].map(links)
columns = [
    "rank",
    "run_name",
    "model_family",
    "recall",
    "precision",
    "roc_auc",
    "pr_auc",
    "f1",
    "business_score",
    "run_id",
]
lines = [
    "Final training batch leaderboard",
    "================================",
    "",
    leaderboard[columns].to_string(index=False),
    "",
    "Run links",
]
lines.extend(f"{row.run_name}: {row.run_link}" for row in leaderboard.itertuples())
text = "\n".join(lines)
(base / "evidence" / "01_leaderboard.txt").write_text(text, encoding="utf-8")
print(text)
