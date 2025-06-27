"""
Try to evaluate the game after it has ended and update the csv files with results.
"""

import pandas as pd
import os
import re
from datetime import datetime
from glob import glob
from pybaseball import pitching_stats_range

def normalize_name(name):
    if pd.isna(name):
        return ""
    n = name.lower().strip()
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", n)
    n = re.sub(r"\s*\([^)]*\)", "", n)
    return re.sub(r"\s+", " ", n).strip()

def evaluate_file(path):
    m = re.search(r"best_lines_(\d{4}-\d{2}-\d{2})\.csv$", path)
    if not m:
        return
    slate_date = m.group(1)

    if slate_date >= datetime.today().strftime("%Y-%m-%d"):
        print(f"Skipping {os.path.basename(path)} — game may not be finished.")
        return

    print(f"Evaluating {os.path.basename(path)} for {slate_date}")

    try:
        logs = pitching_stats_range(slate_date, slate_date)
    except Exception as e:
        print(f"Failed to fetch stats for {slate_date}: {e}")
        return

    if logs.empty:
        print(f"No pitching data found for {slate_date}, skipping.")
        return

    logs = logs[(logs['GS'] > 0) | (logs['IP'] > 0)]
    logs['player_norm'] = logs['Name'].apply(normalize_name)
    actual = logs.set_index('player_norm')['SO'].to_dict()

    df = pd.read_csv(path)
    actual_sos = []
    results = []
    for _, row in df.iterrows():
        pn = normalize_name(row['Player'])
        so = actual.get(pn, None)
        actual_sos.append(so)

        line = row.get('Line (PP)')
        pick = row.get('Best Bet') or row.get('Pick')

        if so is None or pd.isna(line) or pick not in ('OVER','UNDER'):
            results.append('UNKNOWN')
        else:
            if so == line:
                results.append('PUSH')
            elif pick == 'OVER':
                results.append('HIT' if so > line else 'MISS')
            else:
                results.append('HIT' if so < line else 'MISS')

    df['Actual_SO'] = actual_sos
    df['Result']    = results
    df.to_csv(path, index=False)
    print(f"Updated: {path}\n")


def main():
    files = glob("best_lines/best_lines_*.csv")
    for f in sorted(files):
        evaluate_file(f)

if __name__ == "__main__":
    main()
