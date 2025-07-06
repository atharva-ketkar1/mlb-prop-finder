import pandas as pd
import os
import re
from datetime import datetime
from glob import glob
from pybaseball import pitching_stats_range
from find_best_lines import normalize_name


def fetch_actual_stats(slate_date):
    try:
        logs = pitching_stats_range(slate_date, slate_date)
    except Exception as e:
        print(f"Failed to fetch stats for {slate_date}: {e}")
        return {}

    logs = logs[(logs['GS'] > 0) | (logs['IP'] > 0)]
    logs['player_norm'] = logs['Name'].apply(normalize_name)
    return logs.set_index('player_norm')['SO'].to_dict()

def evaluate_best_lines_file(path):
    m = re.search(r"best_lines_(\d{4}-\d{2}-\d{2})\.csv$", path)
    if not m:
        return
    slate_date = m.group(1)

    if slate_date >= datetime.today().strftime("%Y-%m-%d"):
        print(f"Skipping {os.path.basename(path)} — game may not be finished.")
        return

    df_check = pd.read_csv(path)
    if 'Result' in df_check.columns and df_check['Result'].notna().all():
        print(f"Already evaluated {os.path.basename(path)}")
        return

    print(f"Evaluating {os.path.basename(path)} for {slate_date}")
    actual = fetch_actual_stats(slate_date)
    if not actual:
        return

    df = df_check.copy()
    actual_sos = []
    results = []

    for _, row in df.iterrows():
        pn = normalize_name(row['Player'])
        so = actual.get(pn, None)
        actual_sos.append(so)

        line = row.get('Line (PP)')
        pick = row.get('Best Bet') or row.get('Pick')

        if so is None or pd.isna(line) or pick not in ('OVER', 'UNDER'):
            results.append('')
        else:
            if so == line:
                results.append('PUSH')
            elif pick == 'OVER':
                results.append('HIT' if so > line else 'MISS')
            else:
                results.append('HIT' if so < line else 'MISS')

    df['Actual_SO'] = actual_sos
    df['Result'] = results
    df.to_csv(path, index=False)
    print(f"Updated: {path}\n")

    evaluate_slate_file(slate_date, actual)

def evaluate_slate_file(slate_date, actual=None):
    slate_path = f"data/mlb_slates/mlb_pitcher_slate_{slate_date}.csv"
    if slate_date >= datetime.today().strftime("%Y-%m-%d"):
        print(f"Skipping {os.path.basename(slate_path)} — game may not be finished.")
        return

    df = pd.read_csv(slate_path)
    if 'Result' in df.columns and df['Result'].notna().all():
        print(f"Already evaluated {os.path.basename(slate_path)}")
        return

    print(f"Evaluating {os.path.basename(slate_path)}")

    if actual is None:
        actual = fetch_actual_stats(slate_date)
        if not actual:
            return

    actual_sos = []
    results = []

    for _, row in df.iterrows():
        pn = normalize_name(row['player_pp'])
        so = actual.get(pn, None)
        actual_sos.append(so)

        line = row.get('prizepicks_line')
        pick = row.get('dk_label')

        if so is None or pd.isna(line) or pick not in ('Over', 'Under'):
            results.append('')
        else:
            if so == line:
                results.append('PUSH')
            elif pick == 'Over':
                results.append('HIT' if so > line else 'MISS')
            else:
                results.append('HIT' if so < line else 'MISS')

    df['Actual_SO'] = actual_sos
    df['Result'] = results
    df.to_csv(slate_path, index=False)
    print(f"Updated: {slate_path}\n")

def main():
    files = glob("best_lines/best_lines_*.csv")
    for f in sorted(files):
        evaluate_best_lines_file(f)

    slate_files = glob("data/mlb_slates/mlb_pitcher_slate_*.csv")
    for f in sorted(slate_files):
        m = re.search(r"mlb_pitcher_slate_(\d{4}-\d{2}-\d{2})\.csv$", f)
        if not m:
            continue
        slate_date = m.group(1)
        evaluate_slate_file(slate_date)

if __name__ == "__main__":
    main()