import pandas as pd
import datetime
import os

from scrapes.scrape_prizepicks import scrape_prizepicks_mlb
from scrapes.scrape_draftkings import scrape_draftkings_mlb

def normalize_name(name):
    return name.strip().lower()

def odds_to_prob(odds):
    if pd.isna(odds):
        return None
    odds = str(odds).replace('−', '-').strip()
    try:
        odds_val = int(odds)
    except ValueError:
        return None
    if odds_val > 0:
        prob = 100 / (odds_val + 100)
    else:
        prob = -odds_val / (-odds_val + 100)
    return prob

def filter_best_dk_lines(df):
    df['dk_prob'] = df['dk_odds'].apply(odds_to_prob)
    df_sorted = df.sort_values(['player_norm', 'dk_prob'], ascending=[True, False])
    df_filtered = df_sorted.drop_duplicates(subset=['player_norm'], keep='first')
    return df_filtered

def save_props(df, output_dir="data/mlb_slates", filename_prefix="mlb_pitcher_slate"):
    today = datetime.date.today().isoformat()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename_prefix}_{today}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved MLB slate to {output_path}")
    return output_path

def main():
    pp_df = scrape_prizepicks_mlb()
    dk_df = scrape_draftkings_mlb()

    pp_df['player_norm'] = pp_df['player'].apply(normalize_name)
    dk_df['player_norm'] = dk_df['player'].apply(normalize_name)

    mlb_slate = pd.merge(pp_df, dk_df, on='player_norm', how='left', suffixes=('_pp', '_dk'))
    mlb_slate = filter_best_dk_lines(mlb_slate)

    mlb_slate = mlb_slate.drop(columns=['player_norm', 'market_name', 'dk_prob'], errors='ignore')

    save_props(mlb_slate)

if __name__ == "__main__":
    main()
