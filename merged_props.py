import pandas as pd
import datetime
import os
import unicodedata
import re

from scrapes.scrape_prizepicks import scrape_prizepicks_mlb
from scrapes.scrape_draftkings import scrape_draftkings_mlb
from scrapes.scrape_underdog import scrape_underdog_mlb

""" 
There are name mismatches between the sites, so we need a consistent way to generate keys for players.
Ex.) Jacob Latz vs Jake Latz, Germán Márquez vs German Marquez, Logan Allen vs Logan Allen (CLE), etc. 
Use chatgpt for regex help bc Im not tryna do that rn.
"""
def name_key(name):
    name = unicodedata.normalize('NFKD', name)
    name = name.encode('ascii', 'ignore').decode('ascii')
    name = name.lower().strip()
    name = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b\.?', '', name)
    name = re.sub(r'\s*\([^)]*\)', '', name)
    name = re.sub(r'\s+', ' ', name)

    parts = name.split()
    if len(parts) < 2:
        return name

    first_name = parts[0]
    last_name = parts[-1]

    first_initials = first_name[:2]  
    key = first_initials + "_" + last_name
    return key

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
    df_sorted = df.sort_values(['player_key', 'dk_prob'], ascending=[True, False])
    df_filtered = df_sorted.drop_duplicates(subset=['player_key'], keep='first')
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
    ud_df = scrape_underdog_mlb()

    pp_df['player_key'] = pp_df['player'].apply(name_key)
    dk_df['player_key'] = dk_df['player'].apply(name_key)
    ud_df['player_key'] = ud_df['player'].apply(name_key)

    mlb_slate = pd.merge(pp_df, dk_df, on='player_key', how='left', suffixes=('_pp', '_dk'))

    mlb_slate = filter_best_dk_lines(mlb_slate)

    mlb_slate = mlb_slate.merge(
        ud_df.add_suffix('_ud'),
        left_on='player_key',
        right_on='player_key_ud',
        how='left'
    )

    mlb_slate = mlb_slate.drop(columns=[
        'player_dk', 'player_key', 'player_ud', 'market_name', 'dk_prob', 'player_key_ud','payout_multiplier_over_ud', 'payout_multiplier_under_ud'
    ], errors='ignore')

    columns_order = [
        'player_pp', 'team', 'stat_type', 'prizepicks_line',
        'dk_line', 'dk_odds', 'dk_label',
        'line_ud', 'over_odds_ud', 'under_odds_ud',
        'payout_multiplier_over_ud', 'payout_multiplier_under_ud'
    ]
    columns_order = [col for col in columns_order if col in mlb_slate.columns]
    mlb_slate = mlb_slate[columns_order]

    save_props(mlb_slate)

if __name__ == "__main__":
    main()
