from pybaseball import statcast, pitching_stats_range, statcast_pitcher, pitching_stats
import pybaseball
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pandas as pd
import re
import unicodedata

def fix_escaped_unicode(text):
    if pd.isna(text):
        return ""
    try:
        return bytes(text, "utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")
    except Exception:
        return text



def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    n = n.lower().strip()
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", n)
    n = re.sub(r"\s*\([^)]*\)", "", n)
    return re.sub(r"\s+", " ", n).strip()

def save_pitching_stats():
    end_date = datetime.today()
    end_str = end_date.strftime('%Y-%m-%d')

    df_pitching = pitching_stats_range('2023-04-01', end_str)
    df_pitching['Name'] = df_pitching['Name'].apply(fix_escaped_unicode)
    df_pitching['Name'] = df_pitching['Name'].apply(normalize_name)

    df_pitching.to_csv('data/pitcher_stats/pitching_stats_2023-2025.csv', index=False, encoding='utf-8')

def prepare_data():
    df = pd.read_csv('data/pitcher_stats/pitching_stats_2023-2025.csv')
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['SO', 'IP', 'SO9', 'ERA', 'WHIP', 'BB'])

    df['SO_per_IP'] = df['SO'] / df['IP']
    df['K_BB_ratio'] = df['SO'] / df['BB'].replace(0, np.nan)

    features = ['Age', 'IP', 'SO9', 'ERA', 'WHIP', 'K_BB_ratio', 'SO_per_IP']
    df['SO_per_game'] = df['SO'] / df['G'].replace(0, np.nan)
    target = 'SO_per_game'

    df = df.dropna(subset=features)

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    df = df.dropna(subset=features)
    
    return X, y

def predict_today(model):
    curr_date = datetime.today().strftime('%Y-%m-%d')
    df_props = pd.read_csv(f'data/mlb_slates/mlb_pitcher_slate_{curr_date}.csv')
    df_props['Name'] = df_props['player_pp'].apply(fix_escaped_unicode)
    df_props['Name'] = df_props['player_pp'].apply(normalize_name)

    df_stats = pd.read_csv('data/pitcher_stats/pitching_stats_2023-2025.csv')


    df_stats_sorted = df_stats.sort_values(by=['Name', '#days'])
    df_latest = df_stats_sorted.groupby('Name').first().reset_index()

    df_today = df_latest[df_latest['Name'].isin(df_props['Name'])].copy()

    df_today['SO_per_IP'] = df_today['SO'] / df_today['IP'].replace(0, np.nan)
    df_today['K_BB_ratio'] = df_today['SO'] / df_today['BB'].replace(0, np.nan)
    df_today = df_today.replace([np.inf, -np.inf], np.nan).dropna(subset=['Age', 'IP', 'SO9', 'ERA', 'WHIP', 'K_BB_ratio', 'SO_per_IP'])

    features = ['Age', 'IP', 'SO9', 'ERA', 'WHIP', 'K_BB_ratio', 'SO_per_IP']
    df_today['SO_pred'] = model.predict(df_today[features])

    df_final = pd.merge(df_props, df_today[['Name', 'SO_pred']], on='Name', how='left')

    df_final['edge'] = df_final['SO_pred'] - df_final['prizepicks_line']
    df_final['recommendation'] = df_final['edge'].apply(lambda x: 'OVER' if x > 0.5 else ('UNDER' if x < -0.5 else 'NO BET'))
    df_final['abs_edge'] = df_final['edge'].abs()
    df_final = df_final.sort_values(by='abs_edge', ascending=False)

    print(df_final[['player_pp', 'SO_pred', 'prizepicks_line', 'edge', 'recommendation']])
    

    
if __name__ == "__main__":
    save_pitching_stats()
    X, y = prepare_data()
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    predict_today(model)