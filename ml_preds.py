from pybaseball import statcast, pitching_stats_range, statcast_pitcher, pitching_stats
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import re
import unicodedata
import os
from xgboost import XGBRegressor

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

    features = ['Age', 'IP', 'SO9', 'ERA', 'WHIP', 'K_BB_ratio', 'SO_per_IP', 'GS', 'Pit', 'AB', 'BF']
    df['SO_per_game'] = df['SO'] / df['G'].replace(0, np.nan)
    target = 'SO_per_game'

    df = df.dropna(subset=features)

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, features

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
    df_today = df_today.replace([np.inf, -np.inf], np.nan).dropna(subset=['Age', 'IP', 'SO9', 'ERA', 'WHIP', 'K_BB_ratio', 'SO_per_IP', 'GS', 'Pit', 'AB', 'BF'])

    features = ['Age', 'IP', 'SO9', 'ERA', 'WHIP', 'K_BB_ratio', 'SO_per_IP', 'GS', 'Pit', 'AB', 'BF']
    df_today['SO_pred'] = model.predict(df_today[features])

    df_final = pd.merge(df_props, df_today[['Name', 'SO_pred']], on='Name', how='left')

    df_final['date'] = curr_date
    df_final['edge'] = df_final['SO_pred'] - df_final['prizepicks_line']
    df_final['recommendation'] = df_final['edge'].apply(lambda x: 'OVER' if x > 0.5 else ('UNDER' if x < -0.5 else 'NO BET'))
    df_final['abs_edge'] = df_final['edge'].abs()
    df_final = df_final.sort_values(by='abs_edge', ascending=False)

    print(df_final[['player_pp', 'SO_pred', 'prizepicks_line', 'edge', 'recommendation']])
    
    cols = ['date','player_pp', 'SO_pred', 'prizepicks_line', 'edge', 'recommendation']
    df_final = df_final[cols].copy()
    save_predictions(df_final, curr_date)
    
def save_predictions(df, date):
    history_path = 'best_lines_ml/mlb_preds_history.csv'
    df['date'] = date
    df = df[df['recommendation'] != 'NO BET'].copy()

    separator = pd.DataFrame([['---'] * len(df.columns)], columns=df.columns)


    if os.path.exists(history_path):
        existing = pd.read_csv(history_path)
        combined = pd.concat([existing, separator, df], ignore_index=True)
        combined.to_csv(history_path, index=False)
    else:
        df.to_csv(history_path, index=False)

if __name__ == "__main__":
    save_pitching_stats()
    
    X_train, X_test, y_train, y_test, features = prepare_data()
    
    #model = RandomForestRegressor(n_estimators=100, random_state=42)
    model = XGBRegressor(n_estimators=100, learning_rate=0.05)

    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Test R²:", r2_score(y_test, y_pred))
    print("Test MSE:", mean_squared_error(y_test, y_pred))
    
    predict_today(model)
    
    #Results right now
    #Test R²: 0.9199029533044669 , pretty good
    #Test MSE: 0.26693755185171036 , also pretty good for now