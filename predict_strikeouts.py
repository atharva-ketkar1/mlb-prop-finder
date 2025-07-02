import pandas as pd
import re
import unicodedata
from sklearn.ensemble import RandomForestRegressor

def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    n = n.lower().strip()
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", n)
    n = re.sub(r"\s*\([^)]*\)", "", n)
    return re.sub(r"\s+", " ", n).strip()

def build_agg_pitcher_stats(logs):
    logs = logs[logs['GS'] > 0].copy()
    logs['player_norm'] = logs['Name'].apply(normalize_name)
    agg = logs.groupby('player_norm').agg({
        'SO': 'sum',
        'G': 'sum',
        'IP': 'mean',
        'ERA': 'mean',
        'WHIP': 'mean',
        'SO9': 'mean'
    }).reset_index()
    agg['SO_avg'] = agg['SO'] / agg['G']

    return agg

def build_statcast_features(statcast):
    statcast[['last_name', 'first_name']] = statcast['last_name, first_name'].str.split(', ', expand=True)
    statcast['player_norm'] = (statcast['first_name'] + " " + statcast['last_name']).apply(normalize_name)

    for col in ['k_percent', 'whiff_percent', 'woba']:
        statcast[col] = pd.to_numeric(statcast[col], errors='coerce')

    statcast_grouped = statcast.groupby('player_norm').agg({
        'k_percent': 'mean',
        'whiff_percent': 'mean',
        'woba': 'mean'
    }).reset_index()

    return statcast_grouped



def predict_strikeouts(slate):
    logs = pd.read_csv("data/pitcher_stats/logs_last_30_days.csv")
    statcast = pd.read_csv("data/pitcher_stats/pitcher_stats.csv")

    agg_stats = build_agg_pitcher_stats(logs)
    statcast_feats = build_statcast_features(statcast)
    train_df = agg_stats.merge(statcast_feats, on='player_norm', how='left').dropna(subset=['SO_avg'])
    """dupes = train_df['player_norm'][train_df['player_norm'].duplicated()]
    if len(dupes) == 0:
        print("no dupes")
    else:
        print(f"Duplicate player_norm in training data: {dupes.tolist()}")
        
        
    print(train_df['player_norm'].value_counts())"""

    feature_cols = ['IP', 'ERA', 'WHIP', 'SO9', 'k_percent', 'whiff_percent', 'woba']
    X_train = train_df[feature_cols].fillna(train_df.mean(numeric_only=True))
    y_train = train_df['SO_avg']
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    if isinstance(slate, str):
        slate = pd.read_csv(slate)

    slate = slate.copy()

    slate['player_norm'] = slate['player_pp'].apply(normalize_name)

    pred_df = slate.merge(agg_stats, on='player_norm', how='left')
    pred_df = pred_df.merge(statcast_feats, on='player_norm', how='left')

    pred_X = pred_df[feature_cols].fillna(train_df.mean(numeric_only=True))

    pred_df['predicted_ks'] = model.predict(pred_X)

    return pred_df[['player_pp', 'team', 'predicted_ks']]

if __name__ == "__main__":
    preds = predict_strikeouts("data/mlb_slates/mlb_pitcher_slate_2025-07-02.csv")
    print(preds)
