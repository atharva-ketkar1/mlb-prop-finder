import pandas as pd
import os
import re
import unicodedata
from datetime import date
from predict_strikeouts import predict_strikeouts, normalize_name  # Import model + normalizer


def odds_to_prob(odds):
    try:
        o = int(str(odds).replace('−', '-'))
    except:
        return None
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def average_odds(o1, o2):
    vals = []
    for o in (o1, o2):
        try:
            vals.append(int(str(o).replace('−', '-')))
        except:
            pass
    return int(round(sum(vals) / len(vals))) if vals else None


def load_slate(today):
    path = f"data/mlb_slates/mlb_pitcher_slate_{today}.csv"
    df = pd.read_csv(path)
    df['player_norm'] = df['player_pp'].apply(normalize_name)
    return df


def calculate_edges(df, tol=0.5):
    df = df.copy()
    df['dk_line_diff'] = (df['dk_line'] - df['prizepicks_line']).abs()
    df['ud_line_diff'] = (df['line_ud'] - df['prizepicks_line']).abs()
    df['dk_ok'] = df['dk_line_diff'] <= tol
    df['ud_ok'] = df['ud_line_diff'] <= tol

    def pick_edge(r):
        o_ud = odds_to_prob(r['over_odds_ud']) if r.ud_ok else None
        u_ud = odds_to_prob(r['under_odds_ud']) if r.ud_ok else None
        dk_p = odds_to_prob(r['dk_odds']) if r.dk_ok else None

        edges = {}
        if o_ud is not None and u_ud is not None:
            edges['best_bet'] = 'OVER' if o_ud > u_ud else 'UNDER'
            edges['edge'] = max(o_ud, u_ud) - 0.5
        elif dk_p is not None:
            edges['best_bet'] = 'DK_ODDS'
            edges['edge'] = dk_p - 0.5
        else:
            edges['best_bet'] = None
            edges['edge'] = None
        return pd.Series(edges)

    df[['best_bet', 'edge']] = df.apply(pick_edge, axis=1)
    df['avg_line'] = df.apply(
        lambda r: average_odds(
            r['dk_odds'] if r.dk_ok else None,
            (r['over_odds_ud'] if r.best_bet == 'OVER' else
             r['under_odds_ud'] if r.best_bet == 'UNDER' else None)
        ), axis=1
    )
    return df


def get_top_stat(df, n=5):
    df2 = calculate_edges(df).dropna(subset=['edge'])
    top = df2.nlargest(n, 'edge').copy()
    top['Edge'] = (top['edge'] * 100).round(1).astype(str) + '%'
    top['Average Odds'] = top['avg_line'].apply(
        lambda x: f"{x:+d}" if x is not None else None
    )
    return top.rename(columns={
        'player_pp': 'Player', 'team': 'Team',
        'prizepicks_line': 'Line (PP)', 'dk_line': 'Line (DK)',
        'line_ud': 'Line (UD)', 'best_bet': 'Pick'
    })[
        ['Player', 'Team', 'Line (PP)', 'Line (DK)', 'Line (UD)',
         'Pick', 'Average Odds', 'Edge']
    ].assign(Source='Stat')


def get_top_model(slate, preds, n=5):
    df = slate.copy()
    df['player_norm'] = df['player_pp'].apply(normalize_name)
    df = df.merge(preds[['player_norm', 'predicted_ks']], on='player_norm', how='left')

    df['predicted_ks'] = df['predicted_ks'].fillna(df['prizepicks_line'])
    df['Predicted Ks'] = df['predicted_ks'].round(2)

    df['Pick'] = df.apply(
        lambda r: 'OVER' if r.predicted_ks > r.prizepicks_line else 'UNDER',
        axis=1
    )
    df['Edge'] = (df['predicted_ks'] - df['prizepicks_line']).round(2)

    def model_avg_odds(r):
        dk_ok = abs(r['dk_line'] - r['prizepicks_line']) <= 0.5
        dk = r['dk_odds'] if dk_ok else None
        ud = r['over_odds_ud'] if r['Pick'] == 'OVER' else r['under_odds_ud']
        avg = average_odds(dk, ud)
        return f"{avg:+d}" if avg is not None else None

    df['Average Odds'] = df.apply(model_avg_odds, axis=1)

    top = df.nlargest(n, 'Edge').copy()
    return top.rename(columns={'player_pp': 'Player', 'team': 'Team'})[
        ['Player', 'Team', 'prizepicks_line', 'Predicted Ks', 'Pick', 'Average Odds', 'Edge']
    ].rename(columns={'prizepicks_line': 'Line (PP)'}).assign(Source='Model')


def main():
    today = date.today().isoformat()
    slate = load_slate(today)

    # Get top stat-based picks
    stat5 = get_top_stat(slate, n=5)

    # Run model directly (no save/load)
    model_preds = predict_strikeouts(f"data/mlb_slates/mlb_pitcher_slate_{today}.csv")
    print(model_preds)
    model_preds['player_norm'] = model_preds['player_pp'].apply(normalize_name)
    model5 = get_top_model(slate, model_preds, n=5)

    # Divider row
    divider = pd.DataFrame([{
        'Player': '─── MODEL PICKS ───', 'Team': '', 'Line (PP)': '',
        'Pick': '', 'Average Odds': '', 'Edge': '', 'Source': ''
    }])

    # Output
    out = pd.concat([stat5, divider, model5], ignore_index=True)
    os.makedirs("best_lines", exist_ok=True)
    out.to_csv(f"best_lines/best_lines_{today}.csv", index=False)
    print("best_lines updated:", f"best_lines/best_lines_{today}.csv")


if __name__ == '__main__':
    main()
