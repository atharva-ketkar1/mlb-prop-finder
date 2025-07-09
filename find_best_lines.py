import pandas as pd
import os
from datetime import date
from predict_strikeouts import predict_strikeouts, normalize_name  

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
        edges = {}

        try:
            dk_odds = int(str(r['dk_odds']).replace('−', '-'))
        except:
            dk_odds = None

        try:
            over_ud_odds = int(str(r['over_odds_ud']).replace('−', '-'))
        except:
            over_ud_odds = None

        try:
            under_ud_odds = int(str(r['under_odds_ud']).replace('−', '-'))
        except:
            under_ud_odds = None

        best_bet = None
        best_odds = None
        edge = None

        # Evaluate DK signal
        if dk_odds is not None:
            direction = r['dk_label'].upper() if pd.notna(r['dk_label']) else None
            if direction in ("OVER", "UNDER"):
                best_bet = direction
                best_odds = dk_odds

        # Evaluate UD signals
        if over_ud_odds is not None:
            if best_odds is None or abs(over_ud_odds) > abs(best_odds):
                best_bet = "OVER"
                best_odds = over_ud_odds

        if under_ud_odds is not None:
            if best_odds is None or abs(under_ud_odds) > abs(best_odds):
                best_bet = "UNDER"
                best_odds = under_ud_odds

        # Final edge calculation
        if best_odds is not None:
            prob = odds_to_prob(best_odds)
            if prob is not None:
                edge = prob - 0.5

        edges['best_bet'] = best_bet
        edges['edge'] = edge
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
        lambda x: f"{x:+.0f}" if x is not None else None
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

    stat5 = get_top_stat(slate, n=5)

    model_preds = predict_strikeouts(f"data/mlb_slates/mlb_pitcher_slate_{today}.csv")
    #print(model_preds)
    model_preds['player_norm'] = model_preds['player_pp'].apply(normalize_name)
    model5 = get_top_model(slate, model_preds, n=5)

    divider = pd.DataFrame([{
        'Player': '─── MODEL PICKS ───', 'Team': '', 'Line (PP)': '',
        'Pick': '', 'Average Odds': '', 'Edge': '', 'Source': ''
    }])

    out = pd.concat([stat5, divider, model5], ignore_index=True)
    os.makedirs("best_lines", exist_ok=True)
    out.to_csv(f"best_lines/best_lines_{today}.csv", index=False)
    print("best_lines updated:", f"best_lines/best_lines_{today}.csv")


if __name__ == '__main__':
    main()
