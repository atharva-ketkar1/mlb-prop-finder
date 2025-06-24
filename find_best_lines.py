import pandas as pd
import os
from datetime import date

def odds_to_prob(odds):
    try:
        odds = str(odds).replace('−', '-').strip()
        odds = int(odds)
    except (ValueError, TypeError):
        return None
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def average_odds(odds1, odds2):
    try:
        odds1 = int(str(odds1).replace('−', '-')) if odds1 is not None else None
        odds2 = int(str(odds2).replace('−', '-')) if odds2 is not None else None
    except ValueError:
        return None

    if odds1 is not None and odds2 is not None:
        return int(round((odds1 + odds2) / 2))
    elif odds1 is not None:
        return odds1
    elif odds2 is not None:
        return odds2
    else:
        return None

def load_latest_slate(input_dir="data/mlb_slates", prefix="mlb_pitcher_slate"):
    today = date.today().isoformat()
    filename = f"{prefix}_{today}.csv"
    filepath = os.path.join(input_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Slate file not found: {filepath}")
    return pd.read_csv(filepath)

# Use ChatGPT to calculate edges between PrizePicks, DraftKings, and Underdog lines
def calculate_edges(df, line_tolerance=0.5):
    df['dk_line_diff'] = (df['dk_line'] - df['prizepicks_line']).abs()
    df['ud_line_diff'] = (df['line_ud'] - df['prizepicks_line']).abs()

    df['dk_line_match'] = df['dk_line_diff'] <= line_tolerance
    df['ud_line_match'] = df['ud_line_diff'] <= line_tolerance

    def select_odds(row):
        over_prob_ud = under_prob_ud = None

        over_prob_ud = odds_to_prob(row['over_odds_ud']) if row['ud_line_match'] else None
        under_prob_ud = odds_to_prob(row['under_odds_ud']) if row['ud_line_match'] else None

        dk_prob = odds_to_prob(row['dk_odds']) if row['dk_line_match'] else None

        edge_over = over_prob_ud - 0.5 if over_prob_ud is not None else None
        edge_under = under_prob_ud - 0.5 if under_prob_ud is not None else None

        if edge_over is not None and edge_under is not None:
            best_bet = 'OVER' if edge_over > edge_under else 'UNDER'
            edge = max(edge_over, edge_under)
        elif dk_prob is not None:
            edge = dk_prob - 0.5
            best_bet = 'DK_ODDS'
        else:
            best_bet = None
            edge = None

        return pd.Series({
            'edge_over': edge_over,
            'edge_under': edge_under,
            'best_bet': best_bet,
            'edge': edge
        })

    edges = df.apply(select_odds, axis=1)
    df = pd.concat([df, edges], axis=1)

    def get_avg_odds(row):
        odds_to_average = []
        if row['dk_line_match']:
            odds_to_average.append(row['dk_odds'])
        if row['ud_line_match']:
            if row['best_bet'] == 'OVER':
                odds_to_average.append(row['over_odds_ud'])
            elif row['best_bet'] == 'UNDER':
                odds_to_average.append(row['under_odds_ud'])

        return average_odds(*odds_to_average)

    df['avg_line'] = df.apply(get_avg_odds, axis=1)

    df['dk_line_diff'] = df['dk_line_diff'].round(2)
    df['ud_line_diff'] = df['ud_line_diff'].round(2)

    return df

# Use ChatGPT to generate the top props based on the calculated edges
def get_top_props(df, top_n=5):
    filtered = df.dropna(subset=['edge'])
    top_props = filtered.sort_values(by='edge', ascending=False).head(top_n).copy()

    top_props['Edge'] = (top_props['edge'] * 100).round(1).astype(str) + '%'

    def format_odds(o):
        if pd.isna(o):
            return None
        o = int(o)
        return f"+{o}" if o > 0 else str(o)
    top_props['Average Odds'] = top_props['avg_line'].apply(format_odds)

    return top_props[[
        'player_pp', 'team', 'prizepicks_line', 'dk_line', 'dk_line_diff', 'line_ud', 'ud_line_diff',
        'best_bet', 'Average Odds', 'Edge'
    ]].rename(columns={
        'player_pp': 'Player',
        'team': 'Team',
        'prizepicks_line': 'Line (PP)',
        'dk_line': 'Line (DK)',
        'dk_line_diff': 'DK Line Diff',
        'line_ud': 'Line (UD)',
        'ud_line_diff': 'UD Line Diff',
        'best_bet': 'Best Bet'
    })

def save_top_props(df, output_dir="best_lines"):
    today = date.today().isoformat()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"best_lines_{today}.csv")
    df.to_csv(output_path, index=False)
    print(f"Top props saved to {output_path}")

def main():
    df = load_latest_slate()
    df = calculate_edges(df)
    top_props = get_top_props(df)
    save_top_props(top_props)

if __name__ == "__main__":
    main()
