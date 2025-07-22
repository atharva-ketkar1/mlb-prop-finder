import requests
import pandas as pd

def scrape_draftkings_mlb():
    url = "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusil/v1/leagues/84240/categories/1031/subcategories/18195"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "Referer": "https://sportsbook.draftkings.com/",
        "Origin": "https://sportsbook.draftkings.com",
        "Accept": "*/*",
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    market_map = {m['id']: m for m in data['markets']}

    dk_selections = []
    for sel in data['selections']:
        market = market_map.get(sel['marketId'], {})
        market_name = market.get('name', '')
        if 'Strikeouts' not in market_name:
            continue

        participants = sel.get('participants', [])
        participant_name = participants[0]['name'] if participants else None
        if participant_name is None:
            continue

        dk_selections.append({
            "player": participant_name,
            "dk_line": sel.get('points', None),
            "dk_odds": sel.get('displayOdds', {}).get('american', None),
            "dk_label": sel.get('label', ''),
            "market_name": market_name,
        })

    return pd.DataFrame(dk_selections)


if __name__ == "__main__":
    print(scrape_draftkings_mlb())