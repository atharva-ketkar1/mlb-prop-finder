import requests
import pandas as pd

def scrape_underdog_mlb():
    url = "https://api.underdogfantasy.com/beta/v6/over_under_lines?sport_id=mlb"
    response = requests.get(url)
    data = response.json()

    props = []

    for line in data.get("over_under_lines", []):
        over_under = line.get("over_under", {})
        appearance_stat = over_under.get("appearance_stat", {})
        stat = appearance_stat.get("stat")

        if stat != "strikeouts":
            continue

        options = line.get("options", [])
        over_option = next((opt for opt in options if opt.get("choice") == "higher"), None)
        under_option = next((opt for opt in options if opt.get("choice") == "lower"), None)

        props.append({
            "player": over_option.get("selection_header") if over_option else "Unknown",
            "stat_type": stat,
            "line": line.get("stat_value"),
            "over_odds": over_option.get("american_price") if over_option else None,
            "under_odds": under_option.get("american_price") if under_option else None,
            "payout_multiplier_over": over_option.get("payout_multiplier") if over_option else None,
            "payout_multiplier_under": under_option.get("payout_multiplier") if under_option else None,
        })

    df = pd.DataFrame(props)

    return df

if __name__ == "__main__":
    print(scrape_underdog_mlb())
