import requests
import pandas as pd

DESIRED_STATS = ["Pitcher Strikeouts"]

def scrape_prizepicks_mlb():
    url = "https://api.prizepicks.com/projections"
    params = {"league_id": "2", "per_page": "250"}
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    projections = data["data"]
    included = {item["id"]: item for item in data["included"]}

    all_records = []
    for proj in projections:
        attr = proj["attributes"]
        stat_type = attr["stat_type"]
        line_score = attr["line_score"]
        odds_type = attr.get("odds_type", "none")

        if odds_type != "standard" or stat_type not in DESIRED_STATS:
            continue

        player_id = str(proj["relationships"]["new_player"]["data"]["id"])
        player_info = included.get(player_id, {}).get("attributes", {})
        name = player_info.get("name", "Unknown")
        team = player_info.get("team", "Unknown")

        all_records.append({
            "player": name,
            "team": team,
            "stat_type": stat_type,
            "prizepicks_line": line_score,
        })

    return pd.DataFrame(all_records)
