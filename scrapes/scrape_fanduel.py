import requests

def get_market(market_id):
    url = f"https://sbapi.nj.sportsbook.fanduel.com/api/markets/{market_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

market_ids = ["734.129033727", "734.128971594", "734.129033731"]

for mid in market_ids:
    market_data = get_market(mid)
    print(f"Market ID: {mid}")
    print(market_data)
