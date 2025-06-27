from pybaseball import pitching_stats_range
import pandas as pd
from datetime import datetime, timedelta
import os

end_date = datetime.today()
start_date = end_date - timedelta(days=30)
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

df = pitching_stats_range(start_str, end_str)
df = df[df['GS'] > 0]  # only starters

os.makedirs("data/pitcher_stats", exist_ok=True)

output_path = "data/pitcher_stats/logs_last_30_days.csv"
df.to_csv(output_path, index=False)
print(f"Saved logs ({start_str} → {end_str})")
