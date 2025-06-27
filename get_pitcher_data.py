from pybaseball import pitching_stats_range
import pandas as pd
from datetime import datetime, timedelta

end_date = datetime.today()
start_date = end_date - timedelta(days=30)

start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# Get pitcher stats for that date range
df = pitching_stats_range(start_str, end_str)

# Optional: filter for starting pitchers only
df = df[df['GS'] > 0]  # Games Started > 0

# Save locally
df.to_csv(f"data/pitcher_stats/logs_{end_str}.csv", index=False)
print(f"Saved game logs from {start_str} to {end_str}")
