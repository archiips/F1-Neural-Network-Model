import fastf1
import pandas as pd
import os
from datetime import datetime
import pytz

# Create cache directory 
cache_dir = 'f1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Create output directory
os.makedirs('data', exist_ok=True)

# Use timezone-aware today date 
today = datetime.now(pytz.UTC)

all_years_data = []

for year in range(2020, 2026):  # 2020 through 2025 inclusive
    print(f"Processing year: {year}")
    
    try:
        schedule = fastf1.events.get_event_schedule(year)
    except Exception as e:
        print(f"Failed to get schedule for {year}: {e}")
        continue

    # Filter completed races (Session5Date < today)
    completed_races = schedule[schedule['Session5Date'] < today]

    for _, row in completed_races.iterrows():
        grand_prix = row['EventName']
        try:
            qualifying = fastf1.get_session(year, grand_prix, 'Q')
            race = fastf1.get_session(year, grand_prix, 'R')

            qualifying.load()
            race.load()

            # Process qualifying data
            quali_laps = qualifying.laps
            # Manually compute fastest lap per driver
            quali_fastest_laps = quali_laps.groupby('Driver')['LapTime'].min().reset_index()
            quali_df = quali_fastest_laps.copy()
            quali_df['QualifyingPosition'] = quali_df['LapTime'].rank(method='min').astype(int)

            # Process race results
            race_results = race.results[['DriverNumber', 'Abbreviation', 'Position']].copy()
            race_results.rename(columns={'Abbreviation': 'Driver'}, inplace=True)

            merged_df = pd.merge(quali_df, race_results, on='Driver')

            merged_df['Winner'] = (merged_df['Position'] == 1).astype(int)
            merged_df['Race'] = f"{year} {grand_prix}"

            final_df = merged_df[['Driver', 'QualifyingPosition', 'Winner', 'Race']]
            all_years_data.append(final_df)

            print(f"Processed: {year} {grand_prix}")

        except Exception as e:
            print(f"Skipped {year} {grand_prix} due to error: {e}")

if all_years_data:
    full_dataset = pd.concat(all_years_data, ignore_index=True)
    full_dataset.to_csv('data/f1_dataset_2020_2025.csv', index=False)
    print("Full dataset saved as 'data/f1_dataset_2020_2025.csv'")

else:
    print("No race data processed.")
