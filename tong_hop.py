import os
import pandas as pd

OUTPUT_FOLDER = "output_data"
TARGET_YEAR = "2025"
MERGED_FILE = f"weather_merged_{TARGET_YEAR}.csv"

all_dfs = []

for station_folder in os.listdir(OUTPUT_FOLDER):
    station_path = os.path.join(OUTPUT_FOLDER, station_folder)
    if os.path.isdir(station_path):
        year_file = os.path.join(station_path, f"{TARGET_YEAR}.csv")
        if os.path.exists(year_file):
            try:
                df = pd.read_csv(year_file)
                df["STATION_FOLDER"] = station_folder  
                all_dfs.append(df)
            except Exception as e:
                print(f"Lỗi khi đọc {year_file}: {e}")

if all_dfs:
    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df.to_csv(MERGED_FILE, index=False, encoding="utf-8")
    print(f"✅ Đã lưu file tổng hợp tại: {MERGED_FILE} với {len(merged_df)} dòng.")
else:
    print("⚠️ Không có file nào được hợp nhất.")
