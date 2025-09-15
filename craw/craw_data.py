import csv
import requests
import time
import os

API_TOKEN = "NljJwzoKlLtHzqfXLaimhvOyDUXkqpRl"
BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
OUTPUT_FOLDER = "output_data"

headers = {
    "token": API_TOKEN
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

start_year = 2025
end_year = 2025

with open("vietnam_stations.csv", "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        station_id = row["USAF"] + row["WBAN"]
        station_name = row["Station_Name"].replace(" ", "_").replace("/", "-")
        station_folder = os.path.join(OUTPUT_FOLDER, f"{station_name}_{station_id}")

        os.makedirs(station_folder, exist_ok=True)
        print(f"Bắt đầu lấy dữ liệu cho trạm {station_name}_{station_id}")

        all_data = [] 

        for year in range(start_year, end_year + 1):
            start_date = f"{year}-01-01T00:00:00"
            end_date = f"{year}-12-31T23:59:59"
            print(f"Lấy dữ liệu năm {year}")

            params = {
                "dataset": "global-hourly",
                "dataTypes": "WND,CIG,VIS,TMP,DEW,SLP,AA1,AY1,GF1,KA1,MD1,MW1,EQD",
                "stations": station_id,
                "startDate": start_date,
                "endDate": end_date,
                "format": "json",
                "units": "metric",
                "includeAttributes": "false"
            }

            try:
                response = requests.get(BASE_URL, params=params, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    if data:

                        all_data.extend(data)

                        all_keys = set()
                        for entry in data:
                            all_keys.update(entry.keys())
                        fieldnames = list(all_keys)

                        output_path = os.path.join(station_folder, f"{year}.csv")
                        with open(output_path, "w", newline="", encoding="utf-8") as csvfile_out:
                            writer = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
                            writer.writeheader()
                            for entry in data:
                                writer.writerow(entry)

                        print(f"Đã lưu dữ liệu năm {year} cho trạm {station_id} vào {output_path}")
                    else:
                        print(f"Không có dữ liệu cho năm {year} trạm {station_id}")
                else:
                    print(f"Lỗi {response.status_code} khi lấy dữ liệu năm {year} trạm {station_id}")
            except Exception as e:
                print(f"Lỗi khi lấy dữ liệu năm {year} trạm {station_id}: {str(e)}")

            time.sleep(1.2)


        if all_data:

            all_keys = set()
            for entry in all_data:
                all_keys.update(entry.keys())
            fieldnames = list(all_keys)

            output_all_path = os.path.join(station_folder, "all_years.csv")
            with open(output_all_path, "w", newline="", encoding="utf-8") as csvfile_all:
                writer = csv.DictWriter(csvfile_all, fieldnames=fieldnames)
                writer.writeheader()
                for entry in all_data:
                    writer.writerow(entry)
            print(f"Đã lưu file tổng hợp tất cả dữ liệu: {output_all_path}")
        else:
            print("Không có dữ liệu tổng hợp để lưu.")

        print(f"Hoàn thành trạm {station_name}_{station_id}\n")
