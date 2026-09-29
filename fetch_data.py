import json
import urllib.request
import pandas as pd
import datetime

def fetch_september_data():
    api_url = "https://logam-mulia-api.iamutaki.workers.dev/api/prices/logammulia/history?weight=1&length=1000"
    print(f"Mengunduh data dari: {api_url} ...")
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        payload = json.loads(response.read().decode())

    data = payload.get("data", [])
    records = []
    for item in data:
        # Filter hanya Emas Batangan 1 gram Antam
        if item.get("materialType") == "Emas Batangan" and item.get("weight") == 1:
            rec_date = item.get("recordedDate")
            if rec_date and rec_date.startswith("2026-09"):
                price = item.get("sellPrice")
                if price is not None:
                    records.append({
                        "date": rec_date,
                        "price_idr": int(price),
                        "source": "Antam Official (API)"
                    })

    # Urutkan berdasarkan tanggal ascending
    df_sep = pd.DataFrame(records).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df_sep["day_name"] = pd.to_datetime(df_sep["date"]).dt.day_name()
    return df_sep

def main():
    # 1. Baca data training Jan-Agu yang sudah ada
    df_train = pd.read_csv("../harga_emas_antam_2026_jan_agu.csv")
    print(f"Data Training (Jan-Agu 2026): {len(df_train)} baris")

    # 2. Ambil data September 2026
    df_sep = fetch_september_data()
    print(f"Data Testing September 2026 (tercatat s.d. saat ini): {len(df_sep)} baris")
    print(f"Rentang tanggal September: {df_sep['date'].min()} s.d. {df_sep['date'].max()}")

    # Tambahkan day_index lanjutan untuk September
    last_idx = df_train["day_index"].max()
    df_sep["day_index"] = range(last_idx + 1, last_idx + 1 + len(df_sep))

    # Pastikan format kolom konsisten
    cols = ["day_index", "date", "day_name", "price_idr", "source"]
    df_train_clean = df_train[["day_index", "date", "day_name", "price_idr", "source"]]
    df_sep_clean = df_sep[cols]

    # Simpan file masing-masing
    df_train_clean.to_csv("train_jan_agu_2026.csv", index=False)
    df_sep_clean.to_csv("test_sep_2026.csv", index=False)

    # Gabungan penuh Jan - Sep
    df_full = pd.concat([df_train_clean, df_sep_clean], ignore_index=True)
    df_full.to_csv("full_jan_sep_2026.csv", index=False)

    # Hitung dan simpan parameter normalisasi (min-max dari data training)
    p_min = df_train_clean["price_idr"].min()
    p_max = df_train_clean["price_idr"].max()
    print("\n--- Parameter Normalisasi (Dihitung HANYA dari Data Training) ---")
    print(f"Min Harga Latih (Jan-Agu) : Rp {p_min:,}")
    print(f"Max Harga Latih (Jan-Agu) : Rp {p_max:,}")

    with open("normalization_params.json", "w") as f:
        json.dump({
            "min_price": int(p_min),
            "max_price": int(p_max),
            "train_rows": len(df_train_clean),
            "test_rows": len(df_sep_clean)
        }, f, indent=2)

    print("\n[OK] Berhasil menyimpan:")
    print("  - train_jan_agu_2026.csv")
    print("  - test_sep_2026.csv")
    print("  - full_jan_sep_2026.csv")
    print("  - normalization_params.json")

if __name__ == "__main__":
    main()
