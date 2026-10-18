import pandas as pd
import requests
import time
import re
from pathlib import Path

# --- НАСТРОЙКИ ---
INPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "addresses_for_geo.csv"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "leads_with_coords.csv"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "KDVavilov_GeoCoder/1.0"
BASE_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": USER_AGENT}
REQUEST_DELAY = 1.1
CHECKPOINT_INTERVAL = 100

def clean_address_for_nominatim(address: str) -> str:
    if not isinstance(address, str):
        return ""
    addr = address.strip()
    
    # Убираем коды домофонов
    addr = re.sub(r'[#\[\(]?(код\s*)?\d+[#\]\)]?', '', addr, flags=re.IGNORECASE)
    # Убираем упоминания этажей
    addr = re.sub(r'\s*,?\s*\d+\s*эт(аж)?', '', addr, flags=re.IGNORECASE)
    # Убираем номера квартир
    addr = re.sub(r'\s*,?\s*кв\.?\s*\d+', '', addr, flags=re.IGNORECASE)
    # Убираем явные хвосты
    patterns = [r'\s*,\s*самовывоз\s*$', r'\s*,\s*пункт\s+выдачи\s*$', r'\s*,\s*пвз\s*$']
    for p in patterns:
        addr = re.sub(p, '', addr, flags=re.IGNORECASE)
    
    return addr.rstrip(',').strip()

def is_suspicious_address(address: str) -> bool:
    if not isinstance(address, str):
        return True
    t = address.lower()
    suspicious = ["с тортом", "без торта", "хочу торт", "доставка с", "самовывоз", "пвз", "склад"]
    if any(w in t for w in suspicious):
        return True
    if len(t.strip()) < 12:  # слишком короткий адрес
        return True
    return False

def geocode_nominatim(address: str):
    clean_addr = clean_address_for_nominatim(address)
    if not clean_addr:
        return None, None, None, "error: empty_address"

    # Отладка: первые 5 адресов покажем в консоли
    if not hasattr(geocode_nominatim, "debug_count"):
        geocode_nominatim.debug_count = 0
    if geocode_nominatim.debug_count < 5:
        print(f"🔍 DEBUG: '{address[:60]}...' → '{clean_addr[:60]}...'")
        geocode_nominatim.debug_count += 1

    params = {
        "q": clean_addr,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
        "countrycodes": "RU"  # Ищем только по России
    }

    try:
        resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=10)
        if resp.status_code == 429:
            print("⚠️ Лимит: ждём 60 сек...")
            time.sleep(60)
            resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None, None, None, "not_found"
        first = data[0]
        return float(first.get("lat")), float(first.get("lon")), first.get("display_name", ""), "found"
    except Exception as e:
        return None, None, None, f"error: {str(e)}"

def save_checkpoint(df, processed_count):
    filename = f"leads_with_coords_checkpoint_{processed_count:05d}.csv"
    (CHECKPOINT_DIR / filename).mkdir(exist_ok=True, parents=True)
    df.to_csv(CHECKPOINT_DIR / filename, index=False, encoding="utf-8-sig")
    print(f"💾 Чекпоинт: {filename}")

def main():
    if not INPUT_PATH.exists():
        print(f"❌ Файл не найден: {INPUT_PATH}")
        return

    print(f"📂 Читаем: {INPUT_PATH.name}...")
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    cols = ["lat", "lon", "display_name", "geo_status", "geo_error"]
    for c in cols:
        if c not in df.columns:
            if c in ["lat", "lon"]:
                df[c] = None
            elif c == "display_name":
                df[c] = ""
            elif c == "geo_status":
                df[c] = "pending"
            elif c == "geo_error":
                df[c] = ""

    if "address_for_api" not in df.columns:
        print("❌ Нет колонки 'address_for_api'")
        return

    # Маска: геокодируем только строки, где ещё нет координат
    already_done = (df["lat"].notna()) & (df["lat"] != 0) & (df["geo_status"] == "found")
    has_addr = df["address_for_api"].notna() & (df["address_for_api"].astype(str).str.strip() != "")
    not_suspicious = ~df["address_for_api"].apply(is_suspicious_address)

    mask_to_geo = has_addr & not_suspicious & ~already_done
    to_geo_count = mask_to_geo.sum()
    skipped_count = len(df) - to_geo_count

    print(f"📍 На геокодирование: {to_geo_count} адресов.")
    print(f"⏭️ Пропущено (мусор/уже есть): {skipped_count} строк.\n")

    df.loc[~mask_to_geo, "geo_status"] = "skipped"
    df.loc[(~not_suspicious) & has_addr, "geo_error"] = "suspicious_address"

    processed_count = 0
    found_count = 0

    for i, (idx, row) in enumerate(df[mask_to_geo].iterrows()):
        address = str(row["address_for_api"])
        lat, lon, name, status = geocode_nominatim(address)

        df.at[idx, "lat"] = lat
        df.at[idx, "lon"] = lon
        df.at[idx, "display_name"] = name
        df.at[idx, "geo_status"] = status
        if status.startswith("error"):
            df.at[idx, "geo_error"] = status.split(": ", 1)[-1]
        if status == "found":
            found_count += 1

        processed_count += 1
        if processed_count % 50 == 0 or processed_count == to_geo_count:
            pct = (processed_count / to_geo_count * 100) if to_geo_count else 0
            print(f"[{processed_count}/{to_geo_count}] ({pct:.1f}%) | Найдено: {found_count}")

        if processed_count % CHECKPOINT_INTERVAL == 0 or processed_count == to_geo_count:
            save_checkpoint(df, processed_count)
            rate = found_count / processed_count if processed_count else 0
            if rate < 0.3:
                print(f"\n⚠️ Процент находок низкий: {rate:.2f}. Проверь блок DEBUG выше!\n")

        time.sleep(REQUEST_DELAY)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ Готово: {OUTPUT_PATH}")
    print("📊 Статистика:", df["geo_status"].value_counts())

if __name__ == "__main__":
    main()