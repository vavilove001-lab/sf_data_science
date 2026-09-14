import os
import pandas as pd

def load_and_merge_data(taxi_path, osrm_path, holiday_path, weather_path):
    """
    Загружает все датасеты и делает базовый мёрдж.
    Возвращает: taxi_df, holiday_df, weather_df
    """
    # 1. Загружаем такси
    taxi_df = pd.read_csv(taxi_path)
    
    # 2. Загружаем OSRM (если файл существует)
    if os.path.exists(osrm_path):
        osrm_df = pd.read_csv(osrm_path)
        # Тут можно сделать merge, если нужно сразу, но пока вернём отдельно, 
        # чтобы в ноутбуке было прозрачно
        # taxi_df = taxi_df.merge(osrm_df, on='id', how='left')
    else:
        osrm_df = None
        print(f"⚠️ Файл OSRM не найден: {osrm_path}")

    # 3. Праздники
    holiday_df = pd.read_csv(holiday_path)

    # 4. Погода
    weather_df = pd.read_csv(weather_path)

    return taxi_df, holiday_df, weather_df
