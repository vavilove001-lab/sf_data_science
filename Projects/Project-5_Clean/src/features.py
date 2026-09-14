import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# --- ГЕОМЕТРИЯ И РАССТОЯНИЯ ---

def get_haversine_distance(lat1, lng1, lat2, lng2):
    """Считает расстояние в км по формуле гаверсинуса."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    EARTH_RADIUS = 6371
    lat_delta = lat2 - lat1
    lng_delta = lng2 - lng1
    d = np.sin(lat_delta * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng_delta * 0.5) ** 2
    h = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def get_angle_direction(lat1, lng1, lat2, lng2):
    """Считает угол направления движения в градусах."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    lng_delta_rad = lng2 - lng1
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    alpha = np.degrees(np.arctan2(y, x))
    return alpha

def add_geographical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет расстояние и угол."""
    df = df.copy()
    cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    if all(col in df.columns for col in cols):
        df['haversine_distance'] = get_haversine_distance(
            df['pickup_latitude'], df['pickup_longitude'],
            df['dropoff_latitude'], df['dropoff_longitude']
        )
        df['direction'] = get_angle_direction(
            df['pickup_latitude'], df['pickup_longitude'],
            df['dropoff_latitude'], df['dropoff_longitude']
        )
    return df

# --- ВРЕМЕННЫЕ ПРИЗНАКИ ---

def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет базовые временные признаки."""
    df = df.copy()
    if 'pickup_datetime' in df.columns:
        df['pickup_date'] = df['pickup_datetime'].dt.date
        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        df['pickup_day_of_week'] = df['pickup_datetime'].dt.dayofweek
    return df

def add_temporal_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет синусы/косинусы времени и флаги (час пик, выходные)."""
    df = df.copy()
    
    if 'pickup_hour' in df.columns:
        df['is_rush_hour'] = (
            ((df['pickup_hour'] >= 7) & (df['pickup_hour'] <= 9)) |
            ((df['pickup_hour'] >= 16) & (df['pickup_hour'] <= 19))
        ).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)
    
    if 'pickup_day_of_week' in df.columns:
        df['is_weekend'] = (df['pickup_day_of_week'] >= 5).astype(int)
        df['dow_sin'] = np.sin(2 * np.pi * df['pickup_day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['pickup_day_of_week'] / 7)
        
    return df

# --- КЛАСТЕРИЗАЦИЯ ---

def add_cluster_features(df: pd.DataFrame, kmeans: KMeans) -> pd.DataFrame:
    """Добавляет кластеры на основе координат."""
    df = df.copy()
    coords = np.hstack((
        df[['pickup_latitude', 'pickup_longitude']],
        df[['dropoff_latitude', 'dropoff_longitude']]
    ))
    df['geo_cluster'] = kmeans.predict(coords)
    return df

# --- ПРАЗДНИКИ ---

def add_holiday_features(df: pd.DataFrame, holiday_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет признак is_holiday.
    holiday_df должен содержать колонку 'date' (тип datetime) или 'date' как строку.
    """
    df = df.copy()
    
    # Нормализуем даты в справочниках
    if 'date' in holiday_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(holiday_df['date']):
            holiday_df['date'] = pd.to_datetime(holiday_df['date'], format='%Y-%m-%d')
    
    # Создаем множество дат-праздников для быстрого поиска
    holiday_dates = set(holiday_df['date'].dt.date)
    
    if 'pickup_date' in df.columns:
        df['is_holiday'] = df['pickup_date'].isin(holiday_dates).astype(int)
    elif 'pickup_datetime' in df.columns:
        # Если pickup_date еще не создан, берем дату из datetime
        df['is_holiday'] = df['pickup_datetime'].dt.date.isin(holiday_dates).astype(int)
    else:
        # Если нет ни того, ни другого, создаем временно
        if 'pickup_datetime' in df.columns:
            df['is_holiday'] = df['pickup_datetime'].dt.date.isin(holiday_dates).astype(int)
            
    return df

# --- ПОГОДА ---

def fill_null_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет пропуски в погодных данных медианой по колонке."""
    df = df.copy()
    weather_cols = ['temp', 'feels_like', 'humidity', 'wind_speed', 'pressure']
    # Берем только те колонки, которые реально есть в датасете
    existing_cols = [c for c in weather_cols if c in df.columns]
    
    for col in existing_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        
    return df

def add_weather_features(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Присоединяет погодные данные к поездкам.
    Ожидает, что в weather_df есть колонки: 'date' (datetime или str) и 'hour' (int).
    """
    df = df.copy()
    weather_df = weather_df.copy()  # Чтобы не менять оригинал случайно

    # --- 1. Подготовка ключей в ОСНОВНОМ датасете (такси) ---
    if 'pickup_datetime' in df.columns:
        df['merge_date'] = df['pickup_datetime'].dt.date
        df['merge_hour'] = df['pickup_datetime'].dt.hour
    elif 'pickup_date' in df.columns and 'pickup_hour' in df.columns:
        # Если даты уже разбиты на отдельные колонки
        df['merge_date'] = df['pickup_date']
        df['merge_hour'] = df['pickup_hour'].astype(int)
    else:
        print("⚠️ В датасете поездок нет ни pickup_datetime, ни пары pickup_date/pickup_hour. Погода не будет добавлена.")
        return df

    # --- 2. Подготовка ключей в ПОГОДНОМ датасете ---
    # Проверяем, есть ли уже merge_date/merge_hour (на случай, если файл уже обработан)
    if 'merge_date' not in weather_df.columns or 'merge_hour' not in weather_df.columns:
        
        # Приводим дату погоды к формату datetime, если это строка
        if 'date' in weather_df.columns and not pd.api.types.is_datetime64_any_dtype(weather_df['date']):
            try:
                weather_df['date'] = pd.to_datetime(weather_df['date'], format='%Y-%m-%d')
            except Exception:
                weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        # Создаем ключи для мёрджа
        weather_df['merge_date'] = weather_df['date'].dt.date
        
        # ВАЖНО: колонка с часами может называться 'hour' или быть вычисляемой
        if 'hour' in weather_df.columns:
            weather_df['merge_hour'] = weather_df['hour'].astype(int)
        else:
            # Если колонки hour нет, пробуем взять из time или оставить None
            print("⚠️ Колонка 'hour' не найдена в weather_data. Используем 0 как заглушку (это ошибка данных).")
            weather_df['merge_hour'] = 0

    # --- 3. Мёрдж ---
    cols_to_merge = ['temperature', 'humidity', 'pressure', 'wind speed', 'precip', 'merge_date', 'merge_hour']
    # Оставляем только те колонки, которые реально есть в файле погоды
    available_cols = [c for c in cols_to_merge if c in weather_df.columns]
    
    # Убираем дубликаты в погоде по паре (дата, час)
    weather_agg = weather_df[available_cols].drop_duplicates(subset=['merge_date', 'merge_hour'])
    
    df = df.merge(weather_agg, on=['merge_date', 'merge_hour'], how='left')
    
    # Удаляем временные ключи, если они не нужны дальше
    if 'merge_date' in df.columns:
        df.drop(columns=['merge_date', 'merge_hour'], inplace=True)
        
    return df

