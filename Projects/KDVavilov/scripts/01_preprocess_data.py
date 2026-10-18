from pathlib import Path
import sys
import pandas as pd

# Подключаем твой проверенный модуль
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from src.preprocess import preprocess_leads

# --- ПУТИ ---
input_path = project_root / 'data' / 'amocrm_export_leads_2026-09-20.csv'
output_preprocessed_path = project_root / 'outputs' / 'leads_preprocessed.csv'
output_geo_path = project_root / 'outputs' / 'addresses_for_geo.csv'

# Города МО, которые будем распознавать явно
MO_CITIES = [
    'одинцово', 'балашиха', 'химки', 'королёв', 'мытищи', 
    'люберцы', 'красногорск', 'подольск', 'долгопрудный', 'реутов'
]

def is_self_pickup(original_address: str) -> bool:
    """Проверяем оригинал на слова про самовывоз"""
    if not isinstance(original_address, str):
        return False
    
    orig_lower = original_address.lower()
    keywords = [
        'самовывоз', 'заберу сам', 'сам заберу', 'домой', 
        'без доставки', 'яндекс доставка', 'курьер яндекс', 'яндекс.еда'
    ]
    return any(k in orig_lower for k in keywords)

def make_geo_address(norm_addr: str, orig_addr: str) -> str:
    """
    Формирует адрес для геокодера.
    Логика:
      - Если нормализованный адрес 'unknown' -> возвращаем None (не геокодируем).
      - Если есть Москва -> 'Москва, адрес'.
      - Если есть город МО -> 'Московская область, г. Город, адрес'.
      - Иначе -> 'Московская область, адрес' (БЕЗ указания конкретного города).
    """
    # Если твой preprocess уже пометил как unknown — не шлем в геокодер
    if not isinstance(norm_addr, str) or norm_addr == "unknown" or not norm_addr.strip():
        return None
    
    addr_lower = norm_addr.lower()
    orig_lower = orig_addr.lower()

    # 1. Москва
    if any(city in addr_lower for city in ['москва', 'мск']):
        return f"Москва, {norm_addr}"
    
    # 2. Города МО
    for city in MO_CITIES:
        if city in addr_lower or city in orig_lower:
            return f"Московская область, г. {city.capitalize()}, {norm_addr}"
    
    # 3. Дефолт: просто Московская область (без города)
    # Это решает проблему улицы Неделина: геокодер сам найдет нужный город
    return f"Московская область, {norm_addr}"

def main():
    print("🚀 Шаг 1: Запускаем твою очистку (src/preprocess.py)...")
    df = preprocess_leads(str(input_path), str(output_preprocessed_path))
    
    print("🗺️ Шаг 2: Готовим адреса для геокодера...")
    
    # Ставим флаг самовывоза по оригиналу
    df['is_self_pickup'] = df['Адрес доставки'].apply(is_self_pickup)
    
    # Формируем финальную строку для API
    df['address_for_api'] = df.apply(
        lambda row: make_geo_address(row['Адрес доставки_norm'], row['Адрес доставки']), 
        axis=1
    )
    
    # Статистика
    total = len(df)
    pickup_count = df['is_self_pickup'].sum()
    geo_ready_count = df['address_for_api'].notna().sum()
    
    print(f"📊 Статистика:")
    print(f"   Всего заказов: {total}")
    print(f"   → Самовывоз (стоимость 0): {pickup_count}")
    print(f"   → Пойдет в геокодер: {geo_ready_count}")
    
    # Сохраняем только то, что нужно для следующего шага
    geo_df = df[[
        'ID', 
        'Адрес доставки',                 # Оригинал (для проверки)
        'Адрес доставки_norm',            # Твой чистый адрес
        'address_for_api',                # Адрес для геокодера (с регионом)
        'is_self_pickup'                  # Флаг: везти или нет
    ]].copy()
    
    geo_df.to_csv(output_geo_path, index=False, encoding='utf-8-sig')
    print(f"✅ Файл для геокодера готов: {output_geo_path}")
    
    # Покажем примеры, чтобы ты убедился
    print("\n🧪 Примеры данных:")
    sample = geo_df.head(5)
    for _, row in sample.iterrows():
        status = "Самовывоз" if row['is_self_pickup'] else "Доставка"
        api_addr = row['address_for_api'] if row['address_for_api'] else "None (не пойдет в геокодер)"
        print(f"   Статус: {status:<10} | Для API: {api_addr}")

if __name__ == "__main__":
    main()