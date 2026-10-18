import pandas as pd
import re
from pathlib import Path


def filter_addresses(df: pd.DataFrame, addr_col: str = "Адрес доставки_norm") -> pd.DataFrame:
    """
    Лёгкая фильтрация: только очевидный мусор (время и совсем короткие строки).
    Стоп‑слова убраны: «офис», «ресторан», «сова» и т.п. теперь не режутся.
    Возвращает df с колонкой address_reason.
    """
    df = df.copy()
    df["address_reason"] = "valid"

    # 1. Совсем короткие строки (< 3 символов) — точно не адрес
    # Теперь сюда попадают только реально мусорные обрывки, потому что
    # все сокращения уже заменены на длинные фразы.
    mask_short = df[addr_col].str.len() < 3
    df.loc[mask_short, "address_reason"] = "too_short"

    # 2. Паттерн времени (11,00 / 11:00) — это не адрес, это время
    time_pattern = r'^\s*\d{1,2}[\:\,][\d]{2}\s*$'
    df_lower = df[addr_col].str.lower()
    mask_time = df_lower.str.match(time_pattern, na=False)
    df.loc[mask_time & (df["address_reason"] == "valid"), "address_reason"] = "time_pattern"

    return df


def preprocess_leads(input_path, output_path):
    """
    Основной пайплайн предобработки адресов для amoCRM.
    Сохраняет оригинал, нормализованный адрес, причину и финальный статус валидности.
    """
    # 1. Читаем файл
    df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)

    # Сохраняем оригинал, чтобы потом видеть, что было до чистки
    df['Адрес доставки_original'] = df['Адрес доставки']

    def clean_address(addr):
        # Если не строка (например, NaN) -> сразу unknown
        if not isinstance(addr, str):
            return "unknown", "not_string"

        addr_lower = addr.lower().strip()

        # --- Явные случаи самовывоза и курьерских сервисов ---
        # Ловим ДО любых замен, чтобы не пытаться превратить «заберу домой» во что-то другое
        if 'домой' in addr_lower or 'заберу сам' in addr_lower or 'сам заберу' in addr_lower:
            return "unknown", "self_pickup"

        if 'яндекс доставка' in addr_lower or 'курьер яндекс' in addr_lower or 'яндекс.еда' in addr_lower:
            return "unknown", "yandex_courier"

        original_lower = addr_lower

        # Убираем кавычки
        addr = re.sub(r'["\']', '', addr)
        addr = addr.strip()

        # --- ВАЖНО: сначала делаем замены сокращений ---
        abbrev_map = {
            'гб': 'Гуссарская баллада микрорайон Одинцово',
            'бб': 'ресторан Б&Б',
            'фб': 'форт боярд улица Озерная 119',
            'пр-кт': 'проспект',
            'ул ': 'улица ',
            'пер ': 'переулок '
        }
        for short, full in abbrev_map.items():
            # replace чувствителен к регистру, поэтому работаем с addr (не lower)
            addr = addr.replace(short, full)
            # и отдельно для верхнего регистра, если вдруг встретится
            addr = addr.replace(short.upper(), full)
            addr = addr.replace(short.capitalize(), full)

        # Удаляем мусорные фразы про доставку и время
        noise_phrases = [
            'можно доставку до', 'доставка до', 'адрес доставки',
            'заберет сам', 'без доставки', 'клиент заберет',
            'прошу доставить', 'адрес:', 'время ', 'в ', 'на ', 'к '
        ]
        for phrase in noise_phrases:
            addr = addr.replace(phrase, '')

        # Удаляем детали (этажи, квартиры, офисы) — оставляем слово «офис» без номера
        addr = re.sub(r'\ьэтаж\s*\d+', '', addr, flags=re.IGNORECASE)
        addr = re.sub(r'\ькв\.?\s*\d+', '', addr, flags=re.IGNORECASE)
        addr = re.sub(r'\ьофис\s*\d+', 'офис', addr, flags=re.IGNORECASE)

        # Проверка на время и коды (16.30, к 12, №123)
        has_time = bool(re.search(r'\b\d{1,2}[.:]\d{2}\b', addr))
        has_k_number = bool(re.search(r'\bk\s*\d+\b', addr, flags=re.IGNORECASE))
        has_no_number = bool(re.search(r'\b(no|№)\s*\d+\b', addr, flags=re.IGNORECASE))

        if has_time or has_k_number or has_no_number:
            return "unknown", "time_or_code"

        # Чистим лишние пробелы
        addr = re.sub(r'\s+', ' ', addr).strip()

        # Одиночные цифры и пустые значения
        if not addr or addr.isdigit() or re.fullmatch(r'[\d\.\-]+', addr):
            return "unknown", "only_digits"

        # Умная валидация: проверяем по маркерам улицы/заведения
        street_markers = ['улица', 'проспект', 'бульвар', 'переулок', 'шоссе', 'площадь', 'наб.', 'набережная']
        place_keywords = ['ресторан', 'тц ', 'бц ', 'кафе ', 'магазин ', '"', "'"]

        has_street = any(marker in addr for marker in street_markers)
        # Проверяем по original_lower, чтобы не потерять кавычки и слово «ресторан»
        has_place = any(k in original_lower for k in place_keywords)

        if has_street or has_place:
            return addr, "valid"

        # Если ничего нет, но строка не пустая — отдаём геокодеру как есть
        if addr:
            return addr, "no_street_marker"

        return "unknown", "empty"

    # Применяем очистку
    cleaned = df['Адрес доставки'].apply(clean_address)
    df[['Адрес доставки_norm', 'address_reason']] = pd.DataFrame(cleaned.tolist(), index=df.index)

    # Валидным считаем всё, что НЕ равно "unknown" на этом этапе
    df['is_address_valid_pre'] = df['Адрес доставки_norm'] != "unknown"

    # --- ВТОРОЙ СЛОЙ ФИЛЬТРАЦИИ ---
    # Теперь прогоняем через filter_addresses, чтобы отсечь только время и совсем короткие
    df = filter_addresses(df, addr_col='Адрес доставки_norm')

    # Пересчитаем валидность уже по новому reason
    # Считаем валидным только если reason == 'valid' (после filter_addresses)
    df['is_address_valid_final'] = df['address_reason'] == 'valid'

    # Сохраняем результат
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    # Вывод статистики
    total_count = len(df)
    valid_count = df['is_address_valid_final'].sum()
    print(f"Готово! Обработано {total_count} строк. Валидных адресов для геокодирования: {valid_count} ({valid_count/total_count:.1%})")

    invalid_df = df[~df['is_address_valid_final']]
    if not invalid_df.empty:
        print("\nТоп-10 причин, почему адрес отсеян:")
        print(invalid_df['address_reason'].value_counts().head(10))
        print("\nТоп-10 проблемных оригиналов (для ручной проверки):")
        print(invalid_df['Адрес доставки_original'].value_counts().head(10))

    return df