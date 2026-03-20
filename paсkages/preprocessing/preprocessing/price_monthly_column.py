import re
import pandas as pd


def convert_to_monthly(price_str):
    if pd.isna(price_str):
        return None

    price_str = str(price_str)

    # Извлекаем число
    numbers = re.findall(r"[\d,]+", price_str)
    if not numbers:
        return None

    # Убираем запятые и конвертируем в число
    amount = float(numbers[0].replace(",", ""))

    # Проверяем период
    if "weekly" in price_str.lower() or "week" in price_str.lower():
        return amount * 4.33  # среднее количество недель в месяце
    elif "monthly" in price_str.lower() or "month" in price_str.lower():
        return amount
    elif "yearly" in price_str.lower() or "year" in price_str.lower():
        return amount / 12
    else:
        return amount  # предположим что это месячная
