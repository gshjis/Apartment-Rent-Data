import pandas as pd


def find_rare_categories(df, categorical_cols, threshold_percent=0.05):
    """
    Находит 5% самых редких категорий и заменяет их на 'other'.

    Args:
        df (pd.DataFrame): Входной DataFrame
        categorical_cols (list[str]): Категориальные колонки
        threshold_percent (float): Порог редких категорий (по умолчанию 0.05 = 5%)

    Returns:
        pd.DataFrame: DataFrame с замененными редкими категориями
    """
    df_clean = df.copy()

    for col in categorical_cols:
        # Считаем частоты
        value_counts = df_clean[col].value_counts()

        # Сортируем по возрастанию (редкие сначала)
        value_counts = value_counts.sort_values()

        # Считаем кумулятивную сумму процентов
        cumsum_percent = value_counts.cumsum() / value_counts.sum()

        # Находим категории, которые входят в нижние 5%
        rare_categories = value_counts[cumsum_percent <= threshold_percent].index

        if len(rare_categories) > 0:
            print(f"{col}: {len(rare_categories)} редких категорий заменено на 'other'")
            df_clean.loc[:, col] = df_clean[col].replace(rare_categories, "other")

    return df_clean
