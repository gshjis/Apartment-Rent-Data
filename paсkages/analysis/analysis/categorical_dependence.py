import pandas as pd
import matplotlib.pyplot as plt


def plot_categorical_dependencies(
    table: pd.DataFrame,
    categorical_columns: list[str],
    target: str,
    *,
    agg: str = "mean",
    figsize: tuple[int, int] | None = (14, 4),
    sort_by: str | None = None,
):
    """Отрисовывает зависимости категориальных признаков от целевой переменной.

    Пример агрегации: `train.groupby("state")["price"].mean()`.

    Для каждого признака из `categorical_columns` строится отдельный график.

    Args:
        table: Таблица с данными.
        categorical_columns: Список категориальных колонок.
        target: Имя целевой числовой колонки (например, "price").
        agg: Как агрегировать значения target по категориям (например, "mean").
        figsize: Размер одного subplot (если None — matplotlib сам подберёт).
        sort_by: Если не None, сортировать категории по агрегату этого типа.
            По умолчанию сортировка идёт по значению агрегата.
    """

    if target not in table.columns:
        raise ValueError(f"Целевая колонка '{target}' отсутствует в таблице")

    missing = [c for c in categorical_columns if c not in table.columns]
    if missing:
        raise ValueError(f"Отсутствуют категориальные колонки: {missing}")

    n = len(categorical_columns)
    if n == 0:
        raise ValueError("categorical_columns не должен быть пустым")

    fig, axes = plt.subplots(
        1, n, figsize=(figsize[0] * n, figsize[1]) if figsize else None
    )
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, categorical_columns):
        grouped = table.groupby(col)[target]
        if not hasattr(grouped, agg):
            raise ValueError(
                f"Агрегация agg='{agg}' недоступна для pandas GroupBy[{target}]"
            )
        stats = getattr(grouped, agg)()

        # сортировка
        stats = (
            stats.sort_values(ascending=False) if sort_by is not None or True else stats
        )

        ax.bar(stats.index.astype(str), stats.values)
        ax.set_title(f"{col} → {agg}({target})")
        ax.set_xlabel(col)
        ax.set_ylabel(f"{agg}({target})")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
