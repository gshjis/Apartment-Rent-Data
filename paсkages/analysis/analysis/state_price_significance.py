import numpy as np
import pandas as pd

from scipy.stats import f_oneway, kruskal

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def price_significance_by_state(
    table: pd.DataFrame,
    *,
    state_col: str = "state",
    price_col: str = "price",
    alpha: float = 0.05,
    use_nonparametric: bool = True,
) -> dict:
    """Проверяет значимость различий цены между значениями state.

    Проводится тест:
    - если `use_nonparametric=True` (по умолчанию) — Kruskal–Wallis (непараметрический)
    - иначе — one-way ANOVA (f_oneway)

    Args:
        table: Датасет.
        state_col: Колонка с категориями (например, штат).
        price_col: Колонка с ценой.
        alpha: Уровень значимости.
        use_nonparametric: Использовать ли Kruskal–Wallis.

    Returns:
        Словарь с p-value, статистикой теста и флагом значимости.
    """

    if state_col not in table.columns:
        raise ValueError(f"Колонка state '{state_col}' отсутствует в таблице.")
    if price_col not in table.columns:
        raise ValueError(f"Колонка price '{price_col}' отсутствует в таблице.")

    df = table[[state_col, price_col]].copy()
    # Фильтрация NaN без сложных типов (для совместимости с типизацией)
    df[state_col] = df[state_col]
    df = df[[state_col, price_col]]
    df = df[pd.notna(df[state_col])]  # type: ignore[call-arg]
    df = df[pd.notna(df[price_col])]  # type: ignore[call-arg]

    if len(df) == 0:
        raise ValueError("После очистки данных нет строк для анализа.")

    states = np.asarray(df[state_col])
    prices = np.asarray(df[price_col], dtype=float)
    unique_states = pd.unique(states)
    groups = [prices[states == s] for s in unique_states]
    groups = [g for g in groups if g.size > 0]

    if len(groups) < 2:
        raise ValueError(
            "Недостаточно групп: нужно как минимум 2 разных значения state с данными price."
        )

    if use_nonparametric:
        stat, p_value = kruskal(*groups)
        test_name = "Kruskal-Wallis"
    else:
        stat, p_value = f_oneway(*groups)
        test_name = "One-way ANOVA (f_oneway)"

    return {
        "test": test_name,
        "state_col": state_col,
        "price_col": price_col,
        "n_groups": len(groups),
        "statistic": float(stat),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "significant": bool(p_value < alpha),
        "groups_sizes": [int(g.size) for g in groups],
    }


def report_price_significance_by_state(
    table: pd.DataFrame,
    *,
    state_col: str = "state",
    price_col: str = "price",
    alpha: float = 0.05,
    use_nonparametric: bool = True,
    plot: bool = True,
    top_n: int | None = None,
):
    """Красиво выводит результат теста различий цены между state.

    Печатает:
    - итоговую таблицу (state-level test)
    - при `plot=True` — boxplot распределений `price` по state
    - при `top_n` — ограничивает график топ-N штатов по медиане

    Возвращает dict с тем же содержимым, что `price_significance_by_state`,
    плюс агрегированные статистики для печати.
    """

    from matplotlib import pyplot as _plt

    res = price_significance_by_state(
        table,
        state_col=state_col,
        price_col=price_col,
        alpha=alpha,
        use_nonparametric=use_nonparametric,
    )

    df = table[[state_col, price_col]].copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df[pd.notna(df[state_col])]
    df = df[pd.notna(df[price_col])]

    grouped = df.groupby(state_col, sort=False)[price_col]
    stats = grouped.agg(["count", "median", "mean", "std"]).reset_index()
    stats = stats.rename(columns={state_col: "state"})

    if top_n is not None:
        stats = stats.sort_values("median", ascending=False).head(int(top_n))
        df_plot = df[df[state_col].isin(stats["state"])].copy()
    else:
        df_plot = df

    summary_table = pd.DataFrame(
        {
            "test": [res["test"]],
            "statistic": [res["statistic"]],
            "p_value": [res["p_value"]],
            "alpha": [res["alpha"]],
            "significant": [res["significant"]],
            "n_groups": [res["n_groups"]],
        }
    )

    # вывод как pandas таблицы
    summary_table = summary_table.copy()
    stats = stats.sort_values("median", ascending=False).reset_index(drop=True)

    if plot:
        fig, ax = _plt.subplots(1, 1, figsize=(14, 5))
        ax.boxplot(
            [
                df_plot.loc[df_plot[state_col] == s, price_col].to_numpy(dtype=float)
                for s in stats.sort_values("median", ascending=False)["state"].tolist()
            ],
            labels=[
                str(s)
                for s in stats.sort_values("median", ascending=False)["state"].tolist()
            ],
            showfliers=False,
        )
        ax.set_title(f"{price_col} by {state_col} (boxplot)")
        ax.set_xlabel(state_col)
        ax.set_ylabel(price_col)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        _plt.tight_layout()
        _plt.show()

    # В Jupyter вернётся как pandas.DataFrame (красиво), в обычном окружении — через print.
    try:
        from IPython.display import display  # type: ignore

        display(summary_table)
        display(stats)
    except Exception:
        print(summary_table.to_string(index=False))
        print(stats.to_string(index=False))

    return {**res, "state_stats": stats, "summary_table": summary_table}


def kruskal_state_city_homogeneity(
    table: pd.DataFrame,
    *,
    state_col: str = "state",
    city_col: str = "city",
    price_col: str = "price",
    alpha: float = 0.05,
    min_cities: int = 2,
    min_points_per_city: int = 5,
) -> pd.DataFrame:
    """Для каждого штата проверяет однородность распределений price по городам.

    Тест внутри каждого state:
    - собираем группы price по city внутри state
    - Kruskal–Wallis: H0 распределения одинаковы во всех городах штата

    Возвращает таблицу:
        state, city_count, points_count, p_value, significant, effect_size_epsilon2
    """

    for col in (state_col, city_col, price_col):
        if col not in table.columns:
            raise ValueError(f"Колонка '{col}' отсутствует в таблице.")

    df = table[[state_col, city_col, price_col]].copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df[pd.notna(df[state_col])]
    df = df[pd.notna(df[city_col])]
    df = df[pd.notna(df[price_col])]

    results: list[dict] = []

    for st, st_df in df.groupby(state_col, sort=False):
        points_count = int(len(st_df))

        city_groups = []
        city_count = 0
        for city, city_df in st_df.groupby(city_col, sort=False):
            arr = np.asarray(city_df[price_col], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size < min_points_per_city:
                continue
            city_count += 1
            city_groups.append(arr)

        if city_count < min_cities:
            # Автоматически считаем однородным, когда нечего сравнивать
            p_value = 1.0
            significant = False
            effect_size = 0.0
        else:
            stat, p_value = kruskal(*city_groups)
            significant = bool(p_value < alpha)

            # Эффект-размер ε² по аналогии с η² для ранговых тестов:
            # ε² = (H - k + 1) / (n - k) , где H — статистика Kruskal–Wallis, k — число групп, n — общее n.
            n = int(sum(g.size for g in city_groups))
            k = int(city_count)
            if n > k:
                effect_size = float((stat - k + 1) / (n - k))
            else:
                effect_size = 0.0

        results.append(
            {
                state_col: st,
                "city_count": int(city_count),
                "points_count": points_count,
                "p_value": float(p_value),
                "significant": bool(significant),
                "alpha": float(alpha),
                "effect_size_epsilon2": float(effect_size),
            }
        )

    out = pd.DataFrame(results)
    out = out.sort_values(by=["p_value", state_col], ascending=[True, True])
    return out


def dunn_posthoc_for_heterogeneous_states(
    table: pd.DataFrame,
    *,
    state_col: str = "state",
    city_col: str = "city",
    price_col: str = "price",
    alpha: float = 0.05,
    epsilon2_threshold: float = 0.06,
    min_points_per_city: int = 5,
    p_adjust: str = "holm",
    median_deviation_threshold: float = 0.30,
    min_points_city: int = 1000,
    majority_ratio: float = 0.5,
    plot_heatmaps: bool = True,
    show_plots: bool = True,
    save_plots: bool = False,
    plots_dir: str = "./plots",
) -> dict:
    """Для неоднородных штатов (ε² > порога) выполняет Dunn post-hoc тест.

    Шаги:
    1) Для каждого штата: Kruskal–Wallis по городам + ε²
    2) Если ε² > epsilon2_threshold — запускается post-hoc Dunn test по парам городов

    Возвращает:
      - matrix_by_state: {state: DataFrame(pairwise_p_values)}
      - significant_cities_by_state: {state: list[str]}
    """

    # 1) Сначала определяем неоднородные штаты
    homo = kruskal_state_city_homogeneity(
        table,
        state_col=state_col,
        city_col=city_col,
        price_col=price_col,
        alpha=alpha,
        min_points_per_city=min_points_per_city,
    )

    heterogeneous = homo[homo["effect_size_epsilon2"] > epsilon2_threshold]

    matrix_by_state: dict = {}
    significant_cities_by_state: dict = {}
    candidates_cities_by_state: dict = {}
    candidates_summary: list[dict] = []

    # 2) Dunn test
    try:
        import scikit_posthocs as sp  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Для dunn_posthoc_for_heterogeneous_states требуется scikit-posthocs. "
            "Установите пакет scikit-posthocs."
        ) from e

    # min_points_city задан одним значением

    for _, row in heterogeneous.iterrows():
        st = row[state_col]
        st_df = table[table[state_col] == st][[city_col, price_col]].copy()
        st_df[price_col] = pd.to_numeric(st_df[price_col], errors="coerce")
        st_df = st_df[pd.notna(st_df[city_col])]
        st_df = st_df[pd.notna(st_df[price_col])]

        # Фильтрация городов по числу точек (для Dunn)
        counts = st_df.groupby(city_col, sort=False).size()
        cities = counts[counts >= min_points_per_city].index.tolist()
        st_df = st_df[st_df[city_col].isin(cities)]

        if len(cities) < 2:
            continue

        # Dunn test для всех пар городов — получаем матрицу p-value
        # sp.posthoc_dunn returns a square DataFrame indexed/columns by group labels.
        dunn_p = sp.posthoc_dunn(
            st_df,
            val_col=price_col,
            group_col=city_col,
            p_adjust=p_adjust,
        )

        matrix_by_state[st] = dunn_p

        if plot_heatmaps:
            # Разница медиан: median(city_i) - median(city_j)
            city_medians = {}
            for ci in cities:
                mask_ci = np.asarray(st_df[city_col] == ci)
                city_prices = np.asarray(st_df[price_col], dtype=float)[mask_ci]
                city_prices = city_prices[np.isfinite(city_prices)]
                city_medians[ci] = (
                    float(np.median(city_prices)) if city_prices.size else np.nan
                )

            # Матрица разниц медиан
            n_city = len(cities)
            med_diff = np.zeros((n_city, n_city), dtype=float)
            sig_mask = np.zeros((n_city, n_city), dtype=bool)
            for i, ci in enumerate(cities):
                for j, cj in enumerate(cities):
                    med_diff[i, j] = city_medians[ci] - city_medians[cj]
                    if i != j:
                        sig_mask[i, j] = float(dunn_p.loc[ci, cj]) < alpha

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            im = ax.imshow(med_diff, cmap="RdBu")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("median(price_i) - median(price_j)")

            ax.set_xticks(range(n_city))
            ax.set_yticks(range(n_city))
            ax.set_xticklabels([str(c) for c in cities], rotation=45, ha="right")
            ax.set_yticklabels([str(c) for c in cities])

            # рамки на значимых парах
            for i in range(n_city):
                for j in range(n_city):
                    if i == j:
                        continue
                    if sig_mask[i, j]:
                        ax.add_patch(
                            Rectangle(
                                (j - 0.5, i - 0.5),
                                1,
                                1,
                                fill=False,
                                edgecolor="black",
                                linewidth=1.5,
                            )
                        )

            ax.set_title(f"State={st}: median differences (alpha={alpha})")
            plt.tight_layout()

            if save_plots:
                import os

                os.makedirs(plots_dir, exist_ok=True)
                fig_path = os.path.join(plots_dir, f"dunn_heatmap_state_{st}.png")
                fig.savefig(fig_path, dpi=150)
            if show_plots:
                plt.show()
            else:
                plt.close(fig)

        # "значимо отличается от большинства": город считается таким,
        # если против большинства других городов p < alpha.
        sig_lists = []
        for ci in cities:
            # Берём строку ci: p против всех остальных
            row_p = dunn_p.loc[ci]
            other = [c for c in cities if c != ci]
            p_other = row_p[other].astype(float)
            sig_count = int((p_other < alpha).sum())
            if sig_count >= max(1, int(len(other) * majority_ratio)):
                sig_lists.append(ci)
        # Дополнительные пороговые критерии для кандидатов на отдельную модель
        st_prices = pd.to_numeric(
            table.loc[table[state_col] == st, price_col], errors="coerce"
        ).dropna()
        st_median = (
            float(np.median(np.asarray(st_prices, dtype=float)))
            if len(st_prices)
            else np.nan
        )

        candidates = []
        for city in sig_lists:
            city_prices = np.asarray(
                st_df[st_df[city_col] == city][price_col], dtype=float
            )
            city_prices = city_prices[np.isfinite(city_prices)]
            if city_prices.size < min_points_city:
                continue
            city_median = (
                float(np.median(city_prices)) if city_prices.size > 0 else np.nan
            )

            if np.isfinite(st_median) and np.isfinite(city_median) and st_median != 0:
                deviation = abs(city_median - st_median) / abs(st_median)
            else:
                deviation = np.inf

            if deviation > median_deviation_threshold:
                candidates.append(city)

                candidates_summary.append(
                    {
                        state_col: st,
                        city_col: city,
                        "city_median": city_median,
                        "state_median": st_median,
                        "median_deviation": deviation,
                        "city_points": int(city_prices.size),
                    }
                )

        # city также должен быть «кандидатом» только если он значимо отличается от большинства
        # (это уже заложено в sig_lists) и имеет достаточно точек и отклонение медианы.

        significant_cities_by_state[st] = sig_lists
        candidates_cities_by_state[st] = candidates

    # 3) Доля данных в штате, которые попадают в кандидаты
    # candidate definition: город входит в candidates_cities_by_state
    # доля = число строк в (state, city in candidates) / общее число строк в state
    candidate_share_by_state = {}
    for st in heterogeneous[state_col].tolist():
        cand_cities = candidates_cities_by_state.get(st, [])
        if not cand_cities:
            candidate_share_by_state[st] = 0.0
            continue
        st_all = table[table[state_col] == st]
        denom = len(st_all)
        if denom == 0:
            candidate_share_by_state[st] = 0.0
            continue
        st_cand = st_all[st_all[city_col].isin(cand_cities)]
        candidate_share_by_state[st] = float(len(st_cand) / denom)

    # Для удобства: превращаем результаты в единую pandas-таблицу.
    # Каждая строка = кандидат (state, city). Если нужно — можно агрегировать по state.
    candidates_df = pd.DataFrame(candidates_summary)
    if len(candidates_df) > 0:
        candidates_df["candidate_share_by_state"] = candidates_df[state_col].map(
            candidate_share_by_state
        )

    return {
        "candidates": candidates_df,
        "alpha": float(alpha),
        "epsilon2_threshold": float(epsilon2_threshold),
        "matrix_by_state": matrix_by_state,
        "significant_cities_by_state": significant_cities_by_state,
        "candidates_cities_by_state": candidates_cities_by_state,
        "candidate_share_by_state": candidate_share_by_state,
        "heterogeneous_states": heterogeneous[state_col].tolist(),
    }


def plot_price_heatmap_by_state(
    table: pd.DataFrame,
    *,
    state_col: str = "state",
    price_col: str = "price",
    alpha: float = 0.05,
    use_nonparametric: bool = True,
    cmap: str = "RdBu",
    figsize: tuple[int, int] = (10, 8),
) -> dict:
    """Строит тепловую карту попарных сравнений средних/медиан по state.

    Логика:
    - вычисляется парная значимость (Mann–Whitney U по умолчанию)
    - в матрице помечаются значимо отличающиеся пары
    - тепловая карта отображает разницу центров (mean/median)

    Примечание: это именно heatmap попарных различий между state.
    """

    # Local import чтобы не тащить лишние зависимости, если функция не используется
    from scipy.stats import mannwhitneyu

    if state_col not in table.columns:
        raise ValueError(f"Колонка state '{state_col}' отсутствует в таблице.")
    if price_col not in table.columns:
        raise ValueError(f"Колонка price '{price_col}' отсутствует в таблице.")

    df = table[[state_col, price_col]].copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df[[state_col, price_col]]
    df = df[pd.notna(df[state_col])]
    df = df[pd.notna(df[price_col])]

    if len(df) == 0:
        raise ValueError("После очистки данных нет строк для анализа.")

    states = pd.unique(df[state_col])
    states = [
        s
        for s in states
        if pd.Series(df.loc[df[state_col] == s, price_col]).dropna().shape[0] > 0
    ]

    centers = {}
    groups = {}
    for s in states:
        arr = np.asarray(df.loc[df[state_col] == s, price_col], dtype=float)
        arr = arr[np.isfinite(arr)]
        groups[s] = arr
        centers[s] = float(np.median(arr) if use_nonparametric else np.mean(arr))

    if len(states) < 2:
        raise ValueError("Недостаточно разных значений state с данными price.")

    n = len(states)
    diff = np.zeros((n, n), dtype=float)
    sig = np.zeros((n, n), dtype=bool)

    for i, si in enumerate(states):
        for j, sj in enumerate(states):
            if i == j:
                diff[i, j] = 0.0
                sig[i, j] = False
                continue
            diff[i, j] = centers[si] - centers[sj]
            # Парное сравнение
            # Если groups очень маленькие — mannwhitneyu может падать; поэтому защищаем.
            try:
                stat, p = mannwhitneyu(groups[si], groups[sj], alternative="two-sided")
            except Exception:
                p = 1.0
            sig[i, j] = p < alpha

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(diff, cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "median(price_i) - median(price_j)"
        if use_nonparametric
        else "mean(price_i) - mean(price_j)"
    )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(s) for s in states], rotation=45, ha="right")
    ax.set_yticklabels([str(s) for s in states])

    # Подсветка значимых пар рамками
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if sig[i, j]:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.5,
                    )
                )

    ax.set_title(f"Price differences by state (alpha={alpha})")
    ax.grid(False)
    plt.tight_layout()
    plt.show()

    return {
        "state_col": state_col,
        "price_col": price_col,
        "alpha": alpha,
        "use_nonparametric": use_nonparametric,
        "states": states,
        "diff_matrix": diff,
        "significant_matrix": sig,
    }


def plot_boxplots_by_state(
    table: pd.DataFrame,
    *,
    state_col: str = "state",
    price_col: str = "price",
    figsize: tuple[int, int] = (14, 5),
    showfliers: bool = False,
    sort_by: str = "median",
    grid: bool = True,
):
    """Строит boxplot распределения price по state.

    Args:
        table: Датасет.
        state_col: Колонка с штатами.
        price_col: Колонка с ценой.
        figsize: Размер фигуры.
        showfliers: Показывать ли выбросы.
        sort_by: Как сортировать штаты: 'median' или 'mean'.
        grid: Добавить ли сетку.
    """

    if state_col not in table.columns:
        raise ValueError(f"Колонка state '{state_col}' отсутствует в таблице.")
    if price_col not in table.columns:
        raise ValueError(f"Колонка price '{price_col}' отсутствует в таблице.")

    df = table[[state_col, price_col]].copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df[pd.notna(df[state_col])]
    df = df[pd.notna(df[price_col])]

    if len(df) == 0:
        raise ValueError("После очистки данных нет строк для анализа.")

    unique_states = pd.Series(df[state_col]).unique().tolist()

    # При желании сортируем по центру
    # Получаем массивы напрямую и используем маску (без .loc)
    state_values = np.asarray(df[state_col])
    price_values = np.asarray(df[price_col], dtype=float)

    if sort_by in {"mean", "median"}:
        centers = {}
        for s in unique_states:
            mask = state_values == s
            arr = price_values[mask]
            arr = arr[np.isfinite(arr)]
            centers[s] = float(arr.mean() if sort_by == "mean" else np.median(arr))
        unique_states = sorted(unique_states, key=lambda s: centers[s], reverse=True)

    data = []
    for s in unique_states:
        mask = state_values == s
        arr = price_values[mask]
        arr = arr[np.isfinite(arr)]
        data.append(arr)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.boxplot(data, showfliers=showfliers)
    ax.set_xticks(range(1, len(unique_states) + 1))
    ax.set_xticklabels([str(s) for s in unique_states], rotation=45, ha="right")
    ax.set_xlabel(state_col)
    ax.set_ylabel(price_col)
    ax.set_title(f"Price distribution by {state_col}")
    ax.tick_params(axis="x", rotation=45)
    if grid:
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "state_col": state_col,
        "price_col": price_col,
        "states_order": unique_states,
    }
