# Модуль `analysis`

Пакет статистического и визуального анализа датасета аренды (price/география/категориальные признаки).

## Основные функции

### Координаты
- [`plot_coordinates`](paсkages/analysis/analysis/coordinates.py:6) — карта точек по `longitude/latitude` с окраской по цене.

### Категориальные признаки
- [`plot_categorical_dependencies`](paсkages/analysis/analysis/categorical_dependence.py:5) — барчарты агрегированных зависимостей по категориальным признакам.

### Значимость различий по `state`
- [`price_significance_by_state`](paсkages/analysis/analysis/state_price_significance.py:7) — тест различий цен между штатами (Kruskal–Wallis по умолчанию, либо ANOVA).
- [`report_price_significance_by_state`](paсkages/analysis/analysis/state_price_significance.py:67) — красивый вывод результата: таблицы + (опционально) boxplot.
- [`plot_price_heatmap_by_state`](paсkages/analysis/analysis/state_price_significance.py:81) — тепловая карта попарных различий по `state`.
- [`plot_boxplots_by_state`](paсkages/analysis/analysis/state_price_significance.py:202) — boxplot распределений `price` по штатам.

### Двухфакторный ANOVA: город + штат
- [`anova_two_factor_city_state`](paсkages/analysis/analysis/state_price_significance.py:128) — модель `price ~ C(city) + C(state) + C(city):C(state)`.

## Вариант 1: однородность внутри штата по городам

### Шаг 1: Kruskal–Wallis внутри каждого штата
- [`kruskal_state_city_homogeneity`](paсkages/analysis/analysis/state_price_significance.py:275)
  - H0: распределения `price` по городам внутри `state` одинаковы.
  - Возвращает таблицу со `p_value`, `significant` и эффект-размером `effect_size_epsilon2` (ε²).

### Шаг 2: Dunn post-hoc для неоднородных штатов
- [`dunn_posthoc_for_heterogeneous_states`](paсkages/analysis/analysis/state_price_significance.py:161)
  - берёт неоднородные штаты (по умолчанию `effect_size_epsilon2 > 0.06`)
  - проводит Dunn test между городами внутри `state`
  - возвращает:
    - `candidates`: pandas DataFrame с городами-кандидатами на отдельную модель
    - `candidates_summary`: расширенная сводка кандидатов
    - дополнительные структуры (`matrix_by_state`, `significant_cities_by_state`, `candidate_share_by_state`, ...)

## Зависимости

- `statsmodels` — для двухфакторного ANOVA
- `scikit-posthocs` — для Dunn post-hoc

