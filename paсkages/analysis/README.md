# Модуль `analysis`

Набор функций для анализа датасета аренды: визуализации и статистические проверки по географии (штат/город) и цене.

## Быстрый импорт

Все основные функции экспортируются из [`paсkages/analysis/analysis/__init__.py:1`](paсkages/analysis/analysis/__init__.py:1).

## Что внутри

### 1) Визуализация координат

- [`plot_coordinates`](paсkages/analysis/analysis/coordinates.py:6) — scatter по `longitude/latitude`, окраска по цене.

### 2) Категориальные зависимости

- [`plot_categorical_dependencies`](paсkages/analysis/analysis/categorical_dependence.py:5) — барчарт агрегированного `target` по категориальным признакам.

### 3) Проверки значимости по `state`

- [`price_significance_by_state`](paсkages/analysis/analysis/state_price_significance.py:7) — сравнение распределений цены между штатами (Kruskal–Wallis по умолчанию, либо ANOVA).
- [`plot_price_heatmap_by_state`](paсkages/analysis/analysis/state_price_significance.py:81) — тепловая карта попарных различий между штатами с рамками для значимых пар.
- [`plot_boxplots_by_state`](paсkages/analysis/analysis/state_price_significance.py:202) — boxplot распределения цены по штатам.

### 4) Двухфакторный ANOVA: город и штат

- [`anova_two_factor_city_state`](paсkages/analysis/analysis/state_price_significance.py:128) — модель `price ~ C(city) + C(state) + C(city):C(state)` и p-value по эффектам.

## Вариант 1: «Внутри штата распределение цен одинаково для всех городов?»

### Шаг A — однородность внутри штата по городам

- [`kruskal_state_city_homogeneity`](paсkages/analysis/analysis/state_price_significance.py:275)
  - Для каждого `state` строит группы `price` по `city`.
  - Проводит Kruskal–Wallis.
  - Возвращает таблицу с p-value и эффект-размером `effect_size_epsilon2` (ε²).

### Шаг B — Dunn post-hoc для неоднородных штатов

- [`dunn_posthoc_for_heterogeneous_states`](paсkages/analysis/analysis/state_price_significance.py:161)
  - Выбирает неоднородные штаты по условию `effect_size_epsilon2 > epsilon2_threshold`.
  - Запускает Dunn test между городами внутри каждого выбранного штата.
  - Формирует:
    - `matrix_by_state`: матрицы попарных p-value
    - `significant_cities_by_state`: города, отличающиеся от большинства
    - `candidates_cities_by_state`: кандидаты для отдельной модели по правилам:
      - Dunn значимость
      - отклонение медианы города от медианы штата > `median_deviation_threshold`
      - достаточное число точек в городе (`min_points_city`)
    - `candidates_summary`: сводка кандидатов с метриками
    - `candidate_share_by_state`: доля строк в штате, приходящаяся на города-кандидаты

## Заметки по зависимостям

Некоторые функции требуют внешние библиотеки:
- `statsmodels` — для двухфакторного ANOVA
- `scikit-posthocs` — для Dunn post-hoc

