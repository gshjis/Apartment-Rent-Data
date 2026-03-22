"""Preprocessor utilities.

Класс :class:`~paсkages.preprocessing.preprocessing.preprocessor.Preprocessor`
инкапсулирует логику преобразований для train/test пайплайна.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from analysis import LdaTopicModel
from analysis import dunn_posthoc_for_heterogeneous_states
from logger import log_success
from logger.logger import get_logger


logger = get_logger(__name__)


class Preprocessor:
    """Пакетный препроцессор для данных по аренде."""

    def __init__(self) -> None:
        """Инициализация пустого состояния."""

        # редкие категории
        self.rare_categories_map: dict[str, list[Any]] = {}

        # аномальные города
        self.anomaly_cities: list[str] = []
        self.anomaly_types: dict[str, Literal["premium", "budget"]] = {}

        # LDA
        self.lda_model: LdaTopicModel | None = None
        self.lda_topic_correlations: list[float] = []
        self.lda_vocabulary: list[str] = []

    @staticmethod
    def drop_columns(
        df: pd.DataFrame,
        columns: list[str] | tuple[str, ...],
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Удаляет указанные колонки.

        Args:
            df: Входной DataFrame.
            columns: Колонки для удаления.
            inplace: Если False — возвращает копию.

        Returns:
            DataFrame с удалёнными колонками.
        """

        if inplace:
            df.drop(columns=list(columns), inplace=True, errors="ignore")
            return df
        return df.drop(columns=list(columns), errors="ignore").copy()

    @staticmethod
    def merge_columns(
        df: pd.DataFrame,
        col1: str,
        col2: str,
        new_col: str,
        drop_original: bool = True,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Объединяет две колонки через `|`.

        NaN преобразуются в пустую строку.

        Args:
            df: Входной DataFrame.
            col1: Первая колонка.
            col2: Вторая колонка.
            new_col: Имя результирующей колонки.
            drop_original: Удалить исходные колонки.
            inplace: Если False — возвращает копию.

        Returns:
            DataFrame с новой колонкой.
        """

        out = df if inplace else df.copy()
        if col1 not in out.columns or col2 not in out.columns:
            raise ValueError(f"Одна из колонок не найдена: {col1=}, {col2=}")

        s1 = out[col1].fillna("").astype(str)
        s2 = out[col2].fillna("").astype(str)
        out[new_col] = s1 + "|" + s2
        if drop_original:
            out.drop(columns=[col1, col2], inplace=True, errors="ignore")
        return out

    def fit_rare_categories(
        self,
        df: pd.DataFrame,
        columns: list[str] | tuple[str, ...],
        threshold: float = 0.05,
    ) -> pd.DataFrame:
        """Обучает маппинг редких категорий и заменяет их на `'other'`.

        Проводится для каждого столбца в `columns` независимо.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df должен быть pandas.DataFrame")

        out = df.copy()
        self.rare_categories_map = {}
        for col in columns:
            if col not in out.columns:
                continue
            logger.info(
                "fit_rare_categories: start (column=%s, threshold=%s)",
                col,
                threshold,
            )
            vc = out[col].value_counts(dropna=False, normalize=True)
            rare = vc[vc < threshold].index.tolist()
            self.rare_categories_map[col] = rare
            logger.info(
                "fit_rare_categories: %s: %d rare values replaced with 'other' (threshold=%s)",
                col,
                len(rare),
                threshold,
            )
            out[col] = out[col].where(~out[col].isin(rare), other="other")

        log_success(logger, "fit_rare_categories: completed")
        return out

    def transform_rare_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет замену редких категорий на `'other'` для train/test."""

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df должен быть pandas.DataFrame")
        out = df.copy()
        for col, rare_vals in self.rare_categories_map.items():
            if col not in out.columns:
                continue
            logger.info(
                "transform_rare_categories: column=%s (rare_values=%d)",
                col,
                len(rare_vals),
            )
            out[col] = out[col].where(~out[col].isin(rare_vals), other="other")

        log_success(logger, "transform_rare_categories: completed")
        return out

    def fit_anomaly_cities(
        self,
        df: pd.DataFrame,
        city_col: str,
        state_col: str,
        price_col: str = "price",
        min_points_city: int = 30,
        median_deviation_threshold: float = 0.3,
        alpha: float = 0.05,
        majority_ratio: float = 0.3,
    ) -> pd.DataFrame:
        """Выявляет аномальные города и добавляет флаги/типы."""

        out = df.copy()
        logger.info(
            "fit_anomaly_cities: start (city_col=%s, state_col=%s, min_points_city=%d)",
            city_col,
            state_col,
            min_points_city,
        )
        out["city_state"] = (
            out[city_col].fillna("").astype(str)
            + "|"
            + out[state_col].fillna("").astype(str)
        )

        candidates = dunn_posthoc_for_heterogeneous_states(
            out,
            state_col="state",
            city_col="city_state",
            price_col=price_col,
            alpha=alpha,
            min_points_per_city=5,
            epsilon2_threshold=0.06,
            median_deviation_threshold=median_deviation_threshold,
            min_points_city=min_points_city,
            majority_ratio=majority_ratio,
            plot_heatmaps=False,
            show_plots=False,
        )

        self.anomaly_cities = []
        self.anomaly_types = {}
        if isinstance(candidates, dict):
            # Функция возвращает словари по state.
            # Формат верхнего уровня не фиксирован, поэтому делаем максимально совместимую агрегацию.
            significant_by_state = candidates.get("significant_cities_by_state", {})
            candidates_summary = candidates.get("candidates_summary", [])

            # список аномальных city_state (агрегируем по state)
            anomaly_list: list[str] = []
            for _, lst in (significant_by_state or {}).items():
                anomaly_list.extend([str(x) for x in (lst or [])])
            self.anomaly_cities = list(dict.fromkeys(anomaly_list))

            # типизация по dev (если есть в summary)
            if isinstance(candidates_summary, list):
                for row in candidates_summary:
                    if not isinstance(row, dict):
                        continue
                    city = row.get("city_state")
                    dev = row.get("median_deviation")
                    if city is None or dev is None:
                        continue
                    city_s = str(city)
                    try:
                        dev_f = float(dev)
                    except Exception:
                        continue
                    self.anomaly_types[city_s] = "premium" if dev_f > 0 else "budget"

            logger.info(
                "fit_anomaly_cities: found %d anomaly candidates",
                len(self.anomaly_cities),
            )

            log_success(
                logger,
                f"fit_anomaly_cities: detected {len(self.anomaly_cities)} anomaly candidates",
            )

        anomaly_set = set(self.anomaly_cities)
        out["is_anomaly"] = out["city_state"].apply(
            lambda x: 1 if str(x) in anomaly_set else 0
        )
        out["anomaly_type"] = out["city_state"].apply(
            lambda x: self.anomaly_types.get(str(x), "normal")
        )
        return out

    def transform_anomaly_cities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет ранее найденные аномальные города."""

        out = df.copy()
        if "city_state" not in out.columns:
            logger.warning(
                "transform_anomaly_cities: column 'city_state' not found; marking all as normal"
            )
            out["is_anomaly"] = 0
            out["anomaly_type"] = "normal"
            log_success(logger, "transform_anomaly_cities: completed (no city_state)")
            return out
        anomaly_set = set(self.anomaly_cities)
        out["is_anomaly"] = out["city_state"].apply(
            lambda x: 1 if str(x) in anomaly_set else 0
        )
        out["anomaly_type"] = out["city_state"].apply(
            lambda x: self.anomaly_types.get(str(x), "normal")
        )

        log_success(logger, "transform_anomaly_cities: completed")
        return out

    def fit_lda(
        self,
        df: pd.DataFrame,
        text_column: str,
        n_topics: int = 15,
        alpha: float = 0.01,
        eta: float = 0.01,
        drop_text_column: bool = True,
    ) -> pd.DataFrame:
        """Обучает LDA и добавляет вероятности тем в колонки `topic_*`."""

        out = df.copy()
        logger.info(
            "fit_lda: start (n_topics=%d, text_column=%s, alpha=%s, eta=%s)",
            n_topics,
            text_column,
            alpha,
            eta,
        )
        self.lda_model = LdaTopicModel(
            dataset=out,
            text_column=text_column,
            n_topics=n_topics,
            alpha=alpha,
            eta=eta,
        )
        self.lda_model.fit()
        logger.info(
            "fit_lda: trained LDA model (%d topics) on text_column=%s",
            n_topics,
            text_column,
        )
        mat = self.lda_model.transform(out)
        for tid in range(int(n_topics)):
            out[f"topic_{tid}"] = mat[:, tid]
        if drop_text_column and text_column in out.columns:
            out.drop(columns=[text_column], inplace=True)

        log_success(logger, "fit_lda: completed")
        return out

    def transform_lda(
        self, df: pd.DataFrame, drop_text_column: bool = True
    ) -> pd.DataFrame:
        """Добавляет вероятности тем на основе обученной LDA модели."""

        if self.lda_model is None:
            raise RuntimeError("Сначала вызовите fit_lda().")
        out = df.copy()
        logger.info(
            "transform_lda: start (drop_text_column=%s)",
            drop_text_column,
        )
        mat = self.lda_model.transform(out)
        n_topics = int(getattr(self.lda_model, "n_topics", mat.shape[1]))
        for tid in range(n_topics):
            out[f"topic_{tid}"] = mat[:, tid]
        if drop_text_column and self.lda_model.text_column in out.columns:
            out.drop(columns=[self.lda_model.text_column], inplace=True)

        log_success(logger, "transform_lda: completed")
        return out

    def fit_lda_weighted_correlation(
        self,
        df: pd.DataFrame,
        text_column: str,
        price_col: str = "price",
        n_topics: int = 15,
        alpha: float = 0.01,
        eta: float = 0.01,
        drop_text_column: bool = True,
    ) -> pd.DataFrame:
        """Обучает LDA и добавляет `topic_weighted_correlation` (Spearman)."""

        out = self.fit_lda(
            df,
            text_column=text_column,
            n_topics=n_topics,
            alpha=alpha,
            eta=eta,
            drop_text_column=False,
        )
        if self.lda_model is None:
            raise RuntimeError("LDA model не инициализирована")

        corr_df = self.lda_model.topics_price_correlation(
            price_column=price_col,
            correlation_method="spearman",
            significance_test=False,
            return_both_pvalues=False,
        )
        # corr_df содержит: topic, corr
        corr_map = corr_df.set_index("topic")["corr"].to_dict()
        self.lda_topic_correlations = [
            float(corr_map.get(i, 0.0)) for i in range(int(n_topics))
        ]
        logger.info(
            "fit_lda_weighted_correlation: correlations computed for %d topics",
            int(n_topics),
        )

        topic_cols = [f"topic_{i}" for i in range(int(n_topics))]
        weights = out[topic_cols].to_numpy(dtype=float)
        corrs = np.asarray(self.lda_topic_correlations, dtype=float)
        out["topic_weighted_correlation"] = (weights * corrs).sum(axis=1)

        if drop_text_column and text_column in out.columns:
            out.drop(columns=[text_column], inplace=True)

        log_success(logger, "fit_lda_weighted_correlation: completed")
        return out

    def transform_lda_weighted_correlation(
        self, df: pd.DataFrame, drop_text_column: bool = True
    ) -> pd.DataFrame:
        """Применяет `topic_weighted_correlation` для новых данных."""

        if self.lda_model is None:
            raise RuntimeError("Сначала вызовите fit_lda_weighted_correlation().")
        if not self.lda_topic_correlations:
            raise RuntimeError("Сначала вызовите fit_lda_weighted_correlation().")

        out = self.transform_lda(df, drop_text_column=False)
        n_topics = len(self.lda_topic_correlations)
        topic_cols = [f"topic_{i}" for i in range(int(n_topics))]
        weights = out[topic_cols].to_numpy(dtype=float)
        corrs = np.asarray(self.lda_topic_correlations, dtype=float)
        out["topic_weighted_correlation"] = (weights * corrs).sum(axis=1)

        if drop_text_column and self.lda_model.text_column in out.columns:
            out.drop(columns=[self.lda_model.text_column], inplace=True)

        log_success(logger, "transform_lda_weighted_correlation: completed")
        return out

    def fit_lda_ohe(
        self,
        df: pd.DataFrame,
        text_column: str,
        n_top_words: int = 100,
        n_topics: int = 15,
        alpha: float = 0.01,
        eta: float = 0.01,
        drop_text_column: bool = True,
    ) -> pd.DataFrame:
        """Обучает LDA и добавляет OHE бинарные признаки топ-N слов."""

        out = df.copy()
        logger.info(
            "fit_lda_ohe: start (n_top_words=%d, n_topics=%d)",
            n_top_words,
            n_topics,
        )
        self.lda_model = LdaTopicModel(
            dataset=out,
            text_column=text_column,
            n_topics=n_topics,
            alpha=alpha,
            eta=eta,
        )
        self.lda_model.fit()

        words_per_topic = int(n_top_words // max(1, n_topics) + 1)
        topics_words = self.lda_model.get_top_words_per_topic(n_words=words_per_topic)

        vocab: list[str] = []
        for _, words in topics_words.items():
            for w in words:
                if w not in vocab:
                    vocab.append(w)
                if len(vocab) >= int(n_top_words):
                    break
            if len(vocab) >= int(n_top_words):
                break
        self.lda_vocabulary = vocab
        logger.info(
            "fit_lda_ohe: built vocabulary of %d words (n_top_words=%d, n_topics=%d)",
            len(self.lda_vocabulary),
            int(n_top_words),
            int(n_topics),
        )

        texts = out[text_column].fillna("").astype(str).tolist()
        for word in self.lda_vocabulary:
            out[f"lda_word_{word}"] = [1 if word in t else 0 for t in texts]

        if drop_text_column and text_column in out.columns:
            out.drop(columns=[text_column], inplace=True)

        log_success(logger, "fit_lda_ohe: completed")
        return out

    def transform_lda_ohe(
        self, df: pd.DataFrame, drop_text_column: bool = True
    ) -> pd.DataFrame:
        """Добавляет OHE бинарные признаки слов из обученного словаря."""

        if self.lda_model is None:
            raise RuntimeError("Сначала вызовите fit_lda_ohe().")
        if not self.lda_vocabulary:
            raise RuntimeError("Сначала вызовите fit_lda_ohe().")
        if self.lda_model.text_column not in df.columns:
            raise ValueError(
                "Входной df не содержит текстовую колонку для transform_lda_ohe"
            )

        out = df.copy()
        texts = out[self.lda_model.text_column].fillna("").astype(str).tolist()
        for word in self.lda_vocabulary:
            out[f"lda_word_{word}"] = [1 if word in t else 0 for t in texts]

        if drop_text_column and self.lda_model.text_column in out.columns:
            out.drop(columns=[self.lda_model.text_column], inplace=True)

        log_success(logger, "transform_lda_ohe: completed")
        return out

    def get_dataset(
        self, df: pd.DataFrame, split_by_anomaly: bool = False
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Возвращает df или разбивает его на нормальные/аномальные строки."""

        if not split_by_anomaly:
            return df
        if "is_anomaly" not in df.columns:
            raise ValueError("В df отсутствует колонка is_anomaly")

        df_anomaly = df.loc[df["is_anomaly"] == 1].copy()
        df_normal = df.loc[df["is_anomaly"] == 0].copy()
        return df_normal, df_anomaly

    def save(self, path: str | Path) -> None:
        """Сохраняет состояние препроцессора в pickle."""

        p = Path(path)
        with p.open("wb") as f:
            pickle.dump(self, f)

        log_success(logger, f"Preprocessor saved to '{p}'")

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        """Загружает препроцессор из pickle."""

        p = Path(path)
        with p.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError("pickle не содержит объект Preprocessor")

        log_success(logger, f"Preprocessor loaded from '{p}'")
        return obj

    def pipeline(
        self,
        df: pd.DataFrame,
        *,
        mode: Literal["train", "test"] = "train",
        drop_columns_args: dict[str, Any] | None = None,
        merge_columns_args: dict[str, Any] | None = None,
        rare_categories_args: dict[str, Any] | None = None,
        anomaly_args: dict[str, Any] | None = None,
        lda_type: Literal[
            "probabilities", "weighted_correlation", "ohe"
        ] = "probabilities",
        lda_args: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Последовательно применяет преобразования в корректном порядке.

        В train-режиме выполняются методы `fit_*`, в test-режиме — `transform_*`.
        """

        out = df.copy()

        logger.info(
            "pipeline: start (mode=%s, lda_type=%s)",
            mode,
            lda_type,
        )

        drop_columns_args = drop_columns_args or {}
        if "columns" in drop_columns_args:
            out = self.drop_columns(out, inplace=False, **drop_columns_args)

        merge_columns_args = merge_columns_args or {}
        if "col1" in merge_columns_args and "col2" in merge_columns_args:
            out = self.merge_columns(out, inplace=False, **merge_columns_args)

        rare_categories_args = rare_categories_args or {}
        if mode == "train":
            if "columns" in rare_categories_args:
                out = self.fit_rare_categories(out, **rare_categories_args)
        else:
            out = self.transform_rare_categories(out)

        anomaly_args = anomaly_args or {}
        if mode == "train":
            if "city_col" in anomaly_args and "state_col" in anomaly_args:
                out = self.fit_anomaly_cities(out, **anomaly_args)
        else:
            out = self.transform_anomaly_cities(out)

        lda_args = lda_args or {}
        if lda_type == "probabilities":
            out = (
                self.fit_lda(out, **lda_args)
                if mode == "train"
                else self.transform_lda(out, **lda_args)
            )
        elif lda_type == "weighted_correlation":
            out = (
                self.fit_lda_weighted_correlation(out, **lda_args)
                if mode == "train"
                else self.transform_lda_weighted_correlation(out, **lda_args)
            )
        elif lda_type == "ohe":
            out = (
                self.fit_lda_ohe(out, **lda_args)
                if mode == "train"
                else self.transform_lda_ohe(out, **lda_args)
            )
        else:
            raise ValueError(f"Неверный lda_type: {lda_type}")

        log_success(logger, "pipeline: completed")
        return out
