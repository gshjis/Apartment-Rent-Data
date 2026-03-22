from __future__ import annotations

import re
from typing import Any, Literal

import numpy as np
import pandas as pd

try:  # optional heavy deps
    import gensim
    from gensim import corpora
    from gensim.models.ldamodel import LdaModel
except Exception:  # pragma: no cover
    gensim = None
    corpora = None
    LdaModel = None


class LdaTopicModel:
    """LDA тематическая модель для текста + корреляции тем с ценой.

    Реализация опирается на `gensim` (LDA + Dictionary + Corpus) и `nltk` для
    токенизации/стемминга/стоп-слов.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        text_column: str,
        n_topics: int,
        *,
        # предобработка
        lowercase: bool = True,
        strip_extra_spaces: bool = True,
        remove_punctuation: bool = True,
        # стемминг/лемматизация: выбираем стемминг (Porter)
        stem: bool = True,
        lemmatize: bool = False,
        # опциональная фильтрация токенов
        min_token_length: int = 3,
        stopwords_lang: str = "english",
        min_df: float | int = 2,
        max_df: float | int = 0.5,
        min_count: int | None = None,
        max_count: int | None = None,
        # LDA
        random_state: int = 42,
        passes: int = 10,
        alpha: str | float = "auto",
        eta: str | float = "auto",
        max_features: int | None = None,
        # gensim Dictionary/Corpus ограничения (доп.)
        # max_features реализуем через filter_n_most_frequent
    ) -> None:
        if gensim is None or corpora is None or LdaModel is None:
            raise ImportError("Для LdaTopicModel требуется пакет gensim.")

        self.dataset = dataset
        self.text_column = text_column
        self.n_topics = int(n_topics)

        self.lowercase = lowercase
        self.strip_extra_spaces = strip_extra_spaces
        self.remove_punctuation = remove_punctuation
        self.stem = stem
        self.lemmatize = lemmatize

        self.min_token_length = int(min_token_length)
        self.stopwords_lang = stopwords_lang
        self.min_df = min_df
        self.max_df = max_df
        self.min_count = min_count
        self.max_count = max_count

        self.random_state = random_state
        self.passes = passes
        self.alpha = alpha
        self.eta = eta
        self.max_features = max_features

        self._gensim = gensim
        self._corpora = corpora
        self._LdaModel = LdaModel

        self._stopwords: set[str] | None = None
        self._stemmer: Any | None = None
        self._lemmatizer: Any | None = None

        self.dictionary: Any | None = None
        self.corpus: Any | None = None
        self.tokens: list[list[str]] | None = None
        self.model: Any | None = None

        self._prepared = False
        self._prepare()

    def _ensure_required_columns(self) -> None:
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError("dataset должен быть pandas.DataFrame")
        if self.text_column not in self.dataset.columns:
            raise ValueError(f"Отсутствует текстовая колонка: {self.text_column}")

    def _get_stopwords(self) -> set[str]:
        if self._stopwords is not None:
            return self._stopwords
        try:
            import nltk
            from nltk.corpus import stopwords

            try:
                _ = stopwords.words(self.stopwords_lang)
            except LookupError:
                nltk.download("stopwords", quiet=True)
            self._stopwords = set(stopwords.words(self.stopwords_lang))
        except Exception:
            self._stopwords = {
                "the",
                "a",
                "and",
                "to",
                "of",
                "in",
                "is",
                "it",
                "for",
                "on",
            }
        return self._stopwords

    def _get_stemmer(self):
        if self._stemmer is not None:
            return self._stemmer
        try:
            from nltk.stem import PorterStemmer

            self._stemmer = PorterStemmer()
        except Exception:
            self._stemmer = None
        return self._stemmer

    def _get_lemmatizer(self):
        if self._lemmatizer is not None:
            return self._lemmatizer
        try:
            from nltk.stem import WordNetLemmatizer
            import nltk

            try:
                nltk.data.find("corpora/wordnet")
            except LookupError:
                nltk.download("wordnet", quiet=True)
            self._lemmatizer = WordNetLemmatizer()
        except Exception:
            self._lemmatizer = None
        return self._lemmatizer

    def _normalize_text(self, text: str) -> str:
        text = text or ""
        if self.lowercase:
            text = text.lower()
        if self.strip_extra_spaces:
            text = re.sub(r"\s+", " ", text).strip()
        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)
        # Remove numbers
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        return [t for t in text.split(" ") if t]

    def _root_form(self, token: str) -> str:
        if self.lemmatize:
            lem = self._get_lemmatizer()
            if lem is not None:
                return lem.lemmatize(token)
        if self.stem:
            st = self._get_stemmer()
            if st is not None:
                return st.stem(token)
        return token

    def _preprocess_dataset(self) -> list[list[str]]:
        self._ensure_required_columns()
        stopwords_set = self._get_stopwords()
        tokens_all: list[list[str]] = []

        raw_series = self.dataset[self.text_column]
        texts = raw_series.fillna("").astype(str).tolist()
        for raw in texts:
            norm = self._normalize_text(raw)
            toks = self._tokenize(norm)
            toks = [t for t in toks if len(t) >= self.min_token_length]
            toks = [t for t in toks if t not in stopwords_set]
            toks = [self._root_form(t) for t in toks]
            toks = [t for t in toks if len(t) >= self.min_token_length]
            toks = [t for t in toks if t and t not in stopwords_set]
            tokens_all.append(toks)

        return tokens_all

    def _apply_token_frequency_filters(
        self, tokens: list[list[str]]
    ) -> list[list[str]]:
        """Частотные фильтры min_count/max_count не применяются отдельно.

        Всё фильтрование делается через `Dictionary.filter_extremes`.
        """

        if self.min_count is not None or self.max_count is not None:
            raise NotImplementedError(
                "min_count/max_count сейчас не поддерживаются отдельно: используйте min_df/max_df."
            )
        return tokens

    def _prepare(self) -> None:
        self.tokens = self._preprocess_dataset()
        self.tokens = self._apply_token_frequency_filters(self.tokens)

        # формируем Dictionary и corpus
        dictionary = self._corpora.Dictionary(self.tokens)

        # filter_extremes: min_df/max_df как в gensim (no_below/no_above)
        # В gensim это: no_below — минимальная частота по документам,
        # no_above — максимальная доля документов.
        no_below: int
        if isinstance(self.min_df, float):
            # min_df float трактуем как долю документов => переводим в count невозможно без n_docs.
            # Поэтому если float — приводим к int как safe fallback.
            no_below = max(1, int(self.min_df))
        else:
            no_below = int(self.min_df)

        no_above = self.max_df
        if isinstance(no_above, float):
            no_above = float(no_above)
        else:
            no_above = float(no_above)

        dictionary.filter_extremes(no_below=no_below, no_above=no_above)

        if self.max_features is not None:
            dictionary.filter_n_most_frequent(int(self.max_features))

        corpus = [dictionary.doc2bow(doc) for doc in self.tokens]

        self.dictionary = dictionary
        self.corpus = corpus
        self._prepared = True

    def fit(self) -> Any:
        if not self._prepared:
            self._prepare()
        if self.dictionary is None or self.corpus is None:
            raise RuntimeError("Словарь или corpus не сформированы")

        model = self._LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.n_topics,
            random_state=self.random_state,
            passes=self.passes,
            alpha=self.alpha,
            eta=self.eta,
        )
        self.model = model
        return model

    def get_top_words_per_topic(self, n_words: int = 10) -> dict[int, list[str]]:
        if self.model is None:
            raise RuntimeError("Сначала вызовите fit().")
        topics: dict[int, list[str]] = {}
        for topic_id in range(self.n_topics):
            words = self.model.show_topic(topic_id, topn=n_words)
            topics[topic_id] = [w for w, _ in words]
        return topics

    def transform(self, dataset: pd.DataFrame | None = None) -> np.ndarray:
        if self.model is None or self.dictionary is None:
            raise RuntimeError("Сначала вызовите fit().")

        if dataset is None:
            dataset = self.dataset
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("dataset должен быть pandas.DataFrame")
        if self.text_column not in dataset.columns:
            raise ValueError(f"Отсутствует текстовая колонка: {self.text_column}")

        stopwords_set = self._get_stopwords()
        texts = dataset[self.text_column].fillna("").astype(str).tolist()
        token_docs: list[list[str]] = []
        for raw in texts:
            norm = self._normalize_text(raw)
            toks = self._tokenize(norm)
            toks = [t for t in toks if len(t) >= self.min_token_length]
            toks = [t for t in toks if t not in stopwords_set]
            toks = [self._root_form(t) for t in toks]
            toks = [t for t in toks if len(t) >= self.min_token_length]
            toks = [t for t in toks if t and t not in stopwords_set]
            token_docs.append(toks)

        corpus = [self.dictionary.doc2bow(doc) for doc in token_docs]
        n_docs = len(corpus)
        mat = np.zeros((n_docs, self.n_topics), dtype=float)
        for i, bow in enumerate(corpus):
            doc_topics = self.model.get_document_topics(bow, minimum_probability=0.0)
            for tid, prob in doc_topics:
                mat[i, int(tid)] = float(prob)
        return mat

    def topics_price_correlation(
        self,
        price_column: str,
        *,
        topic_aggregation: Literal["distribution"] = "distribution",
        correlation_method: Literal["pearson", "spearman"] = "pearson",
        significance_test: bool = True,
        alpha: float = 0.05,
        n_permutations: int = 1000,
        random_state: int = 42,
        return_both_pvalues: bool = True,
    ) -> pd.DataFrame:
        """Корреляция вероятностей тем по документам с ценой."""

        if self.model is None or self.dictionary is None:
            raise RuntimeError("Сначала вызовите fit().")
        if price_column not in self.dataset.columns:
            raise ValueError(f"Отсутствует колонка цены: {price_column}")

        prices = pd.to_numeric(self.dataset[price_column], errors="coerce")
        mask = prices.notna().to_numpy(dtype=bool)
        prices = prices.to_numpy()[mask].astype(float)

        topic_mat = self.transform(self.dataset)[mask]

        from scipy.stats import pearsonr, spearmanr

        rng = np.random.default_rng(random_state)
        rows: list[dict[str, Any]] = []
        for topic_id in range(self.n_topics):
            x = topic_mat[:, topic_id]
            if correlation_method == "spearman":
                corr_val, p_val = spearmanr(x, prices)
            else:
                corr_val, p_val = pearsonr(x, prices)

            p_perm = np.nan
            do_perm = significance_test and n_permutations > 0
            if do_perm:
                # permutation: перемешиваем price
                perm_corrs = np.zeros(int(n_permutations), dtype=float)
                for k in range(int(n_permutations)):
                    y_perm = rng.permutation(prices)
                    if correlation_method == "spearman":
                        perm_corr, _ = spearmanr(x, y_perm)
                    else:
                        perm_corr, _ = pearsonr(x, y_perm)
                    perm_corrs[k] = perm_corr

                # two-sided p-value
                p_perm = float(
                    (np.sum(np.abs(perm_corrs) >= abs(float(corr_val))) + 1)
                    / (len(perm_corrs) + 1)
                )

            p_out = float(p_perm if do_perm else p_val)

            rows.append(
                {
                    "topic": int(topic_id),
                    "corr": float(corr_val),
                    "p_value": float(p_out),
                    "p_value_theoretical": float(p_val),
                    "p_value_permutation": (
                        float(p_perm) if np.isfinite(p_perm) else np.nan
                    ),
                    "significant": bool(p_out < alpha),
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values("p_value", ascending=True)
            .reset_index(drop=True)
        )

    def save(self, prefix: str) -> None:
        """Сохранение gensim-модели и словаря."""
        if self.model is None or self.dictionary is None:
            raise RuntimeError("Сначала вызовите fit().")
        self.model.save(prefix + ".lda")
        self.dictionary.save(prefix + ".dict")

        import json

        cfg = {
            "text_column": self.text_column,
            "n_topics": self.n_topics,
            "lowercase": self.lowercase,
            "strip_extra_spaces": self.strip_extra_spaces,
            "remove_punctuation": self.remove_punctuation,
            "stem": self.stem,
            "lemmatize": self.lemmatize,
            "min_token_length": self.min_token_length,
            "stopwords_lang": self.stopwords_lang,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "min_count": self.min_count,
            "max_count": self.max_count,
            "random_state": self.random_state,
            "passes": self.passes,
            "alpha": self.alpha,
            "eta": self.eta,
            "max_features": self.max_features,
            "empty_docs_ratio_": getattr(self, "empty_docs_ratio_", None),
        }
        with open(prefix + ".config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, prefix: str, dataset: pd.DataFrame | None = None) -> "LdaTopicModel":
        import json

        with open(prefix + ".config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        if dataset is None:
            dataset = pd.DataFrame({cfg["text_column"]: []})

        obj = cls(
            dataset=dataset,
            text_column=cfg["text_column"],
            n_topics=int(cfg["n_topics"]),
            lowercase=cfg["lowercase"],
            strip_extra_spaces=cfg["strip_extra_spaces"],
            remove_punctuation=cfg["remove_punctuation"],
            stem=cfg["stem"],
            lemmatize=cfg["lemmatize"],
            min_token_length=int(cfg["min_token_length"]),
            stopwords_lang=cfg["stopwords_lang"],
            min_df=cfg["min_df"],
            max_df=cfg["max_df"],
            min_count=cfg["min_count"],
            max_count=cfg["max_count"],
            random_state=int(cfg["random_state"]),
            passes=int(cfg["passes"]),
            alpha=cfg["alpha"],
            eta=cfg["eta"],
            max_features=cfg["max_features"],
        )

        # загружаем обученное
        from gensim import corpora
        from gensim.models.ldamodel import LdaModel

        obj.dictionary = corpora.Dictionary.load(prefix + ".dict")
        obj.model = LdaModel.load(prefix + ".lda")
        obj._prepared = True
        return obj
