# LdaTopicModel

Модуль `LdaTopicModel` реализует тематическое моделирование LDA (Latent Dirichlet Allocation) для текста с возможностью корреляции полученных тем с числовыми переменными (например, ценой аренды).

## Особенности

- Предобработка текста: приведение к нижнему регистру, удаление пунктуации, удаление лишних пробелов, стемминг (Porter) или лемматизация (WordNet).
- Фильтрация токенов по минимальной длине, удаление стоп-слова (поддерживаются языки через NLTK, с fallback на базовый набор английских стоп-слова).
- Формирование словаря и корпуса с использованием `gensim.corpora.Dictionary`.
- Фильтрация экстремальных частот через `min_df`/`max_df` (аналог `no_below`/`no_above` в gensim).
- Ограничение количества признаков через `max_features` (фильтрация самых частых токенов).
- Обучение LDA-модели через `gensim.models.LdaModel`.
- Получение топ-слов для каждой темы.
- Преобразование новых документов в пространство тем (распределение тем по документам).
- Вычисление корреляции (Пирсона или Спирмена) между вероятностями тем и числовой переменной (например, ценой) с опциональной проверкой значимости через перестановки.
- Сохранение и загрузка модели, словаря и конфигурации.

## Требования

- Python 3.12+
- pandas
- numpy
- gensim
- nltk (для стоп-слова, стемминга/лемматизации)
- scipy (для вычисления корреляции и p-значения)


## Использование

### Базовый пример

```python
import pandas as pd
from LdaTopicModel import LdaTopicModel

# Предположим, у нас есть DataFrame с колонкой 'description' содержащей текст объявлений
# и колонкой 'price' с ценой аренды
df = pd.read_csv('apartments.csv')

# Инициализация модели
lda_model = LdaTopicModel(
    dataset=df,
    text_column='description',
    n_topics=10,
    lowercase=True,
    strip_extra_spaces=True,
    remove_punctuation=True,
    stem=True,          # использовать стемминг Porter
    lemmatize=False,    # не использовать лемматизацию
    min_token_length=3,
    stopwords_lang='english',
    min_df=2,           # минимум в 2 документах
    max_df=0.5,         # максимум в 50% документов
    random_state=42,
    passes=10,
    alpha='auto',
    eta='auto',
    max_features=5000   # ограничить словарь 5000 самыми частыми токенами
)

# Обучение модели
lda_model.fit()

# Получение топ-слов для каждой темы
top_words = lda_model.get_top_words_per_topic(n_words=10)
for topic_id, words in top_words.items():
    print(f"Тема {topic_id}: {', '.join(words)}")

# Преобразование документов в пространство тем
topic_distribution = lda_model.transform(df)  # shape: (n_documents, n_topics)

# Корреляция тем с ценой
correlation_df = lda_model.topics_price_correlation(
    price_column='price',
    correlation_method='pearson',
    significance_test=True,
    alpha=0.05,
    n_permutations=1000
)
print(correlation_df)
```

### Сохранение и загрузка

После обучения модель можно сохранить:

```python
lda_model.save('/path/to/model/lda_model')
```

Это создаст три файла:
- `/path/to/model/lda_model.lda` —gensim модель LDA
- `/path/to/model/lda_model.dict` —gensim словарь
- `/path/to/model/lda_model.config.json` —конфигурация инициализации

Загрузка модели:

```python
loaded_model = LdaTopicModel.load('/path/to/model/lda_model', dataset=df)
# После загрузки можно сразу использовать transform и т.д., так как модель уже обучена
```

## Параметры инициализации

| Параметр | Тип | Описание |
|----------|-----|----------|
| `dataset` | pandas.DataFrame | Датасет, содержащий текстовый столбец и, опционально, другие столбцы (например, цена). |
| `text_column` | str | Название столбца с текстом для анализа. |
| `n_topics` | int | Количество тем для извлечения. |
| `lowercase` | bool, default=True | Приводить текст к нижнему регистру. |
| `strip_extra_spaces` | bool, default=True | Удалять лишние пробелы. |
| `remove_punctuation` | bool, default=True | Удалять знаки пунктуации. |
| `stem` | bool, default=True | Выполнять стемминг (PorterStemmer). |
| `lemmatize` | bool, default=False | Выполнять лемматизацию (WordNetLemmatizer). Если True, параметр `stem` игнорируется. |
| `min_token_length` | int, default=3 | Минимальная длина токена после предобработки. |
| `stopwords_lang` | str, default='english' | Язык стоп-слова для загрузки из NLTK. |
| `min_df` | float|int, default=2 | Минимальная частота токена по документам. Если float — доля документов (от 0 до 1). |
| `max_df` | float|int, default=0.5 | Максимальная доля документов, содержащих токен. Если float — доля (от 0 до 1). Если int — абсолютное количество документов. |
| `min_count` | int|None, default=None | Минимальная абсолютная частота токена (не используется, оставлено для совместимости). |
| `max_count` | int|None, default=None | Максимальная абсолютная частота токена (не используется, оставлено для совместимости). |
| `random_state` | int, default=42 | Seed для воспроизводимости. |
| `passes` | int, default=10 | Количество проходов по корпусу во время обучения LDA. |
| `alpha` | str|float, default='auto' | Параметр Dirichlet для распределения тем на документ. |
| `eta` | str|float, default='auto' | Параметр Dirichlet для распределения слов на тему. |
| `max_features` | int|None, default=None | Ограничить размер словаря топ-N самых частых токенов (после применения min_df/max_df). |

## Методы

### `fit()`
Обучает LDA-модель на подготовленном корпусе. Возвращает объект обученной модели gensim.

### `get_top_words_per_topic(n_words=10)`
Возвращает словарь, где ключ — ID темы, значение — список топ-слов для этой темы.

### `transform(dataset=None)`
Преобразует текстовые документы (из переданного датасета или исходного `self.dataset`) в матрицу распределений тем. Возвращает numpy array формы `(n_documents, n_topics)`.

### `topics_price_correlation(...)`
Вычисляет корреляцию между вероятностями тем по документам и числовой переменной (например, ценой). Возвращает DataFrame с колонками:
- `topic`: ID темы
- `corr`: коэффициент корреляции
- `p_value`: p-значение (пермутационное, если включено, иначе теоретическое)
- `p_value_theoretical`: теоретическое p-значение из теста Пирсона/Спирмена
- `p_value_permutation`: пермутационное p-значение (или NaN, если не вычислялось)
- `significant`: булево, значимо ли при заданном alpha

### `save(prefix)`
Сохраняет модель, словарь и конфигурацию в файлы с префиксом `prefix`.

### `load(prefix, dataset=None)`
Классметод для загрузки сохраненной модели. Требует передать датасет (может быть пустым DataFrame с нужной текстовыми колонкой) для инициализации объекта.

## Примечания

- Для работы стемминга/лемматизации требуется установленный NLTK и, возможно, загрузка дополнительных данных (punkt, wordnet, stopwords). При первом запуске модель попытается загрузить их автоматически (если доступно интернет-соединение).
- Если gensim не установлен, при инициализации будет поднято `ImportError`.
- Методы `fit`, `transform`, `topics_price_correlation` требуют предварительного вызова `fit()` (или загрузки через `load`).
- При фильтрации через `min_df`/`max_df` используется gensim's `filter_extremes` с параметрами `no_below` и `no_above`. Если `min_df` задан как float, он преобразуется в int как безопасный fallback (так как без знания общего числа документов невозможно вычислить абсолютную частоту). Аналогично для `max_df`.
- Параметры `min_count` и `max_count` пока не реализованы отдельно; используйте `min_df`/`max_df` для фильтрации по частоте.