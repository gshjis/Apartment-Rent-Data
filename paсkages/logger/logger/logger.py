from __future__ import annotations

import logging
import os
import sys
from typing import Final


_LEVEL_EMOJI: Final[dict[int, str]] = {
    logging.DEBUG: "🧩",
    logging.INFO: "ℹ️",
    logging.WARNING: "⚠️",
    logging.ERROR: "❌",
    logging.CRITICAL: "🚨",
}


class _EmojiFormatter(logging.Formatter):
    """Форматтер с эмодзи, временем и именем файла-источника."""

    default_time_format = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        emoji = _LEVEL_EMOJI.get(record.levelno, "🔔")

        # Имя файла-источника (без директорий). Если вдруг нет атрибутов — подстрахуемся.
        filename = getattr(record, "filename", None) or "unknown"

        # record.asctime уже выставляется базовым Formatter при вызове formatTime
        # (мы вызываем super().format(...) чтобы корректно сформировать asctime)
        base = super().format(record)

        # base начинается с формата: "{asctime} {level} ..." — мы добавим эмодзи и файл.
        # Чтобы не ломать остальной шаблон, вставляем вручную.
        # Пример base: "2026-... INFO [name] message"
        return f"{emoji} {filename}: {base}"


def get_logger(name: str | None = None) -> logging.Logger:
    """Создаёт/возвращает настроенный логгер.

    Требования:
    - эмодзи в начале сообщения
    - имя файла из которого сделан лог
    - разные эмодзи для уровней (info/warning/error/success)

    Для "успехов" используется уровень INFO, но вызывается через `log_success`.
    """

    logger_name = name or "apartment_rent_data"
    logger = logging.getLogger(logger_name)

    # Не дублируем хендлеры при повторных вызовах.
    if getattr(logger, "_emoji_configured", False):
        return logger

    level_str = os.getenv("LOG_LEVEL", "INFO").upper().strip()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)

    # Чтобы formatter мог показывать asctime
    handler.setFormatter(
        _EmojiFormatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt=_EmojiFormatter.default_time_format,
        )
    )

    # Не позволяем корневым логгерам дублировать вывод
    logger.propagate = False
    logger.addHandler(handler)
    logger._emoji_configured = True  # type: ignore[attr-defined]
    return logger


def log_success(logger: logging.Logger, message: str, *args, **kwargs) -> None:
    """Лог с эмодзи успеха, как отдельный метод.

    Реализовано через INFO, чтобы не требовать отдельный уровень.
    """

    # Вставляем эмодзи успеха в начало сообщения.
    # В отчете форматтер добавит ещё эмодзи уровня (ℹ️), но мы оставляем это как
    # наглядный маркер успеха.
    logger.info(f"✅ {message}", *args, **kwargs)


# Глобальный логгер по умолчанию, удобный для импортов
logger = get_logger(__name__)
