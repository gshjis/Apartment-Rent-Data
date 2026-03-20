import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_coordinates(
    table: pd.DataFrame,
    price_col: str = "price",
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    cmap: str = "RdBu",
):
    """Строит scatter-плот по координатам (latitude/longitude).

    Точки окрашиваются по цене (колонка `price_col`):
    чем цена выше — точка краснее, чем ниже — синее.

    Args:
        table: Таблица с данными.
        price_col: Колонка с ценой для цветовой шкалы.
        lat_col: Колонка с широтой.
        lon_col: Колонка с долготой.
        cmap: Колormap (по умолчанию "RdBu").
    """

    if lat_col not in table.columns or lon_col not in table.columns:
        raise ValueError(f"Ожидаются колонки координат: '{lat_col}' и '{lon_col}'.")

    if price_col not in table.columns:
        raise ValueError(f"Колонка с ценой '{price_col}' отсутствует в таблице.")

    data = table[[lat_col, lon_col, price_col]].copy()
    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")

    values = np.array(data[price_col], dtype=float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmin == vmax:
        vmax = vmin + 1e-9

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sc = ax.scatter(
        data[lon_col],
        data[lat_col],
        c=values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
        edgecolors="none",
    )

    ax.set_xlabel(lon_col, fontsize=12)
    ax.set_ylabel(lat_col, fontsize=12)
    ax.set_title("Apartment Locations\nColored by price", fontsize=14)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(sc, ax=ax, label=price_col, fraction=0.046, pad=0.04)
    _ = cbar

    plt.tight_layout()
    plt.show()
