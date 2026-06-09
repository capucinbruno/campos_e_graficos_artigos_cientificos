# -*- coding: utf-8 -*-
"""
Boxes e indices climaticos de TSM (ENSO, IOD, AMO, TNA, TSA, SAD, PDO).

Centraliza o desenho dos boxes e o calculo/rotulagem dos indices sobre um mapa
global de anomalia de TSM, usado pelo s24 (mapa global de indices) e pela area
`globo` do s12. Mantem uma unica fonte de verdade para regioes, cores e valores.

Os boxes sao desenhados em coordenadas geograficas reais (transform=PlateCarree),
portanto independem da projecao. Os rotulos tambem usam coordenadas reais
(transform=PlateCarree), de modo que aparecem na posicao correta em qualquer
`central_longitude` do mapa.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Indice PDO (projecao da SSTA na EOF1)
# ---------------------------------------------------------------------------
PDO_EOF1_FILE = Path('Entrada/pdo_eof/EOF1.csv')


def interpolation_dataset(dataset, lst_lon: list, lst_lat: list):
    """Interpola o dataset para a grade especificada."""
    try:
        dataset = dataset.interp(lon=lst_lon, lat=lst_lat, method='linear')
        dataset = dataset.interpolate_na(dim='lon', method='linear', fill_value='extrapolate')
        dataset = dataset.interpolate_na(dim='lat', method='linear', fill_value='extrapolate')
    except ValueError:
        dataset = dataset.interp(longitude=lst_lon, latitude=lst_lat, method='linear')
        dataset = dataset.interpolate_na(dim='longitude', method='linear', fill_value='extrapolate')
    return dataset


def convert_dataset_to_df(ds_input: xr.Dataset, nome_variavel: str) -> pd.DataFrame:
    """Converte Dataset para DataFrame com colunas (lon, lat, variavel)."""
    df = ds_input.to_dataframe()
    df = (pd.DataFrame(df.index.to_list(), df[nome_variavel])).reset_index(drop=False)
    df = df[[0, 1, nome_variavel]]
    if min(df[0]) > min(df[1]):
        df.rename(columns={0: 'lat', 1: 'lon'}, inplace=True)
        df = df[['lon', 'lat', nome_variavel]]
    else:
        df.rename(columns={0: 'lon', 1: 'lat'}, inplace=True)
    return df.sort_values(by=['lon', 'lat']).reset_index(drop=True).round(2)


def calcula_indice_pdo(ds: xr.Dataset, name_var: str) -> float:
    """Calcula o indice PDO com base na projecao dos dados de SSTA na EOF1."""
    warnings.filterwarnings('ignore', category=UserWarning)

    df_eof1 = pd.read_csv(PDO_EOF1_FILE)

    lst_lon, lst_lat, lst_values_eof = [], [], []
    for _, row in df_eof1.iterrows():
        parts = row.iloc[0].split(';')
        lst_lon.append(float(parts[0]))
        lst_lat.append(float(parts[1]))
        lst_values_eof.append(float(parts[2]))

    lst_lon_eof1 = np.array(list(set(lst_lon)))
    lst_lat_eof1 = np.array(list(set(lst_lat)))

    df_eof = pd.DataFrame({'lat': lst_lat, 'lon': lst_lon, 'var1': lst_values_eof})

    X = np.empty((0, 3739))
    try:
        ds = ds.assign_coords(longitude=(ds.longitude % 360)).sortby('longitude')
    except Exception:
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby('lon')

    ds = interpolation_dataset(ds, lst_lon_eof1, lst_lat_eof1)
    df = convert_dataset_to_df(ds, name_var)
    df = df.dropna(axis=0)
    df.rename(columns={'lat': 'lon', 'lon': 'lat', name_var: 'anom'}, inplace=True)
    df_concat = pd.merge(df_eof, df, on=['lat', 'lon'], how='inner')

    linha_unica = df_concat[['anom']].values.flatten()
    X = np.concatenate([X, linha_unica.reshape(1, -1)], axis=0)

    media = np.mean(X[0])
    desvio = np.std(X[0])
    dados_pad = (X - media) / desvio

    eof1_reshape = np.array(lst_values_eof).reshape(-1, 1)
    indice = np.dot(dados_pad[0].reshape(1, -1), eof1_reshape).item()

    desvio_hist = 1101.995133699693
    media_hist = 1.4019186154496004e-14
    return (indice - media_hist) / desvio_hist


# ---------------------------------------------------------------------------
# Desenho dos boxes e rotulos dos indices
# ---------------------------------------------------------------------------
# zorder alto: boxes e rotulos sempre acima das linhas de continente/features (zorder~100)
_ZORDER_BOX = 1100       # boxes acima dos continentes
_ZORDER_BOX_TOP = 1110   # Nino 3.4: acima dos boxes vizinhos sobrepostos
_ZORDER_LABEL = 1200     # rotulos acima dos boxes


def _box(ax, lon_a, lon_b, lat_a, lat_b, color, zorder=_ZORDER_BOX, lw=2) -> None:
    ax.plot(
        [lon_a, lon_b, lon_b, lon_a, lon_a],
        [lat_a, lat_a, lat_b, lat_b, lat_a],
        color=color, linewidth=lw, zorder=zorder, transform=ccrs.PlateCarree(),
    )


def _label(ax, lon, lat, text, color, stroke_fg, zorder=_ZORDER_LABEL) -> None:
    # Rotulo em coordenadas reais (transform=PlateCarree) → independe da projecao.
    t = ax.text(
        lon, lat, text, fontsize=10, color=color, weight='bold',
        zorder=zorder, transform=ccrs.PlateCarree(),
    )
    t.set_path_effects([
        path_effects.Stroke(linewidth=3, foreground=stroke_fg),
        path_effects.Normal(),
    ])


def _mean(da: xr.DataArray, lon_a, lon_b, lat_a, lat_b) -> float:
    return float(da.sel(lon=slice(lon_a, lon_b), lat=slice(lat_a, lat_b)).mean(dim=('lon', 'lat')))


def desenha_boxes_indices(ax, da_average_data: xr.DataArray, index_pdo: float) -> None:
    """
    Desenha sobre `ax` os boxes e rotulos de todos os indices climaticos de TSM
    (IOD, Nino 1+2/3/3.4/4, AMO, TNA, TSA, SAD, PDO), calculando cada indice a
    partir de `da_average_data` (anomalia de TSM, dims lat/lon, lon em -180..180).

    `index_pdo` deve ser calculado antes via `calcula_indice_pdo` (depende da
    EOF1 na grade nativa, antes de aplicar ponto ciclico).

    Os rotulos usam coordenadas reais (lon convertida da projecao original do s24,
    central_longitude=220, para lon geografica), portanto aparecem corretamente
    em qualquer projecao global.
    """
    # ---- IOD = caixa oceanica oeste - caixa oriental leste ----
    _box(ax, 50, 70, -10, 10, 'black')
    _box(ax, 90, 110, -10, 0, 'black')
    index_iod = round(_mean(da_average_data, 50, 70, -10, 10)
                      - _mean(da_average_data, 90, 110, -10, 0), 2)
    _label(ax, 66.75, -18.64, f'IOD = {index_iod:.2f}', 'white', 'black')

    # ---- Nino 1+2 ----
    _box(ax, -90, -80, -10, 0, 'limegreen')
    val_12 = round(_mean(da_average_data, -90, -80, -10, 0), 2)
    _label(ax, -104.20, -17.64, f'Nino 1+2 = {val_12:.2f}', 'limegreen', 'black')

    # ---- Nino 3 ----
    _box(ax, -149.5, -90, -5, 5, 'blue')
    val_3 = round(_mean(da_average_data, -149.5, -90, -5, 5), 2)
    _label(ax, -135.99, 8.64, f'Nino 3 = {val_3:.2f}', 'blue', 'white')

    # ---- Nino 3.4 ----
    _box(ax, -170, -120, -5, 5, 'black', zorder=_ZORDER_BOX_TOP)
    val_34 = round(_mean(da_average_data, -170, -120, -5, 5), 2)
    _label(ax, -163.03, -11.64, f'Nino 3.4 = {val_34:.2f}', 'white', 'black')

    # ---- Nino 4 ----
    _box(ax, 160, 209.5, -5, 5, 'magenta')
    val_4 = round(_mean(da_average_data, 160, 209.5, -5, 5), 2)
    _label(ax, 168.35, 8.64, f'Nino 4 = {val_4:.2f}', 'magenta', 'black')

    # ---- AMO ----
    _box(ax, -70, -7, 1, 70, 'black')
    val_amo = round(_mean(da_average_data, -70, -7, 1, 70), 2)
    _label(ax, -51.75, 32.15, f'AMO = {val_amo:.2f}', 'black', 'white')

    # ---- TNA ----
    _box(ax, -55, -15, 5, 25, 'black')
    val_tna = round(_mean(da_average_data, -55, -15, 5, 25), 2)
    _label(ax, -48.75, 13.15, f'TNA = {val_tna:.2f}', 'black', 'white')

    # ---- TSA ----
    _box(ax, -30, 10, -20, 0, 'blue')
    val_tsa = round(_mean(da_average_data, -30, 10, -20, 0), 2)
    _label(ax, -23.75, -12.15, f'TSA = {val_tsa:.2f}', 'blue', 'white')

    # ---- SAD = caixa norte - caixa sul ----
    _box(ax, -20, 9, -15, -1, 'black')
    _box(ax, -40, -10, -40, -25, 'black')
    index_sad = round(_mean(da_average_data, -20, 9, -15, -1)
                      - _mean(da_average_data, -40, -10, -40, -25), 2)
    _label(ax, -24.75, -22.15, f'SAD = {index_sad:.2f}', 'white', 'black')

    # ---- PDO (indice calculado externamente via EOF1) ----
    _box(ax, 140, 240, 20, 60, 'black')
    _label(ax, 175.97, 37.00, f'PDO = {index_pdo:.2f}', 'black', 'white')
