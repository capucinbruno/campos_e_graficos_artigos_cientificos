# -*- coding: utf-8 -*-
"""
Nucleo do CHI200 intrasazonal (metodo CPC de remocao de media movel).

Dado o vento anomalo diario em 200 hPa (`anom = u/v200 - LTM_diaria`), isola a
banda intrasazonal removendo a media dos N dias anteriores (default 120, padrao CPC):

    intra(t) = anom(t) - media(anom em [t-N, t-1])

e calcula o potencial de velocidade (chi200) de cada dia por inversao de Poisson
do vento divergente (reusa o solver de `plot_chi200`). A serie diaria de chi200
intrasazonal alimenta os tres produtos do s31: mapas de pentada, Hovmoller e mapa
do periodo.

Como chi e linear no vento, filtrar o vento no tempo e inverter Poisson e
equivalente a inverter Poisson a cada dia e filtrar o chi — usamos a 1a forma.
"""

from __future__ import annotations

import numpy as np

from app.src.uteis.plot_chi200 import _compute_divergence, _solve_poisson_sphere


def _chi_from_wind(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """chi (potencial de velocidade) de um campo de vento 2D, sem logs (uso em loop diario)."""
    div = _compute_divergence(u, v, lat, lon)
    return _solve_poisson_sphere(div, lat, lon)


def div_wind_from_chi(chi2d: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vento divergente (u_div, v_div) associado ao potencial de velocidade chi.

    u_div = (1 / a cos φ) ∂χ/∂λ
    v_div = (1/a) ∂χ/∂φ
    """
    a = 6.371e6
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    dchi_dphi = np.gradient(chi2d, lat_r, axis=0)
    dchi_dlam = np.gradient(chi2d, lon_r, axis=1)
    cos_lat = np.where(np.abs(np.cos(lat_r)) < 1e-10, np.nan, np.cos(lat_r))
    v_div = dchi_dphi / a
    u_div = dchi_dlam / (a * cos_lat[:, None])
    return u_div, v_div


def remove_media_movel(anom: np.ndarray, janela: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove a media movel TRAILING de `janela` dias da serie de anomalia.

    `anom`: (T, lat, lon), diario e contiguo (sem buracos).
    Retorna (intra, idx) onde:
      - intra: (T-janela, lat, lon) = anom(t) - media(anom[t-janela:t])
      - idx: indices (no eixo de tempo original) correspondentes a cada linha de intra
    """
    T = anom.shape[0]
    if T <= janela:
        raise ValueError(
            f'Serie diaria ({T} dias) menor que a janela da media movel ({janela}). '
            f'Aumente o periodo de download (precisa de >{janela} dias antes do alvo).'
        )
    out = np.empty((T - janela,) + anom.shape[1:], dtype=np.float64)
    for k, t in enumerate(range(janela, T)):
        out[k] = anom[t] - anom[t - janela:t].mean(axis=0)
    idx = np.arange(janela, T)
    return out, idx


def chi200_intrasazonal_series(
    u_anom: np.ndarray,
    v_anom: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    janela: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Serie diaria de chi200 intrasazonal e vento divergente associado.

    Retorna (chi_intra, u_div_series, v_div_series, idx):
      - chi_intra: (T-janela, lat, lon) — chi200 intrasazonal por dia
      - u_div_series, v_div_series: (T-janela, lat, lon) — vento divergente por dia
      - idx: indices no eixo de tempo original (para mapear de volta as datas)
    """
    u_is, idx = remove_media_movel(u_anom, janela)
    v_is, _ = remove_media_movel(v_anom, janela)
    chi = np.empty_like(u_is)
    u_div_ser = np.empty_like(u_is)
    v_div_ser = np.empty_like(v_is)
    for k in range(u_is.shape[0]):
        chi[k] = _chi_from_wind(u_is[k], v_is[k], lat, lon)
        u_div_ser[k], v_div_ser[k] = div_wind_from_chi(chi[k], lat, lon)
    return chi, u_div_ser, v_div_ser, idx


def media_faixa_latitude(
    campo: np.ndarray, lat: np.ndarray, lat_sul: float, lat_norte: float,
) -> np.ndarray:
    """Media de `campo` (..., lat, lon) numa faixa de latitude, ponderada por cos(lat).

    Retorna array (..., lon).
    """
    lo, hi = min(lat_sul, lat_norte), max(lat_sul, lat_norte)
    mask = (lat >= lo) & (lat <= hi)
    w = np.cos(np.deg2rad(lat[mask]))
    sub = campo[..., mask, :]
    return np.sum(sub * w[:, None], axis=-2) / np.sum(w)


def agrupa_pentadas(
    chi_series: np.ndarray, dates: np.ndarray, n_pentadas: int, *extras: np.ndarray,
) -> list[tuple]:
    """
    Agrupa as ULTIMAS `n_pentadas` pentadas (blocos de 5 dias, da mais recente p/ tras)
    e tira a media de chi em cada uma.

    `extras`: arrays adicionais (mesma forma de chi_series) cuja media de pentada e retornada
    junto com chi. Cada elemento de extras aparece como campo extra na tupla de retorno.

    Retorna lista [(dia_ini, dia_fim, campo_medio, *extras_medios), ...] da mais antiga p/ a mais recente.
    """
    T = chi_series.shape[0]
    blocos = []
    for p in range(n_pentadas):
        fim = T - p * 5
        ini = fim - 5
        if ini < 0:
            break
        extras_mean = tuple(e[ini:fim].mean(axis=0) for e in extras)
        blocos.append((dates[ini], dates[fim - 1], chi_series[ini:fim].mean(axis=0)) + extras_mean)
    return list(reversed(blocos))
