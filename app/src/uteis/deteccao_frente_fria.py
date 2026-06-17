# app/src/uteis/deteccao_frente_fria.py
# -*- coding: utf-8 -*-
"""
Detecção objetiva de passagens de frente fria por localidade (produto s36).

Método (mudança de vento, Simmonds et al. — Hemisfério Sul) + guarda térmica:
para cada localidade, amostra-se o ponto de grade mais próximo e procura-se, entre
dois tempos sinóticos consecutivos (6 h), o **giro da componente meridional do vento
para o quadrante sul** — `v` passa de `<= 0` (norte/quente, pré-frontal) para `> 0`
(sul/frio, pós-frontal) — com **`|Δv| > DV_MIN`** (default 2 m/s, Simmonds). Para
descartar giros de vento sem frente (brisa, convecção tropical), exige-se ainda uma
**queda da média diária de T2m** em torno do evento (`ΔT_diário < -DTEMP_MIN`), o que
neutraliza o ciclo diurno (não compara 6 h "quentes" com 6 h "frias" do mesmo dia).

Um **período refratário** (`REFRACTORY_H`, default 48 h) garante que uma única frente
conte uma vez por localidade.

Convenção de sinal: `v` é a componente meridional (para norte). `v > 0` = vento de
sul (ar frio vindo do sul, pós-frontal no HS); `v < 0` = vento de norte (pré-frontal).

Nada de download aqui — recebe um `xr.Dataset` já com `u10/v10/t2m[/msl]`
(ver `downloaders_era5_superficie.py`).
"""

from __future__ import annotations

# Bibliotecas padrão
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence

# Bibliotecas de terceiros
import numpy as np
import pandas as pd
import xarray as xr

# -----------------------------------------------------------------------------
# Limiares padrão (calibráveis)
# -----------------------------------------------------------------------------
DV_MIN = 2.0          # m/s — |Δv| mínimo em 6 h (giro p/ sul). Simmonds et al.
DTEMP_MIN = 1.0       # °C  — queda mínima da média diária de T2m em torno do evento
REFRACTORY_H = 48     # h   — 1 frente = 1 contagem por localidade
KELVIN = 273.15


# -----------------------------------------------------------------------------
# Estruturas de saída
# -----------------------------------------------------------------------------
@dataclass
class ResultadoLocalidade:
    """Contagem de frentes e datas dos eventos para uma localidade."""
    nome: str
    lat: float
    lon: float
    lat_grade: float
    lon_grade: float
    n_frentes: int
    eventos: List[pd.Timestamp] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Normalização do dataset
# -----------------------------------------------------------------------------
def _ensure_time_dim(ds: xr.Dataset) -> xr.Dataset:
    """Padroniza a dimensão temporal para 'time'."""
    if 'time' not in ds.dims and 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    if 'time' not in ds.coords and 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    if 'time' not in ds.dims:
        raise KeyError("Dataset sem dimensão temporal ('time'/'valid_time').")
    return ds.sortby('time')


def _steps_per_day(times: pd.DatetimeIndex) -> int:
    """Nº de passos por dia a partir do espaçamento mediano (6 h -> 4)."""
    if len(times) < 2:
        return 4
    diffs_h = np.diff(times.values).astype('timedelta64[m]').astype(float) / 60.0
    step_h = float(np.median(diffs_h))
    if step_h <= 0:
        return 4
    return max(1, int(round(24.0 / step_h)))


# -----------------------------------------------------------------------------
# Amostragem do ponto de grade mais próximo
# -----------------------------------------------------------------------------
def serie_no_ponto(ds: xr.Dataset, lat: float, lon: float) -> pd.DataFrame:
    """
    Extrai a série temporal (v10, t2m em °C, msl em hPa se houver) no ponto de
    grade mais próximo de (lat, lon). Retorna DataFrame indexado pelo tempo.
    """
    ds = _ensure_time_dim(ds)
    pt = ds.sel(latitude=lat, longitude=lon, method='nearest')

    times = pd.DatetimeIndex(pd.to_datetime(pt['time'].values))
    data = {'v': np.asarray(pt['v10'].values, dtype=float)}

    t2m = np.asarray(pt['t2m'].values, dtype=float)
    # ERA5 t2m em Kelvin; converte p/ °C se necessário (heurística robusta)
    if np.nanmean(t2m) > 100.0:
        t2m = t2m - KELVIN
    data['t2m'] = t2m

    if 'msl' in pt:
        msl = np.asarray(pt['msl'].values, dtype=float)
        if np.nanmean(msl) > 2000.0:  # Pa -> hPa
            msl = msl / 100.0
        data['msl'] = msl

    df = pd.DataFrame(data, index=times).sort_index()
    df.attrs['lat_grade'] = float(pt['latitude'].values)
    df.attrs['lon_grade'] = float(pt['longitude'].values)
    return df


# -----------------------------------------------------------------------------
# Detecção no ponto
# -----------------------------------------------------------------------------
def detectar_frentes(
    df: pd.DataFrame,
    *,
    dv_min: float = DV_MIN,
    dtemp_min: float = DTEMP_MIN,
    refractory_h: float = REFRACTORY_H,
    janela_inicio: Optional[datetime] = None,
    janela_fim: Optional[datetime] = None,
) -> List[pd.Timestamp]:
    """
    Detecta os instantes de passagem de frente fria na série de um ponto.

    Critério (entre t e t+6h):
      1. `v(t) <= 0` e `v(t+6h) > 0` (giro p/ quadrante sul) e `Δv > dv_min`
      2. queda da média diária de T2m: `mean(24h após) - mean(24h antes) < -dtemp_min`
      3. refratário: ignora novo evento por `refractory_h` após o último contado

    `janela_inicio`/`janela_fim` restringem QUAIS instantes de giro são contados
    (a guarda térmica ainda usa dados fora da janela, se existirem).
    """
    df = df.sort_index()
    times = df.index
    v = df['v'].to_numpy()
    t2 = df['t2m'].to_numpy()
    n = len(times)
    if n < 2:
        return []

    spd = _steps_per_day(times)  # passos por dia (4 p/ 6-horário)
    win_start = pd.Timestamp(janela_inicio) if janela_inicio is not None else None
    win_end = pd.Timestamp(janela_fim) if janela_fim is not None else None

    def _media_diaria(i: int) -> Optional[float]:
        """ΔT diário em torno do passo i: mean(próx. 24h) - mean(24h ant.)."""
        prev = t2[max(0, i - spd + 1): i + 1]
        nxt = t2[i + 1: i + 1 + spd]
        prev = prev[~np.isnan(prev)]
        nxt = nxt[~np.isnan(nxt)]
        if len(prev) < 2 or len(nxt) < 2:
            return None
        return float(np.mean(nxt) - np.mean(prev))

    eventos: List[pd.Timestamp] = []
    ultimo: Optional[pd.Timestamp] = None

    for i in range(n - 1):
        t = times[i]
        if win_start is not None and t < win_start:
            continue
        if win_end is not None and t > win_end:
            continue

        # espaçamento ~6 h (não atravessar lacuna no arquivo)
        dh = (times[i + 1] - t) / np.timedelta64(1, 'h')
        if dh > 6.5:
            continue

        # 1) giro do vento p/ sul
        dv = v[i + 1] - v[i]
        if not (v[i] <= 0 and v[i + 1] > 0 and dv > dv_min):
            continue

        # 2) guarda térmica (queda da média diária)
        drop = _media_diaria(i)
        if drop is None:
            # sem dados p/ média diária: cai no ΔT imediato de 6 h
            drop = t2[i + 1] - t2[i]
        if not (drop < -dtemp_min):
            continue

        # 3) refratário
        if ultimo is not None and (t - ultimo) / np.timedelta64(1, 'h') < refractory_h:
            continue

        eventos.append(pd.Timestamp(t))
        ultimo = pd.Timestamp(t)

    return eventos


# -----------------------------------------------------------------------------
# Contagem por lista de localidades
# -----------------------------------------------------------------------------
def contar_frentes_localidades(
    ds: xr.Dataset,
    localidades: Sequence[Sequence],
    *,
    dv_min: float = DV_MIN,
    dtemp_min: float = DTEMP_MIN,
    refractory_h: float = REFRACTORY_H,
    janela_inicio: Optional[datetime] = None,
    janela_fim: Optional[datetime] = None,
) -> List[ResultadoLocalidade]:
    """
    Conta passagens de frente fria para cada localidade `[lat, lon, nome]`.
    """
    ds = _ensure_time_dim(ds)
    resultados: List[ResultadoLocalidade] = []

    for loc in localidades:
        lat, lon, nome = float(loc[0]), float(loc[1]), str(loc[2])
        df = serie_no_ponto(ds, lat, lon)
        eventos = detectar_frentes(
            df,
            dv_min=dv_min,
            dtemp_min=dtemp_min,
            refractory_h=refractory_h,
            janela_inicio=janela_inicio,
            janela_fim=janela_fim,
        )
        resultados.append(
            ResultadoLocalidade(
                nome=nome,
                lat=lat,
                lon=lon,
                lat_grade=df.attrs.get('lat_grade', lat),
                lon_grade=df.attrs.get('lon_grade', lon),
                n_frentes=len(eventos),
                eventos=eventos,
            )
        )

    return resultados
