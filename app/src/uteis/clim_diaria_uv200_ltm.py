# -*- coding: utf-8 -*-
"""
Climatologia diaria (LTM) de vento — NCEP Reanalysis (PSL), base 1991-2020.

Fornece a LTM diaria (365 dias-do-ano, sem 29/fev) usada para anomalias verdadeiras
`vento(t) - LTM(dia-do-ano)`, base do metodo CPC de isolamento da intrasazonal:
  - u/v em 200 hPa     -> `clim_uv200_daily`  (s31: anomalia u/v200 para o chi200)
  - u (zonal) em 850 hPa -> `clim_u850_daily`  (s31: anomalia u850 do 3o Hovmoller)

A LTM vem do mesmo arquivo OPeNDAP do PSL (todos os niveis de pressao); seleciona-se
o nivel desejado e guarda-se um NetCDF local pequeno por nivel (cache). Grade nativa
NCEP: 2.5° (lat 90..-90, lon 0..357.5) — a mesma usada no calculo do potencial de
velocidade. Usar a mesma fonte/base/grade para 200 e 850 garante que a anomalia de
u850 seja consistente com a de u200 (mesma climatologia de referencia).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from app.shared.logger import get_logger
from app.shared.settings_factory import settings

logger = get_logger(__name__)

# OPeNDAP (THREDDS/PSL) — LTM diaria NCEP em niveis de pressao (todos os niveis)
LTM_OPENDAP = {
    'u': 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/uwnd.day.ltm.1991-2020.nc',
    'v': 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/vwnd.day.ltm.1991-2020.nc',
}
LTM_VAR = {'u': 'uwnd', 'v': 'vwnd'}


def _local_name(component: str, level_hpa: int) -> str:
    """Nome do cache local por componente+nivel (ex: uwnd200.day.ltm.1991-2020.nc)."""
    return f'{LTM_VAR[component]}{int(level_hpa)}.day.ltm.1991-2020.nc'


def _ensure_local_ltm(component: str, level_hpa: int) -> Path:
    """Garante o NetCDF local do LTM diario no nivel pedido (baixa o nivel via OPeNDAP se faltar)."""
    out_dir = Path(settings.DIR_FILE_NC)
    out_dir.mkdir(parents=True, exist_ok=True)
    local = out_dir / _local_name(component, level_hpa)
    if local.exists():
        logger.info('LTM diaria {}{} ja existe localmente: {}', component, level_hpa, local.name)
        return local

    url = LTM_OPENDAP[component]
    var = LTM_VAR[component]
    logger.info('Baixando LTM diaria {}{} (nivel {}) via OPeNDAP: {}', component, level_hpa, level_hpa, url)
    with xr.open_dataset(url, decode_times=False) as ds:
        # OPeNDAP/THREDDS do PSL: carregar os 365 tempos de uma vez e' instavel
        # (retorna zeros de forma nao-deterministica). Carregar em chunks de tempo
        # e concatenar e' confiavel. Tambem usar isel por indice de nivel.
        ilev = int(np.argmin(np.abs(ds['level'].values - level_hpa)))
        da_lvl = ds[var].isel(level=ilev)
        nt = int(ds.sizes['time'])
        step = 30
        parts = [da_lvl.isel(time=slice(t0, t0 + step)).load() for t0 in range(0, nt, step)]
        da = xr.concat(parts, dim='time')
        t_units = ds['time'].attrs.get('units', 'hours since 1800-01-01 00:00:0.0')
        t_cal = ds['time'].attrs.get('calendar', 'standard')

    if not np.isfinite(da.values).any() or float(np.nanmax(np.abs(da.values))) == 0.0:
        raise RuntimeError(
            f'LTM {component}{level_hpa} baixada veio toda zero/invalida (instabilidade do OPeNDAP PSL). '
            f'Tente novamente.'
        )

    # Reconstroi um dataset limpo (sem atributos herdados que conflitam na escrita)
    out = xr.Dataset(
        {var: (('time', 'lat', 'lon'), np.asarray(da.values, dtype='float32'))},
        coords={
            'time': ('time', np.asarray(da['time'].values)),
            'lat': ('lat', np.asarray(da['lat'].values, dtype='float32')),
            'lon': ('lon', np.asarray(da['lon'].values, dtype='float32')),
        },
    )
    out['time'].attrs['units'] = t_units
    out['time'].attrs['calendar'] = t_cal
    out.to_netcdf(local)
    logger.info('LTM diaria {}{} salva: {} ({} dias)', component, level_hpa, local.name, out.sizes.get('time', 0))
    return local


def _doy_index_map(time_values, units: str, calendar: str = 'standard') -> dict[tuple[int, int], int]:
    """Mapa (mes, dia) -> indice do dia-do-ano, decodificando o tempo via cftime."""
    import cftime

    dates = cftime.num2date(time_values, units, calendar)
    return {(int(d.month), int(d.day)): i for i, d in enumerate(dates)}


def _select_by_doy(
    path: Path, var: str, dates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seleciona a fatia da LTM (mes, dia) de cada data. 29/fev usa 28/fev.

    Retorna (campo (n_dates, lat, lon), lat, lon) na grade nativa da LTM.
    """
    ds = xr.open_dataset(path, decode_times=False)
    try:
        tvar = ds['time']
        idx_by_md = _doy_index_map(
            tvar.values, tvar.attrs['units'], tvar.attrs.get('calendar', 'standard')
        )
        lat = ds['lat'].values
        lon = ds['lon'].values
        arr = ds[var].values  # (365, lat, lon)

        sel = []
        for dt64 in np.asarray(dates):
            d = np.datetime64(dt64, 'D').astype(object)
            md = (d.month, d.day)
            if md == (2, 29) and md not in idx_by_md:
                md = (2, 28)
            sel.append(idx_by_md[md])
        sel = np.asarray(sel, dtype=int)
        return arr[sel], lat, lon
    finally:
        ds.close()


def clim_uv200_daily(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Climatologia diaria de u/v 200 para uma sequencia de datas.

    Para cada data em `dates` (datetime64[D]), seleciona a fatia da LTM correspondente
    ao seu (mes, dia). 29/fev usa 28/fev (a LTM tem 365 dias).

    Retorna (u_clim, v_clim, lat, lon):
      - u_clim, v_clim: arrays (n_dates, lat, lon) na grade nativa 2.5° da LTM
      - lat, lon: coordenadas da grade da LTM
    """
    path_u = _ensure_local_ltm('u', 200)
    path_v = _ensure_local_ltm('v', 200)
    u_clim, lat, lon = _select_by_doy(path_u, LTM_VAR['u'], dates)
    v_clim, _, _ = _select_by_doy(path_v, LTM_VAR['v'], dates)
    return u_clim, v_clim, lat, lon


def clim_u850_daily(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Climatologia diaria de u (zonal) 850 para uma sequencia de datas.

    Mesma fonte (NCEP Reanalysis), base (1991-2020), grade (2.5°) e resolucao
    dia-a-dia da LTM de 200 hPa — garante que a anomalia de u850 use a MESMA
    climatologia de referencia da anomalia de u200 (consistencia do s31).

    Retorna (u_clim, lat, lon) com u_clim (n_dates, lat, lon) na grade nativa da LTM.
    """
    path_u = _ensure_local_ltm('u', 850)
    return _select_by_doy(path_u, LTM_VAR['u'], dates)
