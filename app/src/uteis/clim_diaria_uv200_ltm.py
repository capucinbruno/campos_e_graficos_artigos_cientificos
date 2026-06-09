# -*- coding: utf-8 -*-
"""
Climatologia diaria (LTM) de u/v em 200 hPa — NCEP Reanalysis (PSL), base 1991-2020.

Usada pelo s31 (CHI200 intrasazonal) para obter a anomalia diaria
`u/v200(t) - LTM(dia-do-ano)`, base do metodo CPC de isolamento da intrasazonal.

A LTM tem 365 dias-do-ano (sem 29/fev). Baixa-se SOMENTE o nivel 200 hPa via
OPeNDAP (~15 MB por componente) e guarda-se um NetCDF local pequeno (cache).
Grade nativa NCEP: 2.5° (lat 90..-90, lon 0..357.5) — a mesma usada para o
calculo do potencial de velocidade (escala grande).
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
LTM_LOCAL = {'u': 'uwnd200.day.ltm.1991-2020.nc', 'v': 'vwnd200.day.ltm.1991-2020.nc'}
LTM_LEVEL_HPA = 200


def _ensure_local_ltm200(component: str) -> Path:
    """Garante o NetCDF local do LTM diario de 200 hPa (baixa o nivel 200 via OPeNDAP se faltar)."""
    out_dir = Path(settings.DIR_FILE_NC)
    out_dir.mkdir(parents=True, exist_ok=True)
    local = out_dir / LTM_LOCAL[component]
    if local.exists():
        logger.info('LTM diaria {}200 ja existe localmente: {}', component, local.name)
        return local

    url = LTM_OPENDAP[component]
    var = LTM_VAR[component]
    logger.info('Baixando LTM diaria {}200 (nivel 200) via OPeNDAP: {}', component, url)
    with xr.open_dataset(url, decode_times=False) as ds:
        # OPeNDAP/THREDDS do PSL: carregar os 365 tempos de uma vez e' instavel
        # (retorna zeros de forma nao-deterministica). Carregar em chunks de tempo
        # e concatenar e' confiavel. Tambem usar isel por indice de nivel.
        i200 = int(np.argmin(np.abs(ds['level'].values - LTM_LEVEL_HPA)))
        da_lvl = ds[var].isel(level=i200)
        nt = int(ds.sizes['time'])
        step = 30
        parts = [da_lvl.isel(time=slice(t0, t0 + step)).load() for t0 in range(0, nt, step)]
        da = xr.concat(parts, dim='time')
        t_units = ds['time'].attrs.get('units', 'hours since 1800-01-01 00:00:0.0')
        t_cal = ds['time'].attrs.get('calendar', 'standard')

    if not np.isfinite(da.values).any() or float(np.nanmax(np.abs(da.values))) == 0.0:
        raise RuntimeError(
            f'LTM {component}200 baixada veio toda zero/invalida (instabilidade do OPeNDAP PSL). '
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
    logger.info('LTM diaria {}200 salva: {} ({} dias)', component, local.name, out.sizes.get('time', 0))
    return local


def _doy_index_map(time_values, units: str, calendar: str = 'standard') -> dict[tuple[int, int], int]:
    """Mapa (mes, dia) -> indice do dia-do-ano, decodificando o tempo via cftime."""
    import cftime

    dates = cftime.num2date(time_values, units, calendar)
    return {(int(d.month), int(d.day)): i for i, d in enumerate(dates)}


def clim_uv200_daily(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Climatologia diaria de u/v 200 para uma sequencia de datas.

    Para cada data em `dates` (datetime64[D]), seleciona a fatia da LTM correspondente
    ao seu (mes, dia). 29/fev usa 28/fev (a LTM tem 365 dias).

    Retorna (u_clim, v_clim, lat, lon):
      - u_clim, v_clim: arrays (n_dates, lat, lon) na grade nativa 2.5° da LTM
      - lat, lon: coordenadas da grade da LTM
    """
    path_u = _ensure_local_ltm200('u')
    path_v = _ensure_local_ltm200('v')

    ds_u = xr.open_dataset(path_u, decode_times=False)
    ds_v = xr.open_dataset(path_v, decode_times=False)
    try:
        tvar = ds_u['time']
        idx_by_md = _doy_index_map(
            tvar.values, tvar.attrs['units'], tvar.attrs.get('calendar', 'standard')
        )
        lat = ds_u['lat'].values
        lon = ds_u['lon'].values
        u_all = ds_u[LTM_VAR['u']].values  # (365, lat, lon)
        v_all = ds_v[LTM_VAR['v']].values

        sel = []
        for dt64 in np.asarray(dates):
            d = np.datetime64(dt64, 'D').astype(object)
            md = (d.month, d.day)
            if md == (2, 29) and md not in idx_by_md:
                md = (2, 28)
            sel.append(idx_by_md[md])
        sel = np.asarray(sel, dtype=int)
        return u_all[sel], v_all[sel], lat, lon
    finally:
        ds_u.close()
        ds_v.close()
