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
    'hgt': 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/hgt.day.ltm.1991-2020.nc',
    'air': 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/air.day.ltm.1991-2020.nc',
}
LTM_VAR = {'u': 'uwnd', 'v': 'vwnd', 'hgt': 'hgt', 'air': 'air'}


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


def clim_uv250_daily(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Climatologia diaria de u/v 250 hPa (NCEP 1991-2020, 2.5°) — escoamento basico do WAF
    a 250 hPa (usada no s31 forecast). Mesma fonte/base/grade das LTMs de 200/850.

    Retorna (u_clim, v_clim, lat, lon) com u/v (n_dates, lat, lon) na grade da LTM.
    """
    path_u = _ensure_local_ltm('u', 250)
    path_v = _ensure_local_ltm('v', 250)
    u_clim, lat, lon = _select_by_doy(path_u, LTM_VAR['u'], dates)
    v_clim, _, _ = _select_by_doy(path_v, LTM_VAR['v'], dates)
    return u_clim, v_clim, lat, lon


def clim_u250_daily(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Climatologia diaria de u (zonal) 250 hPa — versao de UMA componente (3-tupla), para a
    anomalia do vento zonal 250 (campo wnd250_zonal_anom do globo 3D s38/s39).

    Mesma fonte/base/grade da `clim_uv250_daily`. Retorna (u_clim, lat, lon) na grade da LTM.
    """
    path_u = _ensure_local_ltm('u', 250)
    return _select_by_doy(path_u, LTM_VAR['u'], dates)


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


def clim_v850_daily(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Climatologia diaria de v (meridional) 850 para uma sequencia de datas.

    Mesma fonte/base/grade da `clim_u850_daily` — par (u850, v850) p/ a anomalia do vento 850
    (streamlines no campo t2m_wnd850 do s34). Retorna (v_clim, lat, lon) na grade da LTM.
    """
    path_v = _ensure_local_ltm('v', 850)
    return _select_by_doy(path_v, LTM_VAR['v'], dates)


def clim_hgt200_daily(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Climatologia diaria de altura geopotencial 200 hPa para uma sequencia de datas.

    Mesma fonte/base/grade (NCEP Reanalysis 1991-2020, 2.5°) e resolucao dia-a-dia
    das LTMs de u/v 200 — garante que a anomalia de hgt200 (WAF, Z200) use a MESMA
    climatologia de referencia das anomalias de u/v200 (consistencia do s34) e seja
    deslizavel por janela movel (diferente da composite PSL por intervalo).

    Retorna (hgt_clim, lat, lon) com hgt_clim (n_dates, lat, lon) na grade da LTM.
    """
    path_h = _ensure_local_ltm('hgt', 200)
    return _select_by_doy(path_h, LTM_VAR['hgt'], dates)


def clim_hgt250_daily(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Climatologia diaria de altura geopotencial 250 hPa (NCEP 1991-2020, 2.5°, gpm).

    Mesma fonte/base/grade das LTMs de 200/500/700 — usada na anomalia de Z250 (ex.: Hovmoller
    do s27). Retorna (hgt_clim, lat, lon) com hgt_clim (n_dates, lat, lon) na grade da LTM."""
    path_h = _ensure_local_ltm('hgt', 250)
    return _select_by_doy(path_h, LTM_VAR['hgt'], dates)


def clim_hgt500_daily(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Climatologia diaria de altura geopotencial 500 hPa para uma sequencia de datas.

    Mesma fonte/base/grade (NCEP Reanalysis 1991-2020, 2.5°) e resolucao dia-a-dia das
    LTMs de u/v/hgt 200 — garante que a anomalia de hgt500 (mapa Z500 do s34) use a MESMA
    climatologia de referencia, deslizavel por janela movel.

    Retorna (hgt_clim, lat, lon) com hgt_clim (n_dates, lat, lon) na grade da LTM.
    """
    path_h = _ensure_local_ltm('hgt', 500)
    return _select_by_doy(path_h, LTM_VAR['hgt'], dates)


def clim_hgt700_daily(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Climatologia diaria de altura geopotencial 700 hPa para uma sequencia de datas.

    Mesma fonte/base/grade (NCEP Reanalysis 1991-2020, 2.5°) das demais LTMs — usada pelo
    s35 (indice AAO) para formar a anomalia diaria de Z700 projetada no padrao EOF.

    Retorna (hgt_clim, lat, lon) com hgt_clim (n_dates, lat, lon) na grade da LTM.
    """
    path_h = _ensure_local_ltm('hgt', 700)
    return _select_by_doy(path_h, LTM_VAR['hgt'], dates)


def clim_t850_daily(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Climatologia diaria de temperatura 850 hPa (NCEP air.day.ltm 1991-2020).

    Mesma fonte/base/grade das LTMs de u/v/hgt 200 — consistencia do s34.
    Retorna (t_clim, lat, lon). Unidade conforme o arquivo NCEP (o s34 normaliza p/ °C).
    """
    path_t = _ensure_local_ltm('air', 850)
    return _select_by_doy(path_t, LTM_VAR['air'], dates)


# LTM diaria de SUPERFICIE (NCEP surface_gauss) — grade gaussiana T62 (~1.9°), nao 2.5°.
# Usada p/ T2m (temperatura do ar a 2 m). Reaproveita _select_by_doy; o regrid p/ 2.5° e
# feito em clim_t2m_daily (a serie do s34 trabalha na grade 2.5° das demais LTMs).
LTM_SURFACE_OPENDAP = {
    # air.2m.day.ltm.nc = LTM diaria de T2m, base 1991-2020 (mesma das LTMs de pressao), grade
    # gaussiana T62 (94x192). O PSL nao publica esse arquivo com a base no nome (so a generica).
    'air2m': 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface_gauss/air.2m.day.ltm.nc',
}
LTM_SURFACE_VAR = {'air2m': 'air'}
# Grade 2.5° padrao NCEP (a mesma das LTMs de pressao) — alvo do regrid do T2m.
_GRID25_LAT = np.arange(90.0, -92.5, -2.5)   # 90..-90 (descendente), 73 pontos
_GRID25_LON = np.arange(0.0, 360.0, 2.5)     # 0..357.5, 144 pontos


def _ensure_local_ltm_surface(component: str) -> Path:
    """Garante o NetCDF local da LTM diaria de SUPERFICIE (sem dim de nivel)."""
    out_dir = Path(settings.DIR_FILE_NC)
    out_dir.mkdir(parents=True, exist_ok=True)
    var = LTM_SURFACE_VAR[component]
    local = out_dir / f'{var}2m.day.ltm.1991-2020.nc'
    if local.exists():
        logger.info('LTM diaria superficie {} ja existe localmente: {}', component, local.name)
        return local
    url = LTM_SURFACE_OPENDAP[component]
    logger.info('Baixando LTM diaria superficie {} via OPeNDAP: {}', component, url)
    with xr.open_dataset(url, decode_times=False) as ds:
        da = ds[var]
        for d in ('level', 'nbnds'):  # remove dims espurias (surface tem level=1 as vezes)
            if d in da.dims:
                da = da.isel({d: 0}, drop=True)
        nt = int(ds.sizes['time'])
        step = 30  # OPeNDAP do PSL e instavel carregando os 365 tempos de uma vez -> chunks
        parts = [da.isel(time=slice(t0, t0 + step)).load() for t0 in range(0, nt, step)]
        da = xr.concat(parts, dim='time')
        t_units = ds['time'].attrs.get('units', 'hours since 1800-01-01 00:00:0.0')
        t_cal = ds['time'].attrs.get('calendar', 'standard')
    if not np.isfinite(da.values).any() or float(np.nanmax(np.abs(da.values))) == 0.0:
        raise RuntimeError(f'LTM superficie {component} veio toda zero/invalida (OPeNDAP PSL). Tente de novo.')
    out = xr.Dataset(
        {var: (('time', 'lat', 'lon'), np.asarray(da.values, dtype='float32'))},
        coords={'time': ('time', np.asarray(da['time'].values)),
                'lat': ('lat', np.asarray(da['lat'].values, dtype='float32')),
                'lon': ('lon', np.asarray(da['lon'].values, dtype='float32'))},
    )
    out['time'].attrs['units'] = t_units
    out['time'].attrs['calendar'] = t_cal
    out.to_netcdf(local)
    logger.info('LTM diaria superficie {} salva: {} ({} dias)', component, local.name, out.sizes.get('time', 0))
    return local


def clim_t2m_daily(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Climatologia diaria de temperatura do ar a 2 m (NCEP air.2m.day.ltm 1991-2020).

    A LTM vem na grade GAUSSIANA T62 (~1.9°); aqui e regridada (interp linear) para a grade
    2.5° padrao do s34 — a mesma das demais LTMs — para a anomalia ser consistente e a serie
    casar posicionalmente com os campos ja regridados a 2.5°. Retorna (t2m_clim, lat25, lon25)
    com lat25 descendente (90..-90), como as outras LTMs. Unidade conforme NCEP (s34 -> °C).
    """
    path = _ensure_local_ltm_surface('air2m')
    arr, lat_g, lon_g = _select_by_doy(path, LTM_SURFACE_VAR['air2m'], dates)
    da = xr.DataArray(
        arr, dims=('time', 'lat', 'lon'),
        coords={'time': np.arange(arr.shape[0]), 'lat': lat_g, 'lon': lon_g},
    ).sortby('lat')  # interp exige coord crescente
    da25 = da.interp(lat=_GRID25_LAT[::-1], lon=_GRID25_LON, method='linear',
                     kwargs={'fill_value': 'extrapolate'})
    da25 = da25.sortby('lat', ascending=False)  # volta p/ 90..-90 (como as outras LTMs)
    return da25.values, da25['lat'].values, da25['lon'].values
