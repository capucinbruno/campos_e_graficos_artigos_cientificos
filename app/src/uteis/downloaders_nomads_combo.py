# app/src/uteis/downloaders_nomads_combo.py
# -*- coding: utf-8 -*-
"""
Download COMBINADO dos modelos NOMADS (GFS/GEFS) para o s34 — UMA requisicao por passo.

Todas as variaveis que o s34 usa (u/v/HGT @ 200/250/500/850, TMP @ 850, T2m, ULWRF/OLR) estao
no MESMO arquivo GRIB do GFS/GEFS. Em vez de ~8 requisicoes separadas ao grib filter por passo
(uma por variavel) — que martelam o NOMADS e disparam o throttle (302) — aqui se faz **uma unica
requisicao** pedindo todos os niveis/variaveis, e o GRIB combinado e **fatiado** localmente nos
mesmos NetCDFs por variavel (mesmos nomes/dirs que os downloaders por-variavel), que o resto do
s34 le sem alteracao.

E uma CAMADA DE PRE-BUSCA (best-effort): o s34 chama o combo primeiro; os downloaders por-variavel
seguem como FALLBACK (cache-hit nos arquivos que o combo ja gravou; baixam individualmente o que
faltar). Pior caso = comportamento atual.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import xarray as xr

from app.common.forecast_download import (
    GRIB_NETCDF_LOCK,
    StepNotAvailable,
    download_days_parallel,
    save_netcdf,
)
from app.shared.logger import get_logger
from app.src.uteis.downloaders_gfs_olr import _open_gfs_olr  # parser ULWRF->olr (model-agnostico)

logger = get_logger(__name__)

# Niveis/variaveis combinados pedidos numa requisicao so (cross-product do grib filter).
_COMBO_LEVELS = ('lev_200_mb', 'lev_250_mb', 'lev_500_mb', 'lev_850_mb',
                 'lev_2_m_above_ground', 'lev_top_of_atmosphere')
_COMBO_VARS = ('var_UGRD', 'var_VGRD', 'var_HGT', 'var_TMP', 'var_ULWRF')


def _rename_latlon(ds: xr.Dataset) -> xr.Dataset:
    ren = {}
    for name in list(ds.dims) + list(ds.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in ds.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in ds.dims:
            ren[name] = 'lon'
    return ds.rename(ren) if ren else ds


def _slice2d(da: xr.DataArray, level: int):
    """Seleciona o nivel de pressao e limpa coords escalares, devolvendo 2D (lat, lon)."""
    if 'isobaricInhPa' not in da.coords and 'isobaricInhPa' not in da.dims:
        return None
    levs = np.round(np.atleast_1d(da['isobaricInhPa'].values)).astype(int)
    if level not in levs.tolist():
        return None
    da = da.sel(isobaricInhPa=level)
    for c in ('isobaricInhPa', 'time', 'step', 'valid_time', 'level'):
        if c in da.coords and c not in da.dims:
            da = da.drop_vars(c, errors='ignore')
    return da


def _extract_all(path: Path) -> dict:
    """Do GRIB combinado, devolve {tipo: Dataset 2D} para os tipos presentes (mesma estrutura dos
    downloaders por-variavel): fcst200(u,v,hgt), uv250(u,v), hgt250(hgt), hgt500(hgt),
    uv850(u,v), tmp850(t), t2m(t2m), olr(olr)."""
    out: dict = {}
    # --- niveis de pressao (u/v/gh/t com dim isobaricInhPa) ---
    ds_pl = None
    with GRIB_NETCDF_LOCK:
        try:
            ds_pl = xr.open_dataset(
                path, engine='cfgrib', backend_kwargs={'indexpath': ''},
                filter_by_keys={'typeOfLevel': 'isobaricInhPa'}).load()
        except Exception as exc:
            logger.warning('  combo: niveis de pressao nao abriram ({})', exc)
    if ds_pl is not None:
        ds_pl = _rename_latlon(ds_pl)
        u, v, gh, t = (ds_pl.get(k) for k in ('u', 'v', 'gh', 't'))
        u200, v200, g200 = (_slice2d(x, 200) if x is not None else None for x in (u, v, gh))
        if u200 is not None and v200 is not None and g200 is not None:
            out['fcst200'] = xr.Dataset({'u': u200, 'v': v200, 'hgt': g200})
        u250, v250 = (_slice2d(x, 250) if x is not None else None for x in (u, v))
        if u250 is not None and v250 is not None:
            out['uv250'] = xr.Dataset({'u': u250, 'v': v250})
        g250 = _slice2d(gh, 250) if gh is not None else None
        if g250 is not None:
            out['hgt250'] = g250.rename('hgt').to_dataset(name='hgt')
        g500 = _slice2d(gh, 500) if gh is not None else None
        if g500 is not None:
            out['hgt500'] = g500.rename('hgt').to_dataset(name='hgt')
        u850, v850 = (_slice2d(x, 850) if x is not None else None for x in (u, v))
        if u850 is not None and v850 is not None:
            out['uv850'] = xr.Dataset({'u': u850, 'v': v850})
        t850 = _slice2d(t, 850) if t is not None else None
        if t850 is not None:
            da = t850.rename('t')
            da.attrs['units'] = 'K'
            out['tmp850'] = da.to_dataset(name='t')
    # --- T2m (heightAboveGround=2) ---
    with GRIB_NETCDF_LOCK:
        try:
            ds_s = xr.open_dataset(
                path, engine='cfgrib', backend_kwargs={'indexpath': ''},
                filter_by_keys={'typeOfLevel': 'heightAboveGround'}).load()
            ds_s = _rename_latlon(ds_s)
            var = next((x for x in ('t2m', 't', '2t', 'tmp') if x in ds_s.data_vars), None)
            if var is not None:
                da = ds_s[var]
                if 'heightAboveGround' in da.dims:  # pode ter 2m e 10m -> pega 2m
                    da = da.sel(heightAboveGround=2)
                for c in ('heightAboveGround', 'time', 'step', 'valid_time', 'surface'):
                    if c in da.coords and c not in da.dims:
                        da = da.drop_vars(c, errors='ignore')
                da = da.rename('t2m')
                da.attrs['units'] = 'K'
                out['t2m'] = da.to_dataset(name='t2m')
        except Exception as exc:
            logger.warning('  combo: T2m nao abriu ({})', exc)
    # --- OLR (ULWRF topo) — reusa o parser do GFS ---
    try:
        out['olr'] = _open_gfs_olr(path)
    except Exception as exc:
        logger.warning('  combo: OLR nao abriu ({})', exc)
    return out


# tipo -> (dir_attr_suffix, nome no prefixo do arquivo). Os DIRs/prefixos vem do model_spec.
_TYPES = ('fcst200', 'uv250', 'hgt250', 'hgt500', 'uv850', 'tmp850', 't2m', 'olr')


def _nc_path(dirs: dict, prefixes: dict, typ: str, init: datetime, day: date) -> Path:
    return dirs[typ] / f'{prefixes[typ]}_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'


def _download_combo_day(
    init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool,
    *, file_fn: Callable, dir_fn: Callable, download_fn: Callable, grb_dir: Path,
    grb_prefix: str, dirs: dict, prefixes: dict,
) -> None:
    """Baixa o GRIB combinado de cada passo do dia (1 request/passo) e grava os NetCDFs por variavel."""
    # se TODOS os NetCDFs do dia ja existem, nao baixa nada
    if not force and all(_nc_path(dirs, prefixes, t, init, day).exists() for t in _TYPES):
        return
    grb_dir.mkdir(parents=True, exist_ok=True)
    acc = {t: [] for t in _TYPES}  # tipo -> lista de (vt, ds2d)
    for fhr, vt in steps:
        grb = grb_dir / f'{grb_prefix}_combo_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        params = {'file': file_fn(init, fhr), 'dir': dir_fn(init)}
        params.update({lv: 'on' for lv in _COMBO_LEVELS})
        params.update({vr: 'on' for vr in _COMBO_VARS})
        if not grb.exists() or force:
            try:
                download_fn(params, grb)
            except StepNotAvailable:
                logger.warning('  combo {} f{:03d} indisponivel (404/throttle) — pulando passo',
                               grb_prefix, fhr)
                continue
        fields = _extract_all(grb)
        for t, ds in fields.items():
            acc[t].append(ds.expand_dims(time=[np.datetime64(vt)]))
        if grb.exists():
            grb.unlink()
    for t, parts in acc.items():
        if not parts:
            continue
        nc = _nc_path(dirs, prefixes, t, init, day)
        if nc.exists() and not force:
            continue
        ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
        for d in (dirs[t],):
            d.mkdir(parents=True, exist_ok=True)
        save_netcdf(ds_day, nc)
    logger.info('combo {} {}: NetCDFs por variavel gravados (init {}Z)', grb_prefix, day, init.hour)


def _ensure_combo(
    init: datetime, lead_hours: int, hours: Sequence[int], force: bool,
    *, file_fn, dir_fn, download_fn, steps_fn, grb_dir, grb_prefix, dirs, prefixes,
) -> None:
    end = init + timedelta(hours=lead_hours)
    jobs = []
    day = init.date()
    while day <= end.date():
        steps = steps_fn(init, day, hours, lead_hours)
        if steps:
            jobs.append((day, steps))
        day += timedelta(days=1)
    download_days_parallel(
        jobs,
        lambda day, steps: _download_combo_day(
            init, day, steps, force, file_fn=file_fn, dir_fn=dir_fn, download_fn=download_fn,
            grb_dir=grb_dir, grb_prefix=grb_prefix, dirs=dirs, prefixes=prefixes),
        logger,
    )
    logger.info('COMBO {}: pre-busca concluida | init {:%Y-%m-%d %H}Z + {}h ({} dias)',
                grb_prefix, init, lead_hours, len(jobs))


# ---------------------------------------------------------------------------
# GFS
# ---------------------------------------------------------------------------
def ensure_gfs_combo_for_period(init, lead_hours, hours=None, force_redownload=False) -> None:
    """Pre-busca COMBINADA do GFS (1 request/passo) -> grava os NetCDFs por variavel do s34."""
    from app.src.uteis.downloaders_gfs_fcst200 import (
        DEFAULT_SYNOPTIC_HOURS, _download_grb2, _steps_for_day,
    )
    from app.src.uteis.downloaders_gfs_fcst200 import DIR_GFS_FCST200
    from app.src.uteis.downloaders_gfs_hgt250 import DIR_GFS_HGT250
    from app.src.uteis.downloaders_gfs_hgt500 import DIR_GFS_HGT500
    from app.src.uteis.downloaders_gfs_olr import DIR_GFS_OLR
    from app.src.uteis.downloaders_gfs_t2m import DIR_GFS_T2M
    from app.src.uteis.downloaders_gfs_tmp850 import DIR_GFS_TMP850
    from app.src.uteis.downloaders_gfs_uv250 import DIR_GFS_UV250
    from app.src.uteis.downloaders_gfs_uv850 import DIR_GFS_UV850
    hours = list(DEFAULT_SYNOPTIC_HOURS) if hours is None else list(hours)
    dirs = {'fcst200': DIR_GFS_FCST200, 'uv250': DIR_GFS_UV250, 'hgt250': DIR_GFS_HGT250,
            'hgt500': DIR_GFS_HGT500, 'uv850': DIR_GFS_UV850, 'tmp850': DIR_GFS_TMP850,
            't2m': DIR_GFS_T2M, 'olr': DIR_GFS_OLR}
    prefixes = {'fcst200': 'gfs_fcst200', 'uv250': 'gfs_uv250', 'hgt250': 'gfs_hgt250',
                'hgt500': 'gfs_hgt500', 'uv850': 'gfs_uv850', 'tmp850': 'gfs_tmp850',
                't2m': 'gfs_t2m', 'olr': 'gfs_olr'}
    _ensure_combo(
        init, lead_hours, hours, force_redownload,
        file_fn=lambda i, f: f'gfs.t{i.hour:02d}z.pgrb2.0p25.f{f:03d}',
        dir_fn=lambda i: f'/gfs.{i.strftime("%Y%m%d")}/{i.hour:02d}/atmos',
        download_fn=_download_grb2, steps_fn=_steps_for_day,
        grb_dir=DIR_GFS_FCST200, grb_prefix='gfs', dirs=dirs, prefixes=prefixes,
    )


# ---------------------------------------------------------------------------
# GEFS (media do ensemble geavg, pgrb2a 0.5°)
# ---------------------------------------------------------------------------
def ensure_gefs_combo_for_period(init, lead_hours, hours=None, force_redownload=False) -> None:
    """Pre-busca COMBINADA do GEFS (1 request/passo) -> grava os NetCDFs por variavel do s34."""
    from app.src.uteis.downloaders_gefs_fcst200 import (
        DEFAULT_SYNOPTIC_HOURS, _download_grb2, _gefs_dir, _gefs_file, _steps_for_day,
    )
    from app.src.uteis.downloaders_gefs_fcst200 import DIR_GEFS_FCST200
    from app.src.uteis.downloaders_gefs_hgt250 import DIR_GEFS_HGT250
    from app.src.uteis.downloaders_gefs_hgt500 import DIR_GEFS_HGT500
    from app.src.uteis.downloaders_gefs_olr import DIR_GEFS_OLR
    from app.src.uteis.downloaders_gefs_t2m import DIR_GEFS_T2M
    from app.src.uteis.downloaders_gefs_tmp850 import DIR_GEFS_TMP850
    from app.src.uteis.downloaders_gefs_uv250 import DIR_GEFS_UV250
    from app.src.uteis.downloaders_gefs_uv850 import DIR_GEFS_UV850
    hours = list(DEFAULT_SYNOPTIC_HOURS) if hours is None else list(hours)
    dirs = {'fcst200': DIR_GEFS_FCST200, 'uv250': DIR_GEFS_UV250, 'hgt250': DIR_GEFS_HGT250,
            'hgt500': DIR_GEFS_HGT500, 'uv850': DIR_GEFS_UV850, 'tmp850': DIR_GEFS_TMP850,
            't2m': DIR_GEFS_T2M, 'olr': DIR_GEFS_OLR}
    prefixes = {'fcst200': 'gefs_fcst200', 'uv250': 'gefs_uv250', 'hgt250': 'gefs_hgt250',
                'hgt500': 'gefs_hgt500', 'uv850': 'gefs_uv850', 'tmp850': 'gefs_tmp850',
                't2m': 'gefs_t2m', 'olr': 'gefs_olr'}
    _ensure_combo(
        init, lead_hours, hours, force_redownload,
        file_fn=_gefs_file, dir_fn=_gefs_dir,
        download_fn=_download_grb2, steps_fn=_steps_for_day,
        grb_dir=DIR_GEFS_FCST200, grb_prefix='gefs', dirs=dirs, prefixes=prefixes,
    )
