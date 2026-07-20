"""Helpers compartilhados de leitura/regrade de campos diarios e de resolucao de
inicializacoes de previsao (lagged ensemble).

Extraido do s34 para ser reusado por outros scripts (ex.: s35 AAO). As funcoes
publicas mantem a mesma assinatura/comportamento que tinham no s34:

    daily_uv200_on_grid(files, dt_ini, dt_fim, target_lat, target_lon, logger)
    daily_scalar_on_grid(files, candidates, dt_ini, dt_fim, target_lat, target_lon, logger)
    synoptic_scalar_on_grid(files, candidates, dt_ini, dt_fim, target_lat, target_lon, logger)
    resolve_run_inits(rodada, num_rodada, forecast_init)
    lagged_ensemble_mean(per_run)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from cartopy.util import add_cyclic_point as _add_cyclic_point

from app.shared.logger import get_logger

logger = get_logger('forecast_daily')

DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)


# ---------------------------------------------------------------------------
# Utilitarios de grade (padrao s31/s33/s34)
# ---------------------------------------------------------------------------
def _ensure_time_coord(obj):
    if hasattr(obj, 'dims') and 'time' not in obj.dims and 'valid_time' in obj.dims:
        obj = obj.rename({'valid_time': 'time'})
    elif hasattr(obj, 'coords') and 'time' not in obj.coords and 'valid_time' in obj.coords:
        obj = obj.rename({'valid_time': 'time'})
    if 'time' not in obj.coords:
        raise KeyError("Nem 'time' nem 'valid_time' encontrados.")
    return obj


def _rename_std_latlon(obj):
    ren = {}
    for name in list(obj.dims) + list(obj.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in obj.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in obj.dims:
            ren[name] = 'lon'
    return obj.rename(ren) if ren else obj


def _drop_expver(ds: xr.Dataset) -> xr.Dataset:
    if 'expver' in ds.dims:
        ds = ds.bfill('expver').ffill('expver').isel(expver=0, drop=True)
    if 'number' in ds.dims:
        ds = ds.isel(number=0, drop=True)
    for c in ('expver', 'number'):
        if c in ds.coords and c not in ds.dims:
            try:
                ds = ds.drop_vars(c)
            except Exception:
                pass
    return ds


def _sort_dedup_time(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.sortby('time')
    t = pd.DatetimeIndex(pd.to_datetime(ds['time'].values))
    _, idx = np.unique(t.values, return_index=True)
    if len(idx) != ds.sizes.get('time', 0):
        ds = ds.isel(time=np.sort(idx))
    return ds


def _find_uv_vars(ds: xr.Dataset) -> Tuple[str, str]:
    u_name = v_name = None
    for name in ('u', 'u_component_of_wind', 'U_GRD_L100', 'uwnd'):
        if name in ds.data_vars:
            u_name = name
            break
    for name in ('v', 'v_component_of_wind', 'V_GRD_L100', 'vwnd'):
        if name in ds.data_vars:
            v_name = name
            break
    if u_name is None or v_name is None:
        raise KeyError(f'u/v nao encontrados. Disponiveis: {list(ds.data_vars)}')
    return u_name, v_name


# ---------------------------------------------------------------------------
# Series diarias regridadas
# ---------------------------------------------------------------------------
def daily_uv200_on_grid(
    files, dt_ini: datetime, dt_fim: datetime,
    target_lat: np.ndarray, target_lon: np.ndarray, logger,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Le ERA5/GDAS, filtra horas sinoticas, media diaria, regrida p/ a grade da LTM.

    Mantem lon 0..360 (igual a LTM NCEP) para a anomalia ser consistente. Retorna
    (u_da, v_da) em (time, lat, lon), lat ascendente.
    """
    t_ini = np.datetime64(dt_ini.date())
    t_fim = np.datetime64(dt_fim.date())
    req = set(DEFAULT_SYNOPTIC_HOURS)
    tgt_lat = xr.DataArray(target_lat, dims=['lat'])
    tgt_lon = xr.DataArray(target_lon, dims=['lon'])

    us, vs = [], []
    for fp in files:
        ds = xr.open_dataset(fp, engine='netcdf4')
        try:
            ds = _sort_dedup_time(_rename_std_latlon(_drop_expver(_ensure_time_coord(ds))))
            ds = ds.assign_coords(lon=(ds['lon'] % 360)).sortby('lon')
            u_var, v_var = _find_uv_vars(ds)
            da_u, da_v = ds[u_var], ds[v_var]
            for dim in ('pressure_level', 'isobaricInhPa', 'level'):
                if dim in da_u.dims:
                    da_u = da_u.isel({dim: 0}, drop=True)
                    da_v = da_v.isel({dim: 0}, drop=True)
            ti = pd.DatetimeIndex(pd.to_datetime(da_u['time'].values))
            mh = np.array([h in req for h in ti.hour], dtype=bool)
            da_u, da_v = da_u.isel(time=mh), da_v.isel(time=mh)
            da_u = da_u.sel(time=slice(t_ini, t_fim))
            da_v = da_v.sel(time=slice(t_ini, t_fim))
            if da_u.sizes.get('time', 0) == 0:
                continue
            da_u = da_u.resample(time='1D').mean()
            da_v = da_v.resample(time='1D').mean()
            keep = ~((da_u['time'].dt.month == 2) & (da_u['time'].dt.day == 29))
            da_u, da_v = da_u.isel(time=keep.values), da_v.isel(time=keep.values)
            da_u = da_u.interp(lat=tgt_lat, lon=tgt_lon, method='linear').reset_coords(drop=True)
            da_v = da_v.interp(lat=tgt_lat, lon=tgt_lon, method='linear').reset_coords(drop=True)
            us.append(da_u.load())
            vs.append(da_v.load())
            logger.info('Serie diaria: {} -> {} dias', fp.name, da_u.sizes['time'])
        finally:
            ds.close()

    if not us:
        raise RuntimeError('Nenhum dado u/v 200 valido no periodo.')

    u_da = xr.concat(us, dim='time', coords='minimal', compat='override').sortby('time')
    v_da = xr.concat(vs, dim='time', coords='minimal', compat='override').sortby('time')
    _, uniq = np.unique(u_da['time'].values, return_index=True)
    return u_da.isel(time=uniq), v_da.isel(time=uniq)


def daily_scalar_on_grid(
    files, candidates, dt_ini: datetime, dt_fim: datetime,
    target_lat: np.ndarray, target_lon: np.ndarray, logger,
) -> xr.DataArray:
    """Serie diaria de um escalar (hgt/olr/tmp) regridada p/ a grade da LTM. lat ascendente.

    `candidates` = nomes possiveis da variavel no arquivo (1o encontrado e usado).
    """
    t_ini, t_fim = np.datetime64(dt_ini.date()), np.datetime64(dt_fim.date())
    req = set(DEFAULT_SYNOPTIC_HOURS)
    tgt_lat = xr.DataArray(target_lat, dims=['lat'])
    tgt_lon = xr.DataArray(target_lon, dims=['lon'])
    hs = []
    for fp in files:
        ds = xr.open_dataset(fp, engine='netcdf4')
        try:
            ds = _sort_dedup_time(_rename_std_latlon(_drop_expver(_ensure_time_coord(ds))))
            ds = ds.assign_coords(lon=(ds['lon'] % 360)).sortby('lon')
            hname = next((v for v in candidates if v in ds.data_vars), None)
            if hname is None:
                continue
            da = ds[hname]
            for dim in ('pressure_level', 'isobaricInhPa', 'level'):
                if dim in da.dims:
                    da = da.isel({dim: 0}, drop=True)
            ti = pd.DatetimeIndex(pd.to_datetime(da['time'].values))
            da = da.isel(time=np.array([h in req for h in ti.hour], dtype=bool))
            da = da.sel(time=slice(t_ini, t_fim))
            if da.sizes.get('time', 0) == 0:
                continue
            da = da.resample(time='1D').mean()
            keep = ~((da['time'].dt.month == 2) & (da['time'].dt.day == 29))
            da = da.isel(time=keep.values)
            da = da.interp(lat=tgt_lat, lon=tgt_lon, method='linear').reset_coords(drop=True)
            hs.append(da.load())
        finally:
            ds.close()
    if not hs:
        raise RuntimeError(f'Nenhum dado valido no periodo para {candidates}.')
    h_da = xr.concat(hs, dim='time', coords='minimal', compat='override').sortby('time')
    _, uniq = np.unique(h_da['time'].values, return_index=True)
    return h_da.isel(time=uniq)


def synoptic_scalar_on_grid(
    files, candidates, dt_ini: datetime, dt_fim: datetime,
    target_lat: np.ndarray, target_lon: np.ndarray, logger,
) -> xr.DataArray:
    """Serie NA HORA SINOTICA (00/06/12/18 UTC, sem resample diario) de um escalar (hgt/olr/tmp)
    regridada p/ a grade da LTM. lat ascendente. Irma de `daily_scalar_on_grid`, mas mantem cada
    passo sinotico como um indice de tempo proprio (usado pelo MP4 do s42 em vez da media diaria).

    `candidates` = nomes possiveis da variavel no arquivo (1o encontrado e usado).
    """
    t_ini = np.datetime64(dt_ini.date())
    # Limite superior estende ate 23h do ultimo dia -- senao o slice cortaria os passos
    # 06/12/18Z de `dt_fim` (que e so a DATA, meia-noite), deixando o ultimo dia so com o 00Z.
    t_fim = np.datetime64(dt_fim.date()) + np.timedelta64(23, 'h')
    req = set(DEFAULT_SYNOPTIC_HOURS)
    tgt_lat = xr.DataArray(target_lat, dims=['lat'])
    tgt_lon = xr.DataArray(target_lon, dims=['lon'])
    hs = []
    for fp in files:
        ds = xr.open_dataset(fp, engine='netcdf4')
        try:
            ds = _sort_dedup_time(_rename_std_latlon(_drop_expver(_ensure_time_coord(ds))))
            ds = ds.assign_coords(lon=(ds['lon'] % 360)).sortby('lon')
            hname = next((v for v in candidates if v in ds.data_vars), None)
            if hname is None:
                continue
            da = ds[hname]
            for dim in ('pressure_level', 'isobaricInhPa', 'level'):
                if dim in da.dims:
                    da = da.isel({dim: 0}, drop=True)
            ti = pd.DatetimeIndex(pd.to_datetime(da['time'].values))
            da = da.isel(time=np.array([h in req for h in ti.hour], dtype=bool))
            da = da.sel(time=slice(t_ini, t_fim))
            if da.sizes.get('time', 0) == 0:
                continue
            keep = ~((da['time'].dt.month == 2) & (da['time'].dt.day == 29))
            da = da.isel(time=keep.values)
            da = da.interp(lat=tgt_lat, lon=tgt_lon, method='linear').reset_coords(drop=True)
            hs.append(da.load())
        finally:
            ds.close()
    if not hs:
        raise RuntimeError(f'Nenhum dado sinotico valido no periodo para {candidates}.')
    h_da = xr.concat(hs, dim='time', coords='minimal', compat='override').sortby('time')
    _, uniq = np.unique(h_da['time'].values, return_index=True)
    h_da = h_da.isel(time=uniq)
    logger.info('Serie sinotica: {} passos (00/06/12/18Z) de {} a {}',
                h_da.sizes['time'], dt_ini.date(), dt_fim.date())
    return h_da


def native_scalar_on_grid(
    files, candidates, dt_ini: datetime, dt_fim: datetime,
    target_lat: np.ndarray, target_lon: np.ndarray, logger,
) -> xr.DataArray:
    """Serie no PASSO NATIVO do modelo (SEM filtro de hora sinotica, sem resample diario) de um
    escalar (tp/ptype/msl) regridada p/ a grade-alvo. Irma de `synoptic_scalar_on_grid`, mas
    mantem TODO passo presente no arquivo (3h/6h nao-uniforme) -- usada por series que animam no
    cadenciamento nativo (ex.: chuva/neve por tipo do ptype ECMWF)."""
    t_ini = np.datetime64(dt_ini.date())
    t_fim = np.datetime64(dt_fim.date()) + np.timedelta64(23, 'h')
    tgt_lat = xr.DataArray(target_lat, dims=['lat'])
    tgt_lon = xr.DataArray(target_lon, dims=['lon'])
    hs = []
    for fp in files:
        ds = xr.open_dataset(fp, engine='netcdf4')
        try:
            ds = _sort_dedup_time(_rename_std_latlon(_drop_expver(_ensure_time_coord(ds))))
            ds = ds.assign_coords(lon=(ds['lon'] % 360)).sortby('lon')
            hname = next((v for v in candidates if v in ds.data_vars), None)
            if hname is None:
                continue
            da = ds[hname]
            for dim in ('pressure_level', 'isobaricInhPa', 'level'):
                if dim in da.dims:
                    da = da.isel({dim: 0}, drop=True)
            da = da.sel(time=slice(t_ini, t_fim))
            if da.sizes.get('time', 0) == 0:
                continue
            # .astype(float32): o interp() promove p/ float64 por padrao -- a fonte (ECMWF/GFS) ja e
            # float32, dobrar a RAM da serie sem ganho de precisao (era um dos gargalos do s46).
            da = (da.interp(lat=tgt_lat, lon=tgt_lon, method='linear')
                  .astype('float32').reset_coords(drop=True))
            hs.append(da.load())
        finally:
            ds.close()
    if not hs:
        raise RuntimeError(f'Nenhum dado no passo nativo valido no periodo para {candidates}.')
    h_da = xr.concat(hs, dim='time', coords='minimal', compat='override').sortby('time')
    _, uniq = np.unique(h_da['time'].values, return_index=True)
    h_da = h_da.isel(time=uniq)
    logger.info('Serie passo nativo: {} passo(s) de {} a {}',
                h_da.sizes['time'], dt_ini.date(), dt_fim.date())
    return h_da


_MSL_VARS = ('msl', 'mean_sea_level_pressure', 'prmsl', 'PRMSL', 'psl', 'sp')


def daily_mslp_on_grid(
    files, dt_ini: datetime, dt_fim: datetime,
    target_lat: np.ndarray, target_lon: np.ndarray, logger,
) -> xr.DataArray:
    """Serie diaria de MSLP (hPa) regridada p/ grade alvo.

    Lê arquivos NetCDF com variavel msl (ERA5), filtra horas sinoticas,
    calcula a media diaria e retorna MSLP em hPa em (time, lat, lon).
    """
    t_ini = np.datetime64(dt_ini.date())
    t_fim = np.datetime64(dt_fim.date())
    req = set(DEFAULT_SYNOPTIC_HOURS)
    tgt_lat = xr.DataArray(target_lat, dims=['lat'])
    tgt_lon = xr.DataArray(target_lon, dims=['lon'])
    slices = []
    for fp in files:
        ds = xr.open_dataset(fp, engine='netcdf4')
        try:
            ds = _sort_dedup_time(_rename_std_latlon(_drop_expver(_ensure_time_coord(ds))))
            ds = ds.assign_coords(lon=(ds['lon'] % 360)).sortby('lon')
            msl_var = next((v for v in _MSL_VARS if v in ds.data_vars), None)
            if msl_var is None:
                continue
            da = ds[msl_var]
            ti = pd.DatetimeIndex(pd.to_datetime(da['time'].values))
            mask = np.array([h in req for h in ti.hour], dtype=bool)
            da = da.isel(time=mask).sel(time=slice(t_ini, t_fim))
            if da.sizes.get('time', 0) == 0:
                continue
            if float(da.mean()) > 10000:  # Pa → hPa
                da = da / 100.0
            da = da.resample(time='1D').mean()
            keep = ~((da['time'].dt.month == 2) & (da['time'].dt.day == 29))
            da = da.isel(time=keep.values)
            _lon_rounded = np.round(da['lon'].values, 6)
            _cyc, _lon_cyc = _add_cyclic_point(da.values, coord=_lon_rounded)
            da = xr.DataArray(
                _cyc, dims=['time', 'lat', 'lon'],
                coords={'time': da['time'].values, 'lat': da['lat'].values, 'lon': _lon_cyc},
            )
            # .astype(float32): mesmo motivo do `native_scalar_on_grid` acima -- interp() promove
            # p/ float64 sem necessidade (hPa nao precisa dessa precisao).
            da = (da.interp(lat=tgt_lat, lon=tgt_lon, method='linear')
                  .astype('float32').reset_coords(drop=True))
            slices.append(da.load())
        finally:
            ds.close()
    if not slices:
        raise RuntimeError('Nenhum dado msl valido no periodo para MSLP.')
    mslp = xr.concat(slices, dim='time', coords='minimal', compat='override').sortby('time')
    _, uniq = np.unique(mslp['time'].values, return_index=True)
    mslp = mslp.isel(time=uniq)
    mslp.name = 'mslp'
    mslp.attrs['units'] = 'hPa'
    return mslp


def daily_wind_speed_on_grid(
    files, dt_ini: datetime, dt_fim: datetime,
    target_lat: np.ndarray, target_lon: np.ndarray, logger,
) -> xr.DataArray:
    """Serie diaria de magnitude do vento (sqrt(u²+v²)) regridada p/ grade alvo.

    Lê arquivos NetCDF com variaveis u/v, filtra horas sinoticas, calcula a
    media diaria de cada componente e retorna sqrt(u²+v²) em (time, lat, lon).
    """
    t_ini = np.datetime64(dt_ini.date())
    t_fim = np.datetime64(dt_fim.date())
    req = set(DEFAULT_SYNOPTIC_HOURS)
    tgt_lat = xr.DataArray(target_lat, dims=['lat'])
    tgt_lon = xr.DataArray(target_lon, dims=['lon'])

    speeds = []
    for fp in files:
        ds = xr.open_dataset(fp, engine='netcdf4')
        try:
            ds = _sort_dedup_time(_rename_std_latlon(_drop_expver(_ensure_time_coord(ds))))
            ds = ds.assign_coords(lon=(ds['lon'] % 360)).sortby('lon')
            u_var, v_var = _find_uv_vars(ds)
            u_da, v_da = ds[u_var], ds[v_var]
            for dim in ('pressure_level', 'isobaricInhPa', 'level'):
                if dim in u_da.dims:
                    u_da = u_da.isel({dim: 0}, drop=True)
                    v_da = v_da.isel({dim: 0}, drop=True)
            ti = pd.DatetimeIndex(pd.to_datetime(u_da['time'].values))
            mask = np.array([h in req for h in ti.hour], dtype=bool)
            u_da, v_da = u_da.isel(time=mask), v_da.isel(time=mask)
            u_da = u_da.sel(time=slice(t_ini, t_fim))
            v_da = v_da.sel(time=slice(t_ini, t_fim))
            if u_da.sizes.get('time', 0) == 0:
                continue
            speed = np.sqrt(u_da ** 2 + v_da ** 2)
            speed = speed.resample(time='1D').mean()
            keep = ~((speed['time'].dt.month == 2) & (speed['time'].dt.day == 29))
            speed = speed.isel(time=keep.values)
            # Adiciona ponto cíclico ANTES do interp para fechar o seam em 0°/360°
            # (mesma correção aplicada à climatologia em _anom_from_clim).
            _spd_cyc, _lon_cyc = _add_cyclic_point(speed.values, coord=np.round(speed['lon'].values, 6))
            speed = xr.DataArray(
                _spd_cyc, dims=['time', 'lat', 'lon'],
                coords={'time': speed['time'].values, 'lat': speed['lat'].values, 'lon': _lon_cyc},
            )
            speed = speed.interp(lat=tgt_lat, lon=tgt_lon, method='linear').reset_coords(drop=True)
            speeds.append(speed.load())
            logger.info('Wind speed: {} -> {} dias', str(fp).rsplit('/', 1)[-1], speed.sizes['time'])
        finally:
            ds.close()

    if not speeds:
        raise RuntimeError('Nenhum dado u/v valido no periodo para wind speed.')
    spd_da = xr.concat(speeds, dim='time', coords='minimal', compat='override').sortby('time')
    _, uniq = np.unique(spd_da['time'].values, return_index=True)
    spd_da = spd_da.isel(time=uniq)
    spd_da.name = 'wind_speed'
    spd_da.attrs['units'] = 'm/s'
    return spd_da


# ---------------------------------------------------------------------------
# Resolucao de inicializacoes de previsao (lagged ensemble)
# ---------------------------------------------------------------------------
def _parse_forecast_init(spec: str, rodada: int) -> datetime:
    """Interpreta a data da rodada de FORECAST_INIT na hora `rodada`.

    Formato preferido: data ISO 'YYYY-MM-DD' (ex.: 2026-06-10) — a hora vem da RODADA.
    Tambem aceita, por compatibilidade, o timestamp completo 'YYYYMMDDHH' (ex.: 2026061000).
    """
    s = spec.strip()
    try:
        d = datetime.fromisoformat(s)  # aceita 'YYYY-MM-DD' (e variantes ISO com hora)
        # data pura (sem hora) -> aplica a hora da RODADA
        if 'T' not in s and ':' not in s:
            d = d.replace(hour=rodada)
        return d
    except ValueError:
        pass
    return datetime.strptime(s, '%Y%m%d%H')  # compatibilidade YYYYMMDDHH


def resolve_run_inits(rodada: int, num_rodada: int, forecast_init) -> list:
    """Lista de ciclos de inicializacao (mais recente primeiro), todos na hora `rodada`.

    forecast_init vazio/None/'latest' -> ciclo `rodada` mais recente disponivel (now-6h).
    Caso contrario, usa a data informada como init0 (ver `_parse_forecast_init`).
    """
    spec = str(forecast_init or '').strip().lower()
    if spec in ('', 'latest'):
        now = datetime.utcnow() - timedelta(hours=6)  # folga p/ latencia do GFS
        init0 = datetime(now.year, now.month, now.day, rodada)
        if init0 > now:
            init0 -= timedelta(days=1)
    else:
        init0 = _parse_forecast_init(spec, rodada)
    return [init0 - timedelta(days=k) for k in range(max(1, num_rodada))]


GEFS_MAX_LEAD_DAYS = 35       # o GEFS so estende ate 35 dias (e so no ciclo 00Z)
GEFS_SAMEDAY_MAX_DAYS = 16    # ate 16d a rodada de HOJE ja esta publicada; acima precisa do D-1
OTHER_MODELS_LEAD_DAYS = 16   # GFS/ECMWF/AIFS/etc.: alcance proprio (~15-16d), DESVINCULADO do settings


def resolve_forecast_lead_init(model: str, *, rodada: int, num_rodada: int, forecast_init,
                               gefs_lead_days: int, cfs_lead_days: int):
    """(run_inits, lead_hours) com HORIZONTE e INIT proprios de cada modelo.

    - **cfs**: sempre `cfs_lead_days` (45, pseudo-ensemble subsazonal); 1 init = D-1 (ou a data de
      FORECAST_INIT). RODADA/NUM_RODADA nao se aplicam.
    - **gefs**: lead = min(`gefs_lead_days`, 35); o init segue a publicacao: se lead > 16 dias
      (precisa do ciclo 00Z estendido, que so fica pronto horas depois) usa o **D-1 00Z**; se <= 16
      usa a rodada mais recente (hoje). Data explicita em FORECAST_INIT e sempre respeitada.
    - **demais** (GFS/ECMWF/AIFS/AIGFS/...): SEMPRE `OTHER_MODELS_LEAD_DAYS` (~16d, o alcance proprio
      deles); NAO dependem do settings — `gefs_lead_days` so vale para o GEFS.
    """
    spec = str(forecast_init or '').strip().lower()
    is_latest = spec in ('', 'latest')
    if model == 'cfs':
        # D-1 no horario LOCAL (Brasilia) — convencao do projeto (gotchas.md). NAO usar utcnow:
        # a noite no Brasil o UTC ja virou o dia seguinte e D-1 cairia na rodada de HOJE.
        D = ((datetime.now() - timedelta(days=1)).date() if is_latest
             else datetime.fromisoformat(spec[:10]).date())
        return [datetime(D.year, D.month, D.day)], int(cfs_lead_days) * 24
    if model == 'gefs':
        lead_days = min(int(gefs_lead_days), GEFS_MAX_LEAD_DAYS)
        if int(rodada) != 0 and lead_days > GEFS_SAMEDAY_MAX_DAYS:
            logger.warning(
                'GEFS lead={}d com RODADA={:02d}Z: o GEFS so estende alem de {}d no ciclo 00Z; '
                'em {:02d}Z o alcance e ~{}d. Use RODADA="00" p/ os {}d completos.',
                lead_days, int(rodada), GEFS_SAMEDAY_MAX_DAYS, int(rodada),
                GEFS_SAMEDAY_MAX_DAYS, lead_days)
        fi = forecast_init
        if is_latest and lead_days > GEFS_SAMEDAY_MAX_DAYS:
            # D-1 no horario LOCAL (Brasilia) — convencao do projeto (gotchas.md). Com utcnow,
            # a noite no Brasil (UTC ja no dia seguinte) D-1 cairia na rodada de HOJE, que ainda
            # esta publicando a extensao de 16->35 dias (so ~22 dias prontos).
            fi = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')  # D-1 00Z (estendido, completo)
        return resolve_run_inits(rodada, num_rodada, fi), lead_days * 24
    return resolve_run_inits(rodada, num_rodada, forecast_init), OTHER_MODELS_LEAD_DAYS * 24


def lagged_ensemble_mean(per_run: list) -> xr.DataArray:
    """Media de lagged ensemble: alinha as series diarias das rodadas por tempo valido
    e faz a media (mais membros nos leads curtos). `per_run` = lista de DataArrays (time,lat,lon)."""
    if len(per_run) == 1:
        return per_run[0]
    all_t = np.unique(np.concatenate([p['time'].values for p in per_run]))
    aligned = [p.reindex(time=all_t) for p in per_run]
    stacked = xr.concat(aligned, dim='run')
    return stacked.mean(dim='run', skipna=True)
