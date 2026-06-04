# app/src/uteis/plot_geop250.py
# -*- coding: utf-8 -*-
"""
Download e processamento de altura geopotencial 250 hPa (ERA5) para anomalia.

Pipeline:
1. Baixa dados ERA5 de geopotencial (z) em 250 hPa via CDS (GRIB)
2. Converte geopotencial (m2/s2) para altura geopotencial (m)
3. Concatena arquivos mensais e calcula media diaria
4. Carrega climatologia 1991-2020 de geopotencial 250 hPa
5. Calcula anomalia = media_periodo - media_climatologica
6. Salva resultado em dados/geop250.nc (variavel 'hgt', 1 timestep)

Chamado por: scripts/s01_geop250_anom.py via `from app.src.uteis.plot_geop250 import main`
"""

from __future__ import annotations

# Bibliotecas padrão
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Bibliotecas de terceiros
import numpy as np
import pandas as pd
import xarray as xr

# Módulos locais
from app.src.uteis.clim_PSL_geop250 import get_clim_geop250_path
from app.src.uteis.downloaders_gdas_hgt250 import ensure_gdas_hgt250_for_period
from app.src.uteis.downloaders_hgt250_ERA5 import (
    ensure_era5_altura_geopotencial_250_global_for_period_grib,
)

# -----------------------------------------------------------------------------
# Integracao com settings
# -----------------------------------------------------------------------------
try:
    # Módulos locais
    from app.shared.settings_factory import settings  # type: ignore

    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
except Exception:
    DIR_DADOS_BASE = Path('dados')

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
LOGGER = logging.getLogger('PLOT_GEOP250')
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

# Horas sinoticas padrao
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)


# -----------------------------------------------------------------------------
# Utilitarios
# -----------------------------------------------------------------------------
def _ensure_time_coord(obj):
    """Normaliza valid_time -> time."""
    if hasattr(obj, 'dims') and 'time' not in obj.dims and 'valid_time' in obj.dims:
        obj = obj.rename({'valid_time': 'time'})
    elif hasattr(obj, 'coords') and 'time' not in obj.coords and 'valid_time' in obj.coords:
        obj = obj.rename({'valid_time': 'time'})
    if 'time' not in obj.coords:
        raise KeyError("Nem 'time' nem 'valid_time' encontrados.")
    return obj


def _drop_or_collapse_expver(ds: xr.Dataset) -> xr.Dataset:
    """Colapsa dim expver/number se existirem (ERA5/ERA5T)."""
    rename_dims = {}
    for d in ds.dims:
        dl = d.lower()
        if dl == 'expver' and d != 'expver':
            rename_dims[d] = 'expver'
        elif dl == 'number' and d != 'number':
            rename_dims[d] = 'number'

    if rename_dims:
        ds = ds.rename(rename_dims)

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


def _normalize_latlon_names(ds):
    """Renomeia latitude/longitude -> lat/lon se necessario."""
    rename = {}
    for name in ds.dims:
        low = name.lower()
        if low in {'latitude'} and 'lat' not in ds.dims:
            rename[name] = 'lat'
        elif low in {'longitude'} and 'lon' not in ds.dims:
            rename[name] = 'lon'
    if rename:
        ds = ds.rename(rename)
    return ds


def _normalize_lon(ds):
    """Converte longitude de 0..360 para -180..180 se necessario."""
    if 'lon' not in ds.coords:
        return ds
    lon_vals = ds['lon'].values
    if np.any(lon_vals > 180):
        ds = ds.assign_coords(lon=(ds['lon'].values + 180) % 360 - 180)
        ds = ds.sortby('lon')
    return ds


def _sort_and_dedup_time(ds: xr.Dataset) -> xr.Dataset:
    """Ordena e remove timestamps duplicados."""
    ds = ds.sortby('time')
    t = pd.DatetimeIndex(pd.to_datetime(ds['time'].values))
    _, idx = np.unique(t.values, return_index=True)
    idx = np.sort(idx)
    if len(idx) != ds.sizes.get('time', 0):
        LOGGER.warning('Removendo %d timestamps duplicados.', ds.sizes['time'] - len(idx))
        ds = ds.isel(time=idx)
    return ds


def _drop_feb29(ds: xr.Dataset) -> xr.Dataset:
    """Remove 29/02 para compatibilizar com climatologia 365d."""
    ds = _ensure_time_coord(ds)
    t = xr.DataArray(ds['time'].values, dims=['time'])
    mask = ~((t.dt.month == 2) & (t.dt.day == 29))
    n_before = ds.sizes['time']
    ds2 = ds.isel(time=mask.values)
    n_after = ds2.sizes['time']
    if n_after < n_before:
        LOGGER.warning(
            'Removendo %d registro(s) de 29/02 para compatibilizar com climatologia 365d.',
            n_before - n_after,
        )
    return ds2


# -----------------------------------------------------------------------------
# Abertura e concatenacao de arquivos convertidos (NetCDF hgt)
# -----------------------------------------------------------------------------
def _open_and_merge_hgt_files(files: Sequence[Path]) -> xr.Dataset:
    """Abre e concatena arquivos mensais de altura geopotencial em NetCDF."""
    if not files:
        raise ValueError('Lista de arquivos de altura geopotencial vazia.')

    dsets = []
    for fp in files:
        LOGGER.info('Abrindo: %s', fp)
        ds = xr.open_dataset(fp, engine='netcdf4')
        ds = _ensure_time_coord(ds)
        ds = _drop_or_collapse_expver(ds)
        ds = _normalize_latlon_names(ds)
        dsets.append(ds)

    if len(dsets) == 1:
        ds_all = dsets[0]
    else:
        ds_all = xr.concat(dsets, dim='time', data_vars='minimal', coords='minimal', compat='override')

    ds_all = _ensure_time_coord(ds_all)
    ds_all = _sort_and_dedup_time(ds_all)

    LOGGER.info(
        'Dataset combinado: %d timestamps | %s -> %s',
        ds_all.sizes.get('time', 0),
        str(pd.to_datetime(ds_all.time.values[0])) if ds_all.sizes.get('time', 0) else 'NA',
        str(pd.to_datetime(ds_all.time.values[-1])) if ds_all.sizes.get('time', 0) else 'NA',
    )
    return ds_all


# -----------------------------------------------------------------------------
# Media diaria a partir de horas sinoticas
# -----------------------------------------------------------------------------
def _compute_daily_mean(
    ds: xr.Dataset,
    required_hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
) -> xr.Dataset:
    """Calcula media diaria filtrando apenas horas sinoticas requeridas."""
    ds = _ensure_time_coord(ds)
    ds = _sort_and_dedup_time(ds)

    required_set = set(int(h) for h in required_hours)
    t_idx = pd.DatetimeIndex(pd.to_datetime(ds['time'].values))

    mask = np.array([h in required_set for h in t_idx.hour], dtype=bool)
    ds = ds.isel(time=mask)

    if ds.sizes.get('time', 0) == 0:
        raise ValueError('Dataset sem timestamps apos filtrar horas sinoticas.')

    ds_daily = ds.resample(time='1D').mean(keep_attrs=True)

    # Remove dias vazios criados pelo resample
    for vn in ds_daily.data_vars:
        da = ds_daily[vn]
        other_dims = [d for d in da.dims if d != 'time']
        if other_dims:
            valid = da.notnull().any(dim=other_dims)
        else:
            valid = da.notnull()
        ds_daily = ds_daily.isel(time=valid.values)
        break

    LOGGER.info(
        'Media diaria: %d dias | %s -> %s',
        ds_daily.sizes.get('time', 0),
        str(pd.to_datetime(ds_daily.time.values[0])) if ds_daily.sizes.get('time', 0) else 'NA',
        str(pd.to_datetime(ds_daily.time.values[-1])) if ds_daily.sizes.get('time', 0) else 'NA',
    )
    return ds_daily


# -----------------------------------------------------------------------------
# Climatologia
# -----------------------------------------------------------------------------
def _load_climatology_geop250(path_clim: Path) -> xr.DataArray:
    """Carrega climatologia de geopotencial 250 hPa e retorna DataArray de hgt."""
    if not path_clim.exists():
        raise FileNotFoundError(f'Climatologia geop250 nao encontrada: {path_clim}')

    LOGGER.info('Abrindo climatologia geop250: %s', path_clim)
    ds = xr.open_dataset(path_clim, engine='netcdf4')
    ds = _normalize_latlon_names(ds)

    LOGGER.info('Variaveis na climatologia: %s', list(ds.data_vars))
    LOGGER.info('Coords na climatologia: %s', list(ds.coords))

    # Procura variavel de altura geopotencial
    for vname in ('hgt', 'z', 'geopotential', 'gh', 'geopotential_height'):
        if vname in ds.data_vars:
            da = ds[vname]
            # Se for geopotential (m2/s2), converte para metros
            units = da.attrs.get('units', '')
            if 'm**2' in units or 'm2' in units or 'J' in units:
                LOGGER.info('Convertendo climatologia de geopotencial (m2/s2) para altura (m)')
                da = da / 9.80665
                da.attrs['units'] = 'm'
                da.attrs['long_name'] = 'geopotential height'
            return da

    # Fallback: primeira variavel numerica 3D
    for vname in ds.data_vars:
        if np.issubdtype(ds[vname].dtype, np.number) and len(ds[vname].dims) >= 2:
            LOGGER.warning('Usando variavel %s como altura geopotencial da climatologia.', vname)
            return ds[vname]

    raise KeyError(
        f'Nao encontrei variavel de altura geopotencial na climatologia: {list(ds.data_vars)}'
    )


def _select_climatology_same_days(
    clim: xr.DataArray,
    daily_dates: xr.DataArray,
) -> xr.DataArray:
    """Seleciona na climatologia os mesmos dias do periodo.

    Suporta 3 formatos de coordenada temporal na climatologia:
    - time com inteiros 1-365 (dayofyear como inteiro)
    - time com datetimes (selecao por month-day)
    - dayofyear como coordenada separada
    """
    target_dates = pd.DatetimeIndex(pd.to_datetime(daily_dates.values))

    # Caso 1: coord 'time' com inteiros 1-365 (dayofyear)
    if 'time' in clim.coords:
        time_vals = clim['time'].values
        if np.issubdtype(type(time_vals[0]), np.integer) or (
            hasattr(time_vals[0], 'dtype') and np.issubdtype(time_vals[0].dtype, np.integer)
        ):
            LOGGER.info('Climatologia com coord time inteira (dayofyear 1-365).')
            target_doy = target_dates.dayofyear.to_numpy()
            clim_sel = clim.sel(time=xr.DataArray(target_doy, dims=['time']))
            clim_sel = clim_sel.assign_coords(time=daily_dates.values)
            LOGGER.info('Climatologia selecionada por dayofyear (inteiro): %d dias.', len(target_doy))
            return clim_sel

        # Caso 2: coord 'time' com datetimes
        try:
            clim_dates = pd.DatetimeIndex(pd.to_datetime(time_vals))
        except Exception:
            raise KeyError(
                f"Coord 'time' da climatologia nao e datetime nem inteiro: {type(time_vals[0])}"
            )

        clim_md = pd.Index(
            [f'{t.month:02d}-{t.day:02d}' for t in clim_dates],
            name='monthday',
        )
        target_md = [f'{t.month:02d}-{t.day:02d}' for t in target_dates]

        md_to_idx = {}
        for i, md in enumerate(clim_md.tolist()):
            md_to_idx[md] = i

        missing = [md for md in target_md if md not in md_to_idx]
        if missing:
            LOGGER.warning(
                'Dias ausentes na climatologia (serao ignorados na anomalia): %s', missing[:10]
            )
            valid_pairs = [(md, i) for i, md in enumerate(target_md) if md in md_to_idx]
            if not valid_pairs:
                raise ValueError('Nenhum dia do periodo encontrado na climatologia.')
            valid_md, valid_target_idx = zip(*valid_pairs)
            idxs = [md_to_idx[md] for md in valid_md]
            clim_sel = clim.isel(time=idxs)
            target_times = daily_dates.values[list(valid_target_idx)]
            clim_sel = clim_sel.assign_coords(time=target_times)
        else:
            idxs = [md_to_idx[md] for md in target_md]
            clim_sel = clim.isel(time=idxs)
            clim_sel = clim_sel.assign_coords(time=daily_dates.values)

        LOGGER.info("Climatologia selecionada por month-day usando coord 'time'.")
        return clim_sel

    # Caso 3: coord 'dayofyear' separada
    if 'dayofyear' in clim.coords:
        target_doy = target_dates.dayofyear.to_numpy()
        clim_sel = clim.sel(dayofyear=xr.DataArray(target_doy, dims=['time']))
        clim_sel = clim_sel.assign_coords(time=daily_dates.values)
        LOGGER.info("Climatologia selecionada por coord 'dayofyear'.")
        return clim_sel

    raise KeyError("Climatologia precisa ter coord 'time' (datetime ou dayofyear int) ou 'dayofyear'.")


# -----------------------------------------------------------------------------
# Seleção de fonte de dados (ERA5 vs GDAS)
# -----------------------------------------------------------------------------
ERA5_LATENCY_DAYS = 7


def _get_data_sources(
    dt_ini: datetime,
    dt_fim: datetime,
) -> Tuple[Optional[Tuple[datetime, datetime]], Optional[Tuple[datetime, datetime]]]:
    """Retorna (periodo_era5, periodo_gdas) com base na latência do ERA5.

    Qualquer extremo pode ser None se a fonte não for necessária.
    """
    cutoff = (datetime.now() - timedelta(days=ERA5_LATENCY_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    if dt_fim < cutoff:
        return (dt_ini, dt_fim), None
    if dt_ini >= cutoff:
        return None, (dt_ini, dt_fim)
    return (dt_ini, cutoff - timedelta(days=1)), (cutoff, dt_fim)


def _load_climatology_psl(path: Path) -> xr.DataArray:
    """Carrega climatologia do PSL (1 tempo, 2.5°) e retorna DataArray hgt 2D."""
    if not path.exists():
        raise FileNotFoundError(f'Climatologia PSL não encontrada: {path}')

    ds = xr.open_dataset(path, engine='netcdf4')
    ds = _normalize_latlon_names(ds)
    da = ds['hgt']

    if 'time' in da.dims:
        da = da.isel(time=0, drop=True)

    LOGGER.info('Climatologia PSL carregada: %s | shape=%s | lon=[%.1f, %.1f]',
                path.name, da.shape, float(da.lon.min()), float(da.lon.max()))
    return da


# -----------------------------------------------------------------------------
# Média do período em streaming (um arquivo por vez)
# -----------------------------------------------------------------------------
def _compute_period_mean_streaming(
    files: Sequence[Path],
    required_hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    dt_ini: Optional[datetime] = None,
    dt_fim: Optional[datetime] = None,
) -> xr.DataArray:
    """Calcula média do período processando um arquivo por vez.

    Mantém no máximo 1 arquivo na RAM por vez — adequado para períodos longos
    com múltiplas variáveis sem estourar memória no WSL.
    """
    required_set = set(int(h) for h in required_hours)
    t_ini = np.datetime64(dt_ini.date()) if dt_ini else None
    t_fim = np.datetime64(dt_fim.date()) if dt_fim else None

    sum_2d: Optional[np.ndarray] = None
    count_2d: Optional[np.ndarray] = None
    ref_lat: Optional[np.ndarray] = None
    ref_lon: Optional[np.ndarray] = None
    total_days = 0

    for fp in files:
        LOGGER.info('Streaming: abrindo %s', fp.name)
        ds = xr.open_dataset(fp, engine='netcdf4')
        try:
            ds = _ensure_time_coord(ds)
            ds = _drop_or_collapse_expver(ds)
            ds = _normalize_latlon_names(ds)
            ds = _normalize_lon(ds)
            ds = _sort_and_dedup_time(ds)

            hgt_var = next(
                (v for v in ('hgt', 'z', 'geopotential') if v in ds.data_vars),
                list(ds.data_vars)[0],
            )
            da = ds[hgt_var]

            for dim_name in ('pressure_level', 'isobaricInhPa', 'level'):
                if dim_name in da.dims:
                    da = da.isel({dim_name: 0}, drop=True)

            # Filtrar horas sinóticas
            t_idx = pd.DatetimeIndex(pd.to_datetime(da['time'].values))
            mask_h = np.array([h in required_set for h in t_idx.hour], dtype=bool)
            da = da.isel(time=mask_h)

            if da.sizes.get('time', 0) == 0:
                LOGGER.warning('Sem horas sinóticas válidas em: %s', fp.name)
                continue

            # Recortar ao período
            if t_ini is not None or t_fim is not None:
                da = da.sel(time=slice(t_ini, t_fim))
            if da.sizes.get('time', 0) == 0:
                LOGGER.warning('Fora do período solicitado: %s', fp.name)
                continue

            # Remover 29/02
            t = xr.DataArray(da['time'].values, dims=['time'])
            da = da.isel(time=(~((t.dt.month == 2) & (t.dt.day == 29))).values)
            if da.sizes.get('time', 0) == 0:
                continue

            # Média diária
            da_daily = da.resample(time='1D').mean(keep_attrs=True)
            valid = da_daily.notnull().any(dim=['lat', 'lon'])
            da_daily = da_daily.isel(time=valid.values)
            n_days_file = da_daily.sizes['time']
            if n_days_file == 0:
                continue

            # Acumular soma e contagem (sem manter o array completo na RAM)
            vals = da_daily.values  # (time, lat, lon)
            if sum_2d is None:
                sum_2d = np.zeros(vals.shape[1:], dtype=np.float64)
                count_2d = np.zeros(vals.shape[1:], dtype=np.int64)
                ref_lat = da_daily['lat'].values.copy()
                ref_lon = da_daily['lon'].values.copy()

            sum_2d += np.nansum(vals, axis=0)
            count_2d += (~np.isnan(vals)).sum(axis=0)
            total_days += n_days_file

            LOGGER.info('Streaming: %s → %d dias (acumulado: %d)', fp.name, n_days_file, total_days)
        finally:
            ds.close()

    if sum_2d is None or total_days == 0:
        raise RuntimeError('Nenhum dado válido encontrado no período solicitado.')

    mean_2d = np.where(count_2d > 0, sum_2d / count_2d, np.nan).astype(np.float32)
    LOGGER.info(
        'Média do período: %d dias | min=%.1f max=%.1f mean=%.1f mgp',
        total_days, float(np.nanmin(mean_2d)), float(np.nanmax(mean_2d)), float(np.nanmean(mean_2d)),
    )

    return xr.DataArray(
        mean_2d,
        dims=['lat', 'lon'],
        coords={'lat': ref_lat, 'lon': ref_lon},
        attrs={'long_name': 'geopotential height', 'units': 'm'},
        name='hgt',
    )


# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Download, processamento e calculo de anomalia de geopotencial 250 hPa.

    Fontes de dados selecionadas automaticamente:
    - ERA5 (CDS): períodos mais antigos que 7 dias
    - GDAS (NOMADS): últimos 7 dias
    - Climatologia: PSL via Playwright (cache local por período MM-DD)

    Resultado: dados/geop250.nc com variavel 'hgt' (anomalia em metros, 1 timestep).
    """
    def _to_datetime(val) -> datetime:
        if isinstance(val, datetime):
            return val
        if hasattr(val, 'year'):  # date object
            return datetime(val.year, val.month, val.day)
        return datetime.strptime(str(val), '%Y-%m-%d')

    dt_ini = _to_datetime(settings.DATA_INICIAL)
    dt_fim = _to_datetime(settings.DATA_FINAL)
    force = getattr(settings, 'FORCE_DOWNLOAD', False)

    # GDAS só tem dados completos até ontem — ajusta DATA_FINAL se necessário
    ontem = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    if dt_fim >= datetime.now().replace(hour=0, minute=0, second=0, microsecond=0):
        LOGGER.warning(
            'DATA_FINAL (%s) é hoje ou futura. '
            'GDAS só tem dados completos até ontem. '
            'Última data considerada: %s',
            dt_fim.strftime('%Y-%m-%d'),
            ontem.strftime('%Y-%m-%d'),
        )
        dt_fim = ontem

    LOGGER.info('=' * 70)
    LOGGER.info('PLOT_GEOP250: Download e anomalia geopotencial 250 hPa')
    LOGGER.info('Periodo: %s a %s', settings.DATA_INICIAL, dt_fim.strftime('%Y-%m-%d'))
    LOGGER.info('=' * 70)

    # 1. Determinar fontes de dados
    era5_period, gdas_period = _get_data_sources(dt_ini, dt_fim)
    if era5_period:
        LOGGER.info('ERA5:  %s → %s', era5_period[0].date(), era5_period[1].date())
    if gdas_period:
        LOGGER.info('GDAS:  %s → %s', gdas_period[0].date(), gdas_period[1].date())

    # 2. Download dos dados
    all_files: List[Path] = []

    if era5_period:
        LOGGER.info('Etapa 2a: Download ERA5 geopotencial 250 hPa')
        era5_files = ensure_era5_altura_geopotencial_250_global_for_period_grib(
            start=era5_period[0],
            end=era5_period[1],
            hours_utc=list(DEFAULT_SYNOPTIC_HOURS),
            force_redownload=force,
            convert_to_height_netcdf=True,
        )
        all_files.extend(era5_files)

    if gdas_period:
        LOGGER.info('Etapa 2b: Download GDAS HGT 250 hPa (NOMADS)')
        gdas_files = ensure_gdas_hgt250_for_period(
            start=gdas_period[0],
            end=gdas_period[1],
            force_redownload=force,
        )
        all_files.extend(gdas_files)

    # 3. Média do período em streaming (um arquivo por vez — sem carregar tudo na RAM)
    LOGGER.info('Etapa 3: Calculando média do período em streaming')
    hgt_period_mean = _compute_period_mean_streaming(
        all_files,
        required_hours=DEFAULT_SYNOPTIC_HOURS,
        dt_ini=dt_ini,
        dt_fim=dt_fim,
    )

    # 4. Climatologia PSL (cache local por MM-DD)
    LOGGER.info('Etapa 4: Climatologia PSL geopotencial 250 hPa')
    clim_path = get_clim_geop250_path(settings.DATA_INICIAL, settings.DATA_FINAL)
    clim_da = _load_climatology_psl(clim_path)
    clim_da = _normalize_lon(clim_da.to_dataset(name='hgt'))['hgt']

    # Adicionar ponto cíclico antes de interpolar: climatologia PSL vai até ~177.5°,
    # mas os dados chegam a 179.75°. Sem o ponto cíclico, a interp gera NaN em
    # 177.75°–179.75°, criando a faixa branca em 180° nos mapas.
    from cartopy.util import add_cyclic_point as _acp
    clim_vals_cyc, clim_lon_cyc = _acp(clim_da.values, coord=clim_da['lon'].values)
    clim_da = xr.DataArray(
        clim_vals_cyc,
        dims=clim_da.dims,
        coords={'lat': clim_da['lat'].values, 'lon': clim_lon_cyc},
    )

    # Interpolação 2.5° → grade ERA5/GDAS 0.25°
    clim_regrid = clim_da.interp(
        lat=hgt_period_mean.lat, lon=hgt_period_mean.lon, method='linear'
    )

    # 5. Anomalia = média do período - climatologia
    LOGGER.info('Etapa 5: Calculando anomalia')
    anomaly = hgt_period_mean - clim_regrid
    anomaly.name = 'hgt'
    anomaly.attrs['long_name'] = 'geopotential height anomaly at 250 hPa'
    anomaly.attrs['units'] = 'm'

    LOGGER.info(
        'Anomalia calculada: min=%.1f, max=%.1f, mean=%.1f mgp',
        float(anomaly.min()),
        float(anomaly.max()),
        float(anomaly.mean()),
    )

    # 6. Salvar dados/geop250.nc (1 timestep — s01 espera ds['hgt'].isel(time=0))
    anomaly_ds = anomaly.expand_dims('time').to_dataset(name='hgt')
    anomaly_ds['time'] = [pd.Timestamp(settings.DATA_FINAL)]

    output_path = DIR_DADOS_BASE / 'geop250.nc'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    anomaly_ds.to_netcdf(output_path, engine='netcdf4')
    LOGGER.info('Anomalia salva em: %s', output_path)
    LOGGER.info('=' * 70)
    LOGGER.info('PLOT_GEOP250: Concluido com sucesso')
    LOGGER.info('=' * 70)


if __name__ == '__main__':
    main()
