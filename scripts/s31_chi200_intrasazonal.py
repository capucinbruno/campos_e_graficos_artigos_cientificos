# -*- coding: utf-8 -*-
"""
s31 - CHI200 intrasazonal (potencial de velocidade 200 hPa, banda intrasazonal/MJO).

Em vez da anomalia media do periodo (s03), isola o sinal INTRASAZONAL pelo metodo
operacional do CPC (remocao de media movel):

    anom(t)  = u/v200(t) - LTM_diaria(dia-do-ano)        # remove o ciclo sazonal
    intra(t) = anom(t) - media(anom em [t-N, t-1])        # remove o interanual (N=120d)
    -> Poisson (vento divergente) -> CHI200 intrasazonal de cada dia

Produtos (todos da mesma serie diaria de chi intrasazonal):
    1. Mapas de pentada (ultimas N_PENTADAS pentadas de 5 dias)
    2. Hovmoller (lon x tempo, media numa faixa equatorial)
    3. Mapa do periodo (media de intra em [DATA_INICIAL, DATA_FINAL])

Dados:
    - ERA5/GDAS u/v 200 hPa diario (downloaders existentes)
    - LTM diaria NCEP u/v 200 (app/src/uteis/clim_diaria_uv200_ltm)

Saida:
    - PNGs em {settings.DIR_OUTPUT}/s31_CHI200_INTRASAZONAL/

Criado em: 2026-06-09
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import time
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.util import add_cyclic_point
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.chi200_intrasazonal import (
    agrupa_pentadas,
    chi200_intrasazonal_series,
    media_faixa_latitude,
)
from app.src.uteis.clim_diaria_uv200_ltm import clim_uv200_daily
from app.src.uteis.downloaders_gdas_uv200 import ensure_gdas_uv200_for_period
from app.src.uteis.downloaders_wind200 import ensure_era5_uv200_for_period
from app.src.uteis.plot_chi200 import (
    _drop_or_collapse_expver,
    _ensure_time_coord,
    _find_uv_vars,
    _normalize_latlon_names,
    _sort_and_dedup_time,
)

SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's31'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
ERA5_LATENCY_DAYS = 5
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)
CHI_SCALE = 1e5  # chi plotado em unidades de 1e5 m2/s (igual ao s03)
LEVELS = np.arange(-90, 100, 10)  # -90..90 de 10 em 10 (×10⁵ m²/s)
# Paleta verde -> bege -> marrom (igual s03): divergencia (verde) x convergencia (marrom)
CHI200_COLORS = [
    '#005a45', '#0f7a6c', '#2e9b96', '#62bdb7', '#9dd8d2', '#dff3f1',
    '#f7f4eb', '#e7d9a9', '#d6b566', '#bd8a35', '#9a6313', '#6f4300',
]


def _cfg(name: str, default):
    return settings.get(name, default)


# ---------------------------------------------------------------------------
# Serie diaria de u/v 200 regridada para a grade da LTM (2.5°)
# ---------------------------------------------------------------------------
def _daily_series_uv200(
    files, start: datetime, end: datetime, target_lat: np.ndarray, target_lon: np.ndarray, logger,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Le os arquivos ERA5/GDAS, filtra horas sinoticas, faz media diaria e interpola
    para a grade alvo (2.5°). Retorna (u_da, v_da) em (time, lat, lon), lat ascendente.
    """
    t_ini = np.datetime64(start.date())
    t_fim = np.datetime64(end.date())
    req = set(DEFAULT_SYNOPTIC_HOURS)
    tgt_lat = xr.DataArray(target_lat, dims=['lat'])
    tgt_lon = xr.DataArray(target_lon, dims=['lon'])

    us, vs = [], []
    for fp in files:
        ds = xr.open_dataset(fp, engine='netcdf4')
        try:
            ds = _sort_and_dedup_time(_normalize_latlon_names(
                _drop_or_collapse_expver(_ensure_time_coord(ds))))
            # Longitude em 0-360 (igual a grade da LTM). NAO usar -180..180 aqui: senao a
            # interpolacao para lon 180-360 cai fora do dominio e gera NaN em metade do globo.
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
            # media diaria + remove 29/02
            da_u = da_u.resample(time='1D').mean()
            da_v = da_v.resample(time='1D').mean()
            keep = ~((da_u['time'].dt.month == 2) & (da_u['time'].dt.day == 29))
            da_u, da_v = da_u.isel(time=keep.values), da_v.isel(time=keep.values)
            # interpola para a grade alvo e descarta coords auxiliares (valid_time, number,
            # expver...) que conflitam no concat entre ERA5 e GDAS
            da_u = da_u.interp(lat=tgt_lat, lon=tgt_lon, method='linear').reset_coords(drop=True)
            da_v = da_v.interp(lat=tgt_lat, lon=tgt_lon, method='linear').reset_coords(drop=True)
            us.append(da_u.load())
            vs.append(da_v.load())
            logger.info('Serie diaria: {} -> {} dias', fp.name, da_u.sizes['time'])
        finally:
            ds.close()

    if not us:
        raise RuntimeError('Nenhum dado u/v 200 valido no periodo de download.')

    u_da = xr.concat(us, dim='time', coords='minimal', compat='override').sortby('time')
    v_da = xr.concat(vs, dim='time', coords='minimal', compat='override').sortby('time')
    # dedup tempo (sobreposicao entre arquivos)
    _, uniq = np.unique(u_da['time'].values, return_index=True)
    u_da = u_da.isel(time=uniq)
    v_da = v_da.isel(time=uniq)
    return u_da, v_da


def _reindex_daily(da: xr.DataArray) -> xr.DataArray:
    """Reindexa para eixo diario contiguo e interpola buracos pequenos (filtro index=calendario)."""
    t0 = pd.Timestamp(da['time'].values[0]).normalize()
    t1 = pd.Timestamp(da['time'].values[-1]).normalize()
    full = pd.date_range(t0, t1, freq='1D')
    da = da.reindex(time=full)
    return da.interpolate_na(dim='time', method='linear', fill_value='extrapolate')


# ---------------------------------------------------------------------------
# Plotagem
# ---------------------------------------------------------------------------
def _cmap_norm():
    # extend='both' exige 2 cores extras (uma por extensao) alem das len(LEVELS)-1 faixas
    cmap = LinearSegmentedColormap.from_list('chi200', CHI200_COLORS, N=len(LEVELS) + 1)
    norm = BoundaryNorm(LEVELS, cmap.N, extend='both')
    return cmap, norm


def _plot_mapa(chi2d, lat, lon, titulo, out_png, input_dir):
    cmap, norm = _cmap_norm()
    arr, lonc = add_cyclic_point(chi2d / CHI_SCALE, coord=lon)
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_xlim([-180, 180])
    ax.set_ylim([-60, 60])
    im = ax.contourf(lonc, lat, arr, levels=LEVELS, cmap=cmap, norm=norm,
                     extend='both', transform=ccrs.PlateCarree(central_longitude=0))
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.8, zorder=10)
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)
    gl.top_labels = gl.right_labels = False
    ax.set_title(titulo, fontsize=15, loc='left')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2.5%', pad=0.08, axes_class=plt.Axes)
    cbar = plt.colorbar(im, cax=cax, ticks=LEVELS[::2], extend='both')
    cbar.set_label('CHI200 intrasazonal (×10⁵ m²/s)', size=12)
    fig.savefig(str(out_png), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


def _plot_hovmoller(hov, lon, dates, titulo, out_png):
    cmap, norm = _cmap_norm()
    hov_s, lonc = add_cyclic_point(hov / CHI_SCALE, coord=lon)
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.contourf(lonc, dates, hov_s, levels=LEVELS, cmap=cmap, norm=norm, extend='both')
    ax.invert_yaxis()  # tempo cresce para baixo (propagacao leste fica inclinada)
    ax.set_xlabel('Longitude', fontsize=13)
    ax.set_ylabel('Data', fontsize=13)
    ax.set_title(titulo, fontsize=14, loc='left')
    cbar = fig.colorbar(im, ax=ax, ticks=LEVELS[::2], extend='both', pad=0.02)
    cbar.set_label('CHI200 intrasazonal (×10⁵ m²/s)', size=12)
    fig.savefig(str(out_png), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = get_logger(SCRIPT_ID)
    logger.info('=' * 80)
    logger.info(f'SCRIPT {SCRIPT_ID.upper()}: {SCRIPT_DESC}')
    logger.info('=' * 80)

    janela = int(_cfg('JANELA_MEDIA_MOVEL', 120))
    n_pentadas = int(_cfg('N_PENTADAS', 6))
    hov_dias = int(_cfg('HOVMOLLER_DIAS', 120))
    faixa = list(_cfg('FAIXA_HOVMOLLER', [-5, 5]))

    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_CHI200_INTRASAZONAL'
    input_dir = Path(settings.DIR_INPUT)

    dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d')
    dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d')

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'janela': janela, 'n_pentadas': n_pentadas, 'hov_dias': hov_dias, 'faixa': faixa,
        'metodo': 'CPC running-mean (LTM diaria + media movel)',
        'script_version': '1.1',
    }
    hov_png = output_dir / 'chi200_intra_hovmoller.png'
    periodo_png = output_dir / 'chi200_intra_periodo.png'
    output_files = [str(hov_png), str(periodo_png)]
    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Pulando execucao.')
        return

    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Janela de download: precisa de `janela` dias antes do inicio da serie de interesse
    inicio_interesse = min(
        dt_ini,
        dt_fim - timedelta(days=max(hov_dias, n_pentadas * 5) - 1),
    )
    start_dl = inicio_interesse - timedelta(days=janela + 2)
    logger.info(f'Periodo de interesse: {dt_ini.date()} a {dt_fim.date()}')
    logger.info(f'Download (com {janela}d de folga p/ media movel): {start_dl.date()} a {dt_fim.date()}')

    # ---- Download ERA5/GDAS u/v 200 ----
    cutoff = (datetime.now() - timedelta(days=ERA5_LATENCY_DAYS)).replace(hour=0, minute=0, second=0, microsecond=0)
    files = []
    if start_dl < cutoff:
        logger.info('Etapa 1a: ERA5 u/v 200 hPa...')
        files += list(ensure_era5_uv200_for_period(
            start=start_dl, end=min(dt_fim, cutoff - timedelta(days=1)),
            hours_utc=list(DEFAULT_SYNOPTIC_HOURS), force_redownload=False,
        ))
    if dt_fim >= cutoff:
        logger.info('Etapa 1b: GDAS u/v 200mb...')
        files += list(ensure_gdas_uv200_for_period(
            start=max(start_dl, cutoff), end=dt_fim, force_redownload=False,
        ))

    # ---- Grade alvo (LTM 2.5°, lat ascendente) ----
    dates_probe = np.array([np.datetime64(dt_fim.date())])
    _, _, ltm_lat, ltm_lon = clim_uv200_daily(dates_probe)
    order = np.argsort(ltm_lat)
    lat, lon = ltm_lat[order], ltm_lon

    # ---- Serie diaria de vento (regridada) ----
    logger.info('Etapa 2: Montando serie diaria de u/v 200 (2.5°)...')
    u_da, v_da = _daily_series_uv200(files, start_dl, dt_fim, lat, lon, logger)
    u_da, v_da = _reindex_daily(u_da), _reindex_daily(v_da)
    dates = np.array([np.datetime64(pd.Timestamp(t).date()) for t in u_da['time'].values])
    logger.info(f'Serie diaria contigua: {len(dates)} dias ({dates[0]} a {dates[-1]})')

    # ---- Anomalia diaria = vento - LTM(dia-do-ano) ----
    logger.info('Etapa 3: Anomalia diaria (- LTM diaria)...')
    u_clim, v_clim, _, _ = clim_uv200_daily(dates)
    u_clim, v_clim = u_clim[:, order, :], v_clim[:, order, :]
    u_anom = u_da.values - u_clim
    v_anom = v_da.values - v_clim

    # ---- Intrasazonal + chi200 por dia ----
    logger.info(f'Etapa 4: Filtro intrasazonal (media movel {janela}d) + Poisson por dia...')
    chi_intra, idx = chi200_intrasazonal_series(u_anom, v_anom, lat, lon, janela=janela)
    dates_intra = dates[idx]
    logger.info(f'Serie chi intrasazonal: {chi_intra.shape[0]} dias ({dates_intra[0]} a {dates_intra[-1]})')

    # ---- Produto 1: mapas de pentada ----
    logger.info('Etapa 5: Mapas de pentada...')
    # remove pentadas de execucoes anteriores (nomes variam com a data) p/ nao acumular
    for antigo in output_dir.glob('chi200_intra_pentada_*.png'):
        antigo.unlink()
    for d_ini, d_fim, campo in agrupa_pentadas(chi_intra, dates_intra, n_pentadas):
        nome = f'chi200_intra_pentada_{str(d_ini)}_a_{str(d_fim)}.png'
        _plot_mapa(campo, lat, lon,
                   f'CHI200 intrasazonal — pentada {d_ini} a {d_fim}',
                   output_dir / nome, input_dir)

    # ---- Produto 2: Hovmoller ----
    logger.info('Etapa 6: Hovmoller...')
    m_hov = dates_intra >= (np.datetime64(dt_fim.date()) - np.timedelta64(hov_dias - 1, 'D'))
    hov = media_faixa_latitude(chi_intra[m_hov], lat, faixa[0], faixa[1])
    _plot_hovmoller(hov, lon, dates_intra[m_hov],
                    f'CHI200 intrasazonal — Hovmoller ({faixa[0]}°..{faixa[1]}°)', hov_png)

    # ---- Produto 3: mapa do periodo ----
    logger.info('Etapa 7: Mapa do periodo...')
    m_per = (dates_intra >= np.datetime64(dt_ini.date())) & (dates_intra <= np.datetime64(dt_fim.date()))
    if m_per.any():
        campo_per = chi_intra[m_per].mean(axis=0)
        _plot_mapa(campo_per, lat, lon,
                   f'CHI200 intrasazonal — media {dt_ini.date()} a {dt_fim.date()}',
                   periodo_png, input_dir)

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)
    logger.info('=' * 80)
    logger.info(f'Script {SCRIPT_ID.upper()} concluido em {execution_time:.1f}s')
    logger.info(f'Saida: {output_dir}')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
