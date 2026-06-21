# -*- coding: utf-8 -*-
"""
s35 — Indice da AAO (Antarctic Oscillation): observado + previsoes multi-modelo.

Metodologia CPC (ver app/src/uteis/aao_eof.py): projeta a anomalia diaria de altura geopotencial
em 700 hPa (HS) no EOF1 proprio (ERA5 1991-2020), normalizado pelo desvio do indice mensal.

Gera UM grafico de linha (estilo CPC):
- linha PRETA = observado (ERA5/GDAS) dos ultimos AAO_HIST_DAYS dias;
- uma linha COLORIDA por modelo de previsao habilitado no settings (so a MEDIA do ensemble);
- eixo x ate o horizonte do modelo mais longo (ex.: CFS pseudo-ensemble ~45 dias); cada
  modelo termina no seu proprio alcance. Modelo sem 700 hPa -> linha simplesmente nao aparece.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # backend nao-interativo (downloads em threads quebram o Tk)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.aao_eof import aao_index_from_height, ensure_aao_loading_pattern
from app.src.uteis.clim_diaria_uv200_ltm import clim_hgt700_daily
from app.src.uteis.forecast_daily import (
    DEFAULT_SYNOPTIC_HOURS,
    daily_scalar_on_grid,
    lagged_ensemble_mean,
    resolve_run_inits,
)

# Downloaders de Z700 (previsao)
from app.src.uteis.downloaders_aifs_ens import ensure_aifs_ens_hgt700_fcst_for_period
from app.src.uteis.downloaders_aifs_hgt700 import ensure_aifs_hgt700_fcst_for_period
from app.src.uteis.downloaders_ai_nomads import (
    ensure_aigefs_hgt700_fcst_for_period,
    ensure_aigfs_hgt700_fcst_for_period,
)
from app.src.uteis.downloaders_cfs_ensemble import CFS_LEAD_DAYS, ensure_cfs_hgt700_for_period
from app.src.uteis.downloaders_ecmwf_ens import ensure_ecmwf_ens_hgt700_fcst_for_period
from app.src.uteis.downloaders_ecmwf_hgt700 import ensure_ecmwf_hgt700_fcst_for_period
from app.src.uteis.downloaders_gefs_hgt700 import ensure_gefs_hgt700_fcst_for_period
from app.src.uteis.downloaders_gfs_hgt700 import ensure_gfs_hgt700_fcst_for_period
# Z700 observado (reanalise/analise)
from app.src.uteis.downloaders_gdas_hgt700 import download_gdas_hgt700_for_date
from app.src.uteis.downloaders_hgt700_ERA5 import (
    ensure_era5_altura_geopotencial_700_global_for_period_grib,
)

logger = get_logger('s35')

SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's35'
HGT_VARS = ('hgt', 'z', 'gh', 'geopotential')
ERA5_LATENCY_DAYS = 7
SCRIPT_VERSION = '2.0'  # titulo sem o sufixo (700 hPa Z, HS)

# Modelos de previsao: flag no settings -> downloader Z700, cor e rotulo.
_MODEL_FLAGS = {
    'gfs': ('RUN_GFS', True), 'gefs': ('RUN_GEFS', False),
    'ecmwf': ('RUN_ECMWF', False), 'ecmwf_ens': ('RUN_ECMWF_ENS', False),
    'aifs': ('RUN_AIFS', False), 'aifs_ens': ('RUN_AIFS_ENS', False),
    'aigfs': ('RUN_AIGFS', False), 'aigefs': ('RUN_AIGEFS', False),
    'cfs': ('RUN_CFS', False),
}
_HGT700_DOWNLOADERS = {
    'gfs': ensure_gfs_hgt700_fcst_for_period,
    'gefs': ensure_gefs_hgt700_fcst_for_period,
    'ecmwf': ensure_ecmwf_hgt700_fcst_for_period,
    'ecmwf_ens': ensure_ecmwf_ens_hgt700_fcst_for_period,
    'aifs': ensure_aifs_hgt700_fcst_for_period,
    'aifs_ens': ensure_aifs_ens_hgt700_fcst_for_period,
    'aigfs': ensure_aigfs_hgt700_fcst_for_period,
    'aigefs': ensure_aigefs_hgt700_fcst_for_period,
    'cfs': ensure_cfs_hgt700_for_period,
}
_MODEL_COLOR = {
    'gfs': '#1f77b4', 'gefs': '#2ca02c', 'ecmwf': '#d62728', 'ecmwf_ens': '#ff7f0e',
    'aifs': '#9467bd', 'aifs_ens': '#e377c2', 'aigfs': '#8c564b', 'aigefs': '#7f7f7f',
    'cfs': '#17becf',
}
_MODEL_LABEL = {
    'gfs': 'GFS', 'gefs': 'GEFS', 'ecmwf': 'ECMWF-HRES', 'ecmwf_ens': 'ECMWF-ENS',
    'aifs': 'AIFS', 'aifs_ens': 'AIFS-ENS', 'aigfs': 'AIGFS', 'aigefs': 'AIGEFS',
    'cfs': 'CFS (45d)',
}


def _enabled_models() -> list:
    """Modelos habilitados pelas flags do settings (ordem de _MODEL_FLAGS)."""
    return [m for m, (flag, default) in _MODEL_FLAGS.items() if bool(settings.get(flag, default))]


def _ltm_grid():
    """Grade COMPLETA da LTM NCEP (lat ascendente, lon 0..360) — alvo do regrid diario."""
    probe = np.array([np.datetime64('2020-01-15')])
    _, lat_raw, lon = clim_hgt700_daily(probe)
    order = np.argsort(lat_raw)
    return lat_raw[order], np.asarray(lon)


def _get_obs_sources(ini: datetime, fim: datetime):
    """(era5_period, gdas_period) para o observado — ERA5 ate o cutoff de latencia, GDAS depois."""
    cutoff = (datetime.now() - timedelta(days=ERA5_LATENCY_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    if fim < cutoff:
        return (ini, fim), None
    if ini >= cutoff:
        return None, (ini, fim)
    return (ini, cutoff - timedelta(days=1)), (cutoff, fim)


def _gdas_best_effort(start: datetime, end: datetime) -> list:
    """Baixa o GDAS dia a dia ate `end`, PARANDO no 1o dia ainda nao publicado.

    Garante que o observado va o mais perto possivel do dia da rodada: se a analise de hoje
    (ou dos ultimos dias) ainda nao saiu no momento da execucao, o observado para no ultimo dia
    disponivel — e a previsao dos modelos cobre o restante ate o init (ver ancora no _plot)."""
    files = []
    d = start.date()
    while d <= end.date():
        try:
            files.append(download_gdas_hgt700_for_date(d))
        except Exception as exc:  # dia recente ainda nao publicado -> para; previsao emenda o resto
            logger.warning('GDAS Z700 {} indisponivel ({}) — observado para aqui; a previsao cobre '
                           'a lacuna ate o init.', d, exc)
            break
        d += timedelta(days=1)
    return files


def _observed_index(lat: np.ndarray, lon: np.ndarray, hist_days: int):
    """Serie do indice AAO observado (ERA5+GDAS) dos ultimos `hist_days` dias.

    Estende ate HOJE usando a analise GDAS (best-effort), de modo que o observado alcance o dia
    do init da rodada — assim a previsao se conecta ao observado num ponto de dado real (sem
    lacuna). Se o GDAS recente ainda nao estiver publicado, o observado para no ultimo dia
    disponivel e a previsao emenda o resto (ver `_plot`).
    """
    fim = datetime.now()                                  # tenta incluir hoje (analise GDAS do dia)
    ini = fim - timedelta(days=hist_days)
    era5_p, gdas_p = _get_obs_sources(ini, fim)
    files = []
    if era5_p:
        files += list(ensure_era5_altura_geopotencial_700_global_for_period_grib(
            start=era5_p[0], end=era5_p[1], hours_utc=list(DEFAULT_SYNOPTIC_HOURS)))
    if gdas_p:
        files += _gdas_best_effort(gdas_p[0], gdas_p[1])
    h_da = daily_scalar_on_grid(files, HGT_VARS, ini, fim, lat, lon, logger)
    idx, dates = aao_index_from_height(h_da, logger)
    return pd.to_datetime(dates), idx


def _resolve_model_inits(model: str):
    """(run_inits, lead_hours) para um modelo — replica a logica do s34 (CFS = pseudo-ensemble)."""
    if model == 'cfs':
        spec = str(settings.get('FORECAST_INIT', '') or '').strip().lower()
        D = ((datetime.utcnow() - timedelta(days=1)).date() if spec in ('', 'latest')
             else datetime.fromisoformat(spec[:10]).date())
        init0 = datetime(D.year, D.month, D.day)
        lead = min(int(settings.get('FORECAST_LEAD_DAYS', 45)), CFS_LEAD_DAYS) * 24
        return [init0], lead
    rodada = int(settings.get('RODADA', 0))
    num_rodada = int(settings.get('NUM_RODADA', 1))
    lead = int(settings.get('FORECAST_LEAD_DAYS', 10)) * 24
    run_inits = resolve_run_inits(rodada, num_rodada, settings.get('FORECAST_INIT', 'latest'))
    return run_inits, lead


def _forecast_index(model: str, lat: np.ndarray, lon: np.ndarray):
    """Serie do indice AAO previsto pela MEDIA do ensemble do `model` (do init0 ao horizonte)."""
    downloader = _HGT700_DOWNLOADERS[model]
    run_inits, lead_hours = _resolve_model_inits(model)
    per_run = []
    for init_k in run_inits:
        files = list(downloader(
            init=init_k, lead_hours=lead_hours, hours=list(DEFAULT_SYNOPTIC_HOURS)))
        if not files:
            continue
        end_k = init_k + timedelta(hours=lead_hours)
        per_run.append(daily_scalar_on_grid(files, HGT_VARS, init_k, end_k, lat, lon, logger))
    if not per_run:
        raise RuntimeError(f'{model}: nenhum dado de Z700 baixado no periodo.')
    h_da = lagged_ensemble_mean(per_run)
    idx, dates = aao_index_from_height(h_da, logger)
    return pd.to_datetime(dates), idx


def _logo_path():
    """Logo conforme settings (SEM_LOGO > LOGO_GREC > LOGO_AMPERE), igual ao s34."""
    entrada = Path(settings.DIR_INPUT)
    if settings.get('SEM_LOGO', False):
        return None
    if settings.get('LOGO_GREC', False):
        cand = entrada / 'logo_grec.png'
    elif settings.get('LOGO_AMPERE', True):
        cand = entrada / 'novo_logo.png'
    else:
        return None
    return cand if cand.exists() else None


def _plot(obs, series_by_model: dict, out_png: Path):
    """Grafico de linha do indice AAO (observado preto + cada modelo uma cor)."""
    fig, ax = plt.subplots(figsize=(13, 4.6), dpi=130)
    xmax = None

    # Fundo das fases da AAO: acima de +1 vermelho bem claro; abaixo de -1 azul claro.
    ax.axhspan(1, 100, facecolor='#ffd9d9', zorder=0)
    ax.axhspan(-100, -1, facecolor='#d6e4ff', zorder=0)

    obs_dates, obs_idx = obs
    if len(obs_dates):
        ax.plot(obs_dates, obs_idx, color='black', lw=2.2, label='Observado (ERA5/GDAS)', zorder=6)
        xmax = obs_dates.max()

    # Ancora: cada previsao "sai" do ultimo ponto observado (fecha a lacuna de 1 dia entre a
    # ultima analise e o init da rodada e evita o degrau visual observado<->previsto).
    anchor = (obs_dates[-1], obs_idx[-1]) if len(obs_dates) else None
    for model, (dates, idx) in series_by_model.items():
        dates = pd.DatetimeIndex(dates)
        idx = np.asarray(idx)
        if anchor is not None and len(dates):
            # Regra uniforme p/ TODOS os modelos (inclusive o CFS, cujo init e de ontem): descarta
            # os dias que se sobrepoem ao observado e faz a previsao SAIR do ultimo ponto observado.
            keep = dates > anchor[0]
            px = pd.DatetimeIndex([anchor[0]]).append(dates[keep])
            py = np.concatenate([[anchor[1]], idx[keep]])
        else:
            px, py = dates, idx
        ax.plot(px, py, color=_MODEL_COLOR[model], lw=1.6, label=_MODEL_LABEL[model], zorder=4)
        xmax = dates.max() if xmax is None else max(xmax, dates.max())

    # Ensemble multi-modelo: media dos modelos por data — linha tracejada preta. So vale enquanto
    # ha MAIS DE UM modelo (>=2): no trecho de lead longo em que sobra so o CFS, o ensemble e
    # cortado (a propria linha do CFS ja mostra essa parte).
    if series_by_model:
        df = pd.concat(
            {m: pd.Series(np.asarray(idx), index=pd.DatetimeIndex(dates))
             for m, (dates, idx) in series_by_model.items()},
            axis=1,
        ).sort_index()
        ens = df.mean(axis=1, skipna=True)[df.notna().sum(axis=1) >= 2]
        edates, eidx = ens.index, ens.values
        if anchor is not None and len(edates):
            keep = edates > anchor[0]
            epx = pd.DatetimeIndex([anchor[0]]).append(edates[keep])
            epy = np.concatenate([[anchor[1]], eidx[keep]])
        else:
            epx, epy = edates, eidx
        ax.plot(epx, epy, color='black', lw=2.4, ls='--',
                label='Ensemble (media dos modelos)', zorder=5)

    xmin = obs_dates.min() if len(obs_dates) else (
        min(d.min() for d, _ in series_by_model.values()) if series_by_model else None)
    ax.axhline(0, color='0.4', lw=1.0, zorder=2)
    ax.grid(True, ls='--', lw=0.5, color='0.8', zorder=0)
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
    # Eixo y: fixo em -4..4 por padrao; expande (simetrico) so se algum valor passar de |4|.
    vals = [np.nanmax(np.abs(obs_idx))] if len(obs_dates) else []
    vals += [np.nanmax(np.abs(idx)) for _, idx in series_by_model.values() if len(idx)]
    lim = max(4.0, float(np.ceil(max(vals)))) if vals else 4.0
    ax.set_ylim(-lim, lim)
    ax.set_yticks(np.arange(-lim, lim + 0.5, 1))  # rotulos do eixo Y de 1 em 1
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.tick_params(axis='both', labelsize=13)  # rotulos de datas (x) e valores (y) maiores
    ax.set_ylabel('Indice AAO (norm.)', fontsize=15)
    ax.set_title('Índice AAO: Observado & Previsões',
                 fontsize=20, fontweight='bold', color='black', pad=14)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=6,
              fontsize=10, framealpha=0.9, borderaxespad=0.0)

    logo = _logo_path()
    if logo is not None:
        try:
            img = Image.open(logo).convert('RGBA')
            bbox = img.getbbox()
            if bbox:
                img = img.crop(bbox)
            fig.figimage(np.asarray(img), xo=fig.bbox.xmax - img.size[0] - 10,
                         yo=10, zorder=10, alpha=0.9)
        except Exception as exc:  # logo nunca derruba o grafico
            logger.warning('Falha ao inserir logo ({}): {}', logo.name, exc)

    out_png.parent.mkdir(parents=True, exist_ok=True)  # cria a pasta so quando ha figura
    fig.tight_layout()
    fig.savefig(str(out_png), bbox_inches='tight')
    plt.close(fig)
    logger.info('Grafico AAO salvo: {}', out_png)


def main():
    """Entry point — chamado pelo CLI sem argumentos."""
    start_time = time.time()
    hist_days = int(settings.get('AAO_HIST_DAYS', 120))
    models = _enabled_models()
    logger.info('=' * 80)
    logger.info('s35 AAO: historico {} dias | modelos: {}', hist_days, ', '.join(models) or '(nenhum)')

    out_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_AAO_INDEX'
    out_png = out_dir / f'aao_index_{datetime.now():%Y%m%d}.png'
    out_nc = out_dir / f'aao_index_{datetime.now():%Y%m%d}.nc'

    cache_params = {
        'hist_days': hist_days,
        'models': models,
        'forecast_init': str(settings.get('FORECAST_INIT', '')),
        'rodada': int(settings.get('RODADA', 0)),
        'num_rodada': int(settings.get('NUM_RODADA', 1)),
        'lead_days': int(settings.get('FORECAST_LEAD_DAYS', 10)),
        'base': '1991-2020',
        'script_version': SCRIPT_VERSION,
    }
    if check_cache_valid(SCRIPT_ID, cache_params, [str(out_png), str(out_nc)]):
        logger.info('CACHE VALIDO! Execucao ja foi realizada com os mesmos parametros.')
        return

    ensure_aao_loading_pattern()  # constroi/cacheia o EOF1 (ERA5 mensal) na 1a vez
    lat, lon = _ltm_grid()

    logger.info('Etapa 1: indice AAO observado (ultimos {} dias)...', hist_days)
    obs = _observed_index(lat, lon, hist_days)

    series_by_model: dict = {}
    for model in models:
        logger.info('Etapa 2: indice AAO previsto — {}', _MODEL_LABEL[model])
        try:
            series_by_model[model] = _forecast_index(model, lat, lon)
        except Exception as exc:  # isolamento por modelo: um modelo sem Z700 nao derruba o grafico
            logger.warning('Modelo {} ignorado (sem indice AAO): {}', model, exc)

    _plot(obs, series_by_model, out_png)

    # Salva as series num NetCDF (observado + cada modelo) para reuso/auditoria.
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds_vars = {}
    obs_dates, obs_idx = obs
    if len(obs_dates):
        ds_vars['observado'] = xr.DataArray(
            obs_idx, dims=['time_obs'], coords={'time_obs': obs_dates.values})
    for model, (dates, idx) in series_by_model.items():
        ds_vars[model] = xr.DataArray(idx, dims=[f'time_{model}'],
                                      coords={f'time_{model}': dates.values})
    if out_nc.exists():
        out_nc.unlink()
    xr.Dataset(ds_vars).to_netcdf(out_nc)

    save_cache_metadata(SCRIPT_ID, cache_params, [str(out_png), str(out_nc)],
                        execution_time_seconds=time.time() - start_time)
    logger.info('s35 concluido em {:.1f}s', time.time() - start_time)


if __name__ == '__main__':
    main()
