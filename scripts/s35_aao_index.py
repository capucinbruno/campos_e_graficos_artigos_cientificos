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
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.common.logo_helper import proportional_logo_zoom, resolve_logo_path
from app.src.uteis.aao_eof import aao_index_from_height, ensure_aao_loading_pattern
from app.src.uteis.aao_verif_archive import append_fcst, existing_init_dates, upsert_obs
from app.src.uteis.backfill_hgt700_aws import BACKFILL_MODELS, backfill_hgt700_aws
from app.src.uteis.clim_diaria_uv200_ltm import clim_hgt700_daily
from app.src.uteis.forecast_daily import (
    DEFAULT_SYNOPTIC_HOURS,
    daily_scalar_on_grid,
    lagged_ensemble_mean,
    resolve_forecast_lead_init,
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
SCRIPT_VERSION = '3.0'  # + arquivo de verificacao (obs/fcst por-init) p/ heatmap/correcao

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
    """(run_inits, lead_hours) por modelo — CFS sempre 45d; GEFS 15/35d com init D vs D-1."""
    return resolve_forecast_lead_init(
        model,
        rodada=int(settings.get('RODADA', 0)),
        num_rodada=int(settings.get('NUM_RODADA', 1)),
        forecast_init=settings.get('FORECAST_INIT', 'latest'),
        gefs_lead_days=int(settings.get('GEFS_FORECAST_LEAD_DAYS', settings.get('FORECAST_LEAD_DAYS', 35))),
        cfs_lead_days=CFS_LEAD_DAYS,
    )


def _forecast_index(model: str, lat: np.ndarray, lon: np.ndarray):
    """Serie do indice AAO previsto pela MEDIA do ensemble do `model` (do init0 ao horizonte).

    Retorna `(dates, idx, per_init)`: a serie defasada (blend, para o grafico) MAIS a lista
    `per_init` de `(init, dates, idx)` POR rodada — esta com lead bem definido (`alvo - init`),
    usada pelo arquivo de verificacao (heatmap/correcao). O blend nao serve para verificar pois
    mistura varios inits (lead ambiguo)."""
    downloader = _HGT700_DOWNLOADERS[model]
    run_inits, lead_hours = _resolve_model_inits(model)
    per_run = []
    per_init = []
    for init_k in run_inits:
        files = list(downloader(
            init=init_k, lead_hours=lead_hours, hours=list(DEFAULT_SYNOPTIC_HOURS)))
        if not files:
            continue
        end_k = init_k + timedelta(hours=lead_hours)
        h_k = daily_scalar_on_grid(files, HGT_VARS, init_k, end_k, lat, lon, logger)
        per_run.append(h_k)
        idx_k, dates_k = aao_index_from_height(h_k, logger)
        per_init.append((init_k, pd.to_datetime(dates_k), idx_k))
    if not per_run:
        raise RuntimeError(f'{model}: nenhum dado de Z700 baixado no periodo.')
    h_da = lagged_ensemble_mean(per_run)
    idx, dates = aao_index_from_height(h_da, logger)
    return pd.to_datetime(dates), idx, per_init


def _backfill_archive(lat: np.ndarray, lon: np.ndarray, days: int, lead_days: int):
    """Preenche o passado do `fcst_archive.csv` (GFS/GEFS via AWS S3, que retem meses).

    Para cada um dos ultimos `days` inits 00Z ainda ausentes no arquivo, baixa o Z700 ate
    `lead_days`, calcula o indice AAO por-init e anexa. Idempotente: inits ja gravados sao
    pulados (NetCDFs ja baixados tambem sao reusados do cache local)."""
    lead_hours = lead_days * 24
    today0 = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    for model in BACKFILL_MODELS:
        have = existing_init_dates(model)
        n_add = 0
        for d in range(1, days + 1):
            init_k = today0 - timedelta(days=d)
            if init_k.strftime('%Y-%m-%d') in have:
                continue
            try:
                files = backfill_hgt700_aws(model, init_k, lead_hours)
                if not files:
                    continue
                h_k = daily_scalar_on_grid(
                    files, HGT_VARS, init_k, init_k + timedelta(hours=lead_hours), lat, lon, logger)
                idx_k, dates_k = aao_index_from_height(h_k, logger)
                append_fcst(model, [(init_k, pd.to_datetime(dates_k), idx_k)])
                n_add += 1
            except Exception as exc:  # um init falho nao derruba o backfill
                logger.warning('Backfill {} init {:%Y-%m-%d} falhou: {}', model, init_k, exc)
        logger.info('Backfill {}: {} inits novos adicionados ({} ja existiam)',
                    model.upper(), n_add, len(have))


def _logo_path():
    """Logo conforme settings (LOGO_CAPUCIN > LOGO_GREC > LOGO_AMPERE; todas false = sem logo)."""
    p = resolve_logo_path(settings.DIR_INPUT)
    return p if (p is not None and p.exists()) else None


def _plot(obs, series_by_model: dict, out_png: Path):
    """Grafico de linha do indice AAO (observado preto + cada modelo uma cor)."""
    fig, ax = plt.subplots(figsize=(13, 4.6), dpi=130)
    xmax = None

    # Fundo das fases da AAO: acima de +1 vermelho bem claro; abaixo de -1 azul claro.
    ax.axhspan(1, 100, facecolor='#ffd9d9', zorder=0)
    ax.axhspan(-100, -1, facecolor='#d6e4ff', zorder=0)

    obs_dates, obs_idx = obs
    if len(obs_dates):
        ax.plot(obs_dates, obs_idx, color='black', lw=2.2, label='Observado', zorder=6)
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
                label='Ensemble (média dos modelos)', zorder=5)

    xmin = obs_dates.min() if len(obs_dates) else (
        min(d.min() for d, _ in series_by_model.values()) if series_by_model else None)
    ax.axhline(0, color='0.4', lw=1.0, zorder=2)
    ax.grid(True, ls='--', lw=0.5, color='0.8', zorder=0)
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
    # Eixo y RESPONSIVO e ASSIMETRICO: default -4..4; estende (em passos inteiros) SO o lado que
    # ultrapassar — positivo e negativo independentes. Ex.: pico +4.3 -> topo 5 (base segue -4).
    vals = [np.asarray(obs_idx)] if len(obs_dates) else []
    vals += [np.asarray(idx) for _, idx in series_by_model.values() if len(idx)]
    allv = np.concatenate(vals) if vals else np.array([0.0])
    allv = allv[np.isfinite(allv)]
    ymax = max(4.0, float(np.ceil(np.nanmax(allv)))) if allv.size else 4.0
    ymin = min(-4.0, float(np.floor(np.nanmin(allv)))) if allv.size else -4.0
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.arange(ymin, ymax + 0.5, 1))  # rotulos do eixo Y de 1 em 1
    # Eixo x: marca o dia 1 e o dia 15 de cada mes (de 15 em 15 dias / quinzenas).
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
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
            im = Image.open(logo).convert('RGBA')
            b = im.getbbox()
            if b:
                im = im.crop(b)
            # canto inferior ESQUERDO da area de plotagem
            ab = AnnotationBbox(OffsetImage(np.asarray(im), zoom=proportional_logo_zoom(
                                    ax, np.asarray(im).shape[1])), (0, 0),
                                xycoords=ax.transAxes, xybox=(5, 5), boxcoords='offset points',
                                box_alignment=(0, 0), frameon=False, pad=0, zorder=10, clip_on=False)
            ax.add_artist(ab)
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

    # Backfill (one-off): preenche o passado do arquivo de verificacao via AWS S3 (GFS/GEFS).
    # Roda ANTES do cache (acao de manutencao, nao gerada pelo cache do grafico). Idempotente.
    if bool(settings.get('S35_BACKFILL', False)):
        ensure_aao_loading_pattern()
        lat_bf, lon_bf = _ltm_grid()
        logger.info('S35_BACKFILL=true: preenchendo {} dias de passado (AWS S3, GFS/GEFS)...',
                    int(settings.get('S35_BACKFILL_DAYS', 60)))
        _backfill_archive(lat_bf, lon_bf,
                          int(settings.get('S35_BACKFILL_DAYS', 60)),
                          int(settings.get('S35_BACKFILL_LEAD_DAYS', 16)))

    out_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_AAO_INDEX'
    out_png = out_dir / f'aao_index_{datetime.now():%Y%m%d}.png'
    out_nc = out_dir / f'aao_index_{datetime.now():%Y%m%d}.nc'

    cache_params = {
        'hist_days': hist_days,
        'models': models,
        'forecast_init': str(settings.get('FORECAST_INIT', '')),
        'rodada': int(settings.get('RODADA', 0)),
        'num_rodada': int(settings.get('NUM_RODADA', 1)),
        'lead_days': int(settings.get('GEFS_FORECAST_LEAD_DAYS', settings.get('FORECAST_LEAD_DAYS', 35))),
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
    # Arquivo de verificacao: registra o observado do dia (ERA5 supera GDAS numa re-rodada).
    # Em cache-hit o main retorna antes daqui, mas a 1a rodada do dia ja gravou (dedup protege).
    upsert_obs(obs[0], obs[1])

    series_by_model: dict = {}
    for model in models:
        logger.info('Etapa 2: indice AAO previsto — {}', _MODEL_LABEL[model])
        try:
            dates, idx, per_init = _forecast_index(model, lat, lon)
            series_by_model[model] = (dates, idx)
            append_fcst(model, per_init)  # acumula previsao por-init p/ heatmap/correcao
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
