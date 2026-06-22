# -*- coding: utf-8 -*-
"""
s36 — Indice da MJO: diagrama de fase RMM1xRMM2 (Wheeler & Hendon 2004).

EOF combinado PROPRIO (ver app/src/uteis/mjo_rmm.py): OLR (CPC Blended) + U850 + U200 (NCEP),
media meridional 15S-15N, anomalia (clima do dia-do-ano) - media dos 120 dias anteriores,
normalizada e projetada nos 2 primeiros EOFs combinados -> RMM1, RMM2.

Gera UM diagrama de fase (estilo CPC/BoM):
- trajetoria PRETA = observado (CPC OLR + NCEP) dos ultimos MJO_HIST_DAYS dias (default 40);
- uma trajetoria COLORIDA por modelo de previsao habilitado (so a MEDIA do ensemble), do init
  ate o horizonte proprio do modelo (GEFS 35d / CFS 45d / demais ~16d);
- linha tracejada preta = ENSEMBLE (media dos modelos por data, onde ha >=2 modelos).

So entram os 5 modelos FISICOS com OLR (GFS, GEFS, ECMWF-HRES, ECMWF-ENS, CFS); os modelos de
IA nao publicam OLR no open data -> sem RMM (pulados, como no s34).
"""

from __future__ import annotations

import matplotlib

matplotlib.use('Agg')  # backend nao-interativo (downloads em threads quebram o Tk)

import time
from datetime import datetime, timedelta
from pathlib import Path

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
from app.src.uteis.forecast_daily import (
    DEFAULT_SYNOPTIC_HOURS,
    daily_scalar_on_grid,
    daily_uv200_on_grid,
    resolve_forecast_lead_init,
)
from app.src.uteis.mjo_rmm import (
    BASE_FIM,
    BASE_INI,
    RUNNING_DAYS,
    _load_olr_band,
    band_mean_on_grid,
    ensure_mjo_eof,
    rmm_from_bands,
)
# Ventos do OBSERVADO recente: ERA5 (latencia ~5 dias) — a reanalise NCEP R1 atrasa ~3 meses,
# entao serve so p/ TREINAR o EOF (historico), nao para o observado em tempo real.
from app.src.uteis.downloaders_wind200 import ensure_era5_uv200_for_period
from app.src.uteis.downloaders_wind850 import ensure_era5_uv850_for_period

# Downloaders de previsao (OLR, U850, U200) — so os 5 modelos com OLR.
from app.src.uteis.downloaders_cfs_ensemble import (
    CFS_LEAD_DAYS,
    ensure_cfs_fcst200_for_period,
    ensure_cfs_olr_for_period,
    ensure_cfs_uv850_for_period,
)
from app.src.uteis.downloaders_ecmwf_ens import (
    ensure_ecmwf_ens_fcst200_for_period,
    ensure_ecmwf_ens_olr_fcst_for_period,
    ensure_ecmwf_ens_uv850_fcst_for_period,
)
from app.src.uteis.downloaders_ecmwf_fcst200 import ensure_ecmwf_fcst200_for_period
from app.src.uteis.downloaders_ecmwf_olr import ensure_ecmwf_olr_fcst_for_period
from app.src.uteis.downloaders_ecmwf_uv850 import ensure_ecmwf_uv850_fcst_for_period
from app.src.uteis.downloaders_gefs_fcst200 import ensure_gefs_fcst200_for_period
from app.src.uteis.downloaders_gefs_olr import ensure_gefs_olr_fcst_for_period
from app.src.uteis.downloaders_gefs_uv850 import ensure_gefs_uv850_fcst_for_period
from app.src.uteis.downloaders_gfs_fcst200 import ensure_gfs_fcst200_for_period
from app.src.uteis.downloaders_gfs_olr import ensure_gfs_olr_fcst_for_period
from app.src.uteis.downloaders_gfs_uv850 import ensure_gfs_uv850_fcst_for_period

logger = get_logger('s36')

SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's36'
SCRIPT_VERSION = '1.6'  # lead/init por modelo (CFS 45d; GEFS 15/35d, init D vs D-1)
OLR_VARS = ('olr', 'OLR', 'ulwrf', 'avg_ulwrf', 'ttr', 'sulwrf')
U_VARS = ('u', 'u_component_of_wind', 'U_GRD_L100', 'ugrd', 'UGRD', 'uwnd')  # so o zonal (RMM nao usa v)
OBS_DAYS_FETCH = RUNNING_DAYS + 90  # janela observada baixada (120d p/ a media movel + folga)
ERA5_LATENCY_DAYS = 8               # ERA5 so tem dia sinotico completo ate ~hoje-7

# Modelos: flag -> (downloader OLR, downloader U850, downloader U200), cor e rotulo. So os com OLR.
_MODEL_FLAGS = {
    'gfs': ('RUN_GFS', True), 'gefs': ('RUN_GEFS', False),
    'ecmwf': ('RUN_ECMWF', False), 'ecmwf_ens': ('RUN_ECMWF_ENS', False),
    'cfs': ('RUN_CFS', False),
}
_DOWNLOADERS = {
    'gfs': (ensure_gfs_olr_fcst_for_period, ensure_gfs_uv850_fcst_for_period,
            ensure_gfs_fcst200_for_period),
    'gefs': (ensure_gefs_olr_fcst_for_period, ensure_gefs_uv850_fcst_for_period,
             ensure_gefs_fcst200_for_period),
    'ecmwf': (ensure_ecmwf_olr_fcst_for_period, ensure_ecmwf_uv850_fcst_for_period,
              ensure_ecmwf_fcst200_for_period),
    'ecmwf_ens': (ensure_ecmwf_ens_olr_fcst_for_period, ensure_ecmwf_ens_uv850_fcst_for_period,
                  ensure_ecmwf_ens_fcst200_for_period),
    'cfs': (ensure_cfs_olr_for_period, ensure_cfs_uv850_for_period,
            ensure_cfs_fcst200_for_period),
}
_MODEL_COLOR = {
    'gfs': '#1f77b4', 'gefs': '#2ca02c', 'ecmwf': '#d62728', 'ecmwf_ens': '#ff7f0e',
    'cfs': '#17becf',
}
_MODEL_LABEL = {
    'gfs': 'GFS', 'gefs': 'GEFS', 'ecmwf': 'ECMWF-HRES', 'ecmwf_ens': 'ECMWF-ENS',
    'cfs': 'CFS (45d)',
}

# Rotulos das fases (octantes) e regioes, na convencao Wheeler-Hendon (= figura do CPC).
_PHASE_ANGLE = {5: 22.5, 6: 67.5, 7: 112.5, 8: 157.5, 1: 202.5, 2: 247.5, 3: 292.5, 4: 337.5}
_REGION_LABELS = [
    (0, -3.75, 'Indian Ocean', 0),
    (3.75, 0, 'Maritime Continent', 90),
    (0, 3.75, 'Western Pacific', 0),
    (-3.75, 0, 'West. Hem. and Africa', 90),
]


def _enabled_models() -> list:
    """Modelos habilitados pelas flags do settings (ordem de _MODEL_FLAGS) — so os com OLR."""
    return [m for m, (flag, default) in _MODEL_FLAGS.items() if bool(settings.get(flag, default))]


def _target_grid():
    """Grade global 2.5 (lat ascendente, lon 0..357.5) — alvo do regrid dos campos de previsao."""
    eof = ensure_mjo_eof()
    lon = eof['lon'].values
    lat = np.arange(-90.0, 90.01, 2.5)
    return lat, lon, eof


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


def _obs_bands(end: datetime, lat: np.ndarray, lon: np.ndarray):
    """Series observadas (time, lon) media-meridional 15S-15N: OLR (CPC) + U850/U200 (ERA5)."""
    ini = end - timedelta(days=OBS_DAYS_FETCH)
    wind_end = end - timedelta(days=ERA5_LATENCY_DAYS)  # ERA5 nao tem dia sinotico completo ate hoje
    olr = _load_olr_band().sel(time=slice(ini.date().isoformat(), end.date().isoformat()))
    hrs = list(DEFAULT_SYNOPTIC_HOURS)
    u850_files = list(ensure_era5_uv850_for_period(start=ini, end=wind_end, hours_utc=hrs))
    u200_files = list(ensure_era5_uv200_for_period(start=ini, end=wind_end, hours_utc=hrs))
    u850 = band_mean_on_grid(daily_scalar_on_grid(u850_files, U_VARS, ini, end, lat, lon, logger))
    u200 = band_mean_on_grid(daily_scalar_on_grid(u200_files, U_VARS, ini, end, lat, lon, logger))
    return olr, u850, u200


def _forecast_bands(model: str, lat: np.ndarray, lon: np.ndarray):
    """Series de previsao (time, lon) media-meridional do `model`: (olr, u850, u200)."""
    dl_olr, dl_u850, dl_u200 = _DOWNLOADERS[model]
    run_inits, lead_hours = _resolve_model_inits(model)
    init0 = run_inits[0]
    end = init0 + timedelta(hours=lead_hours)
    hrs = list(DEFAULT_SYNOPTIC_HOURS)

    olr_files = list(dl_olr(init=init0, lead_hours=lead_hours, hours=hrs))
    u850_files = list(dl_u850(init=init0, lead_hours=lead_hours, hours=hrs))
    u200_files = list(dl_u200(init=init0, lead_hours=lead_hours, hours=hrs))
    if not (olr_files and u850_files and u200_files):
        raise RuntimeError(f'{model}: faltou OLR/U850/U200 de previsao.')

    olr_g = daily_scalar_on_grid(olr_files, OLR_VARS, init0, end, lat, lon, logger)
    u850_g = daily_scalar_on_grid(u850_files, U_VARS, init0, end, lat, lon, logger)
    u200_g = daily_scalar_on_grid(u200_files, U_VARS, init0, end, lat, lon, logger)
    return (band_mean_on_grid(olr_g), band_mean_on_grid(u850_g), band_mean_on_grid(u200_g), init0)


def _concat_band(obs: xr.DataArray, fcst: xr.DataArray, init: datetime) -> xr.DataArray:
    """Emenda observado (ate init-1) + previsao (do init) numa serie (time, lon)."""
    o = obs.sel(time=slice(None, (init - timedelta(days=1)).date().isoformat()))
    return xr.concat([o, fcst], dim='time').sortby('time')


def _phase(rmm1: np.ndarray, rmm2: np.ndarray) -> np.ndarray:
    """Fase 1..8 (octante) na convencao Wheeler-Hendon."""
    ang = (np.degrees(np.arctan2(rmm2, rmm1)) % 360)
    edges = {5: (0, 45), 6: (45, 90), 7: (90, 135), 8: (135, 180),
             1: (180, 225), 2: (225, 270), 3: (270, 315), 4: (315, 360)}
    out = np.zeros(len(ang), dtype=int)
    for ph, (a0, a1) in edges.items():
        out[(ang >= a0) & (ang < a1)] = ph
    return out


def _logo_path():
    """Logo conforme settings (LOGO_CAPUCIN > LOGO_GREC > LOGO_AMPERE; todas false = sem logo)."""
    p = resolve_logo_path(settings.DIR_INPUT)
    return p if (p is not None and p.exists()) else None


def _draw_phase_space(ax):
    """Desenha o fundo do diagrama de fase: circulo unitario, divisoes, rotulos de fase/regiao."""
    lim = 4.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color='0.5', lw=1.0, zorder=1)  # circulo |RMM|=1
    # divisoes: eixos e diagonais (so fora do circulo unitario)
    for ang in (0, 45, 90, 135):
        dx, dy = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        for s in (1, -1):
            ax.plot([s * dx, s * dx * lim], [s * dy, s * dy * lim], color='0.8', lw=0.8, zorder=1)
    ax.axhline(0, color='0.8', lw=0.8, zorder=0)
    ax.axvline(0, color='0.8', lw=0.8, zorder=0)
    for ph, a in _PHASE_ANGLE.items():
        ax.text(3.4 * np.cos(np.deg2rad(a)), 3.4 * np.sin(np.deg2rad(a)), str(ph),
                ha='center', va='center', fontsize=13, color='0.45', zorder=2)
    for x, y, txt, rot in _REGION_LABELS:
        ax.text(x, y, txt, ha='center', va='center', rotation=rot, fontsize=12,
                fontweight='bold', color='0.3', zorder=2)
    ax.set_xlabel('RMM1', fontsize=14)
    ax.set_ylabel('RMM2', fontsize=14)
    ax.set_xticks(range(-4, 5))
    ax.set_yticks(range(-4, 5))
    ax.tick_params(labelsize=11)


def _plot(obs, series_by_model: dict, init_label: str, out_png: Path):
    """Diagrama de fase RMM (observado preto + cada modelo uma cor + ensemble tracejado)."""
    fig, ax = plt.subplots(figsize=(9.2, 9.6), dpi=130)
    _draw_phase_space(ax)

    obs_dates, o1, o2 = obs
    anchor = None
    if len(o1):
        ax.plot(o1, o2, color='black', lw=2.4, zorder=6, label='Observado')
        ax.scatter(o1, o2, color='black', s=14, zorder=6)
        ax.scatter([o1[-1]], [o2[-1]], color='black', s=80, zorder=7, edgecolor='white')
        anchor = (o1[-1], o2[-1])
        ax.annotate(pd.Timestamp(obs_dates[-1]).strftime('%d/%m'),
                    (o1[-1], o2[-1]), textcoords='offset points', xytext=(6, 6), fontsize=9)

    # Ensemble (media dos modelos por data, onde ha >=2 modelos) — linha tracejada preta.
    frames = {}
    for model, (dates, r1, r2) in series_by_model.items():
        if len(r1) == 0:
            continue
        idx = pd.DatetimeIndex(dates)
        frames[model] = pd.DataFrame({'r1': r1, 'r2': r2}, index=idx)
        x = np.concatenate([[anchor[0]], r1]) if anchor is not None else r1
        y = np.concatenate([[anchor[1]], r2]) if anchor is not None else r2
        ax.plot(x, y, color=_MODEL_COLOR[model], lw=1.7, zorder=4, label=_MODEL_LABEL[model])
        ax.scatter([r1[-1]], [r2[-1]], color=_MODEL_COLOR[model], s=45, zorder=5, edgecolor='white')

    if len(frames) >= 2:
        r1m = pd.concat({m: f['r1'] for m, f in frames.items()}, axis=1).sort_index()
        r2m = pd.concat({m: f['r2'] for m, f in frames.items()}, axis=1).sort_index()
        keep = r1m.notna().sum(axis=1) >= 2
        e1, e2 = r1m[keep].mean(axis=1).values, r2m[keep].mean(axis=1).values
        x = np.concatenate([[anchor[0]], e1]) if anchor is not None else e1
        y = np.concatenate([[anchor[1]], e2]) if anchor is not None else e2
        ax.plot(x, y, color='black', lw=2.2, ls='--', zorder=5, label='Ensemble (média dos modelos)')

    ax.set_title(f'Índice MJO (RMM) — Observado & Previsões\ninit {init_label}',
                 fontsize=17, fontweight='bold', pad=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=9,
              framealpha=0.9, borderaxespad=0.0)

    logo = _logo_path()
    if logo is not None:
        try:
            im = Image.open(logo).convert('RGBA')
            b = im.getbbox()
            if b:
                im = im.crop(b)
            # FORA da area do diagrama: canto inferior-ESQUERDO da figura, abaixo das informacoes
            ab = AnnotationBbox(OffsetImage(np.asarray(im), zoom=proportional_logo_zoom(
                                    ax, np.asarray(im).shape[1])), (0.0, 0.0),
                                xycoords='figure fraction', xybox=(8, 8), boxcoords='offset points',
                                box_alignment=(0, 0), frameon=False, pad=0, zorder=10, clip_on=False)
            fig.add_artist(ab)
        except Exception as exc:  # logo nunca derruba o grafico
            logger.warning('Falha ao inserir logo ({}): {}', logo.name, exc)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_png), bbox_inches='tight')
    plt.close(fig)
    logger.info('Diagrama MJO (RMM) salvo: {}', out_png)


def main():
    """Entry point — chamado pelo CLI sem argumentos."""
    start_time = time.time()
    hist_days = int(settings.get('MJO_HIST_DAYS', 40))
    models = _enabled_models()
    logger.info('=' * 80)
    logger.info('s36 MJO (RMM): historico {} dias | modelos: {}', hist_days, ', '.join(models) or '(nenhum)')

    out_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_MJO_INDEX'
    out_png = out_dir / f'mjo_rmm_{datetime.now():%Y%m%d}.png'
    out_nc = out_dir / f'mjo_rmm_{datetime.now():%Y%m%d}.nc'

    cache_params = {
        'hist_days': hist_days,
        'models': models,
        'forecast_init': str(settings.get('FORECAST_INIT', '')),
        'rodada': int(settings.get('RODADA', 0)),
        'num_rodada': int(settings.get('NUM_RODADA', 1)),
        'lead_days': int(settings.get('GEFS_FORECAST_LEAD_DAYS', settings.get('FORECAST_LEAD_DAYS', 35))),
        'base': f'{BASE_INI}-{BASE_FIM}',
        'script_version': SCRIPT_VERSION,
    }
    if check_cache_valid(SCRIPT_ID, cache_params, [str(out_png), str(out_nc)]):
        logger.info('CACHE VALIDO! Execucao ja foi realizada com os mesmos parametros.')
        return

    lat, lon, eof = _target_grid()  # constroi/cacheia o EOF combinado na 1a vez

    logger.info('Etapa 1: RMM observado (CPC OLR + NCEP), ultimos {} dias...', hist_days)
    obs_olr, obs_u850, obs_u200 = _obs_bands(datetime.now(), lat, lon)
    od, oo1, oo2 = rmm_from_bands(obs_olr, obs_u850, obs_u200, eof, logger)
    od = pd.DatetimeIndex(od)
    if len(od):
        msk = od >= (od.max() - pd.Timedelta(days=hist_days))
        obs = (od[msk].values, oo1[msk], oo2[msk])
        logger.info('  Observado RMM: {} dias ({} a {})', int(msk.sum()),
                    od[msk][0].strftime('%d/%m'), od[msk][-1].strftime('%d/%m'))
    else:
        obs = (np.array([], dtype='datetime64[ns]'), np.array([]), np.array([]))
        logger.warning('Observado RMM vazio (sem janela de 120 dias completa).')

    series_by_model: dict = {}
    init_label = '—'
    for model in models:
        logger.info('Etapa 2: RMM previsto — {}', _MODEL_LABEL[model])
        try:
            f_olr, f_u850, f_u200, init0 = _forecast_bands(model, lat, lon)
            c_olr = _concat_band(obs_olr, f_olr, init0)
            c_u850 = _concat_band(obs_u850, f_u850, init0)
            c_u200 = _concat_band(obs_u200, f_u200, init0)
            d, r1, r2 = rmm_from_bands(c_olr, c_u850, c_u200, eof, logger)
            d = pd.DatetimeIndex(d)
            # Descarta o dia do init (lead 0): a OLR em f000 e degenerada (acumulacao comeca do
            # zero) e gera um ponto RMM espurio. A trajetoria comeca no 1o dia cheio (init+1).
            keep = d > pd.Timestamp(init0.date())
            if not keep.any():
                logger.warning('Modelo {} sem datas de previsao validas (janela de 120d) — pulado.', model)
                continue
            series_by_model[model] = (d[keep].values, r1[keep], r2[keep])
            init_label = pd.Timestamp(init0).strftime('%d/%m/%Y %HZ')
            logger.info('  {} RMM: {} dias previstos ({} a {})', _MODEL_LABEL[model], int(keep.sum()),
                        pd.Timestamp(d[keep][0]).strftime('%d/%m'),
                        pd.Timestamp(d[keep][-1]).strftime('%d/%m'))
        except Exception as exc:  # isolamento por modelo: um modelo sem RMM nao derruba o grafico
            logger.warning('Modelo {} ignorado (sem RMM): {}', model, exc)

    _plot(obs, series_by_model, init_label, out_png)

    # Salva as series RMM num NetCDF (observado + cada modelo).
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds_vars = {}
    if len(obs[0]):
        ds_vars['obs_rmm1'] = xr.DataArray(obs[1], dims=['time_obs'], coords={'time_obs': obs[0]})
        ds_vars['obs_rmm2'] = xr.DataArray(obs[2], dims=['time_obs'], coords={'time_obs': obs[0]})
    for model, (dates, r1, r2) in series_by_model.items():
        ds_vars[f'{model}_rmm1'] = xr.DataArray(r1, dims=[f't_{model}'], coords={f't_{model}': dates})
        ds_vars[f'{model}_rmm2'] = xr.DataArray(r2, dims=[f't_{model}'], coords={f't_{model}': dates})
    if out_nc.exists():
        out_nc.unlink()
    xr.Dataset(ds_vars).to_netcdf(out_nc)

    save_cache_metadata(SCRIPT_ID, cache_params, [str(out_png), str(out_nc)],
                        execution_time_seconds=time.time() - start_time)
    logger.info('s36 concluido em {:.1f}s', time.time() - start_time)


if __name__ == '__main__':
    main()
