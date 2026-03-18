"""s00 - Vento 100m + MSLP (ERA5).

Baixa dados de vento a 100m (u/v) e pressao ao nivel do mar (MSLP) da reanalise ERA5
via Copernicus CDS, processa os campos e gera mapas de magnitude do vento com barbelas
e isobaras de MSLP para diversas regioes geograficas configuradas em settings.json.

Tambem gera serie temporal de vento 100m e compara com a climatologia 1991-2020.

Dados de entrada:
    - ERA5 (CDS): vento 100m (u100/v100) e MSLP
    - Climatologia vento 100m (arquivo fixo em Entrada/)

Saida:
    - Mapas PNG em Saida/ (um por regiao)

Criado em:     2026-02-23
Atualizado em: 2026-03-18
"""

# Bibliotecas padrão
import re
import time
from datetime import datetime
from pathlib import Path

# Bibliotecas de terceiros
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.ndimage
import xarray as xr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from numpy import ma

# Módulos locais
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.downloaders_wind100m_ERA5 import (
    ensure_era5_mslp_u100_v100_for_period,
)
from app.src.uteis.processa_wind100m_ERA5 import (
    build_daily_mean_dataset_from_monthly_files,
)

# ---------------------------------------------------------------------------
# Identidade do script (derivada do nome do arquivo)
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's00'
SCRIPT_NAME = Path(__file__).stem  # 's00_plotagem_vento_eraa5'
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME


# =============================================================================
# Helpers
# =============================================================================
def _parse_date(s: str) -> datetime:
    """
    Espera formato YYYY-MM-DD.
    """
    return datetime.strptime(s, '%Y-%m-%d')


def _normalize_lon(ds: xr.Dataset) -> xr.Dataset:
    """
    Converte longitude para [-180, 180] e ordena, se houver coord lon/longitude.
    """
    lon_name = None
    for cand in ('lon', 'longitude'):
        if cand in ds.coords:
            lon_name = cand
            break

    if lon_name is None:
        return ds

    ds = ds.copy()
    ds.coords[lon_name] = ((ds.coords[lon_name] + 180) % 360) - 180
    ds = ds.sortby(lon_name)
    return ds


def _normalize_latlon_names(ds: xr.Dataset) -> xr.Dataset:
    """
    Padroniza nomes de coordenadas para lat/lon.
    """
    rename_map = {}
    if 'latitude' in ds.coords and 'lat' not in ds.coords:
        rename_map['latitude'] = 'lat'
    if 'longitude' in ds.coords and 'lon' not in ds.coords:
        rename_map['longitude'] = 'lon'

    if rename_map:
        ds = ds.rename(rename_map)

    return ds


def _ensure_time_coord(obj):
    """
    valid_time -> time + decode CF
    """
    if hasattr(obj, 'dims') and 'time' not in obj.dims and 'valid_time' in obj.dims:
        obj = obj.rename({'valid_time': 'time'})
    elif hasattr(obj, 'coords') and 'time' not in obj.coords and 'valid_time' in obj.coords:
        obj = obj.rename({'valid_time': 'time'})

    if 'time' not in obj.coords:
        raise KeyError("Nem 'time' nem 'valid_time' encontrados.")

    try:
        if not np.issubdtype(obj['time'].dtype, np.datetime64):
            if isinstance(obj, xr.DataArray):
                obj = xr.decode_cf(obj.to_dataset(name='_tmp'))['_tmp']
            else:
                obj = xr.decode_cf(obj)
    except Exception:
        pass

    return obj


def _subset_period(ds: xr.Dataset, dt_ini: datetime, dt_fim: datetime) -> xr.Dataset:
    ds = _ensure_time_coord(ds)
    return ds.sel(time=slice(np.datetime64(dt_ini.date()), np.datetime64(dt_fim.date())))


def _drop_feb29_if_present(ds: xr.Dataset, logger) -> xr.Dataset:
    """
    Remove 29/02 para compatibilizar com climatologia 365d.
    """
    ds = _ensure_time_coord(ds)
    t = xr.DataArray(ds['time'].values, dims=['time'])
    mask = ~((t.dt.month == 2) & (t.dt.day == 29))
    n_before = ds.sizes.get('time', 0)
    ds2 = ds.isel(time=mask.values)
    n_after = ds2.sizes.get('time', 0)

    if n_after < n_before:
        logger.warning(
            'Removendo %d registro(s) de 29/02 da série diária para compatibilizar com climatologia 365d.',
            n_before - n_after,
        )
    return ds2


def _pick_var(ds: xr.Dataset, candidates: list[str], required: bool = True):
    for vn in candidates:
        if vn in ds.data_vars:
            return ds[vn]
    if required:
        raise KeyError(f'Nenhuma variável encontrada. Candidatas: {candidates}')
    return None


def _load_climatology_wind100m(path_clim: Path, logger) -> xr.Dataset:
    """
    Lê climatologia de vento 100m e tenta identificar:
    - velocidade diretamente
    ou
    - componentes u/v para derivar velocidade

    Espera time diário (365 dias) OU dayofyear.
    """
    if not path_clim.exists():
        raise FileNotFoundError(f'Climatologia não encontrada: {path_clim}')

    logger.info('Abrindo climatologia de vento 100m: %s', path_clim)
    ds = xr.open_dataset(path_clim, engine='netcdf4')
    ds = _normalize_latlon_names(ds)
    ds = _normalize_lon(ds)

    try:
        ds = _ensure_time_coord(ds)
    except Exception:
        pass

    logger.info('Variáveis na climatologia: %s', list(ds.data_vars))
    logger.info('Coords na climatologia: %s', list(ds.coords))
    return ds


def _get_clim_speed_da(ds_clim: xr.Dataset, logger) -> xr.DataArray:
    """
    Retorna DataArray de velocidade climatológica 100m.
    Aceita:
    - velocidade pronta (ws100 etc.)
    - componentes u/v (u100/v100)
    - nomes genéricos (ex.: var246/var247) -> tenta derivar velocidade
    """

    speed = _pick_var(
        ds_clim,
        candidates=[
            'ws100',
            'si100',
            'wind_speed_100m',
            'vento100m',
            'wind100m',
            'mag100',
            'speed',
        ],
        required=False,
    )
    if speed is not None:
        logger.info('Velocidade climatológica encontrada diretamente: %s', speed.name)
        return speed

    u = _pick_var(
        ds_clim,
        candidates=['u100', '100m_u_component_of_wind', 'uwnd', 'u'],
        required=False,
    )
    v = _pick_var(
        ds_clim,
        candidates=['v100', '100m_v_component_of_wind', 'vwnd', 'v'],
        required=False,
    )

    if u is not None and v is not None:
        logger.info('Derivando velocidade climatológica a partir de %s e %s', u.name, v.name)
        speed = np.hypot(u, v)
        speed.name = 'ws100_clim'
        return speed

    if 'var246' in ds_clim.data_vars and 'var247' in ds_clim.data_vars:
        logger.warning(
            'Climatologia com nomes genéricos detectada (var246/var247). '
            'Assumindo componentes de vento 100m e derivando velocidade.'
        )
        u = ds_clim['var246']
        v = ds_clim['var247']
        speed = np.hypot(u, v)
        speed.name = 'ws100_clim'
        return speed

    numeric_3d = []
    for vn, da in ds_clim.data_vars.items():
        if np.issubdtype(da.dtype, np.number) and len(da.dims) == 3:
            numeric_3d.append((vn, da))

    if len(numeric_3d) == 2:
        logger.warning(
            'Assumindo que as duas variáveis 3D numéricas da climatologia são componentes U/V: %s e %s',
            numeric_3d[0][0],
            numeric_3d[1][0],
        )
        u = numeric_3d[0][1]
        v = numeric_3d[1][1]
        speed = np.hypot(u, v)
        speed.name = 'ws100_clim'
        return speed

    raise KeyError(
        'Não foi possível identificar velocidade climatológica 100m nem componentes u/v na climatologia. '
        f'Variáveis disponíveis: {list(ds_clim.data_vars)}'
    )


def _select_climatology_same_days(
    clim_speed: xr.DataArray,
    daily_dates: xr.DataArray,
    logger,
) -> xr.DataArray:
    """
    Seleciona na climatologia os mesmos dias do período (por mês-dia).
    Funciona para climatologia com:
    - coord time (365 dias, ex. ano 2001)
    - coord dayofyear (1..365)
    """
    if 'time' in clim_speed.coords:
        clim_speed = _ensure_time_coord(clim_speed)
        clim_dates = xr.DataArray(clim_speed['time'].values, dims=['time'])

        clim_md = pd.Index(
            [f'{pd.Timestamp(t).month:02d}-{pd.Timestamp(t).day:02d}' for t in clim_dates.values],
            name='monthday',
        )
        target_md = [
            f'{pd.Timestamp(t).month:02d}-{pd.Timestamp(t).day:02d}' for t in daily_dates.values
        ]

        md_to_idx = {}
        for i, md in enumerate(clim_md.tolist()):
            md_to_idx[md] = i

        missing = [md for md in target_md if md not in md_to_idx]
        if missing:
            raise KeyError(
                f'Climatologia não contém todos os month-days necessários. Faltando (ex.): {missing[:5]}'
            )

        idxs = [md_to_idx[md] for md in target_md]
        clim_sel = clim_speed.isel(time=idxs)

        clim_sel = clim_sel.assign_coords(time=daily_dates.values)
        logger.info("Climatologia selecionada por month-day usando coord 'time'.")
        return clim_sel

    if 'dayofyear' in clim_speed.coords:
        target_doy = pd.DatetimeIndex(pd.to_datetime(daily_dates.values)).dayofyear.to_numpy()

        if np.any(target_doy == 60) and np.any(
            (pd.DatetimeIndex(pd.to_datetime(daily_dates.values)).month == 2)
            & (pd.DatetimeIndex(pd.to_datetime(daily_dates.values)).day == 29)
        ):
            raise ValueError('Há 29/02 no período, mas a climatologia parece ser 365d (dayofyear).')

        clim_sel = clim_speed.sel(dayofyear=xr.DataArray(target_doy, dims=['time']))
        clim_sel = clim_sel.assign_coords(time=daily_dates.values)
        logger.info("Climatologia selecionada por coord 'dayofyear'.")
        return clim_sel

    raise KeyError("Climatologia precisa ter coord 'time' ou 'dayofyear'.")


def _save_daily_temp_file(base_dir: Path, dt_ini: datetime, dt_fim: datetime) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f'era5_mslp_u100_v100_dailymean_{dt_ini:%Y%m%d}_{dt_fim:%Y%m%d}.nc'


def _set_extent_from_area(ax, area_cfg: dict):
    """
    Define a área do mapa a partir da configuração interna do script.
    """
    ax.set_extent(
        [
            area_cfg['lon_min'],
            area_cfg['lon_max'],
            area_cfg['lat_min'],
            area_cfg['lat_max'],
        ],
        crs=ccrs.PlateCarree(),
    )


def _add_auto_gridlines(ax):
    """
    Adiciona gridlines com rótulos automáticos e fonte 14.
    """
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.7,
        color='gray',
        alpha=0.5,
        linestyle='--',
    )

    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 22}
    gl.ylabel_style = {'size': 22}
    gl.xlocator = mticker.AutoLocator()
    gl.ylocator = mticker.AutoLocator()

    return gl


def _sanitize_filename(name: str) -> str:
    """
    Sanitiza nome para uso em arquivo.
    """
    name = str(name).strip()
    name = re.sub(r'[^\w\-]+', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')


def _extract_point_series(
    ws100_daily: xr.DataArray,
    clim_speed_period: xr.DataArray,
    lat_point: float,
    lon_point: float,
    logger,
):
    """
    Extrai série temporal no ponto mais próximo da grade para:
    - vento diário observado
    - vento climatológico diário
    """
    if 'lat' not in ws100_daily.coords or 'lon' not in ws100_daily.coords:
        raise KeyError("ws100_daily precisa ter coordenadas 'lat' e 'lon'.")

    if 'lat' not in clim_speed_period.coords or 'lon' not in clim_speed_period.coords:
        raise KeyError("clim_speed_period precisa ter coordenadas 'lat' e 'lon'.")

    ws_point = ws100_daily.sel(lat=lat_point, lon=lon_point, method='nearest')
    clim_point = clim_speed_period.sel(lat=lat_point, lon=lon_point, method='nearest')

    lat_sel = float(ws_point['lat'].values)
    lon_sel = float(ws_point['lon'].values)

    logger.info(
        'Ponto solicitado (lat=%.4f, lon=%.4f) -> grade ERA5 mais próxima (lat=%.4f, lon=%.4f)',
        lat_point,
        lon_point,
        lat_sel,
        lon_sel,
    )

    df = pd.DataFrame({
        'data': pd.to_datetime(ws_point['time'].values),
        'vento_observado': ws_point.values.astype(float),
        'vento_climatologico': clim_point.values.astype(float),
    })

    return df, lat_sel, lon_sel


def _plot_daily_wind_timeseries(
    df: pd.DataFrame,
    ponto_cfg: dict,
    output_path: Path,
    logger,
    lat_grid: float,
    lon_grid: float,
):
    """
    Gera gráfico com:
    - barras: vento diário observado
    - linha: vento climatológico diário
    """
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.bar(
        df['data'],
        df['vento_observado'],
        width=0.85,
        alpha=1,
        color='#060054',
        label='Vento diário observado (100 m)',
    )

    ax.plot(
        df['data'],
        df['vento_climatologico'],
        linewidth=2.5,
        marker='o',
        markersize=3,
        color='#00f2ff',
        label='Climatologia diária (1991–2020)',
    )

    # ✅ pega a escala que o matplotlib escolheu pros teus dados
    ymin, ymax = ax.get_ylim()

    # ✅ pinta 3–20 sem influenciar a escala final
    ax.axhspan(3, 20, color='skyblue', alpha=0.18, zorder=0)

    # ✅ restaura a escala original (não “puxa” pro 20)
    ax.set_ylim(ymin, ymax)

    ax.set_ylabel('Velocidade do vento (m/s)', fontsize=18)
    ax.set_xlabel('Data', fontsize=18)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    titulo = (
        f'{ponto_cfg["nome"]} | {ponto_cfg["subsistema"]} | Potência: {ponto_cfg["potencia_mw"]:.3f} MW\n'
        # f"Lat/Lon informada: ({ponto_cfg['lat']:.4f}, {ponto_cfg['lon']:.4f}) | "
        # f"Grade ERA5 usada: ({lat_grid:.4f}, {lon_grid:.4f})\n"
        f'Período: {settings.DATA_INICIAL} a {settings.DATA_FINAL}'
    )
    ax.set_title(titulo, fontsize=22, loc='left')

    n_dias = len(df)
    if n_dias <= 15:
        locator = mdates.DayLocator(interval=1)
        formatter = mdates.DateFormatter('%d/%m')
    elif n_dias <= 45:
        locator = mdates.DayLocator(interval=2)
        formatter = mdates.DateFormatter('%d/%m')
    elif n_dias <= 120:
        locator = mdates.DayLocator(interval=5)
        formatter = mdates.DateFormatter('%d/%m')
    else:
        locator = mdates.AutoDateLocator()
        formatter = mdates.DateFormatter('%d/%m/%Y')

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=18)

    ax.legend(
        fontsize=12,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info('Gráfico diário salvo em: %s', output_path)


# =============================================================================
# Main
# =============================================================================
def main():
    logger = get_logger(SCRIPT_ID)

    logger.info('=' * 80)
    logger.info('📊 SCRIPT %s: %s', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)
    start_time = time.time()

    plot_areas = {
        'america_do_sul_atlantico': {
            'lon_min': -80,
            'lon_max': 15,
            'lat_min': -40,
            'lat_max': 10,
            'cbar_label': 'Wind Speed Anomaly (%)',
            'title': f'Vento em 100 metros e PNMM (Período: {settings.DATA_INICIAL} - {settings.DATA_FINAL})\nReanálise ERA5',
            'logo_zoom': 1.5,
            'logo_x': 80,
            'logo_y': 165,
        },
        'nordeste_brasil': {
            'lon_min': -50,
            'lon_max': -25,
            'lat_min': -20,
            'lat_max': 5,
            'cbar_label': 'Wind Speed Anomaly (%)',
            'title': f'Vento em 100 metros e PNMM\nPeríodo: {settings.DATA_INICIAL} - {settings.DATA_FINAL}\nReanálise ERA5',
            'logo_zoom': 1.5,
            'logo_x': 80,
            'logo_y': 165,
        },
    }

    empreendimentos = [
        {
            'nome': 'Conj. Caju',
            'potencia_mw': 686,
            'lat': -5.747012868,
            'lon': -36.29595588,
            'subsistema': 'NE',
            'cor': '#00f7ff',
        },
        {
            'nome': 'Conj. Lagoa dos Ventos',
            'potencia_mw': 717,
            'lat': -8.651041667,
            'lon': -41.48511905,
            'subsistema': 'NE',
            'cor': '#00ff59',
        },
        {
            'nome': 'Conj. São Roque',
            'potencia_mw': 795,
            'lat': -8.901654412,
            'lon': -41.61948529,
            'subsistema': 'NE',
            'cor': '#fffb00',
        },
        {
            'nome': 'Conj. Campo Largo',
            'potencia_mw': 688,
            'lat': -10.45987216,
            'lon': -41.48153409,
            'subsistema': 'NE',
            'cor': '#ffae00',
        },
        {
            'nome': 'Conj. Serra do Assuruá',
            'potencia_mw': 858,
            'lat': -11.50455729,
            'lon': -42.57942708,
            'subsistema': 'NE',
            'cor': '#ff0000',
        },
    ]

    lst_areas = list(plot_areas.keys())

    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_VENTO_EOLICAS_SEMOP'
    output_dir_graphs = output_dir / 'graficos_diarios'

    map_output_files = [str(output_dir / f'vento_SEMOP_{area}.png') for area in lst_areas]
    graph_output_files = [
        str(output_dir_graphs / f'vento_diario_{_sanitize_filename(emp["nome"])}.png')
        for emp in empreendimentos
    ]
    output_files = map_output_files + graph_output_files

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'script_version': '2.3_era5_legenda_pontos_ne',
        'fonte': 'ERA5',
        'horas_sinoticas': [0, 6, 12, 18],
        'daily_mean': True,
        'anomalia_clim_file': 'climatologia_1991_2020_vento100m_ERA5.nc',
        'empreendimentos': empreendimentos,
    }

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('🎯 CACHE VÁLIDO! Execução já foi realizada com os mesmos parâmetros.')
        logger.info('   📅 Período: %s a %s', settings.DATA_INICIAL, settings.DATA_FINAL)
        logger.info('   📊 %d arquivo(s) já existe(m)', len(output_files))
        logger.info('   📁 Diretório: %s', output_dir)
        logger.info('   ⏭️  Pulando execução')
        logger.info('=' * 80)
        return

    logger.info('🔄 Cache inválido ou inexistente. Executando processamento...')

    dt_ini = _parse_date(settings.DATA_INICIAL)
    dt_fim = _parse_date(settings.DATA_FINAL)

    if dt_fim < dt_ini:
        raise ValueError('DATA_FINAL não pode ser anterior à DATA_INICIAL.')

    logger.info('📅 Período de análise: %s a %s', dt_ini.date(), dt_fim.date())

    try:
        logger.info('⬇️ Iniciando download/garantia dos arquivos ERA5 (mslp/u100/v100)...')
        files_hourly = ensure_era5_mslp_u100_v100_for_period(
            start=dt_ini,
            end=dt_fim,
            area=[10, -80, -40, 15],
            hours_utc=[0, 6, 12, 18],
            grid='0.25/0.25',
            force_redownload=False,
        )
        logger.info('✅ Arquivos ERA5 garantidos: %d', len(files_hourly))
    except Exception as err:
        logger.exception('ERROR: Falha no download/garantia dos arquivos ERA5')
        raise Exception('ERROR: Falha no download ERA5 (mslp/u100/v100)') from err

    daily_tmp_dir = Path(settings.DIR_DADOS) / 'ERA5_VENTO_PRESSAO' / 'processados'
    daily_nc = _save_daily_temp_file(daily_tmp_dir, dt_ini, dt_fim)

    try:
        logger.info('🧮 Calculando média diária (00/06/12/18)...')
        ds_daily, saved_path, coverage = build_daily_mean_dataset_from_monthly_files(
            files_hourly,
            required_hours_utc=(0, 6, 12, 18),
            require_complete_days=True,
            output_path=daily_nc,
            compress=True,
        )
        logger.info('✅ Média diária concluída. Arquivo: %s', saved_path)
        logger.info(
            '📌 Dias completos encontrados: %d/%d', int(coverage['is_complete'].sum()), len(coverage)
        )
    except Exception as err:
        logger.exception('ERROR: Falha no processamento da média diária')
        raise Exception('ERROR: Falha no processamento da média diária ERA5') from err

    ds_daily = _normalize_latlon_names(ds_daily)
    ds_daily = _normalize_lon(ds_daily)
    ds_daily = _ensure_time_coord(ds_daily)

    ds_period = _subset_period(ds_daily, dt_ini, dt_fim)
    ds_period = _drop_feb29_if_present(ds_period, logger=logger)

    if ds_period.sizes.get('time', 0) == 0:
        raise ValueError('Sem dados diários no período selecionado após filtros.')

    logger.info(
        '📦 Série diária do período: %d dias | %s -> %s',
        ds_period.sizes['time'],
        str(pd.to_datetime(ds_period.time.values[0]).date()),
        str(pd.to_datetime(ds_period.time.values[-1]).date()),
    )

    clim_path = Path(settings.FILE_CLIMATOLOGIA_VENTO100M)

    try:
        ds_clim = _load_climatology_wind100m(clim_path, logger=logger)
        clim_speed = _get_clim_speed_da(ds_clim, logger=logger)

        if isinstance(clim_speed, xr.DataArray):
            clim_speed = clim_speed.to_dataset(name=clim_speed.name or 'ws100_clim')
            clim_speed = _normalize_latlon_names(clim_speed)
            clim_speed = _normalize_lon(clim_speed)
            clim_speed = list(clim_speed.data_vars.values())[0]

        clim_speed_period = _select_climatology_same_days(
            clim_speed=clim_speed,
            daily_dates=ds_period['time'],
            logger=logger,
        )
    except Exception as err:
        logger.exception('ERROR: Falha ao ler/selecionar climatologia de vento 100m')
        raise Exception('ERROR: Falha na climatologia de vento 100m ERA5') from err

    u100 = _pick_var(ds_period, ['u100', '100m_u_component_of_wind'])
    v100 = _pick_var(ds_period, ['v100', '100m_v_component_of_wind'])
    msl = _pick_var(ds_period, ['msl', 'mean_sea_level_pressure'])

    msl_units = str(getattr(msl, 'units', '')).lower()
    if ('pa' in msl_units and 'hpa' not in msl_units) or float(msl.max()) > 2000:
        logger.info('Convertendo MSLP de Pa para hPa.')
        msl = msl / 100.0
        msl.attrs['units'] = 'hPa'

    ws100_daily = np.hypot(u100, v100)
    ws100_daily.name = 'ws100'

    if set(['lat', 'lon']).issubset(set(ws100_daily.coords)) and set(['lat', 'lon']).issubset(
        set(clim_speed_period.coords)
    ):
        if (
            ws100_daily.sizes.get('lat') != clim_speed_period.sizes.get('lat')
            or ws100_daily.sizes.get('lon') != clim_speed_period.sizes.get('lon')
            or not np.array_equal(ws100_daily['lat'].values, clim_speed_period['lat'].values)
            or not np.array_equal(ws100_daily['lon'].values, clim_speed_period['lon'].values)
        ):
            logger.warning(
                'Grid da climatologia difere do ERA5 diário. Interpolando climatologia para o grid do ERA5.'
            )
            clim_speed_period = clim_speed_period.interp_like(ws100_daily)

    ws100_current_mean = ws100_daily.mean(dim='time')
    ws100_clim_mean = clim_speed_period.mean(dim='time')

    ws100_clim_mean_safe = xr.where(ws100_clim_mean <= 0.1, np.nan, ws100_clim_mean)

    anomalia_percentual = ((ws100_current_mean - ws100_clim_mean_safe) / ws100_clim_mean_safe) * 100.0
    anomalia_percentual.name = 'anomalia_percentual_vento100m'

    zonal = u100.mean(dim='time')
    meridional = v100.mean(dim='time')
    mslp = msl.mean(dim='time')
    mslp_smooth = scipy.ndimage.gaussian_filter(mslp.values, sigma=1)

    mag_current = np.hypot(zonal, meridional)
    zonal_vector = ma.masked_where(mag_current < 2, zonal)
    meridional_vector = ma.masked_where(mag_current < 2, meridional)

    ds_plot_ref = _normalize_latlon_names(ds_period)
    ds_plot_ref = _normalize_lon(ds_plot_ref)

    lon = ds_plot_ref['lon'].values
    lat = ds_plot_ref['lat'].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    colors = [
        '#c8334d',
        '#db5054',
        '#ea6e5d',
        '#f68b67',
        '#fea973',
        '#ffc887',
        '#ffe5a5',
        '#ffffff',
        '#ffffff',
        '#d2f2df',
        '#b7dfd9',
        '#a1cbd3',
        '#8db8cc',
        '#7ba4c5',
        '#6b91be',
        '#5b7db7',
    ]
    cmap = LinearSegmentedColormap.from_list('wind_colormap', colors)
    shaded_levels = np.arange(-100, 105, 5)

    output_dir.mkdir(exist_ok=True, parents=True)

    for area_name, area_cfg in plot_areas.items():
        logger.info('Gerando figura para a área: %s', area_name)

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        ax.add_feature(cfeature.STATES.with_scale('50m'))
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
        ax.add_feature(cfeature.BORDERS.with_scale('50m'))
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='beige')
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='steelblue')

        _set_extent_from_area(ax, area_cfg)

        im = ax.contourf(
            lon2d,
            lat2d,
            anomalia_percentual.values,
            cmap=cmap,
            extend='both',
            levels=shaded_levels,
            transform=ccrs.PlateCarree(),
        )

        if area_name == 'nordeste_brasil':
            skip = slice(None, None, 10)
            quiver_scale = 20
            quiver_width = 0.0050
            quiver_headwidth = 3.2
            quiver_headlength = 5.2
        else:
            skip = slice(None, None, 9)
            quiver_scale = 40
            quiver_width = 0.002
            quiver_headwidth = 3
            quiver_headlength = 5

        ax.quiver(
            lon2d[skip, skip],
            lat2d[skip, skip],
            zonal_vector[skip, skip],
            meridional_vector[skip, skip],
            scale_units='inches',
            color='k',
            headwidth=quiver_headwidth,
            scale=quiver_scale,
            headlength=quiver_headlength,
            width=quiver_width,
            transform=ccrs.PlateCarree(),
        )

        # Pontos coloridos com borda preta + legenda no canto superior esquerdo
        if area_name == 'nordeste_brasil':
            handles = []

            for emp in empreendimentos:
                h = ax.scatter(
                    emp['lon'],
                    emp['lat'],
                    s=160,
                    c=emp['cor'],
                    edgecolors='black',
                    linewidths=2.0,
                    marker='o',
                    transform=ccrs.PlateCarree(),
                    zorder=100,
                    label=emp['nome'],
                )
                handles.append(h)

            legenda = ax.legend(
                handles=handles,
                loc='upper left',
                bbox_to_anchor=(0.015, 0.985),
                fontsize=12,
                frameon=True,
                fancybox=True,
                framealpha=0.90,
                borderpad=0.4,
                labelspacing=0.35,
                handlelength=0.8,
                handletextpad=0.4,
            )

            legenda.get_frame().set_facecolor('white')
            legenda.get_frame().set_edgecolor('black')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)

        cbar = plt.colorbar(
            im,
            cax=cax,
            fraction=0.02,
            orientation='horizontal',
            pad=0.05,
            extend='both',
            boundaries=shaded_levels,
            ticks=(-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100),
        )
        cbar.set_label(label=area_cfg['cbar_label'], labelpad=18, size=18)
        cbar.ax.tick_params(labelsize=18)

        im2 = ax.contour(
            lon2d,
            lat2d,
            mslp_smooth,
            levels=range(980, 1045, 3),
            colors='black',
            linewidths=1.0,
            transform=ccrs.PlateCarree(),
        )

        clabels = ax.clabel(im2, inline=True, fmt='%1.0f', fontsize=12, colors='black')
        for txt in clabels:
            txt.set_bbox(dict(facecolor='white', edgecolor='white', pad=0.8))

        _add_auto_gridlines(ax)

        ax.set_title(area_cfg['title'], fontsize=22, loc='left')

        filename_fig = output_dir / f'vento_SEMOP_{area_name}.png'
        logger.info('Salvando a figura %s', filename_fig)

        logo_path = Path(settings.DIR_INPUT) / 'logo_ampere.png'
        if logo_path.exists():
            try:
                img_logotipo_ampere = plt.imread(logo_path)

                fator_zoom = area_cfg['logo_zoom']

                if img_logotipo_ampere.ndim == 3:
                    img_logotipo_ampere = scipy.ndimage.zoom(
                        img_logotipo_ampere,
                        (fator_zoom, fator_zoom, 1),
                        order=1,
                    )
                else:
                    img_logotipo_ampere = scipy.ndimage.zoom(
                        img_logotipo_ampere,
                        (fator_zoom, fator_zoom),
                        order=1,
                    )

                fig.figimage(
                    img_logotipo_ampere,
                    area_cfg['logo_x'],
                    area_cfg['logo_y'],
                    zorder=3,
                    alpha=1,
                )

            except Exception as e:
                logger.warning('Não foi possível inserir logo da Ampere: %s', e)
        else:
            logger.warning('Logo da Ampere não encontrada em %s', logo_path)

        plt.savefig(
            filename_fig,
            dpi=fig.dpi,
            bbox_inches='tight',
        )
        plt.close(fig)

    logger.info('Gerando gráficos diários dos empreendimentos...')
    output_dir_graphs.mkdir(exist_ok=True, parents=True)

    for emp in empreendimentos:
        try:
            df_point, lat_grid, lon_grid = _extract_point_series(
                ws100_daily=ws100_daily,
                clim_speed_period=clim_speed_period,
                lat_point=emp['lat'],
                lon_point=emp['lon'],
                logger=logger,
            )

            filename_graph = output_dir_graphs / f'vento_diario_{_sanitize_filename(emp["nome"])}.png'

            _plot_daily_wind_timeseries(
                df=df_point,
                ponto_cfg=emp,
                output_path=filename_graph,
                logger=logger,
                lat_grid=lat_grid,
                lon_grid=lon_grid,
            )

        except Exception as err:
            logger.exception('Falha ao gerar gráfico diário do empreendimento %s', emp['nome'])
            raise Exception(f'ERROR: Falha ao gerar gráfico diário de {emp["nome"]}') from err

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)

    logger.info('=' * 80)
    logger.info('✅ Script %s concluído com sucesso!', SCRIPT_ID.upper())
    logger.info('⏱️  Tempo de execução: %.1fs (%.1f min)', execution_time, execution_time / 60)
    logger.info('📊 %d arquivo(s) gerado(s)', len(output_files))
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
