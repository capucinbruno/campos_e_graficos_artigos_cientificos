"""s37: Frequência de passagens de frente fria por localidade (heatmap).

Baixa ERA5 superfície 6-horária (u10/v10/t2m/msl), detecta as passagens de frente
fria por cidade pelo método de mudança de vento (Simmonds et al., HS) + guarda
térmica — ver `app/src/uteis/deteccao_frente_fria.py` — e plota um mapa em que cada
localidade é colorida pelo número de passagens no período [DATA_INICIAL, DATA_FINAL].

Modo único: REANÁLISE (ERA5). O modo previsão (emendar a ponta com GDAS/modelos do
s34) fica como evolução futura.
"""

from __future__ import annotations

# Bibliotecas padrão
import time
from datetime import datetime, timedelta
from pathlib import Path

# Bibliotecas de terceiros
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Módulos locais
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.deteccao_frente_fria import (
    DTEMP_MIN,
    DV_MIN,
    REFRACTORY_H,
    contar_frentes_localidades,
)
from app.src.uteis.downloaders_era5_superficie import ensure_era5_superficie_for_period

logger = get_logger('s37')

# Dias extra baixados após DATA_FINAL para alimentar a guarda térmica (média
# diária das 24 h posteriores ao evento). Sem essa folga, frentes no último dia
# da janela cairiam no ΔT imediato de 6 h (menos robusto).
_DIAS_FOLGA_TERMICA = 2


def _abrir_dataset(files: list[Path]) -> xr.Dataset:
    """Abre os arquivos mensais de superfície num único Dataset."""
    paths = [str(f) for f in files if Path(f).exists()]
    if not paths:
        raise RuntimeError('Nenhum arquivo de superfície ERA5 disponível para o período.')
    if len(paths) == 1:
        return xr.open_dataset(paths[0])
    return xr.open_mfdataset(paths, combine='by_coords')


def _plotar_heatmap(resultados, dt_ini, dt_fim, out_png: Path) -> None:
    """Plota o mapa de frequência de frentes frias por localidade."""
    lats = np.array([r.lat for r in resultados], dtype=float)
    lons = np.array([r.lon for r in resultados], dtype=float)
    counts = np.array([r.n_frentes for r in resultados], dtype=float)
    vmax = max(1.0, float(counts.max()) if counts.size else 1.0)

    # Extensão: cobre todas as localidades com uma folga
    margin = 3.0
    extent = [
        float(lons.min()) - margin, float(lons.max()) + margin,
        float(lats.min()) - margin, float(lats.max()) + margin,
    ]

    fig = plt.figure(figsize=(12, 14))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='whitesmoke')
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='white')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5, edgecolor='gray', zorder=50)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.0, edgecolor='black', zorder=60)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.0, edgecolor='black', zorder=60)

    sc = ax.scatter(
        lons, lats, c=counts, cmap='YlOrRd', vmin=0, vmax=vmax,
        s=320, edgecolor='black', linewidth=0.8, zorder=200,
        transform=ccrs.PlateCarree(),
    )

    # Número de passagens no marcador + nome da cidade ao lado
    for r in resultados:
        ax.text(
            r.lon, r.lat, str(r.n_frentes),
            ha='center', va='center', fontsize=7, fontweight='bold',
            color='black', zorder=300, transform=ccrs.PlateCarree(),
        )
        nome_curto = r.nome.split('-')[0].strip()
        ax.text(
            r.lon + 0.45, r.lat + 0.25, nome_curto,
            ha='left', va='bottom', fontsize=5.5, color='dimgray',
            zorder=300, transform=ccrs.PlateCarree(),
        )

    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False

    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', shrink=0.6, pad=0.02)
    cbar.set_label('Nº de passagens de frente fria', fontsize=11)

    titulo = (
        'Frequência de Frentes Frias por Localidade (ERA5)\n'
        f'{dt_ini:%d/%m/%Y} a {dt_fim:%d/%m/%Y}'
    )
    ax.set_title(titulo, fontsize=14, fontweight='bold')

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _salvar_csv(resultados, out_csv: Path) -> None:
    """Salva a contagem por localidade (ordenada desc) e as datas dos eventos."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ordenados = sorted(resultados, key=lambda r: r.n_frentes, reverse=True)
    linhas = ['nome,lat,lon,lat_grade,lon_grade,n_frentes,eventos']
    for r in ordenados:
        eventos = ' '.join(ev.strftime('%Y-%m-%dT%H') for ev in r.eventos)
        linhas.append(
            f'"{r.nome}",{r.lat:.4f},{r.lon:.4f},{r.lat_grade:.4f},'
            f'{r.lon_grade:.4f},{r.n_frentes},"{eventos}"'
        )
    out_csv.write_text('\n'.join(linhas) + '\n', encoding='utf-8')


def main():
    """Entry point - chamado pelo CLI sem argumentos."""
    start_time = time.time()

    logger.info('=' * 80)
    logger.info('SCRIPT s37: FREQUÊNCIA DE FRENTES FRIAS POR LOCALIDADE (ERA5)')
    logger.info('=' * 80)

    output_dir = Path(settings.DIR_OUTPUT) / 's37_FRENTES_FRIAS'
    out_png = output_dir / 's37_heatmap_frentes_frias.png'
    out_csv = output_dir / 's37_frentes_frias.csv'

    localidades = settings.get('LOCALIDADES', [])
    if not localidades:
        raise ValueError('LOCALIDADES vazio no settings. Defina [lat, lon, nome] no settings.')

    dv_min = float(settings.get('FRENTE_DV_MIN', DV_MIN))
    dtemp_min = float(settings.get('FRENTE_DTEMP_MIN', DTEMP_MIN))
    refractory_h = float(settings.get('FRENTE_REFRACTORY_H', REFRACTORY_H))

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'n_localidades': len(localidades),
        'dv_min': dv_min,
        'dtemp_min': dtemp_min,
        'refractory_h': refractory_h,
        'script_version': '1.0',
    }
    output_files = [str(out_png), str(out_csv)]

    if check_cache_valid('s37', cache_params, output_files):
        logger.info('CACHE VALIDO - pulando execucao')
        return

    dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d')
    dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d')

    # Baixa o período + folga térmica (24h após o último dia da janela)
    dt_download_fim = dt_fim + timedelta(days=_DIAS_FOLGA_TERMICA)
    logger.info(
        f'Janela de contagem: {dt_ini:%Y-%m-%d} a {dt_fim:%Y-%m-%d} '
        f'(download até {dt_download_fim:%Y-%m-%d} p/ guarda térmica)'
    )

    files = ensure_era5_superficie_for_period(start=dt_ini, end=dt_download_fim)
    ds = _abrir_dataset(files)

    logger.info(
        f'Detectando frentes em {len(localidades)} localidades '
        f'(DV_MIN={dv_min} m/s, DTEMP_MIN={dtemp_min} °C, REFRATARIO={refractory_h} h)'
    )
    resultados = contar_frentes_localidades(
        ds,
        localidades,
        dv_min=dv_min,
        dtemp_min=dtemp_min,
        refractory_h=refractory_h,
        janela_inicio=dt_ini,
        janela_fim=dt_fim,
    )

    # Log resumido (ordenado por nº de frentes)
    for r in sorted(resultados, key=lambda x: x.n_frentes, reverse=True):
        eventos = ', '.join(ev.strftime('%d/%m %Hz') for ev in r.eventos)
        logger.info(f'  {r.nome:35s} {r.n_frentes:2d}  [{eventos}]')

    _salvar_csv(resultados, out_csv)
    _plotar_heatmap(resultados, dt_ini, dt_fim, out_png)
    logger.info(f'Mapa salvo: {out_png}')
    logger.info(f'CSV salvo:  {out_csv}')

    execution_time = time.time() - start_time
    save_cache_metadata('s37', cache_params, output_files, execution_time)
    logger.info(f'Script s37 concluido em {execution_time:.1f}s')
