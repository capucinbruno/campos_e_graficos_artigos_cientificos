# -*- coding: utf-8 -*-
"""Video MP4 (OLR intrasazonal shaded + vento 850 hPa em vetores) para o s32.

Modulo dedicado — NAO reaproveita `mapa_2d_anim.py`/`globo_3d_anim.py` (motor do
s38-s43), que sao acoplados ao sistema de fichas/`VARIAVEIS` (um unico campo
escalar por vez, jato-raster, icones de pressao), incompativel com o caso aqui
(OLR shaded + vetores REAIS de vento sobrepostos). Reaproveita so a TECNICA de
gravacao/crossfade daquele motor (imageio + interpolacao linear i0/i1/w entre
passos de tempo consecutivos).

Estilo visual: replica o composto OLR+vento da NCICS (`Entrada/intrasazonal.png`),
medido por pixel (nao "no olho"):
  - Continente cinza solido (#999493), oceano branco, SEM fronteiras de
    pais/estado (so o contorno do continente).
  - Projecao PlateCarree GLOBAL (360°), mas girada para a costura cair no
    Atlântico (~30°W) em vez de Greenwich — `central_longitude=150` (calibrado
    batendo a posicao em pixel dos rotulos "0°E" e "180" da imagem de
    referencia: erro < 1px em ambos).
  - Latitude -60..60 (confirmado pela posicao em pixel de "15°N"/"Eq."/"15°S").
  - Vento 850 hPa em TODOS os pontos da grade, sem mascara/clip de magnitude
    (diferente do `_plot_mapa_olr_wind` estatico do s32), com legenda de escala
    (quiverkey).
  - Legendas em caixas no topo (colorbar horizontal + legenda de vento), nao
    colorbar lateral. Sem fase da MJO (removida a pedido).
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from cartopy.util import add_cyclic_point
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

from app.common.logo_helper import proportional_logo_zoom, resolve_logo_path
from app.shared.logger import get_logger

logger = get_logger('s32')

# Mesma escala/paleta dos mapas OLR+vento850 estaticos do s32 (_plot_mapa_olr_wind):
# BrBG_r → negativo (mais chuva) = teal/verde, positivo (menos chuva) = castanho.
LEVELS = np.arange(-40, 44, 4)
CMAP_NAME = 'BrBG_r'

# Geometria do mapa (calibrada em cima de `Entrada/intrasazonal.png`, ver docstring).
CENTRAL_LON = 150.0
LAT_MIN, LAT_MAX = -60.0, 60.0
FIG_W_IN, FIG_H_IN = 12.8, 4.32   # 1280x432 (multiplo de 8, evita resize do libx264); ~razao 3.0
FIG_DPI = 100

COR_CONTINENTE = '#999493'
COR_COSTA = '#726d67'
COR_VENTO = '#333333'
SCALE_QUIVER = 600.0   # m/s por unidade de comprimento da seta (sem clip de magnitude)
QUIVER_SUBSAMPLE = 2   # a cada N pontos da grade (2.5°xN) — grade nativa e densa demais p/ ~13px/°

LAT_LABELS = [(-15.0, '15°S'), (0.0, 'Eq.'), (15.0, '15°N')]
LON_LABELS_ABS = list(range(0, 301, 30))  # 0°E .. 60°W (300°E), mesma faixa rotulada do print


def _cmap_norm():
    cmap = plt.get_cmap(CMAP_NAME, len(LEVELS) + 1)
    norm = BoundaryNorm(LEVELS, cmap.N, extend='both')
    return cmap, norm


def _lon_label(lon_abs: int) -> str:
    if lon_abs == 0:
        return '0°E'
    if lon_abs == 180:
        return '180'
    return f'{lon_abs}°E' if lon_abs < 180 else f'{360 - lon_abs}°W'


def _lon_to_xfrac(lon_abs: float) -> float:
    """Posicao (0..1, fracao da largura dos eixos) de uma longitude ABSOLUTA, dada a rotacao
    `CENTRAL_LON` — mesma conta usada para calibrar `CENTRAL_LON` a partir da imagem de
    referencia (rotulos '0°E' e '180' bateram com erro < 1px)."""
    local_x = ((lon_abs - CENTRAL_LON + 180.0) % 360.0) - 180.0
    return (local_x + 180.0) / 360.0


def _add_logo(ax, logo_path: Path) -> None:
    logo = Image.open(logo_path).convert('RGBA')
    bbox = logo.getbbox()
    if bbox is not None:
        logo = logo.crop(bbox)
    img = np.array(logo)
    imagebox = OffsetImage(img, zoom=proportional_logo_zoom(ax, img.shape[1]))
    ab = AnnotationBbox(
        imagebox, (0, 0), xycoords=ax.transAxes, xybox=(4, 4),
        boxcoords='offset points', box_alignment=(0, 0),
        frameon=False, pad=0, zorder=500, clip_on=False,
    )
    ax.add_artist(ab)


def _draw_axis_labels(ax) -> None:
    """Rotulos de lat/lon DENTRO do mapa (nao numa margem externa) — mesma posicao/estilo
    do print de referencia: so 15°S/Eq./15°N em latitude, 0°E..60°W (a cada 30°) em longitude."""
    for lat_val, label in LAT_LABELS:
        y_frac = (lat_val - LAT_MIN) / (LAT_MAX - LAT_MIN)
        ax.text(0.006, y_frac, label, transform=ax.transAxes, ha='left', va='center',
                fontsize=8, color='#444444', zorder=50)
    for lon_abs in LON_LABELS_ABS:
        x_frac = _lon_to_xfrac(float(lon_abs))
        ax.text(x_frac, 0.012, _lon_label(lon_abs), transform=ax.transAxes, ha='center', va='bottom',
                fontsize=8, color='#444444', zorder=50)


def _draw_olr_colorbar_box(fig, im) -> None:
    """Caixa no topo-esquerdo com a colorbar horizontal do OLR (em vez de colorbar lateral)."""
    cax = fig.add_axes([0.015, 0.855, 0.24, 0.055])
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=LEVELS[::2], extend='both')
    cbar.ax.tick_params(labelsize=6.5, length=2, pad=1)
    cbar.outline.set_linewidth(0.6)
    for spine in cax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.6)
    fig.text(0.135, 0.925, 'OLR intrasazonal (W/m²)', ha='center', va='center', fontsize=7.5,
             fontweight='bold')


def _draw_titulo_box(fig, titulo: str) -> None:
    fig.text(0.5, 0.925, titulo, ha='center', va='center', fontsize=8.5, fontweight='bold',
             bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', linewidth=0.6))


def _draw_wind_legend_box(fig, Q) -> None:
    """Caixa no topo-direito com seta de referencia (m/s) — analoga a 'Surface wind anomaly'."""
    fig.patches.append(plt.Rectangle((0.78, 0.855), 0.205, 0.075, transform=fig.transFigure,
                                     facecolor='white', edgecolor='black', linewidth=0.6, zorder=40))
    fig.text(0.8825, 0.912, 'Vento 850 hPa (m/s)', ha='center', va='center', fontsize=7.5,
             fontweight='bold', zorder=41)
    Q.axes.quiverkey(Q, X=0.885, Y=0.875, U=5.0, label='5 m/s', labelpos='S',
                     coordinates='figure', fontproperties={'size': 7}, zorder=42)


def _build_frame(olr2d: np.ndarray, u2d: np.ndarray, v2d: np.ndarray,
                 lat: np.ndarray, lon: np.ndarray, titulo: str, logo_path: Path | None) -> np.ndarray:
    """Um frame: continente cinza + OLR shaded (contourf) + vento 850 hPa (quiver, sem clip),
    projecao global com costura no Atlântico — mesmo estilo do composto NCICS de referencia."""
    cmap, norm = _cmap_norm()
    # Transparencia perto de zero (|OLR| < 1o nivel): deixa o continente cinza aparecer por
    # baixo, igual ao "buraco" -5..+5 da legenda do print de referencia — sem isso o contourf
    # cobre o globo INTEIRO (inclusive ~0, quase-branco mas opaco) e esconde o continente.
    olr2d = np.where(np.abs(olr2d) < LEVELS[1] - LEVELS[0], np.nan, olr2d)
    arr, lonc = add_cyclic_point(olr2d, coord=lon)
    u_cyc, lon_u_cyc = add_cyclic_point(u2d, coord=lon)
    v_cyc, _ = add_cyclic_point(v2d, coord=lon)
    s = QUIVER_SUBSAMPLE
    lon_q, lat_q = lon_u_cyc[::s], lat[::s]
    u_q, v_q = u_cyc[::s, ::s], v_cyc[::s, ::s]

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=FIG_DPI)
    proj = ccrs.PlateCarree(central_longitude=CENTRAL_LON)
    data_transform = ccrs.PlateCarree()
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection=proj)
    ax.set_xlim([-180, 180])
    ax.set_ylim([LAT_MIN, LAT_MAX])
    ax.set_facecolor('white')

    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor=COR_CONTINENTE,
                   edgecolor=COR_COSTA, linewidth=0.4, zorder=1)

    im = ax.contourf(lonc, lat, arr, levels=LEVELS, cmap=cmap, norm=norm,
                     extend='both', transform=data_transform, zorder=2)

    # Vento 850 hPa: grade regular (sem mascara/clip de magnitude, estilo NCICS) — subamostrada
    # p/ nao sobrepor setas (grade nativa de 2.5° e densa demais p/ a resolucao do frame).
    Q = ax.quiver(lon_q, lat_q, u_q, v_q, transform=data_transform,
                 color=COR_VENTO, pivot='middle', scale=SCALE_QUIVER,
                 width=0.0009, headwidth=3.0, headlength=3.6, headaxislength=3.2, zorder=5)

    ax.gridlines(xlocs=range(-180, 181, 30), ylocs=range(-60, 61, 15),
                linestyle=':', linewidth=0.5, color='gray', alpha=0.6, zorder=3)
    _draw_axis_labels(ax)

    _draw_olr_colorbar_box(fig, im)
    _draw_titulo_box(fig, titulo)
    _draw_wind_legend_box(fig, Q)

    if logo_path is not None and logo_path.exists():
        _add_logo(ax, logo_path)

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def gerar_video_olr_vento850(
    olr: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dates: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    out_path: Path,
    titulo_base: str,
    input_dir: Path,
    fps: int,
    frames_por_passo: int,
    boundary_date: datetime | None = None,
) -> Path:
    """Anima OLR intrasazonal + vento 850 hPa dia a dia, com crossfade (`frames_por_passo`
    frames por dia, interpolacao linear entre passos consecutivos) — sem fase da MJO.

    `olr`/`u`/`v`: (T, lat, lon), ja alinhados com `dates` (T,), mesma banda (com ou sem
    filtro Lanczos conforme o chamador). `boundary_date`: se dado, rotula cada frame como
    'observado' ou 'previsto' (comparando a data do frame com essa fronteira).
    """
    n = olr.shape[0]
    if n < 2:
        raise RuntimeError('gerar_video_olr_vento850: serie precisa de pelo menos 2 dias.')
    logo_path = resolve_logo_path(input_dir)
    total_frames = (n - 1) * frames_por_passo + 1
    boundary = pd.Timestamp(boundary_date).date() if boundary_date is not None else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec='libx264', quality=8, macro_block_size=8)
    t0 = time.time()
    try:
        for f in range(total_frames):
            pos = f / frames_por_passo
            i0 = min(int(np.floor(pos)), n - 1)
            i1 = min(i0 + 1, n - 1)
            w = pos - i0
            olr2d = (1.0 - w) * olr[i0] + w * olr[i1]
            u2d = (1.0 - w) * u[i0] + w * u[i1]
            v2d = (1.0 - w) * v[i0] + w * v[i1]

            data_ref = pd.Timestamp(dates[i1 if w >= 0.5 else i0]).date()
            tag = ''
            if boundary is not None:
                tag = ' (previsto)' if data_ref > boundary else ' (observado)'
            titulo = f'{titulo_base} — {data_ref}{tag}'

            writer.append_data(_build_frame(olr2d, u2d, v2d, lat, lon, titulo, logo_path))
            if (f + 1) % 20 == 0 or f == total_frames - 1:
                logger.info('  frame {}/{}', f + 1, total_frames)
    finally:
        writer.close()
    logger.info('MP4 OLR+vento850 salvo: {} ({:.1f}s, {} frames)', out_path, time.time() - t0, total_frames)
    return out_path
