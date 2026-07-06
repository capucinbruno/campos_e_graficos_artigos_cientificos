# -*- coding: utf-8 -*-
"""Motor de mapa 2D PLANO (regiao fixa, sem voo de camera) — usado pelo s43.

Irmao de `globo_3d_anim.py` (motor do globo 3D, s38-s42): mesma "marca" visual (caixa
The Weather Channel, corrente de jato, icones de pressao animados, caixa de texto livre,
credito), mas plotado numa projecao RETANGULAR fixa (PlateCarree, tipo carta sinotica do
GFS) em vez do globo giratorio com camera voando.

Reaproveita INTEGRALMENTE a camada de dados/geometria agnostica de projecao do
`globo_3d_anim.py` (VARIAVEIS, builders de serie, geometria da corrente de jato, icones de
pressao, texto/overlay) — so o pipeline de RENDER e novo e bem mais simples, pois nao ha
camera, halo de atmosfera/estrelas, vinheta circular nem reprojecao esferica: os eixos JA
SAO a projecao final, entao o shaded/jato/icones sao desenhados DIRETO neles.

Sem multiprocessing/cache de fundo (v1): o render de um mapa plano e leve (sem
reprojecao/regrid caros da esfera), entao um loop sequencial simples e suficiente.
"""

from __future__ import annotations

import time as _time
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
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.globo_3d_anim import (
    VARIAVEIS,
    _agg_estatico,
    _build_var_series,
    _build_var_series_synoptic,
    _draw_icones_pressao,
    _fmt_data_br,
    _fmt_data_pentada,
    _jato_raster,
    _load_gif_frames,
    _output_plan,
    _resolve_family,
    _overlay_guillaume,
)

logger = get_logger('s43')


# ---------------------------------------------------------------------------
# Regiao fixa (GLOBO_2D_AREA -> settings.json["areas_plotagem"])
# ---------------------------------------------------------------------------
def _area_extent(area_key: str) -> tuple[float, float, float, float]:
    """(lon_esq, lon_dir, lat_inf, lat_sup) da area cadastrada em settings.json."""
    areas = settings.get('areas_plotagem', {}) or {}
    cfg = areas.get(area_key)
    if cfg is None:
        raise ValueError(
            f'GLOBO_2D_AREA="{area_key}" nao encontrada em settings["areas_plotagem"]. '
            f'Disponiveis: {sorted(areas.keys())}')
    return (float(cfg['lon_esq']), float(cfg['lon_dir']),
            float(cfg['lat_inf']), float(cfg['lat_sup']))


# ---------------------------------------------------------------------------
# Corrente(s) de jato — MESMAS settings globais GLOBO_3D_JATO*/JET_STREAM*/SUBTROPICAL_JET*
# compartilhadas com s38-s42 (master unico "para todos os globos", agora tambem o mapa 2D).
# ---------------------------------------------------------------------------
def _build_jatos_cfg(estatico: bool = False) -> list[dict]:
    if not bool(settings.get('GLOBO_3D_JATO', False)):
        return []
    hem_sul = bool(settings.get('GLOBO_3D_JATO_HEMISFERIO_SUL', True))
    hem_norte = bool(settings.get('GLOBO_3D_JATO_HEMISFERIO_NORTE', True))
    hemisferios = tuple(h for h, on in (('N', hem_norte), ('S', hem_sul)) if on)
    if not hemisferios:
        return []
    # Unidade de fase por frame (mesmo mecanismo do globo, `_render_clip`): sem isso, `velocidade`
    # inteiro (1.0, 2.0...) faz `frac = frame_idx * vel` (em `_jato_flat_overlay`) ser sempre um
    # numero inteiro, e a etapa seguinte (`_jet_flow_sequence`) usa `(frac*espacamento) %
    # espacamento`, que da ZERO pra qualquer `frac` inteiro -- o jato congela (setas/texto param).
    # PNG (estatico) fica parado de proposito (fase 0); GIF/MP4 usam a MESMA base de frames do
    # loop do GIF (GLOBO_2D_GIF_FRAMES), igual o globo faz com GLOBO_3D_GE_GIF_FRAMES.
    _phase_unit = 0.0 if estatico else 1.0 / float(settings.get('GLOBO_2D_GIF_FRAMES', 48))
    estilo_comum = {
        'alpha': float(settings.get('GLOBO_3D_JATO_ALPHA', 1.0)),
        'stripe_alpha': float(settings.get('GLOBO_3D_JATO_STRIPE_ALPHA', 0.35)),
        'stripe_n': int(settings.get('GLOBO_3D_JATO_STRIPE_N', 3)),
        'stripe_gap0': float(settings.get('GLOBO_3D_JATO_STRIPE_GAP0', 1.0)),
        'stripe_gap': float(settings.get('GLOBO_3D_JATO_STRIPE_GAP', 0.7)),
        'setas_entre': int(settings.get('GLOBO_3D_JATO_SETAS_ENTRE', 2)),
        'setas_passo': float(settings.get('GLOBO_3D_JATO_SETAS_PASSO', 6.0)),
        'arrow_cor': str(settings.get('GLOBO_3D_JATO_ARROW_COR', 'white')),
        'texto_cor': str(settings.get('GLOBO_3D_JATO_TEXTO_COR', 'white')),
        'texto_max_curva': float(settings.get('GLOBO_3D_JATO_TEXTO_MAX_CURVA', 110.0)),
        'min_pts': int(settings.get('GLOBO_3D_JATO_MIN_PTS', 12)),
        'lat_band': (float(settings.get('GLOBO_3D_JATO_LAT_MIN', 15.0)),
                     float(settings.get('GLOBO_3D_JATO_LAT_MAX', 60.0))),
        'hemisferios': hemisferios,
        'largura_deg': float(settings.get('GLOBO_3D_JATO_LARGURA_DEG', 2.2)),
        'stripe_largura_deg': float(settings.get('GLOBO_3D_JATO_STRIPE_LARGURA_DEG', 0.5)),
        'texto_tam_deg': float(settings.get('GLOBO_3D_JATO_TEXTO_TAM_DEG', 1.7)),
        'arrow_tam_deg': float(settings.get('GLOBO_3D_JATO_ARROW_TAM_DEG', 2.4)),
    }
    jatos = []
    if bool(settings.get('GLOBO_3D_JET_STREAM', True)):
        jatos.append({**estilo_comum, 'nome': 'JET_STREAM',
                      'nivel': float(settings.get('GLOBO_3D_JET_STREAM_NIVEL', 10200.0)),
                      'cor': str(settings.get('GLOBO_3D_JET_STREAM_COR', '#1787ad')),
                      'texto': str(settings.get('GLOBO_3D_JET_STREAM_TEXTO', 'JET STREAM')),
                      'velocidade': float(settings.get('GLOBO_3D_JET_STREAM_VELOCIDADE', 1.0)) * _phase_unit})
    if bool(settings.get('GLOBO_3D_SUBTROPICAL_JET', False)):
        jatos.append({**estilo_comum, 'nome': 'SUBTROPICAL_JET',
                      'nivel': float(settings.get('GLOBO_3D_SUBTROPICAL_JET_NIVEL', 10600.0)),
                      'cor': str(settings.get('GLOBO_3D_SUBTROPICAL_JET_COR', '#2e8b57')),
                      'texto': str(settings.get('GLOBO_3D_SUBTROPICAL_JET_TEXTO', 'SUBTROPICAL JET')),
                      'velocidade': float(settings.get('GLOBO_3D_SUBTROPICAL_JET_VELOCIDADE', 1.0)) * _phase_unit})
    return jatos


# ---------------------------------------------------------------------------
# Icones de pressao — MESMA settings global GLOBO_3D_ICONES_PRESSAO (s38-s42 + s43).
# ---------------------------------------------------------------------------
def _build_icones_pressao() -> list[dict]:
    cfgs = list(settings.get('GLOBO_3D_ICONES_PRESSAO', []) or [])
    if not cfgs:
        return []
    dir_icones = Path(settings.get('DIR_INPUT', 'Entrada')) / 'icones_pressao'
    out = []
    for ic in cfgs:
        tipo = str(ic.get('tipo', ''))
        if not tipo:
            continue
        gif_path = dir_icones / f'{tipo}.gif'
        if not gif_path.exists():
            logger.warning('Ícone de pressão não encontrado: {}', gif_path)
            continue
        frames = _load_gif_frames(gif_path)
        if not frames:
            continue
        out.append({**dict(ic), '_frames': frames})
        logger.info('Ícone de pressão: {} ({} frames, tamanho {:.1f}°)',
                    tipo, len(frames), float(ic.get('tamanho_deg', 8.0)))
    return out


# ---------------------------------------------------------------------------
# Caixa de texto livre — MESMA settings global GLOBO_3D_CAIXA_LIVRE* (s38-s42 + s43).
# ---------------------------------------------------------------------------
def _build_caixa_livre() -> dict | None:
    if not (bool(settings.get('GLOBO_3D_CAIXA_LIVRE', False))
            and str(settings.get('GLOBO_3D_CAIXA_LIVRE_TEXTO', ''))):
        return None
    from app.src.uteis.globo_3d_anim import _wrap_balanceado  # local: evita import p/ quem nao usa
    return {
        'lat': float(settings.get('GLOBO_3D_CAIXA_LIVRE_LAT', 0.0)),
        'lon': float(settings.get('GLOBO_3D_CAIXA_LIVRE_LON', 0.0)),
        'texto': str(settings.get('GLOBO_3D_CAIXA_LIVRE_TEXTO', '')),
        'cor_box': str(settings.get('GLOBO_3D_CAIXA_LIVRE_COR_BOX', 'black')),
        'cor_texto': str(settings.get('GLOBO_3D_CAIXA_LIVRE_COR_TEXTO', 'white')),
        'contorno_cor': str(settings.get('GLOBO_3D_CAIXA_LIVRE_CONTORNO_COR', 'white')),
        'contorno_lw': float(settings.get('GLOBO_3D_CAIXA_LIVRE_CONTORNO_LW', 0.0)),
        'fontsize': float(settings.get('GLOBO_3D_CAIXA_LIVRE_FONTSIZE', 14.0)),
        'largura': int(settings.get('GLOBO_3D_CAIXA_LIVRE_LARGURA', 22)),
        'pad': float(settings.get('GLOBO_3D_CAIXA_LIVRE_PAD', 0.30)),
        'alpha_max': float(settings.get('GLOBO_3D_CAIXA_LIVRE_ALPHA_MAX', 1.0)),
        '_wrap_fn': _wrap_balanceado,
    }


# ---------------------------------------------------------------------------
# Colormap discreto/continuo a partir da ficha (mesma receita do globo_3d_anim._render_clip).
# ---------------------------------------------------------------------------
def _build_cmap(ficha: dict, variavel_key: str, vals: np.ndarray):
    vmax = float(ficha.get('vmax', float(np.nanmax(np.abs(vals)))))
    if ficha.get('vmin') is not None:
        vmin = float(ficha['vmin'])
    elif ficha.get('simetrico', True):
        vmin = -vmax
    else:
        vmin = float(np.nanmin(vals))
    paleta = settings.get(f'GLOBO_3D_PALETA_{variavel_key.upper()}', None) or ficha['cmap_colors']
    niveis = int(settings.get(f'GLOBO_3D_NIVEIS_{variavel_key.upper()}', None)
                 or ficha.get('niveis') or getattr(settings, 'GLOBO_3D_NIVEIS', 16))
    levels = np.linspace(vmin, vmax, niveis + 1)
    if len(paleta) == niveis:
        cmap_plot = ListedColormap(list(paleta))
        cmap_plot.set_under(paleta[0])
        cmap_plot.set_over(paleta[-1])
        norm_fn = BoundaryNorm(levels, ncolors=niveis)
    else:
        cmap_plot = LinearSegmentedColormap.from_list('mapa2d', list(paleta))
        norm_fn = plt.Normalize(vmin=vmin, vmax=vmax)
    logger.info('Escala {}: shaded de {:+.1f} a {:+.1f} {} | {} bandas',
                variavel_key, vmin, vmax, ficha.get('unidade', ''), niveis)
    return levels, cmap_plot, norm_fn


# ---------------------------------------------------------------------------
# Um frame: eixos fixos + shaded (crossfade i0/i1/w) + costa + jato + icones + caixa livre + TWC/credito.
# ---------------------------------------------------------------------------
def _build_frame_2d(f: int, ctx: dict) -> np.ndarray:
    n = ctx['n_dias']
    pos = min(f * ctx['vel_var'] / ctx['frames_por_passo'], n - 1) if n > 1 else 0.0
    i0 = min(int(np.floor(pos)), n - 1)
    i1 = min(i0 + 1, n - 1)
    w = pos - i0
    campo = (1.0 - w) * ctx['vals_cyc'][i0] + w * ctx['vals_cyc'][i1]

    fig = plt.figure(figsize=ctx['figsize'], dpi=ctx['dpi'])
    proj = ccrs.PlateCarree(central_longitude=0.0)
    data_transform = ccrs.PlateCarree()
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection=proj)
    ax.set_extent(ctx['extent'], crs=data_transform)

    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor=ctx['cor_oceano'], zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor=ctx['cor_continente'], zorder=0)

    ax.contourf(ctx['lon_cyc'], ctx['lat'], campo, levels=ctx['levels'], cmap=ctx['cmap_plot'],
               norm=ctx['norm_fn'], transform=data_transform, extend='both', zorder=2)

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor=ctx['cor_fronteiras'],
                   linewidth=ctx['lw_coast'], zorder=6)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor=ctx['cor_fronteiras'],
                   linewidth=ctx['lw_border'], zorder=6)
    ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor=ctx['cor_fronteiras'],
                   linewidth=ctx['lw_states'], zorder=6)
    ax.spines['geo'].set_visible(False)

    # ── Fade da cauda (jato/icones/caixa esmaecem no INICIO da cauda, ficam visiveis ate o fim) ──
    fade = 1.0
    if ctx.get('cauda_on'):
        total = max(ctx['total_frames'] - 1, 1)
        prog = f / total
        fi, fd = ctx['cauda_fade_inicio'], max(ctx['cauda_fade_dur'], 1e-6)
        fade = float(np.clip((prog - fi) / fd, 0.0, 1.0))

    # ── Corrente(s) de jato: `i1` ja fica travado no ultimo passo durante a cauda (o `pos` usa
    # `n_dias`, nao `total_frames`) -- o campo-guia "congela" automaticamente, sem caso especial.
    # Renderiza num RASTER PLANO (mesmo mecanismo do globo, `_jato_raster`) e "cola" via imshow
    # COM `transform=` -- desenhar direto nos eixos reais (sem transform) quebra na costura
    # lon 0/360 (meridiano de Greenwich) sempre que a regiao do mapa cruza esse ponto, porque a
    # grade cheia vem em 0..360 e os eixos aqui usam a janela -30..60 (ranges numericos
    # diferentes); soh o imshow com transform normaliza isso corretamente. ──
    if ctx['jatos'] and ctx.get('hgt_jato_cyc') is not None and fade > 0.003:
        hgt_field = ctx['hgt_jato_cyc'][min(i1, ctx['hgt_jato_cyc'].shape[0] - 1)]
        _rgba, _ext = _jato_raster(ctx['lon_cyc'], ctx['lat'], hgt_field, ctx['jatos'], f,
                                   ctx['jato_drape_px'], ctx['jato_drape_pad'])
        if _rgba is not None:
            ax.imshow(_rgba, origin='upper', transform=data_transform, extent=_ext, zorder=9,
                     interpolation='bilinear')

    # ── Icones de pressao (reaproveita _draw_icones_pressao tal qual do globo) ──
    if ctx['icones_pressao']:
        _ic_ctx = {
            'icones_pressao': ctx['icones_pressao'], 'icone_pressao_regrid': ctx['icone_pressao_regrid'],
            'total_frames': ctx['total_frames'], 'jatos': ctx['jatos'],
            'hgt_z250_abs_cyc': ctx.get('hgt_jato_cyc'), 'jato_drape': False,
            'fade_cauda_on': ctx.get('cauda_on', False),
            'fade_cauda_inicio': ctx.get('cauda_fade_inicio', 0.0),
            'fade_cauda_dur': ctx.get('cauda_fade_dur', 0.0),
            'mapa_plano': True,   # PlateCarree: sem foreshortening esferica, nao compensar cos(lat)
        }
        _draw_icones_pressao(ax, _ic_ctx, f, data_transform, proj)

    # ── Caixa de texto livre ──
    cxl = ctx.get('caixa_livre')
    if cxl is not None:
        from matplotlib.colors import to_rgba
        a = fade * cxl['alpha_max']
        if a > 0.003:
            txt = cxl['_wrap_fn'](cxl['texto'], cxl['largura'], cxl['fontsize'],
                                  ctx.get('font_twc') or ctx.get('font_legenda'))
            lw = cxl['contorno_lw']
            bbox = dict(boxstyle=f"square,pad={cxl['pad']}", facecolor=to_rgba(cxl['cor_box'], a),
                       edgecolor=(to_rgba(cxl['contorno_cor'], a) if lw > 0 else 'none'), linewidth=lw)
            ax.text(cxl['lon'], cxl['lat'], txt, transform=data_transform,
                   color=to_rgba(cxl['cor_texto'], a), fontsize=cxl['fontsize'], fontweight='bold',
                   ha='center', va='center', ma='center', linespacing=1.2,
                   family=ctx.get('font_twc') or ctx.get('font_legenda'), zorder=12, bbox=bbox,
                   clip_on=False)

    # ── Caixa "The Weather Channel" (ou legenda completa) + credito ──
    _overlay_guillaume(fig, ctx, ctx['cmap_legend'], ctx['dates_full'][i1], ctx['dates_br'][i1])

    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def _render_clip_2d(serie, ficha: dict, variavel_key: str, output_dir: Path, fonte_label: str,
                    hgt_jato_serie, *, estatico: bool = False, png_path: Path | None = None,
                    gif: bool = False, gif_path: Path | None = None) -> Path:
    """Renderiza uma serie (sinotica ou diaria) no mapa 2D fixo: MP4 (crossfade entre passos +
    cauda final), ou (estatico/gif) 1 unico campo (media do periodo, camera fixa)."""
    vals = serie.values.astype(np.float32)
    vals_cyc, lon_cyc = add_cyclic_point(vals, coord=serie['lon'].values)
    lat = serie['lat'].values
    dates = pd.DatetimeIndex(pd.to_datetime(serie['time'].values))
    n_dias = vals_cyc.shape[0]
    pent_dias = int(serie.attrs.get('pentada_dias', 0))
    mostrar_hora = bool(ficha.get('sinotico_mp4', False)) and not estatico and not gif

    hgt_jato_cyc = None
    if hgt_jato_serie is not None:
        hgt_vals = hgt_jato_serie.values.astype(np.float32)
        hgt_jato_cyc, _ = add_cyclic_point(hgt_vals, coord=hgt_jato_serie['lon'].values)

    levels, cmap_plot, norm_fn = _build_cmap(ficha, variavel_key, vals_cyc)
    cmap_legend = LinearSegmentedColormap.from_list('mapa2d_legend', list(ficha['cmap_colors']))

    area_key = str(settings.get('GLOBO_2D_AREA', 'europa'))
    lon_esq, lon_dir, lat_inf, lat_sup = _area_extent(area_key)

    # Altura da figura DERIVADA da proporcao geografica real da area (lon/lat), nao um valor fixo
    # independente -- a GeoAxes do cartopy preserva a proporcao verdadeira do mapa (nao distorce
    # paises); se o canvas (fig_w x fig_h) tiver proporcao DIFERENTE da area, o cartopy encolhe o
    # mapa dentro do canvas pra manter a proporcao certa, sobrando faixa BRANCA em cima/embaixo
    # (ou nas laterais). Calculando fig_h a partir da area, o canvas sempre bate com o mapa —
    # sem faixas, em QUALQUER area escolhida via GLOBO_2D_AREA (nao so "europa").
    fig_w = float(settings.get('GLOBO_2D_FIGSIZE_W', 15.0))
    fig_h = fig_w / ((lon_dir - lon_esq) / (lat_sup - lat_inf))
    dpi = int(settings.get('GLOBO_2D_DPI', 150))
    deg_per_pt = (lon_dir - lon_esq) / (fig_w * 72.0)

    font_legenda = _resolve_family(str(settings.get('GLOBO_3D_FONTE_LEGENDA', '')), 'sans-serif')
    font_twc = _resolve_family(str(settings.get('GLOBO_3D_FONTE_TWC', '')), font_legenda)

    frames_por_passo = int(settings.get('GLOBO_2D_FRAMES_POR_PASSO', 5))
    vel_var = float(settings.get('GLOBO_2D_VELOCIDADE_VAR', 1.0))
    fps = int(settings.get('GLOBO_2D_FPS', 8))

    if estatico or gif:
        total_frames = 1 if estatico else max(2, int(settings.get('GLOBO_2D_GIF_FRAMES', 48)))
        cauda_on = False
        cauda_fade_inicio = cauda_fade_dur = 0.0
        base_frames = total_frames
    else:
        base_frames = (n_dias - 1) * frames_por_passo + 1 if n_dias > 1 else max(frames_por_passo, 1)
        cauda_seg = float(settings.get('GLOBO_2D_JATO_MP4_CAUDA_SEG', 0.0))
        tail = int(round(cauda_seg * fps)) if cauda_seg > 0 else 0
        total_frames = base_frames + tail
        cauda_on = tail > 0
        cauda_fade_inicio = (base_frames - 1) / total_frames if cauda_on else 0.0
        cauda_fade_dur = ((float(settings.get('GLOBO_2D_FADE_CAUDA_DUR_SEG', 1.5)) * fps) / total_frames
                          if cauda_on else 0.0)

    ctx = {
        'vals_cyc': vals_cyc, 'lon_cyc': lon_cyc, 'lat': lat, 'n_dias': n_dias,
        'frames_por_passo': frames_por_passo, 'vel_var': vel_var,
        'levels': levels, 'cmap_plot': cmap_plot, 'norm_fn': norm_fn, 'cmap_legend': cmap_legend,
        'extent': [lon_esq, lon_dir, lat_inf, lat_sup],
        'figsize': (fig_w, fig_h), 'dpi': dpi, 'deg_per_pt': deg_per_pt,
        'cor_oceano': str(ficha.get('cor_oceano', '#0e426d')),
        'cor_continente': str(ficha.get('cor_continente', '#2b2b2b')),
        'cor_fronteiras': str(ficha.get('cor_fronteiras', '#444444')),
        'lw_coast': float(ficha.get('lw_coast', settings.get('GLOBO_3D_COASTLINE_LW', 0.7))),
        'lw_border': float(ficha.get('lw_border', settings.get('GLOBO_3D_BORDERS_LW', 0.5))),
        'lw_states': float(ficha.get('lw_states', settings.get('GLOBO_3D_STATES_LW', 0.35))),
        'jatos': _build_jatos_cfg(estatico), 'hgt_jato_cyc': hgt_jato_cyc,
        'jato_drape_px': int(settings.get('GLOBO_3D_JATO_DRAPE_PX', 5000)),
        'jato_drape_pad': float(settings.get('GLOBO_3D_JATO_DRAPE_PAD', 6.0)),
        'icones_pressao': _build_icones_pressao(),
        'icone_pressao_regrid': int(settings.get('GLOBO_3D_ICONE_PRESSAO_REGRID', 2048)),
        'caixa_livre': _build_caixa_livre(),
        'total_frames': total_frames, 'cauda_on': cauda_on,
        'cauda_fade_inicio': cauda_fade_inicio, 'cauda_fade_dur': cauda_fade_dur,
        'font_legenda': font_legenda, 'font_twc': font_twc,
        'credito': str(settings.get('GLOBO_3D_CREDITO', 'Bruno Capucin')),
        'so_credito': bool(settings.get('GLOBO_3D_SO_CREDITO', False)),
        'titulo_twc': str(settings.get('TITULO_THE_WEATHER_CHANNEL', '')),
        'titulo_box': ficha.get('rotulo_box', ficha['titulo']),
        'subtitulo_dir': ficha.get('subtitulo_dir', ''),
        'clim_ref': '',
        'rodada_label': fonte_label,
        'legenda5_labels': [], 'legenda_unidade': ficha.get('unidade', ''),
        'legenda_numerica': True, 'legenda_num_step': (levels[1] - levels[0]) * 5,
        'dates_en': [_fmt_data_pentada(d, pent_dias, com_ano=False, mostrar_hora=mostrar_hora) for d in dates],
        'dates_full': [_fmt_data_pentada(d, pent_dias, com_ano=True, mostrar_hora=mostrar_hora) for d in dates],
        'dates_br': [_fmt_data_br(d, pent_dias, mostrar_hora=mostrar_hora) for d in dates],
    }

    if ctx['jatos']:
        logger.info('Corrente(s) de jato ({}): {}', 'GIF' if gif else ('PNG' if estatico else 'MP4'),
                    ', '.join(j['nome'] for j in ctx['jatos']))
    if ctx['icones_pressao']:
        logger.info('{} ícone(s) de pressão ativo(s)', len(ctx['icones_pressao']))
    if ctx['caixa_livre']:
        logger.info('Caixa de texto livre ATIVA: "{}"', ctx['caixa_livre']['texto'])

    t0 = _time.time()
    if estatico:
        if png_path is None:
            raise ValueError('_render_clip_2d(estatico=True) exige png_path.')
        png_path.parent.mkdir(parents=True, exist_ok=True)
        frame = _build_frame_2d(0, ctx)
        imageio.imwrite(str(png_path), frame)
        logger.info('PNG salvo: {} ({:.1f}s)', png_path, _time.time() - t0)
        return png_path

    if gif:
        if gif_path is None:
            raise ValueError('_render_clip_2d(gif=True) exige gif_path.')
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        frames_buf = [_build_frame_2d(f, ctx) for f in range(total_frames)]
        imageio.mimsave(str(gif_path), frames_buf, format='GIF',
                        fps=int(settings.get('GLOBO_2D_GIF_FPS', 12)), loop=0)
        logger.info('GIF salvo: {} ({:.1f}s)', gif_path, _time.time() - t0)
        return gif_path

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f's43_{variavel_key}.mp4'
    writer = imageio.get_writer(str(out_path), fps=fps, codec='libx264', quality=8, macro_block_size=8)
    try:
        for f in range(total_frames):
            writer.append_data(_build_frame_2d(f, ctx))
            if (f + 1) % 20 == 0 or f == total_frames - 1:
                logger.info('  frame {}/{}', f + 1, total_frames)
    finally:
        writer.close()
    logger.info('MP4 salvo: {} ({:.1f}s)', out_path, _time.time() - t0)
    return out_path


# ---------------------------------------------------------------------------
# Orquestrador (espelha `globo_3d_anim.gerar_animacao`, sem camera/MSLP/OLR/contorno-serie).
# ---------------------------------------------------------------------------
def gerar_animacao_2d(variaveis: list[str], output_base: Path, script_id: str = 's43') -> list[Path]:
    plano, dt_ini, dt_fim = _output_plan(variaveis, output_base)
    logger.info('=' * 70)
    logger.info('MAPA 2D (s43): {} a {} | {} clipe(s)', dt_ini.date(), dt_fim.date(), len(plano))
    logger.info('=' * 70)

    outputs: list[Path] = []
    for item in plano:
        ficha = VARIAVEIS[item['var']]
        logger.info('--- {} | {} ({}) ---', item['label'], item['var'], ficha['titulo'])
        serie = _build_var_series(ficha, item['model'], dt_ini, dt_fim)
        serie_mp4 = (_build_var_series_synoptic(ficha, item['model'], dt_ini, dt_fim)
                    if ficha.get('sinotico_mp4') else serie)

        # Campo-guia do jato: SO suportado (v1) quando a propria variavel JA e uma altura
        # geopotencial absoluta (kind in ('hgt250','hgt500')) -- reusa o proprio campo, sem
        # download nem reconstrucao extra (mesmo atalho do globo). Outras variaveis: jato
        # desligado com aviso (reconstrucao via z250_anom+climatologia fica para uma proxima
        # iteracao, ainda nao pedida para o s43).
        hgt_jato_serie = None
        hgt_jato_mp4 = None
        e_hgt_absoluto = bool(ficha.get('absoluto')) and ficha['spec'].get('kind') in ('hgt250', 'hgt500')
        jato_quer = bool(settings.get('GLOBO_3D_JATO', False))
        if jato_quer and e_hgt_absoluto:
            hgt_jato_serie = serie
            hgt_jato_mp4 = serie_mp4
        elif jato_quer:
            logger.warning('⚠ Corrente de jato pedida, mas {} não é altura geopotencial absoluta '
                           '(kind={}) — v1 do mapa 2D só reaproveita Z250/Z500 absoluto próprio. '
                           'Jato NÃO será plotado para {}.',
                           item['var'], ficha['spec'].get('kind'), item['var'])

        outputs.append(_render_clip_2d(serie_mp4, ficha, item['var'], item['dir'], item['label'],
                                       hgt_jato_mp4))

        _dts = pd.DatetimeIndex(pd.to_datetime(serie['time'].values))
        _n = len(_dts)
        serie_m = _agg_estatico(serie, _dts[0], _dts[-1], _n)
        if serie_m is not None:
            hgt_m = _agg_estatico(hgt_jato_serie, _dts[0], _dts[-1], _n) if hgt_jato_serie is not None else None
            png_path = item['dir'] / f"s43_{item['var']}_media.png"
            gif_path = item['dir'] / f"s43_{item['var']}_media.gif"
            outputs.append(_render_clip_2d(serie_m, ficha, item['var'], item['dir'], item['label'],
                                           hgt_m, estatico=True, png_path=png_path))
            outputs.append(_render_clip_2d(serie_m, ficha, item['var'], item['dir'], item['label'],
                                           hgt_m, gif=True, gif_path=gif_path))
    return outputs
