# app/src/uteis/aao_heatmap.py
# -*- coding: utf-8 -*-
"""Heatmap de desempenho dos modelos no indice AAO (s35).

Le o arquivo de verificacao (``dados/s35_verif/``) e monta uma matriz modelo x data-alvo
colorida pelo **skill score vs climatologia**, numa **faixa de lead fixa** (a faixa da a
amostra do MSE de cada celula):

    SS = 1 - MSE_modelo / VAR_clim   (MSSS — Mean Squared Skill Score vs climatologia)
    MSE_modelo = media_{lead in faixa} ( fcst_lead - obs )^2
    VAR_clim   = MSE de prever SEMPRE a climatologia (= 0, indice normalizado) = media(obs^2)

Verde = bate a climatologia (SS>0), vermelho = pior que a climatologia (SS<0), ~amarelo em 0.
A referencia e a VARIANCIA climatologica CONSTANTE (uma so para todo o painel), nao o obs^2 do
dia: usar obs^2 por dia explodia em AAO neutra (|obs|~0) — um erro pequeno virava SS=-1 espurio.
O SS e clipado em [-1, 1].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.src.uteis.aao_verif_archive import fcst_archive_path, obs_archive_path

# Piso da variancia de referencia: so age se o arquivo de obs for minusculo (evita div/0).
_VAR_CLIM_FLOOR = 0.25


def build_skill_matrix(window_days: int, lead_lo: int, lead_hi: int,
                       model_order: list[str] | None = None):
    """Monta (models, dates, SS) lendo os CSVs de verificacao.

    - `models`: linhas (so as que tem >=1 celula verificavel), na ordem de `model_order`.
    - `dates`: colunas = datas-alvo COM observado, na janela [max_obs - window_days, max_obs].
    - `SS`: matriz (len(models) x len(dates)) de skill score clipado; NaN = sem dado.
    """
    fp, op = fcst_archive_path(), obs_archive_path()
    if not fp.exists() or not op.exists():
        return [], [], np.empty((0, 0))

    obs = pd.read_csv(op)
    obs['date'] = pd.to_datetime(obs['date'])
    obs_map = dict(zip(obs['date'], obs['obs_idx']))
    if not obs_map:
        return [], [], np.empty((0, 0))
    # Referencia CONSTANTE = MSE de prever sempre a climatologia (0) = media(obs^2) no arquivo.
    var_clim = max(float(np.mean(obs['obs_idx'].to_numpy() ** 2)), _VAR_CLIM_FLOOR)

    fc = pd.read_csv(fp)
    fc['valid_date'] = pd.to_datetime(fc['valid_date'])
    band = fc[(fc['lead_days'] >= lead_lo) & (fc['lead_days'] <= lead_hi)]

    max_obs = max(obs_map)
    start = max_obs - pd.Timedelta(days=window_days)
    dates = sorted(d for d in obs_map if start <= d <= max_obs)

    present = set(band['model'].unique())
    order = model_order or sorted(present)
    candidates = [m for m in order if m in present]

    # Calcula a linha de skill de cada modelo e MANTEM so as que tem >=1 celula verificavel
    # (modelo so com init de hoje preve datas futuras sem observado -> linha toda NaN -> ocultada).
    models, rows = [], []
    for m in candidates:
        bm = band[band['model'] == m]
        per_date = {d: g['fcst_idx'].to_numpy() for d, g in bm.groupby('valid_date')}
        row = np.full(len(dates), np.nan)
        for c, d in enumerate(dates):
            f = per_date.get(d)
            if f is None or d not in obs_map:
                continue
            o = obs_map[d]
            mse_model = float(np.mean((f - o) ** 2))
            row[c] = np.clip(1.0 - mse_model / var_clim, -1.0, 1.0)
        if np.isfinite(row).any():
            models.append(m)
            rows.append(row)
    ss = np.array(rows) if rows else np.empty((0, len(dates)))
    return models, dates, ss


def plot_skill_heatmap(out_png: Path, window_days: int, lead_lo: int, lead_hi: int,
                       model_order: list[str], model_labels: dict[str, str],
                       logo_path: Path | None = None, title_suffix: str = ''):
    """Gera o PNG do heatmap. Retorna o Path salvo, ou None se nao ha celula verificavel."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    models, dates, ss = build_skill_matrix(window_days, lead_lo, lead_hi, model_order)
    if not models or not dates or np.all(np.isnan(ss)):
        return None

    labels = [model_labels.get(m, m.upper()) for m in models]
    nrow, ncol = len(models), len(dates)
    fig_w = max(8.0, 0.34 * ncol + 3.0)
    fig_h = max(2.6, 0.46 * nrow + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)

    cmap = plt.get_cmap('RdYlGn').copy()
    cmap.set_bad('#e6e6e6')  # NaN (sem dado) = cinza claro
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    masked = np.ma.masked_invalid(ss)
    im = ax.imshow(masked, aspect='auto', cmap=cmap, norm=norm)

    ax.set_xticks(range(ncol))
    ax.set_xticklabels([d.strftime('%d/%m') for d in dates], rotation=90, fontsize=8)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xticks(np.arange(-0.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrow, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    ax.tick_params(which='minor', length=0)

    # Valor do skill em cada celula (texto preto/branco conforme o fundo).
    for r in range(nrow):
        for c in range(ncol):
            v = ss[r, c]
            if np.isnan(v):
                continue
            ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=7,
                    color='black' if -0.55 < v < 0.7 else 'white')

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cb.set_label('Skill score vs climatologia  (1 = perfeito, 0 = clima, <0 pior)', fontsize=9)
    ax.set_title(
        f'Desempenho dos modelos — Índice AAO  (lead {lead_lo}–{lead_hi} d){title_suffix}',
        fontsize=13, fontweight='bold', pad=10)

    if logo_path is not None and Path(logo_path).exists():
        try:
            from PIL import Image
            from matplotlib.offsetbox import AnnotationBbox, OffsetImage
            im_logo = Image.open(logo_path).convert('RGBA')
            b = im_logo.getbbox()
            if b:
                im_logo = im_logo.crop(b)
            arr = np.asarray(im_logo)
            zoom = 38.0 / arr.shape[0]
            ab = AnnotationBbox(OffsetImage(arr, zoom=zoom), (1, 1), xycoords='axes fraction',
                                xybox=(-4, 16), boxcoords='offset points', box_alignment=(1, 0),
                                frameon=False, pad=0, clip_on=False)
            ax.add_artist(ab)
        except Exception:
            pass

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_png), bbox_inches='tight')
    plt.close(fig)
    return out_png
