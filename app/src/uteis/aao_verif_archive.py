# -*- coding: utf-8 -*-
"""Arquivo persistente de verificacao do indice AAO (s35).

Acumula, a cada rodada do s35, os escalares necessarios para o heatmap de desempenho dos
modelos e a futura correcao de vies/MOS:

- ``obs_archive.csv``  : indice AAO observado por data (ERA5/GDAS). Numa re-rodada o ERA5
  (definitivo) SOBRESCREVE o GDAS (best-effort) da ponta recente — vale o valor mais novo.
- ``fcst_archive.csv`` : indice AAO previsto por ``(modelo, init, data_alvo)``. O lead em dias
  e ``data_alvo - init``, o que permite verificar/corrigir por faixa de lead.

Sao so escalares (KB): o heatmap e a correcao LEEM deste arquivo, nunca re-baixam o passado.
Cada modelo acumula um ponto novo por rodada; com o tempo a janela de ~60 dias se completa.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.shared.settings_factory import settings

_OBS_COLS = ['date', 'obs_idx']
_FCST_COLS = ['model', 'init_date', 'valid_date', 'lead_days', 'fcst_idx']


def _verif_dir() -> Path:
    d = Path(settings.DIR_DADOS) / 's35_verif'
    d.mkdir(parents=True, exist_ok=True)
    return d


def obs_archive_path() -> Path:
    return _verif_dir() / 'obs_archive.csv'


def fcst_archive_path() -> Path:
    return _verif_dir() / 'fcst_archive.csv'


def _read_csv(path: Path, cols: list[str]) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)


def upsert_obs(dates, idx) -> Path:
    """Atualiza/insere o indice observado por data, mantendo a ULTIMA ocorrencia (a re-rodada
    mais recente, ja calculada da melhor fonte disponivel, supera o valor antigo do GDAS)."""
    new = pd.DataFrame({
        'date': pd.to_datetime(pd.Index(dates)).strftime('%Y-%m-%d'),
        'obs_idx': pd.to_numeric(pd.Series(list(idx)), errors='coerce').to_numpy(),
    }).dropna(subset=['obs_idx'])
    path = obs_archive_path()
    merged = pd.concat([_read_csv(path, _OBS_COLS), new], ignore_index=True)
    merged = merged.drop_duplicates(subset=['date'], keep='last').sort_values('date')
    merged.to_csv(path, index=False)
    return path


def existing_init_dates(model: str) -> set[str]:
    """Conjunto de `init_date` (YYYY-MM-DD) ja gravados para o modelo — usado pelo backfill
    para ser idempotente (nao re-baixar inits ja presentes)."""
    path = fcst_archive_path()
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=['model', 'init_date'])
    return set(df.loc[df['model'] == model, 'init_date'].astype(str))


def append_fcst(model: str, per_init) -> Path:
    """Anexa as previsoes POR-INIT de um modelo. ``per_init`` = lista de ``(init_datetime,
    dates, idx)``. Dedup por ``(model, init_date, valid_date)`` mantendo o registro mais recente
    (rodar 2x no mesmo dia nao duplica). Descarta leads negativos por seguranca."""
    rows = []
    for init_dt, dates, idx in per_init:
        init_ts = pd.Timestamp(init_dt).normalize()
        init_str = init_ts.strftime('%Y-%m-%d')
        for d, v in zip(pd.to_datetime(pd.Index(dates)), list(idx)):
            if pd.isna(v):
                continue
            lead = int((d.normalize() - init_ts).days)
            if lead < 0:
                continue
            rows.append((model, init_str, d.strftime('%Y-%m-%d'), lead, float(v)))
    path = fcst_archive_path()
    if not rows:
        return path
    new = pd.DataFrame(rows, columns=_FCST_COLS)
    merged = pd.concat([_read_csv(path, _FCST_COLS), new], ignore_index=True)
    merged = merged.drop_duplicates(subset=['model', 'init_date', 'valid_date'], keep='last')
    merged = merged.sort_values(['model', 'init_date', 'valid_date'])
    merged.to_csv(path, index=False)
    return path
