# app/src/uteis/downloaders_aifs_precip.py
# -*- coding: utf-8 -*-
"""
Downloader AIFS-single (IA deterministica do ECMWF) de CHUVA (precipitacao acumulada).

Mesma FONTE e mesma CONVENCAO do HRES: ECMWF Open Data, `tp` acumulado desde o init em METROS,
sem reset -> chuva[D] = (tp(D+1 00Z) - tp(D 00Z)) * 1000. Muda so o caminho do modelo
(`aifs-single/0p25`), entao este modulo e um wrapper fino sobre o motor `ensure_tp_precip_daily`
de `downloaders_ecmwf_precip` -- a logica de acumulacao vive num lugar so.

Confirmado na fonte (2026-07-17, init 00Z): o .index do AIFS traz `tp` com levtype=sfc, ao lado de
`cp` e `sf`. NAO confundir com PWAT (vapor na coluna). Saida = ACUMULADO DIARIO em mm.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

from app.src.uteis.downloaders_aifs_fcst200 import AIFS_MODEL, AIFS_STREAM, AIFS_TYPE
from app.src.uteis.downloaders_ecmwf_fcst200 import DIR_DADOS_BASE
from app.src.uteis.downloaders_ecmwf_precip import ensure_tp_precip_daily

DIR_AIFS_PRECIP: Path = DIR_DADOS_BASE / 'AIFS_PRECIP'

AIFS_MAX_FHR = 360  # AIFS-single vai a 360 h (15 dias), como o HRES


def ensure_aifs_precip_fcst_for_period(
    init: datetime, lead_hours: int, hours=None, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs de CHUVA ACUMULADA DIARIA (mm) do AIFS-single p/ os dias UTC completos em [init, init+lead].

    `hours` e' ignorado (compat com a assinatura dos demais downloaders do globo): a chuva e'
    ACUMULADO DIARIO (00-24 UTC), nao snapshot sinotico."""
    return ensure_tp_precip_daily(
        init, lead_hours, dir_out=DIR_AIFS_PRECIP, prefixo='aifs_precip', tag='AIFS',
        max_fhr=AIFS_MAX_FHR, model=AIFS_MODEL, stream=AIFS_STREAM, ftype=AIFS_TYPE,
        force_redownload=force_redownload)
