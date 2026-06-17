# app/common/forecast_download.py
# -*- coding: utf-8 -*-
"""
Helper para baixar os DIAS de previsao em paralelo (download e I/O-bound -> threads).

Os downloads diarios de cada modelo (GFS/GEFS/ECMWF HRES) sao INDEPENDENTES — cada dia
valido gera seu proprio NetCDF — entao rodam num ThreadPool limitado por `DOWNLOAD_WORKERS`
(default 6). O pool e unico e modesto de proposito: o NOMADS limita/bane IPs que abrem
conexoes demais (o ECMWF/S3 tolera mais). Modelos e variaveis (200/OLR/T850) seguem em serie,
entao a concorrencia total fica limitada a `DOWNLOAD_WORKERS`.

Diferente do `parallel_helper.parallel_process`, este helper PROPAGA excecoes (nao engole):
um dia que falhe deve estourar o erro, e nao gerar dados incompletos silenciosamente.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from app.shared.settings_factory import settings

# ecCodes (cfgrib) e HDF5 (netCDF4) NAO sao thread-safe: chamadas concorrentes dessas libs
# entre os downloads paralelos de dias corrompem arquivos silenciosamente (coords faltando).
# Este lock serializa SO as chamadas de lib (abrir GRIB / escrever NetCDF); o download de rede
# (a parte cara) segue paralelo. Reentrante por seguranca se algum caminho aninhar.
GRIB_NETCDF_LOCK = threading.RLock()


class StepNotAvailable(Exception):
    """Passo de previsao ainda nao publicado na fonte (HTTP 404). NAO e erro fatal: o
    downloader pula esse passo/dia e segue com o que ja existe (ex.: trecho estendido do
    GEFS 00Z que leva horas para subir no NOMADS)."""


def save_netcdf(ds, path, engine: str = 'netcdf4', **kwargs) -> None:
    """Escreve NetCDF sob o lock global (HDF5 nao e thread-safe entre downloads paralelos)."""
    with GRIB_NETCDF_LOCK:
        ds.to_netcdf(path, engine=engine, **kwargs)

# (day, steps) onde steps = lista de (step_h, valid_time)
DayJob = Tuple[object, Sequence]


def download_workers() -> int:
    """Nº de downloads diarios simultaneos (settings.DOWNLOAD_WORKERS, default 6)."""
    return max(1, int(settings.get('DOWNLOAD_WORKERS', 6)))


def download_days_parallel(
    jobs: List[DayJob],
    download_fn: Callable[[object, Sequence], Optional[Path]],
    logger=None,
    workers: Optional[int] = None,
) -> List[Path]:
    """Roda `download_fn(day, steps)` para cada job num ThreadPool (<= `workers`).

    Preserva a ordem dos dias e descarta retornos None (dia sem passos validos). Excecoes
    de qualquer dia sao propagadas (apos as demais threads encerrarem o bloco `with`).
    """
    if not jobs:
        return []
    n = download_workers() if workers is None else max(1, workers)
    n = min(n, len(jobs))
    if n == 1:
        return [p for p in (download_fn(d, s) for d, s in jobs) if p is not None]

    if logger is not None:
        logger.info('Download paralelo: {} dias em ate {} threads', len(jobs), n)
    results: List[Tuple[int, Optional[Path]]] = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        fut_to_idx = {ex.submit(download_fn, day, steps): i for i, (day, steps) in enumerate(jobs)}
        for fut in as_completed(fut_to_idx):
            results.append((fut_to_idx[fut], fut.result()))  # .result() re-lanca a excecao
    results.sort(key=lambda t: t[0])
    return [p for _, p in results if p is not None]
