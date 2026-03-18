# app/src/uteis/processa_era5_vento_pressao.py
# -*- coding: utf-8 -*-
"""
Processamento de ERA5 single-levels (mslp/u100/v100):
- leitura de arquivos mensais NetCDF
- concatenação temporal
- validação de horas sinóticas (00/06/12/18)
- cálculo de média diária
- salvamento em NetCDF

Este módulo foi pensado para trabalhar com os arquivos gerados por:
downloaders_era5_vento_pressao.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, List, Optional, Tuple
import logging
import re

import xarray as xr
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Integração opcional com seu projeto (settings)
# -----------------------------------------------------------------------------
try:
    from app.shared.settings_factory import settings  # type: ignore
    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
except Exception:
    DIR_DADOS_BASE = Path("dados")

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
LOGGER = logging.getLogger("PROC_ERA5_VENTO_PRESSAO")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Diretórios padrão
# -----------------------------------------------------------------------------
DIR_ERA5_BASE = DIR_DADOS_BASE / "ERA5_VENTO_PRESSAO"
DIR_IN = DIR_ERA5_BASE / "mslp_u100_v100_hourly"
DIR_OUT = DIR_ERA5_BASE / "processados"

MIN_BYTES_NETCDF = 50_000
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)


# -----------------------------------------------------------------------------
# Utilitários básicos
# -----------------------------------------------------------------------------
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _file_ok(path: Path, min_bytes: int = MIN_BYTES_NETCDF) -> bool:
    try:
        return path.stat().st_size >= min_bytes
    except FileNotFoundError:
        return False


def _ensure_time_coord(obj):
    """Normaliza valid_time -> time e decodifica CF se necessário."""
    if hasattr(obj, "dims") and "time" not in obj.dims and "valid_time" in obj.dims:
        obj = obj.rename({"valid_time": "time"})
    elif hasattr(obj, "coords") and "time" not in obj.coords and "valid_time" in obj.coords:
        obj = obj.rename({"valid_time": "time"})

    if "time" not in obj.coords:
        raise KeyError("Nem 'time' nem 'valid_time' encontrados.")

    if not np.issubdtype(obj["time"].dtype, np.datetime64):
        if isinstance(obj, xr.DataArray):
            obj = xr.decode_cf(obj.to_dataset(name="_tmp"))["_tmp"]
        else:
            obj = xr.decode_cf(obj)
    return obj


def _drop_or_collapse_expver(ds: xr.Dataset) -> xr.Dataset:
    """
    Colapsa dim expver/number se existirem (comum em ERA5/ERA5T).
    """
    rename_dims = {}
    for d in ds.dims:
        dl = d.lower()
        if dl == "expver" and d != "expver":
            rename_dims[d] = "expver"
        elif dl == "number" and d != "number":
            rename_dims[d] = "number"

    if rename_dims:
        ds = ds.rename(rename_dims)

    if "expver" in ds.dims:
        ds = ds.bfill("expver").ffill("expver").isel(expver=0, drop=True)

    if "number" in ds.dims:
        ds = ds.isel(number=0, drop=True)

    for c in ("expver", "number"):
        if c in ds.coords and c not in ds.dims:
            try:
                ds = ds.drop_vars(c)
            except Exception:
                pass

    return ds


def _sort_and_dedup_time(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.sortby("time")
    # Remove timestamps duplicados se existirem
    t = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    _, idx = np.unique(t.values, return_index=True)
    idx = np.sort(idx)
    if len(idx) != ds.sizes.get("time", 0):
        LOGGER.warning("Removendo %d timestamps duplicados.", ds.sizes["time"] - len(idx))
        ds = ds.isel(time=idx)
    return ds


def _subset_target_vars(ds: xr.Dataset) -> xr.Dataset:
    """
    Mantém preferencialmente msl/u100/v100 (ou nomes longos equivalentes).
    """
    mapping_candidates = {
        "msl": ["msl", "mean_sea_level_pressure"],
        "u100": ["u100", "100m_u_component_of_wind"],
        "v100": ["v100", "100m_v_component_of_wind"],
    }

    selected = {}
    for std_name, candidates in mapping_candidates.items():
        for c in candidates:
            if c in ds.data_vars:
                selected[std_name] = ds[c]
                break

    if not selected:
        # fallback: mantém tudo que for numérico com dimensão time
        keep = []
        for vn in ds.data_vars:
            if "time" in ds[vn].dims and np.issubdtype(ds[vn].dtype, np.number):
                keep.append(vn)
        if not keep:
            raise KeyError("Nenhuma variável temporal numérica encontrada no dataset.")
        LOGGER.warning(
            "Não encontrei msl/u100/v100 por nome. Mantendo variáveis numéricas com tempo: %s",
            keep,
        )
        return ds[keep]

    out = xr.Dataset(selected, coords=ds.coords, attrs=ds.attrs)
    return out


# -----------------------------------------------------------------------------
# Descoberta de arquivos
# -----------------------------------------------------------------------------
def _extract_yyyymm_from_name(name: str) -> Optional[Tuple[int, int]]:
    # exemplo: era5_mslp_u100_v100_hourly_202602_h00061218.nc
    m = re.search(r"_(\d{4})(\d{2})_h\d+\.nc$", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def list_hourly_files(
    input_dir: Path = DIR_IN,
    start_yyyymm: Optional[Tuple[int, int]] = None,
    end_yyyymm: Optional[Tuple[int, int]] = None,
) -> List[Path]:
    """
    Lista arquivos mensais horários (.nc) do downloader.
    Pode filtrar por intervalo YYYYMM.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {input_dir}")

    files = sorted(input_dir.glob("era5_mslp_u100_v100_hourly_*_h*.nc"))
    files = [f for f in files if _file_ok(f)]

    out: List[Path] = []
    for f in files:
        ym = _extract_yyyymm_from_name(f.name)
        if ym is None:
            continue

        if start_yyyymm is not None and ym < start_yyyymm:
            continue
        if end_yyyymm is not None and ym > end_yyyymm:
            continue

        out.append(f)

    LOGGER.info("Arquivos horários encontrados para processamento: %d", len(out))
    for f in out:
        LOGGER.info(" - %s", f.name)

    return out


# -----------------------------------------------------------------------------
# Leitura / merge
# -----------------------------------------------------------------------------
def open_and_merge_hourly_files(
    files: Sequence[Path],
    *,
    chunks: Optional[dict] = None,
) -> xr.Dataset:
    """
    Abre e concatena arquivos mensais em um único Dataset.
    """
    if not files:
        raise ValueError("Lista de arquivos vazia.")

    dsets: List[xr.Dataset] = []

    for fp in files:
        LOGGER.info("Abrindo: %s", fp)
        ds = xr.open_dataset(fp, engine="netcdf4", chunks=chunks)
        ds = _ensure_time_coord(ds)
        ds = _drop_or_collapse_expver(ds)
        ds = _subset_target_vars(ds)
        dsets.append(ds)

    if len(dsets) == 1:
        ds_all = dsets[0]
    else:
        ds_all = xr.concat(dsets, dim="time", data_vars="minimal", coords="minimal", compat="override")

    ds_all = _ensure_time_coord(ds_all)
    ds_all = _sort_and_dedup_time(ds_all)

    LOGGER.info(
        "Dataset combinado: %d timestamps | %s -> %s",
        ds_all.sizes.get("time", 0),
        str(pd.to_datetime(ds_all.time.values[0])) if ds_all.sizes.get("time", 0) else "NA",
        str(pd.to_datetime(ds_all.time.values[-1])) if ds_all.sizes.get("time", 0) else "NA",
    )
    return ds_all


# -----------------------------------------------------------------------------
# Validação de sinóticas e média diária
# -----------------------------------------------------------------------------
def summarize_daily_synoptic_coverage(
    ds: xr.Dataset,
    required_hours_utc: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
) -> pd.DataFrame:
    """
    Retorna dataframe com cobertura diária:
    - n_times (número de horários sinóticos presentes no dia)
    - hours_present (lista de horas)
    - is_complete (True/False)
    """
    ds = _ensure_time_coord(ds)

    t_idx = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    required_set = set(int(h) for h in required_hours_utc)
    t_sel = t_idx[t_idx.hour.isin(required_set)]

    if len(t_sel) == 0:
        return pd.DataFrame(columns=["date", "n_times", "hours_present", "is_complete"]).set_index("date")

    s = pd.Series(t_sel.hour, index=t_sel.floor("D"))
    grouped = s.groupby(level=0)

    rows = []
    for dt, vals in grouped:
        hours_present = sorted(set(int(v) for v in vals.tolist()))
        rows.append(
            {
                "date": pd.Timestamp(dt),
                "n_times": len(hours_present),
                "hours_present": hours_present,
                "is_complete": set(hours_present) >= required_set and len(hours_present) >= len(required_set),
            }
        )

    cov = pd.DataFrame(rows).set_index("date").sort_index()
    return cov


def compute_daily_mean_from_synoptic(
    ds: xr.Dataset,
    *,
    required_hours_utc: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    require_complete_days: bool = True,
    keep_only_required_hours: bool = True,
) -> xr.Dataset:
    """
    Calcula média diária a partir das horas sinóticas.

    Parâmetros
    ----------
    require_complete_days : bool
        Se True, só mantém dias com todas as horas requeridas (ex.: 00/06/12/18).
        Se False, calcula média com o que houver no dia (entre as horas requeridas).
    keep_only_required_hours : bool
        Se True, filtra o dataset para as horas requeridas antes da média.
    """
    ds = _ensure_time_coord(ds)
    ds = _sort_and_dedup_time(ds)

    required_set = set(int(h) for h in required_hours_utc)
    t_idx = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))

    if keep_only_required_hours:
        mask = np.array([h in required_set for h in t_idx.hour], dtype=bool)
        ds = ds.isel(time=mask)
        t_idx = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))

    if ds.sizes.get("time", 0) == 0:
        raise ValueError("Dataset sem timestamps após filtrar horas requeridas.")

    coverage = summarize_daily_synoptic_coverage(ds, required_hours_utc=required_hours_utc)

    if require_complete_days:
        complete_days = coverage.index[coverage["is_complete"]].normalize()
        LOGGER.info(
            "Cobertura diária: %d dias completos de %d dias com registros.",
            int(coverage["is_complete"].sum()),
            len(coverage),
        )

        if len(complete_days) == 0:
            raise ValueError("Nenhum dia completo encontrado para as horas sinóticas requeridas.")

        day_index = pd.DatetimeIndex(pd.to_datetime(ds["time"].values)).floor("D")
        mask_complete = np.array([d in set(complete_days) for d in day_index], dtype=bool)
        ds = ds.isel(time=mask_complete)

        # Segurança extra: para dias incompletos remanescentes (não deveria ocorrer), elimina via groupby count
        coverage2 = summarize_daily_synoptic_coverage(ds, required_hours_utc=required_hours_utc)
        bad_days = coverage2.index[~coverage2["is_complete"]]
        if len(bad_days) > 0:
            LOGGER.warning("Ainda havia %d dias incompletos após filtro. Removendo explicitamente.", len(bad_days))
            day_index2 = pd.DatetimeIndex(pd.to_datetime(ds["time"].values)).floor("D")
            mask_good = np.array([d not in set(bad_days) for d in day_index2], dtype=bool)
            ds = ds.isel(time=mask_good)

    else:
        LOGGER.info(
            "Calculando média diária sem exigir dias completos (pode haver média com < %d horários).",
            len(required_set),
        )

    # Média diária (resample diário)
    ds_daily = ds.resample(time="1D").mean(keep_attrs=True)

    # Remove dias totalmente vazios eventualmente criados pelo resample
    if ds_daily.data_vars:
        valid_any = None
        for vn in ds_daily.data_vars:
            da = ds_daily[vn]
            # reduz em todas as dims exceto time
            other_dims = [d for d in da.dims if d != "time"]
            if other_dims:
                mask_v = da.notnull().any(dim=other_dims)
            else:
                mask_v = da.notnull()

            valid_any = mask_v if valid_any is None else (valid_any | mask_v)

        if valid_any is not None:
            ds_daily = ds_daily.isel(time=valid_any.values)

    ds_daily.attrs = dict(ds.attrs)
    ds_daily.attrs["daily_mean_from_synoptic_hours"] = ",".join(f"{h:02d}" for h in sorted(required_set))
    ds_daily.attrs["daily_mean_require_complete_days"] = str(require_complete_days)

    LOGGER.info(
        "Média diária concluída: %d dias | %s -> %s",
        ds_daily.sizes.get("time", 0),
        str(pd.to_datetime(ds_daily.time.values[0])) if ds_daily.sizes.get("time", 0) else "NA",
        str(pd.to_datetime(ds_daily.time.values[-1])) if ds_daily.sizes.get("time", 0) else "NA",
    )

    return ds_daily


# -----------------------------------------------------------------------------
# Salvamento
# -----------------------------------------------------------------------------
def save_dataset_netcdf(
    ds: xr.Dataset,
    output_path: Path,
    *,
    compress: bool = True,
) -> Path:
    _ensure_dir(output_path.parent)

    encoding = {}
    if compress:
        for vn in ds.data_vars:
            encoding[vn] = {"zlib": True, "complevel": 4}

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    if tmp_path.exists():
        tmp_path.unlink()
    if output_path.exists():
        output_path.unlink()

    LOGGER.info("Salvando NetCDF: %s", output_path)
    ds.to_netcdf(tmp_path, engine="netcdf4", encoding=encoding)
    tmp_path.rename(output_path)

    LOGGER.info("[OK] Arquivo salvo: %s", output_path)
    return output_path


# -----------------------------------------------------------------------------
# Pipeline pronto (hourly -> daily mean)
# -----------------------------------------------------------------------------
def build_daily_mean_dataset_from_monthly_files(
    files: Sequence[Path],
    *,
    required_hours_utc: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    require_complete_days: bool = True,
    output_path: Optional[Path] = None,
    compress: bool = True,
) -> tuple[xr.Dataset, Optional[Path], pd.DataFrame]:
    """
    Pipeline completo:
    1) abre e concatena arquivos
    2) sumariza cobertura diária
    3) calcula média diária
    4) salva (opcional)

    Retorna:
    - ds_daily
    - output_path (ou None)
    - coverage_df (da série horária original)
    """
    ds_hourly = open_and_merge_hourly_files(files)
    coverage = summarize_daily_synoptic_coverage(ds_hourly, required_hours_utc=required_hours_utc)

    n_complete = int(coverage["is_complete"].sum()) if len(coverage) else 0
    n_total = int(len(coverage))
    LOGGER.info("Resumo cobertura: %d/%d dias completos.", n_complete, n_total)

    ds_daily = compute_daily_mean_from_synoptic(
        ds_hourly,
        required_hours_utc=required_hours_utc,
        require_complete_days=require_complete_days,
        keep_only_required_hours=True,
    )

    saved = None
    if output_path is not None:
        saved = save_dataset_netcdf(ds_daily, output_path, compress=compress)

    return ds_daily, saved, coverage


# -----------------------------------------------------------------------------
# Exemplo de uso
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Exemplo: processar fev/2026 (arquivos mensais já baixados)
    # Ajuste se quiser filtrar um intervalo maior.
    files = list_hourly_files(
        input_dir=DIR_IN,
        start_yyyymm=(2026, 2),
        end_yyyymm=(2026, 2),
    )

    if not files:
        raise SystemExit("Nenhum arquivo encontrado para processar.")

    _ensure_dir(DIR_OUT)
    out_nc = DIR_OUT / "era5_mslp_u100_v100_dailymean_202602_h00061218.nc"

    ds_daily, saved_path, coverage = build_daily_mean_dataset_from_monthly_files(
        files,
        required_hours_utc=(0, 6, 12, 18),
        require_complete_days=True,   # só dias com 4 sinóticas
        output_path=out_nc,
        compress=True,
    )

    print("\n=== RESUMO COBERTURA (top 10) ===")
    print(coverage.head(10))
    print("\n=== RESUMO COBERTURA (tail 10) ===")
    print(coverage.tail(10))
    print(f"\nArquivo diário salvo em: {saved_path}")
    print(f"Dims do diário: {ds_daily.sizes}")