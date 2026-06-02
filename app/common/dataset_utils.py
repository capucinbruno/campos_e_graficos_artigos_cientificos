"""
==============================================================================
Módulo: dataset_utils.py
==============================================================================

Descrição:
    Utilitários para gerenciamento seguro de datasets NetCDF com xarray.
    Garante o fechamento automático de arquivos e previne vazamentos de memória.

Funcionalidades:
    - load_dataset: Carrega datasets com fechamento automático
    - Ajuste automático de longitude (-180 a 180)
    - Suporte a filtros de tempo
    - Carregamento completo em memória antes de fechar arquivo

Autor: Sistema GREC
Criado: 2025-10-19
==============================================================================
"""

# Bibliotecas padrão
from pathlib import Path
from typing import Any, Optional, Union

# Bibliotecas de terceiros
import numpy as np
import xarray as xr


def load_dataset(
    filepath: Union[str, Path],
    adjust_lon: bool = True,
    time_slice: Optional[Any] = None,
    decode_times: bool = True,
    chunks: Optional[dict] = None,
) -> xr.Dataset:
    """
    Carrega um dataset NetCDF com ajustes padrão e fecha o arquivo automaticamente.

    Esta função garante que o arquivo NetCDF seja fechado adequadamente após o
    carregamento, prevenindo vazamentos de memória em scripts que processam
    múltiplos arquivos.

    Args:
        filepath: Caminho do arquivo NetCDF
        adjust_lon: Se True, ajusta longitude para o intervalo -180 a 180
        time_slice: Período para filtrar (ex: slice('2024-01-01', '2024-01-31'))
        decode_times: Se True, decodifica as datas automaticamente
        chunks: Dicionário de chunks para carregamento com Dask (opcional)

    Returns:
        xr.Dataset: Dataset carregado completamente em memória

    Exemplos:
        >>> # Uso básico
        >>> ds = load_dataset('dados.nc')

        >>> # Com filtro de tempo
        >>> ds = load_dataset('dados.nc', time_slice=slice('2024-01-01', '2024-01-31'))

        >>> # Sem ajuste de longitude
        >>> ds = load_dataset('dados.nc', adjust_lon=False)

        >>> # Com chunks para datasets grandes
        >>> ds = load_dataset('dados.nc', chunks={'time': 100})

    Notas:
        - O dataset é carregado completamente em memória (.load())
        - O arquivo é fechado automaticamente após o carregamento
        - Ideal para arquivos de tamanho pequeno a médio
        - Para arquivos muito grandes, considere usar chunks ou processar em partes
    """
    # Converter para Path se necessário
    filepath = Path(filepath) if not isinstance(filepath, Path) else filepath

    # Verificar se o arquivo existe
    if not filepath.exists():
        raise FileNotFoundError(f'Arquivo não encontrado: {filepath}')

    # Abrir e processar o dataset
    with xr.open_dataset(filepath, decode_times=decode_times, chunks=chunks) as ds:
        # Ajustar longitude se solicitado e aplicável
        if adjust_lon and 'lon' in ds.dims:
            # Converter longitude de 0-360 para -180 a 180
            ds['lon'] = ((ds['lon'] + 180) % 360) - 180
            ds = ds.sortby(ds.lon)

        # Aplicar filtro de tempo se fornecido
        if time_slice is not None and 'time' in ds.dims:
            ds = ds.sel(time=time_slice)

        # Carregar completamente em memória antes de fechar o arquivo
        # Isso garante que todos os dados estejam disponíveis após o fechamento
        return ds.load()


def load_multiple_datasets(
    filepaths: list,
    adjust_lon: bool = True,
    time_slice: Optional[Any] = None,
    concat_dim: Optional[str] = None,
) -> Union[list, xr.Dataset]:
    """
    Carrega múltiplos datasets NetCDF de forma segura.

    Args:
        filepaths: Lista de caminhos dos arquivos NetCDF
        adjust_lon: Se True, ajusta longitude para -180 a 180
        time_slice: Período para filtrar em todos os datasets
        concat_dim: Se fornecido, concatena os datasets nesta dimensão

    Returns:
        Lista de datasets ou dataset concatenado se concat_dim fornecido

    Exemplos:
        >>> # Carregar múltiplos arquivos
        >>> datasets = load_multiple_datasets(['u250.nc', 'v250.nc'])

        >>> # Concatenar ao longo do tempo
        >>> ds = load_multiple_datasets(['jan.nc', 'feb.nc'], concat_dim='time')
    """
    datasets = []

    for filepath in filepaths:
        ds = load_dataset(filepath, adjust_lon=adjust_lon, time_slice=time_slice)
        datasets.append(ds)

    # Concatenar se solicitado
    if concat_dim and datasets:
        return xr.concat(datasets, dim=concat_dim)

    return datasets


def safe_open_mfdataset(paths: Union[str, list], adjust_lon: bool = True, **kwargs) -> xr.Dataset:
    """
    Versão segura de xr.open_mfdataset com fechamento automático.

    Args:
        paths: Padrão glob ou lista de arquivos
        adjust_lon: Se True, ajusta longitude para -180 a 180
        **kwargs: Argumentos adicionais para xr.open_mfdataset

    Returns:
        Dataset combinado carregado em memória

    Exemplo:
        >>> ds = safe_open_mfdataset('data/*.nc')
    """
    with xr.open_mfdataset(paths, **kwargs) as ds:
        if adjust_lon and 'lon' in ds.dims:
            ds['lon'] = ((ds['lon'] + 180) % 360) - 180
            ds = ds.sortby(ds.lon)
        return ds.load()


def arquivo_cobre_periodo(
    filepath: Union[str, Path],
    start_date: np.datetime64,
    end_date: np.datetime64,
) -> bool:
    """
    Verifica se um arquivo NetCDF existe e cobre o periodo solicitado.

    Util para decidir se e necessario re-baixar um dataset cuja origem
    e atualizada periodicamente (ex.: PSL/NOAA OLR diario).

    Args:
        filepath: Caminho do arquivo NetCDF
        start_date: Data inicial do periodo desejado
        end_date: Data final do periodo desejado

    Returns:
        True se o arquivo existe, abre, possui dimensao 'time' e cobre
        completamente o intervalo [start_date, end_date]. False caso contrario
        (arquivo ausente, corrompido, sem dim time, ou periodo nao coberto).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return False

    try:
        with xr.open_dataset(filepath) as ds:
            if 'time' not in ds.dims:
                return False
            file_start = np.datetime64(ds.time.values[0], 'D')
            file_end = np.datetime64(ds.time.values[-1], 'D')
            inicio = np.datetime64(start_date, 'D')
            fim = np.datetime64(end_date, 'D')
            return file_start <= inicio and fim <= file_end
    except Exception:
        return False


def validar_cobertura_temporal(
    ds: xr.Dataset,
    start_date: np.datetime64,
    end_date: np.datetime64,
    nome: str = 'arquivo',
) -> None:
    """
    Valida que o dataset cobre completamente o periodo solicitado.

    Levanta RuntimeError com mensagem clara mostrando primeira/ultima data
    do dataset quando o periodo solicitado nao esta disponivel ou tem gaps.

    Args:
        ds: Dataset xarray com dimensao 'time'
        start_date: Data inicial do periodo solicitado
        end_date: Data final do periodo solicitado
        nome: Identificador do dataset usado nas mensagens de erro

    Raises:
        RuntimeError: Se o periodo nao esta disponivel ou tem dias faltando
    """
    file_start = np.datetime64(ds.time.values[0], 'D')
    file_end = np.datetime64(ds.time.values[-1], 'D')
    inicio = np.datetime64(start_date, 'D')
    fim = np.datetime64(end_date, 'D')

    if inicio < file_start or fim > file_end:
        raise RuntimeError(
            f'Periodo solicitado nao esta disponivel no {nome}.\n'
            f'   Solicitado:           {inicio} a {fim}\n'
            f'   Disponivel no {nome}: {file_start} a {file_end}\n'
            f'   Acao: ajuste DATA_FINAL para no maximo {file_end} '
            f'ou aguarde a atualizacao do servidor de origem.'
        )

    # Checagem extra: arquivo cobre, mas pode ter gaps no meio
    subset = ds.sel(time=slice(inicio, fim))
    dias_esperados = int((fim - inicio).astype('timedelta64[D]').astype(int)) + 1
    dias_obtidos = len(subset.time)

    if dias_obtidos < dias_esperados:
        raise RuntimeError(
            f'{nome.capitalize()} tem dias faltando no periodo solicitado.\n'
            f'   Periodo:    {inicio} a {fim}\n'
            f'   Esperado:   {dias_esperados} dias\n'
            f'   Disponivel: {dias_obtidos} dias'
        )


# Aliases para compatibilidade
abrir_dataset = load_dataset  # Versão em português
carregar_dataset = load_dataset  # Alternativa em português
