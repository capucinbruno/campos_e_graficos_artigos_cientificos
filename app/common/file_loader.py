"""
Módulo de carregamento de arquivos local/remoto de forma transparente.

Este módulo abstrai o acesso a arquivos, permitindo que scripts funcionem
tanto em ambiente de desenvolvimento (arquivos remotos via SFTP) quanto em
produção (arquivos locais).

Author: @Bruno Capucin
Created: 2025-10-17
"""

# Bibliotecas padrão
import tempfile
from pathlib import Path
from typing import Optional

# Bibliotecas de terceiros
import xarray as xr
import numpy as np

# Módulos locais
from app.config import logger, settings
from app.common.connections import is_remote_file, download_remote_file


def load_netcdf(file_path: str, sftp=None, cache_locally: bool = True) -> xr.Dataset:
    """
    Carrega arquivo NetCDF local ou remoto de forma transparente.

    Detecta automaticamente se deve carregar do filesystem local ou baixar
    via SFTP baseado na configuração settings.RUN_FOR_SFTP.

    Args:
        file_path (str): Caminho do arquivo (local ou remoto)
        sftp (paramiko.SFTPClient, optional): Cliente SFTP (None se modo local)
        cache_locally (bool): Se True, salva arquivo baixado localmente (default: True)
                             Útil para climatologias grandes que serão reutilizadas

    Returns:
        xr.Dataset: Dataset xarray carregado

    Raises:
        FileNotFoundError: Se arquivo não for encontrado (local ou remoto)
        IOError: Se falhar ao baixar arquivo remoto

    Exemplo:
        >>> # Modo development (remoto com SFTP) - baixa e salva localmente
        >>> ds = load_netcdf('Entrada/arquivos_nc/climatologia_CPC_1995_2024.nc', sftp=sftp_client)
        >>>
        >>> # Modo production (local)
        >>> ds = load_netcdf('Entrada/arquivos_nc/climatologia_CPC_1995_2024.nc', sftp=None)
    """
    # Verifica se arquivo existe localmente PRIMEIRO (arquivos baixados por Playwright, etc)
    if Path(file_path).exists():
        logger.info(f'📁 Carregando arquivo NetCDF local: {file_path}')
        ds = xr.open_dataset(file_path)
        logger.info(f'✅ Dataset NetCDF carregado com sucesso: {Path(file_path).name}')
        logger.debug(f'   Dimensões: {dict(ds.dims)}')
        logger.debug(f'   Variáveis: {list(ds.data_vars)}')
        return ds

    # Se não existe localmente e estamos em modo SFTP, tenta buscar remotamente
    if settings.RUN_FOR_SFTP and sftp:
        logger.info(f'🌐 Arquivo não encontrado localmente: {Path(file_path).name}')

        # Ajusta caminho para buscar no servidor remoto
        # Para climatologias, busca em /home/ubuntu/resources/meteorologia/campos-observados/
        remote_path = _adjust_climatology_path(file_path)

        logger.info(f'📡 Tentando baixar do servidor: {remote_path}')

        # Verifica se arquivo existe remotamente
        if not is_remote_file(sftp, remote_path):
            msg = f'❌ Arquivo NetCDF remoto não encontrado: {remote_path}'
            logger.error(msg)
            logger.error(f'   Caminho original: {file_path}')
            raise FileNotFoundError(msg)

        # Determina onde salvar o arquivo
        if cache_locally:
            # Salva no mesmo caminho local (cache permanente)
            local_save_path = file_path
            # Cria diretório se não existir
            Path(local_save_path).parent.mkdir(parents=True, exist_ok=True)

            logger.info(f'💾 Baixando para cache local: {local_save_path}')

            if download_remote_file(sftp, remote_path, local_save_path):
                ds = xr.open_dataset(local_save_path)
                logger.info(f'✅ Arquivo baixado e salvo localmente!')
                logger.info(f'   📂 Cache: {local_save_path}')
                logger.info(f'   📊 Dataset carregado: {Path(file_path).name}')
                logger.debug(f'   Dimensões: {dict(ds.dims)}')
                logger.debug(f'   Variáveis: {list(ds.data_vars)}')
                return ds
            else:
                msg = f'❌ Falha ao baixar arquivo NetCDF remoto: {remote_path}'
                logger.error(msg)
                raise IOError(msg)
        else:
            # Usa arquivo temporário (será deletado após uso)
            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
                tmp_path = tmp.name

            try:
                if download_remote_file(sftp, remote_path, tmp_path):
                    ds = xr.open_dataset(tmp_path)
                    logger.info(f'✅ Dataset NetCDF carregado com sucesso: {Path(file_path).name}')
                    logger.debug(f'   Dimensões: {dict(ds.dims)}')
                    logger.debug(f'   Variáveis: {list(ds.data_vars)}')
                    return ds
                else:
                    msg = f'❌ Falha ao baixar arquivo NetCDF remoto: {remote_path}'
                    logger.error(msg)
                    raise IOError(msg)
            finally:
                # Limpa arquivo temporário
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f'⚠️  Não foi possível remover arquivo temporário {tmp_path}: {e}')

    else:
        # Modo local mas arquivo não encontrado
        msg = f'❌ Arquivo NetCDF local não encontrado: {file_path}'
        logger.error(msg)
        raise FileNotFoundError(msg)


def load_image(file_path: str, sftp=None) -> np.ndarray:
    """
    Carrega imagem PNG/JPG local ou remota.

    Args:
        file_path (str): Caminho da imagem
        sftp (paramiko.SFTPClient, optional): Cliente SFTP (None se modo local)

    Returns:
        numpy.ndarray: Imagem carregada como array NumPy

    Raises:
        FileNotFoundError: Se imagem não for encontrada
        IOError: Se falhar ao baixar imagem remota

    Exemplo:
        >>> img = load_image('Entrada/logo_grec.png', sftp=sftp_client)
        >>> plt.imshow(img)
    """
    import matplotlib.pyplot as plt

    # Verifica se arquivo existe localmente PRIMEIRO
    if Path(file_path).exists():
        logger.debug(f'📁 Carregando imagem local: {file_path}')
        img = plt.imread(file_path)
        logger.debug(f'✅ Imagem carregada: {Path(file_path).name}')
        return img

    # Se não existe localmente e estamos em modo SFTP, tenta buscar remotamente
    if settings.RUN_FOR_SFTP and sftp:
        # Modo remoto
        logger.info(f'🌐 Carregando imagem remota: {Path(file_path).name}')

        remote_path = _adjust_remote_path(file_path)

        if not is_remote_file(sftp, remote_path):
            msg = f'❌ Imagem remota não encontrada: {remote_path}'
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Determina extensão
        suffix = Path(file_path).suffix or '.png'

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        try:
            if download_remote_file(sftp, remote_path, tmp_path):
                img = plt.imread(tmp_path)
                logger.info(f'✅ Imagem carregada: {Path(file_path).name}')
                return img
            else:
                msg = f'❌ Falha ao baixar imagem remota: {remote_path}'
                logger.error(msg)
                raise IOError(msg)
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f'⚠️  Não foi possível remover arquivo temporário {tmp_path}: {e}')

    else:
        # Modo local mas arquivo não encontrado
        msg = f'❌ Imagem local não encontrada: {file_path}'
        logger.error(msg)
        raise FileNotFoundError(msg)


def load_numpy_array(file_path: str, sftp=None, **kwargs) -> np.ndarray:
    """
    Carrega arquivo NumPy (.npy, .npz, ou texto) local ou remoto.

    Args:
        file_path (str): Caminho do arquivo
        sftp (paramiko.SFTPClient, optional): Cliente SFTP
        **kwargs: Argumentos adicionais para np.load() ou np.loadtxt()

    Returns:
        numpy.ndarray: Array carregado

    Raises:
        FileNotFoundError: Se arquivo não for encontrado
        IOError: Se falhar ao baixar arquivo remoto

    Exemplo:
        >>> # Carregar arquivo .npy
        >>> data = load_numpy_array('dados.npy', sftp=sftp_client)
        >>>
        >>> # Carregar arquivo de texto
        >>> data = load_numpy_array('dados.txt', sftp=sftp_client, delimiter=',')
    """
    # Verifica se arquivo existe localmente PRIMEIRO
    if Path(file_path).exists():
        logger.debug(f'📁 Carregando array NumPy local: {file_path}')
        suffix = Path(file_path).suffix

        if suffix in ['.npy', '.npz']:
            data = np.load(file_path, **kwargs)
        else:
            data = np.loadtxt(file_path, **kwargs)

        logger.debug(f'✅ Array NumPy carregado: {Path(file_path).name}')
        logger.debug(f'   Shape: {data.shape}')
        logger.debug(f'   Dtype: {data.dtype}')
        return data

    # Se não existe localmente e estamos em modo SFTP, tenta buscar remotamente
    if settings.RUN_FOR_SFTP and sftp:
        # Modo remoto
        logger.info(f'🌐 Carregando array NumPy remoto: {Path(file_path).name}')

        remote_path = _adjust_remote_path(file_path)

        if not is_remote_file(sftp, remote_path):
            msg = f'❌ Arquivo NumPy remoto não encontrado: {remote_path}'
            logger.error(msg)
            raise FileNotFoundError(msg)

        suffix = Path(file_path).suffix or '.dat'

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        try:
            if download_remote_file(sftp, remote_path, tmp_path):
                # Detecta tipo de arquivo
                if suffix in ['.npy', '.npz']:
                    data = np.load(tmp_path, **kwargs)
                else:
                    # Assume arquivo de texto
                    data = np.loadtxt(tmp_path, **kwargs)

                logger.info(f'✅ Array NumPy carregado: {Path(file_path).name}')
                logger.debug(f'   Shape: {data.shape}')
                logger.debug(f'   Dtype: {data.dtype}')
                return data
            else:
                msg = f'❌ Falha ao baixar arquivo NumPy remoto: {remote_path}'
                logger.error(msg)
                raise IOError(msg)
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f'⚠️  Não foi possível remover arquivo temporário {tmp_path}: {e}')

    else:
        # Modo local mas arquivo não encontrado
        msg = f'❌ Arquivo NumPy local não encontrado: {file_path}'
        logger.error(msg)
        raise FileNotFoundError(msg)


def _adjust_climatology_path(local_path: str) -> str:
    """
    Converte caminho local de climatologia para caminho no servidor de resources.

    Climatologias grandes são armazenadas em:
    /home/ubuntu/resources/meteorologia/campos-observados/Entrada/arquivos_nc/

    Args:
        local_path (str): Caminho local (ex: 'Entrada/arquivos_nc/climatologia_CPC_1995_2024.nc')

    Returns:
        str: Caminho remoto no servidor de resources

    Exemplo:
        >>> _adjust_climatology_path('Entrada/arquivos_nc/climatologia_CPC_1995_2024.nc')
        '/home/ubuntu/resources/meteorologia/campos-observados/Entrada/arquivos_nc/climatologia_CPC_1995_2024.nc'
    """
    local_path = str(local_path)

    # Se já é caminho absoluto remoto, retorna como está
    if local_path.startswith('/home/ubuntu/resources'):
        return local_path

    # Extrai apenas o nome do arquivo se for um caminho
    if '/' in local_path:
        # Ex: 'Entrada/arquivos_nc/climatologia_CPC_1995_2024.nc'
        # Mantém a estrutura relativa
        base_resources = '/home/ubuntu/resources/meteorologia/campos-observados'
        return f'{base_resources}/{local_path}'
    else:
        # Apenas nome do arquivo - assume que está em arquivos_nc
        base_resources = '/home/ubuntu/resources/meteorologia/campos-observados/Entrada/arquivos_nc'
        return f'{base_resources}/{local_path}'


def _adjust_remote_path(local_path: str) -> str:
    """
    Converte caminho local para caminho remoto baseado em configurações.

    Esta função mapeia caminhos relativos locais para caminhos absolutos remotos
    usando as configurações REMOTE_DIR_INPUT e REMOTE_DIR_OUTPUT do settings.toml.

    Args:
        local_path (str): Caminho local (relativo ou absoluto)

    Returns:
        str: Caminho remoto absoluto

    Exemplos:
        >>> _adjust_remote_path('Entrada/psi850.nc')
        '/home/ubuntu/meteorologia/Entrada/psi850.nc'
        >>>
        >>> _adjust_remote_path('Saida/s01_PSI850/mapa.png')
        '/home/ubuntu/meteorologia/Saida/s01_PSI850/mapa.png'
    """
    local_path = str(local_path)

    # Se já é caminho absoluto remoto, retorna como está
    if local_path.startswith('/home/ubuntu'):
        return local_path

    # Substitui diretório base local por remoto
    if local_path.startswith('Entrada'):
        remote_base = settings.get(
            'REMOTE_DIR_INPUT',
            '/home/ubuntu/meteorologia/Entrada'
        )
        return local_path.replace('Entrada', remote_base, 1)

    elif local_path.startswith('Saida'):
        remote_base = settings.get(
            'REMOTE_DIR_OUTPUT',
            '/home/ubuntu/meteorologia/Saida'
        )
        return local_path.replace('Saida', remote_base, 1)

    # Se tem HOME_DIR, substitui
    if settings.HOME_DIR in local_path:
        return local_path.replace(settings.HOME_DIR, '/home/ubuntu')

    # Fallback: assume que é relativo a Entrada
    logger.warning(f'⚠️  Caminho não reconhecido: {local_path}. Assumindo relativo a Entrada/')
    remote_base = settings.get(
        'REMOTE_DIR_INPUT',
        '/home/ubuntu/meteorologia/Entrada'
    )
    return f'{remote_base}/{local_path}'


# Script de teste standalone
if __name__ == "__main__":
    import sys

    print("\n" + "=" * 80)
    print("🧪 TESTE DO MÓDULO FILE_LOADER")
    print("=" * 80 + "\n")

    print("Este módulo requer um cliente SFTP ativo para testes completos.")
    print("Execute os scripts principais para testar em contexto real.")
    print("\n✅ Módulo file_loader.py importado com sucesso!")

    sys.exit(0)
