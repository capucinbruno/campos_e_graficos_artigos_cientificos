"""
Helper otimizado para downloads de arquivos NetCDF com progress bar.

Suporta múltiplas engines de download:
- FTP (3.8x mais rápido que HTTP para PSL/NOAA) ← RECOMENDADO
- HTTP/2 via HTTPX (fallback automático para HTTP/1.1)
- HTTP/1.1 via requests (fallback)
- PyCURL (opcional, alta performance)
- Aria2 (opcional, download paralelo)

Features:
- Seleção automática da melhor engine disponível
- Streaming com TQDM progress bar
- Retry automático com backoff exponencial
- Connection pooling
- Timeout configurável
- Verificação de arquivo existente
- Detecção automática de protocolo (FTP vs HTTP)
- Conversão automática HTTP → FTP para PSL/NOAA

Usage:
    from app.common.download_helper import download_with_progress, DownloadEngine

    # Automático (usa melhor engine disponível)
    download_with_progress(url, output_path)

    # Forçar engine específica
    download_with_progress(url, output_path, engine=DownloadEngine.FTP)
    download_with_progress(url, output_path, engine=DownloadEngine.HTTPX)
    download_with_progress(url, output_path, engine=DownloadEngine.PYCURL)
    download_with_progress(url, output_path, engine=DownloadEngine.ARIA2)
"""

import os
import time
import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from ftplib import FTP
from enum import Enum

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import pycurl
    PYCURL_AVAILABLE = True
except ImportError:
    PYCURL_AVAILABLE = False

# Verificar se aria2c está disponível
ARIA2_AVAILABLE = False
try:
    result = subprocess.run(['aria2c', '--version'], capture_output=True, timeout=2)
    ARIA2_AVAILABLE = (result.returncode == 0)
except:
    pass

from tqdm import tqdm
from app.common.logger import get_logger

logger = get_logger("download")


class DownloadEngine(Enum):
    """Engines de download disponíveis."""
    AUTO = "auto"          # Seleciona automaticamente a melhor
    FTP = "ftp"            # FTP nativo (3.8x mais rápido para NOAA)
    HTTPX = "httpx"        # HTTP/2 via HTTPX
    REQUESTS = "requests"  # HTTP/1.1 via requests (fallback)
    PYCURL = "pycurl"      # PyCURL (C library, alta performance)
    ARIA2 = "aria2"        # Aria2 (download paralelo)


def convert_http_to_ftp(url: str) -> str:
    """
    Converte URL HTTP do PSL/NOAA para FTP (3-4x mais rápido).

    URLs suportadas:
    - https://downloads.psl.noaa.gov/... → ftp://ftp.cdc.noaa.gov/...
    - https://psl.noaa.gov/thredds/fileServer/Datasets/... → ftp://ftp.cdc.noaa.gov/Datasets/...

    Args:
        url: URL HTTP original

    Returns:
        URL FTP equivalente, ou URL original se não houver conversão disponível

    Example:
        >>> convert_http_to_ftp("https://downloads.psl.noaa.gov/Datasets/file.nc")
        'ftp://ftp.cdc.noaa.gov/Datasets/file.nc'
        >>> convert_http_to_ftp("https://psl.noaa.gov/thredds/fileServer/Datasets/cpc_global_temp/tmax.2021.nc")
        'ftp://ftp.cdc.noaa.gov/Datasets/cpc_global_temp/tmax.2021.nc'
    """
    parsed = urlparse(url)

    # Conversão para PSL/NOAA
    if 'psl.noaa.gov' in parsed.netloc or 'cdc.noaa.gov' in parsed.netloc:
        # Remover /thredds/fileServer/ do path se presente
        path = parsed.path
        if '/thredds/fileServer/' in path:
            path = path.replace('/thredds/fileServer/', '/')

        ftp_url = f"ftp://ftp.cdc.noaa.gov{path}"
        logger.debug(f"🔄 Convertendo HTTP → FTP para melhor performance")
        logger.debug(f"   Original: {url}")
        logger.debug(f"   FTP:      {ftp_url}")
        return ftp_url

    # Retornar URL original se não houver conversão
    return url


def _select_engine(
    requested_engine: DownloadEngine,
    protocol: str,
    url: str
) -> DownloadEngine:
    """
    Seleciona a melhor engine de download disponível.

    Args:
        requested_engine: Engine solicitada pelo usuário
        protocol: Protocolo da URL (ftp, http, https)
        url: URL completa para download

    Returns:
        Engine selecionada

    Ordem de preferência (AUTO):
    1. FTP para protocolos FTP (3.8x mais rápido)
    2. Aria2 se disponível (download paralelo)
    3. PyCURL se disponível (C library, rápido)
    4. HTTPX se disponível (HTTP/2)
    5. requests (fallback HTTP/1.1)
    """
    # Se engine específica foi solicitada, validar e usar
    if requested_engine != DownloadEngine.AUTO:
        if requested_engine == DownloadEngine.FTP and protocol != 'ftp':
            logger.warning(f"⚠️  FTP solicitado mas URL é {protocol}. Usando AUTO.")
            requested_engine = DownloadEngine.AUTO
        elif requested_engine == DownloadEngine.ARIA2 and not ARIA2_AVAILABLE:
            logger.warning("⚠️  Aria2 não disponível. Usando AUTO.")
            requested_engine = DownloadEngine.AUTO
        elif requested_engine == DownloadEngine.PYCURL and not PYCURL_AVAILABLE:
            logger.warning("⚠️  PyCURL não disponível. Usando AUTO.")
            requested_engine = DownloadEngine.AUTO
        elif requested_engine == DownloadEngine.HTTPX and not HTTPX_AVAILABLE:
            logger.warning("⚠️  HTTPX não disponível. Usando AUTO.")
            requested_engine = DownloadEngine.AUTO

        # Se ainda não é AUTO, usar a solicitada
        if requested_engine != DownloadEngine.AUTO:
            return requested_engine

    # Seleção automática
    if protocol == 'ftp':
        return DownloadEngine.FTP

    # Para HTTP/HTTPS, escolher a melhor biblioteca disponível
    if ARIA2_AVAILABLE:
        return DownloadEngine.ARIA2
    elif PYCURL_AVAILABLE:
        return DownloadEngine.PYCURL
    elif HTTPX_AVAILABLE:
        return DownloadEngine.HTTPX
    else:
        return DownloadEngine.REQUESTS


def download_with_progress(
    url: str,
    output_path: str,
    description: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 5,
    chunk_size: int = 8192,
    force: bool = False,
    prefer_ftp: bool = True,
    engine: DownloadEngine = DownloadEngine.AUTO,
) -> bool:
    """
    Baixa arquivo com barra de progresso e retry automático.

    Args:
        url: URL do arquivo para download
        output_path: Caminho completo onde salvar o arquivo
        description: Descrição para a barra de progresso (padrão: nome do arquivo)
        timeout: Timeout em segundos (padrão: 120)
        max_retries: Número máximo de tentativas (padrão: 5)
        chunk_size: Tamanho dos chunks em bytes (padrão: 8KB)
        force: Se True, baixa mesmo se arquivo já existir (padrão: False)
        prefer_ftp: Se True, converte HTTP para FTP quando possível (padrão: True)
        engine: Engine de download (AUTO, FTP, HTTPX, REQUESTS, PYCURL, ARIA2)

    Returns:
        bool: True se download foi bem-sucedido, False caso contrário

    Example:
        >>> download_with_progress(
        ...     url="https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.anom.2025.nc",
        ...     output_path="Saida/s50_COMPOSITE_SST_ANOM_OLR/sst_anom_2025.nc",
        ...     description="SST 2025"
        ... )
    """
    # Criar diretório se não existir
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Verificar variável de ambiente FORCE_DOWNLOAD
    force_env = os.environ.get('FORCE_DOWNLOAD', 'false').lower() in ('true', '1', 'yes')
    force = force or force_env

    # Verificar se arquivo já existe
    if not force and os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"✅ Arquivo já existe: {output_path} ({file_size_mb:.1f} MB)")
        return True

    # Se force=True e arquivo existe, avisar que vai re-baixar
    if force and os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.warning(f"🔄 Forçando re-download: {output_path} ({file_size_mb:.1f} MB)")

    # Usar nome do arquivo como descrição padrão
    if description is None:
        description = Path(output_path).name

    # Converter HTTP → FTP se preferível e em modo AUTO (3-4x mais rápido)
    if prefer_ftp and engine == DownloadEngine.AUTO:
        url = convert_http_to_ftp(url)

    # Detectar protocolo da URL
    parsed_url = urlparse(url)
    protocol = parsed_url.scheme.lower()

    # Selecionar engine de download
    selected_engine = _select_engine(engine, protocol, url)
    logger.info(f"🔧 Engine selecionada: {selected_engine.value.upper()}")

    # Despachar para a engine apropriada
    if selected_engine == DownloadEngine.FTP:
        return _download_ftp(
            url, output_path, description, timeout, max_retries, chunk_size
        )
    elif selected_engine == DownloadEngine.ARIA2:
        return _download_aria2(
            url, output_path, description, timeout, max_retries
        )
    elif selected_engine == DownloadEngine.PYCURL:
        return _download_pycurl(
            url, output_path, description, timeout, max_retries, chunk_size
        )
    elif selected_engine == DownloadEngine.HTTPX:
        return _download_httpx(
            url, output_path, description, timeout, max_retries, chunk_size
        )
    elif selected_engine == DownloadEngine.REQUESTS:
        return _download_requests(
            url, output_path, description, timeout, max_retries, chunk_size
        )
    else:
        logger.error(f"❌ Engine não suportada: {selected_engine}")
        return False


def _download_httpx(
    url: str,
    output_path: str,
    description: str,
    timeout: int,
    max_retries: int,
    chunk_size: int,
) -> bool:
    """Download usando HTTPX com HTTP/2."""
    logger.info(f"⬇️  Iniciando download (HTTP/2): {description}")
    logger.info(f"   URL: {url}")

    # Client com HTTP/2, connection pooling e timeouts
    client = httpx.Client(
        http2=True,  # Habilita HTTP/2 (fallback automático para HTTP/1.1)
        timeout=httpx.Timeout(timeout, connect=30),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        follow_redirects=True,
    )

    for tentativa in range(max_retries):
        try:
            logger.info(f"   Tentativa {tentativa + 1}/{max_retries}...")

            # Requisição com streaming
            with client.stream("GET", url) as response:
                response.raise_for_status()

                # Detectar protocolo usado
                protocol = "HTTP/2" if response.http_version == "HTTP/2" else "HTTP/1.1"
                logger.info(f"   Protocolo: {protocol}")

                # Obter tamanho total
                total_size = int(response.headers.get("content-length", 0))
                file_size_mb = total_size / (1024 * 1024)
                logger.info(f"   Tamanho: {file_size_mb:.1f} MB")

                # Download com progress bar
                with open(output_path, "wb") as file:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"   {description}",
                        ncols=100,
                        bar_format="   {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                    ) as pbar:
                        for chunk in response.iter_bytes(chunk_size=chunk_size):
                            if chunk:
                                file.write(chunk)
                                pbar.update(len(chunk))

            logger.info(f"✅ Download concluído: {output_path}")
            client.close()
            return True

        except httpx.TimeoutException:
            logger.error(f"   ⏱️  Timeout na tentativa {tentativa + 1}")
            if tentativa < max_retries - 1:
                wait_time = 2 ** tentativa  # Backoff exponencial: 1s, 2s, 4s, 8s...
                logger.info(f"   ⏳ Aguardando {wait_time}s antes de tentar novamente...")
                time.sleep(wait_time)

        except httpx.HTTPStatusError as e:
            logger.error(f"   ❌ Erro HTTP {e.response.status_code}: {e}")
            client.close()
            return False

        except Exception as e:
            logger.error(f"   ❌ Erro: {e}")
            if tentativa < max_retries - 1:
                wait_time = 2 ** tentativa
                logger.info(f"   ⏳ Aguardando {wait_time}s antes de tentar novamente...")
                time.sleep(wait_time)

    logger.error(f"❌ Falha no download após {max_retries} tentativas")
    client.close()
    return False


def _download_requests(
    url: str,
    output_path: str,
    description: str,
    timeout: int,
    max_retries: int,
    chunk_size: int,
) -> bool:
    """Download usando requests (fallback para HTTP/1.1)."""
    logger.info(f"⬇️  Iniciando download (HTTP/1.1): {description}")
    logger.info(f"   URL: {url}")

    for tentativa in range(max_retries):
        try:
            logger.info(f"   Tentativa {tentativa + 1}/{max_retries}...")

            # Requisição com streaming
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            # Obter tamanho total
            total_size = int(response.headers.get("content-length", 0))
            file_size_mb = total_size / (1024 * 1024)
            logger.info(f"   Tamanho: {file_size_mb:.1f} MB")

            # Download com progress bar
            with open(output_path, "wb") as file:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"   {description}",
                    ncols=100,
                    bar_format="   {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"✅ Download concluído: {output_path}")
            return True

        except requests.Timeout:
            logger.error(f"   ⏱️  Timeout na tentativa {tentativa + 1}")
            if tentativa < max_retries - 1:
                wait_time = 2 ** tentativa
                logger.info(f"   ⏳ Aguardando {wait_time}s antes de tentar novamente...")
                time.sleep(wait_time)

        except requests.HTTPError as e:
            logger.error(f"   ❌ Erro HTTP {e.response.status_code}: {e}")
            return False

        except Exception as e:
            logger.error(f"   ❌ Erro: {e}")
            if tentativa < max_retries - 1:
                wait_time = 2 ** tentativa
                logger.info(f"   ⏳ Aguardando {wait_time}s antes de tentar novamente...")
                time.sleep(wait_time)

    logger.error(f"❌ Falha no download após {max_retries} tentativas")
    return False


def _download_ftp(
    url: str,
    output_path: str,
    description: str,
    timeout: int,
    max_retries: int,
    chunk_size: int,
) -> bool:
    """Download usando FTP (3-4x mais rápido que HTTP para PSL/NOAA)."""
    logger.info(f"⬇️  Iniciando download (FTP): {description}")
    logger.info(f"   URL: {url}")

    # Parse URL FTP
    parsed = urlparse(url)
    host = parsed.netloc
    path = parsed.path
    directory = str(Path(path).parent)
    filename = Path(path).name

    for tentativa in range(max_retries):
        try:
            logger.info(f"   Tentativa {tentativa + 1}/{max_retries}...")

            # Conectar ao servidor FTP
            ftp = FTP(host, timeout=timeout)
            ftp.login()  # Login anônimo
            logger.info(f"   ✅ Conectado ao servidor FTP: {host}")

            # Mudar para o diretório correto
            ftp.cwd(directory)

            # Obter tamanho do arquivo
            total_size = ftp.size(filename)
            file_size_mb = total_size / (1024 * 1024)
            logger.info(f"   Tamanho: {file_size_mb:.1f} MB")

            # Download com progress bar
            downloaded = [0]  # Lista para evitar problema com escopo

            class DownloadInterrupt(Exception):
                pass

            with open(output_path, "wb") as file:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"   {description}",
                    ncols=100,
                    bar_format="   {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                ) as pbar:
                    def callback(data):
                        file.write(data)
                        downloaded[0] += len(data)
                        pbar.update(len(data))

                    ftp.retrbinary(f"RETR {filename}", callback, blocksize=chunk_size)

            ftp.quit()
            logger.info(f"✅ Download concluído: {output_path}")
            return True

        except Exception as e:
            logger.error(f"   ❌ Erro na tentativa {tentativa + 1}: {e}")
            if tentativa < max_retries - 1:
                wait_time = 2 ** tentativa  # Backoff exponencial
                logger.info(f"   ⏳ Aguardando {wait_time}s antes de tentar novamente...")
                time.sleep(wait_time)

    logger.error(f"❌ Falha no download FTP após {max_retries} tentativas")
    return False


def _download_pycurl(
    url: str,
    output_path: str,
    description: str,
    timeout: int,
    max_retries: int,
    chunk_size: int,
) -> bool:
    """Download usando PyCURL (C library, alta performance)."""
    if not PYCURL_AVAILABLE:
        logger.error("❌ PyCURL não está disponível")
        return False

    logger.info(f"⬇️  Iniciando download (PyCURL): {description}")
    logger.info(f"   URL: {url}")

    for tentativa in range(max_retries):
        try:
            logger.info(f"   Tentativa {tentativa + 1}/{max_retries}...")

            # Criar objeto curl
            c = pycurl.Curl()
            c.setopt(c.URL, url)
            c.setopt(c.FOLLOWLOCATION, True)
            c.setopt(c.TIMEOUT, timeout)

            # Progress callback
            downloaded = [0]
            pbar = [None]

            def progress_callback(download_total, downloaded_bytes, upload_total, uploaded_bytes):
                if download_total > 0:
                    if pbar[0] is None:
                        file_size_mb = download_total / (1024 * 1024)
                        logger.info(f"   Tamanho: {file_size_mb:.1f} MB")
                        pbar[0] = tqdm(
                            total=int(download_total),
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=f"   {description}",
                            ncols=100,
                            bar_format="   {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                        )
                    pbar[0].update(int(downloaded_bytes - downloaded[0]))
                    downloaded[0] = downloaded_bytes

            c.setopt(c.NOPROGRESS, False)
            c.setopt(c.XFERINFOFUNCTION, progress_callback)

            # Download
            with open(output_path, "wb") as f:
                c.setopt(c.WRITEDATA, f)
                c.perform()

            # Fechar progress bar
            if pbar[0]:
                pbar[0].close()

            # Verificar status
            status_code = c.getinfo(c.RESPONSE_CODE)
            c.close()

            if status_code == 200:
                logger.info(f"✅ Download concluído: {output_path}")
                return True
            else:
                logger.error(f"   ❌ Status HTTP {status_code}")

        except Exception as e:
            logger.error(f"   ❌ Erro na tentativa {tentativa + 1}: {e}")
            if tentativa < max_retries - 1:
                wait_time = 2 ** tentativa
                logger.info(f"   ⏳ Aguardando {wait_time}s antes de tentar novamente...")
                time.sleep(wait_time)

    logger.error(f"❌ Falha no download PyCURL após {max_retries} tentativas")
    return False


def _download_aria2(
    url: str,
    output_path: str,
    description: str,
    timeout: int,
    max_retries: int,
) -> bool:
    """Download usando Aria2 (download paralelo, muito rápido)."""
    if not ARIA2_AVAILABLE:
        logger.error("❌ Aria2 não está disponível")
        return False

    logger.info(f"⬇️  Iniciando download (Aria2): {description}")
    logger.info(f"   URL: {url}")

    # Diretório de saída
    output_dir = str(Path(output_path).parent)
    output_file = Path(output_path).name

    for tentativa in range(max_retries):
        try:
            logger.info(f"   Tentativa {tentativa + 1}/{max_retries}...")

            # Comando aria2c
            # -x16: 16 conexões paralelas
            # -s16: 16 splits por conexão
            # -k1M: chunk size 1MB
            # --file-allocation=none: sem pré-alocação (mais rápido)
            # --console-log-level=warn: menos verboso
            cmd = [
                'aria2c',
                '--max-connection-per-server=16',
                '--split=16',
                '--min-split-size=1M',
                '--file-allocation=none',
                '--console-log-level=warn',
                '--summary-interval=1',
                f'--max-tries={max_retries}',
                f'--timeout={timeout}',
                f'--dir={output_dir}',
                f'--out={output_file}',
                '--allow-overwrite=true',
                url
            ]

            logger.info(f"   Executando aria2c com 16 conexões paralelas...")

            # Executar aria2c
            result = subprocess.run(
                cmd,
                capture_output=False,  # Mostrar output direto
                timeout=timeout * 2
            )

            if result.returncode == 0:
                if Path(output_path).exists():
                    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
                    logger.info(f"✅ Download concluído: {output_path} ({file_size_mb:.1f} MB)")
                    return True
                else:
                    logger.error(f"   ❌ Arquivo não foi criado")
            else:
                logger.error(f"   ❌ Aria2 retornou código {result.returncode}")

        except subprocess.TimeoutExpired:
            logger.error(f"   ⏱️  Timeout na tentativa {tentativa + 1}")
        except Exception as e:
            logger.error(f"   ❌ Erro na tentativa {tentativa + 1}: {e}")

        if tentativa < max_retries - 1:
            wait_time = 2 ** tentativa
            logger.info(f"   ⏳ Aguardando {wait_time}s antes de tentar novamente...")
            time.sleep(wait_time)

    logger.error(f"❌ Falha no download Aria2 após {max_retries} tentativas")
    return False


def download_multiple(
    downloads: list[dict],
    timeout: int = 120,
    max_retries: int = 5,
) -> dict[str, bool]:
    """
    Baixa múltiplos arquivos sequencialmente.

    Args:
        downloads: Lista de dicts com 'url', 'output_path' e 'description' (opcional)
        timeout: Timeout em segundos
        max_retries: Número máximo de tentativas por arquivo

    Returns:
        dict: Mapeamento {output_path: sucesso}

    Example:
        >>> results = download_multiple([
        ...     {
        ...         "url": "https://example.com/file1.nc",
        ...         "output_path": "Saida/file1.nc",
        ...         "description": "SST 2025"
        ...     },
        ...     {
        ...         "url": "https://example.com/file2.nc",
        ...         "output_path": "Saida/file2.nc",
        ...         "description": "OLR Anomaly"
        ...     }
        ... ])
    """
    results = {}

    for download in downloads:
        url = download["url"]
        output_path = download["output_path"]
        description = download.get("description")

        success = download_with_progress(
            url=url,
            output_path=output_path,
            description=description,
            timeout=timeout,
            max_retries=max_retries,
        )

        results[output_path] = success

    return results
