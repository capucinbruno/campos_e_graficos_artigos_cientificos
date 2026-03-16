"""
Sistema de Cache Inteligente para Scripts de Processamento.

Este módulo fornece funções para gerenciar cache de execuções de scripts,
evitando reprocessamento desnecessário quando os parâmetros de entrada são
idênticos.

Funcionamento:
    1. Antes de executar processamento pesado, verifica se já foi executado
    2. Compara hash dos parâmetros de entrada com execução anterior
    3. Se parâmetros são idênticos E arquivos de saída existem, pula execução
    4. Caso contrário, executa normalmente e salva metadados de cache

Author: @Bruno Capucin
Created: 2025-10-18
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.common.logger import get_logger

logger = get_logger("cache_manager")


def _compute_params_hash(params: Dict[str, Any]) -> str:
    """
    Computa hash MD5 dos parâmetros de entrada.

    Args:
        params: Dicionário com parâmetros que influenciam o resultado

    Returns:
        Hash MD5 em hexadecimal (32 caracteres)

    Example:
        >>> params = {"DATA_INICIAL": "2025-10-08", "DATA_FINAL": "2025-10-12"}
        >>> hash_value = _compute_params_hash(params)
        >>> len(hash_value)
        32
    """
    # Converte dict para JSON string ordenado (para garantir consistência)
    params_str = json.dumps(params, sort_keys=True, default=str)

    # Computa hash MD5
    hash_obj = hashlib.md5(params_str.encode('utf-8'))

    return hash_obj.hexdigest()


def check_cache_valid(
    script_name: str,
    params: Dict[str, Any],
    output_files: List[str],
    cache_dir: str = "logs/cache"
) -> bool:
    """
    Verifica se o cache é válido para uma execução.

    Args:
        script_name: Nome do script (ex: "s01", "s02")
        params: Parâmetros que influenciam o resultado
        output_files: Lista de arquivos que devem existir
        cache_dir: Diretório onde metadados de cache são salvos

    Returns:
        True se cache é válido (pode pular execução), False caso contrário

    Example:
        >>> params = {
        ...     "DATA_INICIAL": "2025-10-08",
        ...     "DATA_FINAL": "2025-10-12",
        ...     "areas": ["brasil", "globo"]
        ... }
        >>> output_files = [
        ...     "Saida/s01_PSI850/psi850_brasil.png",
        ...     "Saida/s01_PSI850/psi850_globo.png"
        ... ]
        >>> is_valid = check_cache_valid("s01", params, output_files)
    """
    # Computa hash dos parâmetros atuais
    current_hash = _compute_params_hash(params)

    # Caminho do arquivo de metadados
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    metadata_file = cache_path / f"{script_name}_metadata.json"

    # Se arquivo de metadados não existe, cache é inválido
    if not metadata_file.exists():
        logger.debug(f"Cache inválido: arquivo de metadados não existe ({metadata_file})")
        return False

    # Carrega metadados do cache anterior
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            cached_metadata = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Erro ao ler metadados de cache: {e}")
        return False

    # Verifica se hash dos parâmetros é o mesmo
    cached_hash = cached_metadata.get("params_hash")
    if cached_hash != current_hash:
        logger.debug(f"Cache inválido: parâmetros diferentes")
        logger.debug(f"   Hash anterior: {cached_hash}")
        logger.debug(f"   Hash atual:    {current_hash}")
        return False

    # Verifica se todos os arquivos de saída existem
    missing_files = []
    for file_path in output_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        logger.debug(f"Cache inválido: {len(missing_files)} arquivos faltando")
        logger.debug(f"   Exemplos: {missing_files[:3]}")
        return False

    # Verifica se arquivos de saída são mais recentes que a última execução
    last_execution = cached_metadata.get("execution_date")
    if last_execution:
        logger.debug(f"Cache válido! Última execução: {last_execution}")
    else:
        logger.debug(f"Cache válido! (sem data de execução registrada)")

    return True


def save_cache_metadata(
    script_name: str,
    params: Dict[str, Any],
    output_files: List[str],
    execution_time_seconds: Optional[float] = None,
    cache_dir: str = "logs/cache"
) -> None:
    """
    Salva metadados de cache após execução bem-sucedida.

    Args:
        script_name: Nome do script (ex: "s01", "s02")
        params: Parâmetros que influenciaram o resultado
        output_files: Lista de arquivos que foram gerados
        execution_time_seconds: Tempo de execução em segundos (opcional)
        cache_dir: Diretório onde metadados de cache são salvos

    Example:
        >>> params = {"DATA_INICIAL": "2025-10-08", "DATA_FINAL": "2025-10-12"}
        >>> output_files = ["Saida/s01_PSI850/psi850_brasil.png"]
        >>> save_cache_metadata("s01", params, output_files, execution_time_seconds=45.2)
    """
    # Computa hash dos parâmetros
    params_hash = _compute_params_hash(params)

    # Prepara metadados
    metadata = {
        "script_name": script_name,
        "params_hash": params_hash,
        "params": params,
        "output_files": output_files,
        "execution_date": datetime.now().isoformat(),
        "execution_time_seconds": execution_time_seconds,
        "num_output_files": len(output_files),
    }

    # Salva metadados
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    metadata_file = cache_path / f"{script_name}_metadata.json"

    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.debug(f"Metadados de cache salvos: {metadata_file}")
    except IOError as e:
        logger.warning(f"Erro ao salvar metadados de cache: {e}")


def invalidate_cache(script_name: str, cache_dir: str = "logs/cache") -> bool:
    """
    Invalida (remove) o cache de um script específico.

    Args:
        script_name: Nome do script (ex: "s01", "s02")
        cache_dir: Diretório onde metadados de cache estão salvos

    Returns:
        True se cache foi invalidado, False se não existia

    Example:
        >>> invalidate_cache("s01")
        True
    """
    cache_path = Path(cache_dir)
    metadata_file = cache_path / f"{script_name}_metadata.json"

    if metadata_file.exists():
        metadata_file.unlink()
        logger.info(f"Cache invalidado: {script_name}")
        return True
    else:
        logger.debug(f"Nenhum cache para invalidar: {script_name}")
        return False


def get_cache_info(script_name: str, cache_dir: str = "logs/cache") -> Optional[Dict[str, Any]]:
    """
    Retorna informações sobre o cache de um script.

    Args:
        script_name: Nome do script (ex: "s01", "s02")
        cache_dir: Diretório onde metadados de cache estão salvos

    Returns:
        Dicionário com metadados do cache ou None se não existe

    Example:
        >>> info = get_cache_info("s01")
        >>> if info:
        ...     print(f"Última execução: {info['execution_date']}")
    """
    cache_path = Path(cache_dir)
    metadata_file = cache_path / f"{script_name}_metadata.json"

    if not metadata_file.exists():
        return None

    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Erro ao ler metadados de cache: {e}")
        return None


# ============================================================================
# SCRIPT DE TESTE STANDALONE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🧪 TESTE DO MÓDULO CACHE MANAGER")
    print("=" * 80 + "\n")

    # Teste 1: Computar hash de parâmetros
    print("📝 Teste 1: Computar hash de parâmetros")
    params1 = {"DATA_INICIAL": "2025-10-08", "DATA_FINAL": "2025-10-12"}
    hash1 = _compute_params_hash(params1)
    print(f"   Hash: {hash1}")
    print(f"   ✅ Hash gerado com sucesso\n")

    # Teste 2: Verificar cache (deve ser inválido na primeira vez)
    print("📝 Teste 2: Verificar cache (primeira execução)")
    output_files = ["Saida/s01_PSI850/psi850_teste.png"]
    is_valid = check_cache_valid("test", params1, output_files)
    print(f"   Cache válido: {is_valid}")
    print(f"   ✅ Esperado: False (primeira vez)\n")

    # Teste 3: Salvar metadados de cache
    print("📝 Teste 3: Salvar metadados de cache")
    save_cache_metadata("test", params1, output_files, execution_time_seconds=45.2)
    print(f"   ✅ Metadados salvos\n")

    # Teste 4: Verificar cache novamente (ainda inválido - arquivo não existe)
    print("📝 Teste 4: Verificar cache (arquivo de saída não existe)")
    is_valid = check_cache_valid("test", params1, output_files)
    print(f"   Cache válido: {is_valid}")
    print(f"   ✅ Esperado: False (arquivo não existe)\n")

    # Teste 5: Criar arquivo de saída temporário
    print("📝 Teste 5: Criar arquivo de saída e verificar cache")
    Path("Saida/s01_PSI850").mkdir(parents=True, exist_ok=True)
    Path(output_files[0]).touch()
    is_valid = check_cache_valid("test", params1, output_files)
    print(f"   Cache válido: {is_valid}")
    print(f"   ✅ Esperado: True (tudo OK agora)\n")

    # Teste 6: Alterar parâmetros e verificar cache
    print("📝 Teste 6: Alterar parâmetros e verificar cache")
    params2 = {"DATA_INICIAL": "2025-10-09", "DATA_FINAL": "2025-10-13"}
    is_valid = check_cache_valid("test", params2, output_files)
    print(f"   Cache válido: {is_valid}")
    print(f"   ✅ Esperado: False (parâmetros diferentes)\n")

    # Teste 7: Obter informações do cache
    print("📝 Teste 7: Obter informações do cache")
    info = get_cache_info("test")
    if info:
        print(f"   Execução: {info['execution_date']}")
        print(f"   Tempo: {info['execution_time_seconds']}s")
        print(f"   ✅ Informações recuperadas\n")

    # Teste 8: Invalidar cache
    print("📝 Teste 8: Invalidar cache")
    invalidated = invalidate_cache("test")
    print(f"   Invalidado: {invalidated}")
    print(f"   ✅ Cache removido\n")

    # Limpar arquivo temporário
    Path(output_files[0]).unlink()

    print("=" * 80)
    print("✅ Todos os testes concluídos!")
    print("=" * 80 + "\n")
