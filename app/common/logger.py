"""
Logger configurado com Factory Pattern, Loguru e separação por módulo/script.

Este módulo fornece funções para criar loggers com três modos principais de operação:

1. Logger Geral (default):
   - Chamado via `get_logger()`
   - Todos os logs são enviados para o arquivo "logs/campos_observados.log"
   - Ideal para uso consolidado do sistema
   - Exemplo de uso:
        from app.common.logger import get_logger
        logger = get_logger()
        logger.info("Log geral da aplicação")

2. Logger por Módulo/Script:
   - Chamado via `get_logger("nome_modulo")`
   - Cria logs separados por script, gravando em arquivos como "logs/s01.log", "logs/s02.log"
   - Ideal para depuração de scripts específicos
   - Exemplo de uso:
        from app.common.logger import get_logger
        logger = get_logger("s01")
        logger.info("Log exclusivo do script s01")

3. Logger automático com base no módulo chamador:
   - Use a função `get_auto_logger(criar_log_separado=True)`
   - Se `criar_log_separado=True` (padrão), cria log específico baseado em `__name__`
   - Se `criar_log_separado=False`, retorna o logger geral
   - Exemplo de uso:
        from app.common.logger import get_auto_logger
        logger = get_auto_logger()  # Cria logger específico para o módulo atual
        logger.debug("Log automático baseado no módulo chamador")

Uso nos Scripts Setups/sXX:
    # No início de cada script (ex: Setups/s01_psi850_anom.py)
    # NÃO criar logger no nível do módulo!

    def main(sftp=None):
        from app.common.logger import get_logger
        logger = get_logger("s01")

        logger.info("Iniciando script s01...")
        # ... código do script ...
        logger.info("Script s01 concluído!")

Arquitetura:
    - Factory Pattern: Funções ao invés de classes Singleton
    - Thread-Safe: Usa locks para prevenir race conditions
    - Lazy Initialization: Sinks só adicionados quando necessário
    - Singleton de Sinks: Cada sink adicionado apenas uma vez globalmente
    - Cache de Loggers: Loggers já criados são reutilizados
"""

# Bibliotecas padrão
import inspect
import sys
import threading
from typing import Any, Optional

# Bibliotecas de terceiros
from loguru import logger

# ============================================================================
# VARIÁVEIS GLOBAIS DO MÓDULO (Thread-Safe)
# ============================================================================

# Track de IDs dos sinks adicionados (para evitar duplicação)
_sink_ids: dict[str, list[int]] = {}

# Cache de loggers criados (para reutilização)
_loggers_cache: dict[str, Any] = {}

# Lock para thread safety
_lock = threading.Lock()

# Flag para indicar se logger geral já foi inicializado
_general_logger_initialized = False

# Nível de log padrão
_LOG_LEVEL = 'DEBUG'


# ============================================================================
# FUNÇÕES PRINCIPAIS (Factory Pattern)
# ============================================================================


def get_logger(module_name: str = 'geral') -> Any:
    """
    Factory function para criar/recuperar loggers.

    Esta função substitui a classe LoggerService usando Factory Pattern.
    É thread-safe e garante que sinks não sejam duplicados.

    Args:
        module_name: Nome do módulo/script (ex: "s01", "s02", "geral")
                    Default: "geral" para logger principal

    Returns:
        Logger configurado e pronto para uso

    Example:
        >>> # Logger geral
        >>> logger = get_logger()
        >>> logger.info("Log geral")
        >>>
        >>> # Logger específico do script S01
        >>> logger = get_logger("s01")
        >>> logger.info("Processing PSI850 data...")
        >>> # Grava em logs/s01.log E no console
    """
    global _general_logger_initialized

    with _lock:
        # Se logger já existe no cache, retorna
        if module_name in _loggers_cache:
            return _loggers_cache[module_name]

        # Inicializa logger geral na primeira chamada
        if not _general_logger_initialized:
            _initialize_general_logger()
            _general_logger_initialized = True

        # Cria logger específico do módulo
        if module_name == 'geral':
            # Logger geral já foi inicializado acima
            module_logger = logger.bind(module='geral')
        else:
            # Logger de módulo específico
            module_logger = _create_module_logger(module_name)

        # Cacheia para reutilização
        _loggers_cache[module_name] = module_logger

        return module_logger


def get_auto_logger(criar_log_separado: bool = True) -> Any:
    """
    Retorna um logger baseado no módulo chamador.

    Args:
        criar_log_separado: Se True, cria log dedicado por módulo.
                           Se False, retorna o logger geral.

    Returns:
        Logger específico ou geral baseado na configuração

    Example:
        >>> # Em qualquer módulo
        >>> from app.common.logger import get_auto_logger
        >>> logger = get_auto_logger()
        >>> logger.info("Auto logger detectou o módulo automaticamente")
    """
    if not criar_log_separado:
        return get_logger()

    # Detecta o nome do módulo chamador
    caller = inspect.stack()[1]
    module_name = caller.frame.f_globals['__name__']

    # Limpa o nome do módulo para usar como nome do arquivo de log
    # Ex: "Setups.s01_psi850_anom" -> "Setups_s01_psi850_anom"
    clean_name = module_name.replace('.', '_').replace('.py', '')

    return get_logger(clean_name)


# ============================================================================
# FUNÇÕES INTERNAS (Não usar diretamente)
# ============================================================================


def _initialize_general_logger():
    """
    Inicializa o logger geral pela primeira vez.

    Remove todos os handlers padrão do loguru e adiciona apenas os sinks
    para o logger geral (console + arquivo).

    IMPORTANTE: Esta função só deve ser chamada UMA VEZ!
    """
    # Remove todos os sinks padrão do loguru
    logger.remove()

    log_format_geral = (
        '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | '
        '<level>{level: <8}</level> | '
        '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | '
        '<level>{message}</level>'
    )

    # Sink para console do logger geral
    sink_id_console = logger.add(
        sys.stdout,
        format=log_format_geral,
        level=_LOG_LEVEL,
        backtrace=True,
        diagnose=True,
        filter=lambda record: record['extra'].get('module') == 'geral',
        enqueue=False,  # Flush imediato
    )

    # Sink para arquivo do logger geral
    sink_id_file = logger.add(
        'logs/campos-observados.log',
        format=log_format_geral,
        level=_LOG_LEVEL,
        rotation='500 MB',
        retention='30 days',
        compression='zip',
        backtrace=True,
        diagnose=True,
        filter=lambda record: record['extra'].get('module') == 'geral',
        enqueue=False,  # Flush imediato
    )

    # Registra IDs dos sinks
    _sink_ids['geral'] = [sink_id_console, sink_id_file]


def _create_module_logger(module_name: str) -> Any:
    """
    Cria um logger específico para um módulo/script.

    Adiciona sinks (console + arquivo) APENAS se ainda não foram adicionados
    para este módulo.

    Args:
        module_name: Nome do módulo (ex: "s01", "s02")

    Returns:
        Logger vinculado ao módulo
    """
    # Cria logger vinculado ao módulo
    module_logger = logger.bind(module=module_name)

    # Se sinks já foram adicionados para este módulo, apenas retorna o logger
    if module_name in _sink_ids:
        return module_logger

    # Formato de log para módulos (mais compacto)
    log_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | '
        '<level>{level: <8}</level> | '
        f'<cyan>{module_name}</cyan> | '
        '<level>{message}</level>'
    )

    # Sink para console do módulo
    sink_id_console = logger.add(
        sys.stdout,
        format=log_format,
        level=_LOG_LEVEL,
        backtrace=False,
        diagnose=False,
        filter=lambda record: record['extra'].get('module') == module_name,
        enqueue=False,  # Flush imediato
    )

    # Sink para arquivo específico do módulo
    sink_id_file = logger.add(
        f'logs/{module_name}.log',
        format=log_format,
        level=_LOG_LEVEL,
        rotation='100 MB',
        retention='15 days',
        compression='zip',
        backtrace=True,
        diagnose=True,
        filter=lambda record: record['extra'].get('module') == module_name,
        enqueue=False,  # Flush imediato
    )

    # Registra IDs dos sinks para evitar duplicação
    _sink_ids[module_name] = [sink_id_console, sink_id_file]

    return module_logger


# ============================================================================
# COMPATIBILIDADE COM CÓDIGO LEGADO (LoggerService)
# ============================================================================


class LoggerService:
    """
    Classe de compatibilidade para código legado que usa LoggerService.

    DEPRECATED: Use get_logger() diretamente ao invés de LoggerService().

    Esta classe ainda funciona mas é apenas um wrapper para get_logger().
    """

    @staticmethod
    def get_logger() -> Any:
        """Retorna o logger geral."""
        return get_logger('geral')

    @staticmethod
    def get_module_logger(module_name: Optional[str] = None) -> Any:
        """Retorna um logger específico por módulo/script."""
        if not module_name:
            return get_logger('geral')
        return get_logger(module_name)


# ============================================================================
# SCRIPT DE TESTE STANDALONE
# ============================================================================

if __name__ == '__main__':
    print('\n' + '=' * 80)
    print('🧪 TESTE DO MÓDULO LOGGER (Factory Pattern)')
    print('=' * 80 + '\n')

    # Teste 1: Logger geral
    print('📝 Teste 1: Logger Geral')
    logger_geral = get_logger()
    logger_geral.info('Este é um log GERAL')
    logger_geral.debug('Debug do logger geral')

    # Teste 2: Logger de módulo
    print('\n📝 Teste 2: Logger de Módulo (s01)')
    logger_s01 = get_logger('s01')
    logger_s01.info('Este é um log do S01')
    logger_s01.warning('Warning do S01')

    # Teste 3: Outro logger de módulo
    print('\n📝 Teste 3: Logger de Módulo (s02)')
    logger_s02 = get_logger('s02')
    logger_s02.info('Este é um log do S02')
    logger_s02.error('Error do S02')

    # Teste 4: Reutilização de logger (deve usar cache)
    print('\n📝 Teste 4: Reutilização de Logger (s01 novamente)')
    logger_s01_again = get_logger('s01')
    logger_s01_again.info('Log do S01 reutilizado (mesmo logger do cache)')

    # Teste 5: Logger automático
    print('\n📝 Teste 5: Logger Automático')
    logger_auto = get_auto_logger()
    logger_auto.info('Logger automático detectou o módulo __main__')

    # Teste 6: Compatibilidade com LoggerService
    print('\n📝 Teste 6: Compatibilidade com LoggerService (deprecated)')
    service = LoggerService()
    logger_legacy = service.get_module_logger('test')
    logger_legacy.info('Logger criado via LoggerService (modo compatibilidade)')

    print('\n' + '=' * 80)
    print('✅ Todos os testes concluídos!')
    print('📁 Verifique os arquivos de log criados em logs/')
    print('=' * 80 + '\n')

    sys.exit(0)
