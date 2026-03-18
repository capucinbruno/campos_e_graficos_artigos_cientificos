"""LoggerService — Singleton com Loguru."""

# Bibliotecas padrão
import logging
import sys
from pathlib import Path

# Bibliotecas de terceiros
from loguru import logger as _loguru_logger

# Módulos locais
from app.shared.singleton import SingletonMeta

# Silencia logs verbosos de bibliotecas externas
logging.getLogger('paramiko').setLevel(logging.ERROR)
logging.getLogger('cdsapi').setLevel(logging.WARNING)


class LoggerService(metaclass=SingletonMeta):
    """Configura Loguru uma unica vez e fornece loggers por modulo."""

    def __init__(self) -> None:
        # Módulos locais
        from app.shared.settings_factory import get_settings

        self._settings = get_settings()
        self._configure()

    def _configure(self) -> None:
        _loguru_logger.remove()

        level = self._settings.get('LEVEL_LOGGING', 'INFO')
        backtrace = self._settings.get('LOGGER_BACKTRACE', False)
        diagnose = self._settings.get('LOGGER_DIAGNOSE', False)

        # Console
        _loguru_logger.add(
            sys.stderr,
            level=level,
            format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
            '<level>{level:<8}</level> | '
            '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - '
            '<level>{message}</level>',
            backtrace=backtrace,
            diagnose=diagnose,
        )

        # Arquivo
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        _loguru_logger.add(
            str(log_dir / 'campos_observados.log'),
            level=level,
            rotation='10 MB',
            retention='30 days',
            compression='gz',
            backtrace=backtrace,
            diagnose=diagnose,
        )

    def get_logger(self, name: str | None = None):
        """Retorna logger com binding de modulo."""
        if name:
            return _loguru_logger.bind(name=name)
        return _loguru_logger


def get_logger(name: str | None = None):
    """Atalho para obter um logger."""
    return LoggerService().get_logger(name)


logger = get_logger()
