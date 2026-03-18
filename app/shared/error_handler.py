"""Handler global de excecoes para CLI.

Estrategia: um unico ponto de captura no entry point traduz excecoes
tecnicas em mensagens amigaveis para o usuario. Nao precisa de
try/except espalhado pelo codigo — basta usar este decorator ou
chamar handle_cli_error() no main().

Como funciona:
    Toda excecao nao tratada sobe ate o topo da pilha (Python garante isso).
    O decorator @friendly_errors intercepta ali e decide:
    1. Se e uma excecao conhecida → mostra mensagem amigavel com solucao
    2. Se e desconhecida → mostra resumo + sugere --verbose para traceback
"""

import sys
import functools
from pathlib import Path

from app.shared.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Mapa de erros conhecidos: (tipo_excecao, substring_na_mensagem) → dica
# Adicione novas entradas aqui conforme novos erros surgirem.
# ──────────────────────────────────────────────────────────────────────
_ERROR_HINTS: list[tuple[type, str | None, str]] = [
    # SFTP / SSH
    (
        FileNotFoundError,
        "No such file",
        "Arquivo nao encontrado no servidor SFTP.\n"
        "  Verifique se o caminho remoto esta correto no SCRIPTS dict (app/cli/run_script.py)\n"
        "  ou copie o arquivo manualmente para o caminho local esperado.",
    ),
    (
        FileNotFoundError,
        "Arquivo obrigatorio",
        "Arquivo obrigatorio nao encontrado localmente.\n"
        "  Leia a mensagem acima para saber onde copiar o arquivo.",
    ),
    (
        ConnectionError,
        "Falha na conexao SFTP",
        "Falha ao conectar no servidor SFTP.\n"
        "  Leia a mensagem acima — ela mostra IP, porta e chave utilizados.\n"
        "  Verifique se os dados em .secrets.toml estao corretos.",
    ),
    (
        ConnectionRefusedError,
        None,
        "Conexao SSH recusada.\n"
        "  Verifique SSH_HOST e SSH_PORT em app/settings/.secrets.toml\n"
        "  e se o servidor esta acessivel.",
    ),
    (
        TimeoutError,
        None,
        "Timeout na conexao SSH.\n"
        "  Verifique se o servidor esta online e se voce tem acesso de rede.",
    ),
    # Paramiko
    (
        Exception,
        "Authentication failed",
        "Autenticacao SSH falhou.\n"
        "  Verifique SSH_USERNAME e SSH_KEY_PATH em app/settings/.secrets.toml\n"
        "  e se a chave .pem esta correta.",
    ),
    (
        Exception,
        "No such file or directory",
        "Chave SSH nao encontrada.\n"
        "  Verifique SSH_KEY_PATH em app/settings/.secrets.toml\n"
        "  Caminho atual pode estar incorreto.",
    ),
    # CDS / API
    (
        Exception,
        "KEY_CDS",
        "Chave do Copernicus CDS nao configurada.\n"
        "  Preencha KEY_CDS em app/settings/.secrets.toml\n"
        "  Obtenha sua chave em: https://cds.climate.copernicus.eu/",
    ),
    (
        Exception,
        "403",
        "Acesso negado pela API do Copernicus CDS (HTTP 403).\n"
        "  Sua KEY_CDS pode estar invalida ou expirada.\n"
        "  Verifique em: https://cds.climate.copernicus.eu/",
    ),
    # NetCDF / dados
    (
        RuntimeError,
        "corrompido mesmo apos re-download",
        "O arquivo de suporte esta corrompido tanto localmente quanto no servidor.\n"
        "  Gere uma nova versao do arquivo ou entre em contato com o responsavel pelo servidor.",
    ),
    (
        RuntimeError,
        "corrompido",
        "Arquivo de suporte corrompido.\n"
        "  Leia a mensagem acima para instrucoes de como resolver.",
    ),
    (
        RuntimeError,
        "Re-download falhou",
        "O re-download via SFTP falhou.\n"
        "  O arquivo remoto pode nao existir ou o servidor esta inacessivel.",
    ),
    (
        OSError,
        "Unknown file format",
        "Arquivo NetCDF corrompido ou formato incompativel.\n"
        "  Delete o arquivo e re-baixe: uv run python run_script.py sNN --force-download\n"
        "  Ou copie novamente do servidor se for arquivo de suporte.",
    ),
    # Imports
    (
        ImportError,
        "jinja2",
        "Dependencia jinja2 nao instalada.\n"
        "  Execute: uv sync",
    ),
    (
        ModuleNotFoundError,
        None,
        "Modulo Python nao encontrado.\n"
        "  Execute: uv sync\n"
        "  Ou verifique se o virtualenv esta ativado: source .venv/bin/activate",
    ),
]

# Cores ANSI
_RED = "\033[31m"
_YELLOW = "\033[33m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _find_hint(exc: Exception) -> str | None:
    """Busca dica amigavel para a excecao no mapa de erros."""
    msg = str(exc)
    for exc_type, substring, hint in _ERROR_HINTS:
        if isinstance(exc, exc_type):
            if substring is None or substring in msg:
                return hint
    return None


def _format_friendly_error(exc: Exception, hint: str | None, show_traceback: bool) -> None:
    """Formata e exibe erro amigavel no terminal."""
    print(f"\n{_RED}{_BOLD}ERRO:{_RESET} {exc}\n", file=sys.stderr)

    if hint:
        print(f"{_YELLOW}Solucao:{_RESET}", file=sys.stderr)
        for line in hint.split("\n"):
            print(f"  {line}", file=sys.stderr)
        print(file=sys.stderr)

    if show_traceback:
        # --verbose ativo: mostra traceback completo
        import traceback
        print(f"{_DIM}Traceback completo:{_RESET}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    else:
        print(
            f"{_DIM}Dica: use --verbose para ver o traceback completo{_RESET}",
            file=sys.stderr,
        )


def friendly_errors(func):
    """Decorator que captura excecoes e exibe mensagens amigaveis.

    Uso no entry point:

        @friendly_errors
        def main():
            ...

    Isso e o unico ponto de captura necessario. Todos os erros que
    subirem de qualquer lugar do codigo serao interceptados aqui.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SystemExit:
            raise  # argparse usa sys.exit() — nao interceptar
        except KeyboardInterrupt:
            print(f"\n{_YELLOW}Execucao interrompida pelo usuario.{_RESET}", file=sys.stderr)
            sys.exit(130)
        except Exception as exc:
            hint = _find_hint(exc)
            # Verifica se --verbose foi passado
            show_traceback = "--verbose" in sys.argv
            _format_friendly_error(exc, hint, show_traceback)
            sys.exit(1)

    return wrapper
