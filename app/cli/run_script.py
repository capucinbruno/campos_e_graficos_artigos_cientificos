"""CLI para execucao de scripts meteorologicos."""

import argparse
import importlib
import sys
from datetime import datetime
from pathlib import Path

from app.shared.logger import get_logger
from app.shared.settings_factory import get_settings

logger = get_logger(__name__)
settings = get_settings()

REMOTE_BASE = "/home/ubuntu/resources/meteorologia/campos-observados"

SCRIPTS = {
    "s00": {
        "module": "Scripts.s00_plotagem_vento_eraa5",
        "description": "Vento 100m + MSLP (ERA5)",
        "setting_flag": "RUN_S00",
        "support_files": [
            {
                "local": "Entrada/arquivos_nc/climatologia_1991_2020_vento100m_ERA5.nc",
                "remote": f"{REMOTE_BASE}/Entrada/arquivos_nc/climatologia_1991_2020_vento100m_ERA5.nc",
                "description": "Climatologia vento 100m (1991-2020)",
            },
        ],
    },
    "s01": {
        "module": "Scripts.s01_geop250_anom",
        "description": "Anomalia Geopotencial 250hPa",
        "setting_flag": "RUN_S01",
        "support_files": [],
        "required_files": [
            {
                "local": "Entrada/legenda_atlantic.png",
                "description": "Legenda mapa Atlantico",
            },
        ],
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Configura e retorna argumentos CLI."""
    parser = argparse.ArgumentParser(
        prog="run_script",
        description="Executa scripts de campos observados ERA5",
    )

    parser.add_argument(
        "script",
        nargs="?",
        choices=list(SCRIPTS.keys()),
        help="Script a executar (ex: s00, s01)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lista scripts disponiveis",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Executa todos os scripts habilitados",
    )
    parser.add_argument(
        "--data-inicial",
        type=str,
        help="Data inicial (YYYY-MM-DD), sobrescreve settings",
    )
    parser.add_argument(
        "--data-final",
        type=str,
        help="Data final (YYYY-MM-DD), sobrescreve settings",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Ativa logging DEBUG",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Forca re-download dos dados",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Limpa cache antes de executar",
    )

    return parser.parse_args(argv)


def list_scripts() -> None:
    """Exibe scripts disponiveis com cores ANSI."""
    sftp_enabled = settings.get("SFTP_ENABLED", False)

    # Cores ANSI
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    RESET = "\033[0m"

    print(f"\n{BOLD}Scripts disponiveis:{RESET}")
    print("-" * 70)
    for key, info in SCRIPTS.items():
        flag = info["setting_flag"]
        enabled = settings.get(flag, True)
        if enabled:
            status = f"{GREEN}ON{RESET}"
        else:
            status = f"{RED}OFF{RESET}"
        print(f"  {BOLD}{CYAN}{key}{RESET}    {info['description']:40s} [{status}]")

        # Mostra arquivos de suporte (SFTP) e seu status
        for sf in info.get("support_files", []):
            local_exists = Path(sf["local"]).exists()
            if local_exists:
                sf_status = f"{GREEN}OK{RESET}"
            elif sftp_enabled:
                sf_status = f"{YELLOW}SFTP: sera baixado automaticamente{RESET}"
            else:
                sf_status = f"{RED}FALTANDO - copie manualmente ou ative SFTP{RESET}"
            print(f"         {DIM}Requisito:{RESET} {sf['description']} [{sf_status}]")

        # Mostra arquivos obrigatorios (sem SFTP) e seu status
        for rf in info.get("required_files", []):
            local_exists = Path(rf["local"]).exists()
            if local_exists:
                rf_status = f"{GREEN}OK{RESET}"
            else:
                rf_status = f"{RED}FALTANDO - copie manualmente para {rf['local']}{RESET}"
            print(f"         {DIM}Requisito:{RESET} {rf['description']} [{rf_status}]")

    print()
    sftp_color = GREEN if sftp_enabled else YELLOW
    print(f"  SFTP_ENABLED = {sftp_color}{sftp_enabled}{RESET}  |  Ambiente: {MAGENTA}{settings.current_env}{RESET}")
    print()
    print(f"{BOLD}Comandos:{RESET}")
    print()
    print(f"  {GREEN}uv run python run_script.py s00{RESET}")
    print(f"    {DIM}Baixa dados ERA5 (vento 100m + MSLP) via API CDS, salva .nc em Entrada/{RESET}")
    print(f"    {DIM}e gera mapas de vento e graficos diarios em Saida/s00_VENTO_EOLICAS_SEMOP/{RESET}")
    print(f"    {YELLOW}Requisito: climatologia 1991-2020 (baixada via SFTP ou copiada manualmente){RESET}")
    print()
    print(f"  {GREEN}uv run python run_script.py s01{RESET}")
    print(f"    {DIM}Baixa geopotencial 250hPa via API CDS, salva .grb em Entrada/{RESET}")
    print(f"    {DIM}e gera mapas de anomalia em Saida/s04_GEOP250/{RESET}")
    print(f"    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}")
    print()
    print(f"  {GREEN}uv run python run_script.py s00 --verbose{RESET}")
    print(f"    {DIM}Executa com logging DEBUG (mostra detalhes de download e processamento){RESET}")
    print()
    print(f"  {GREEN}uv run python run_script.py s00 --force-download{RESET}")
    print(f"    {DIM}Forca re-download dos dados mesmo que o .nc/.grb ja exista em Entrada/{RESET}")
    print()
    print(f"  {GREEN}uv run python run_script.py s00 --data-inicial 2026-03-01 --data-final 2026-03-12{RESET}")
    print(f"    {DIM}Sobrescreve as datas do settings.local.toml para este periodo{RESET}")
    print()
    print(f"  {GREEN}uv run python run_script.py --all{RESET}")
    print(f"    {DIM}Executa todos os scripts habilitados (RUN_S00=true, RUN_S01=true no settings){RESET}")
    print()
    print(f"  {GREEN}uv run python run_script.py --clear-cache{RESET}")
    print(f"    {DIM}Limpa cache de execucoes anteriores (forca reprocessamento na proxima execucao){RESET}")
    print()


def _validate_date(date_str: str) -> str:
    """Valida formato de data YYYY-MM-DD."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        logger.error(f"Data invalida: {date_str}. Use formato YYYY-MM-DD.")
        sys.exit(1)


def _apply_overrides(args: argparse.Namespace) -> None:
    """Aplica overrides do CLI nas settings."""
    if args.data_inicial:
        settings.set("DATA_INICIAL", _validate_date(args.data_inicial))
    if args.data_final:
        settings.set("DATA_FINAL", _validate_date(args.data_final))
    if args.verbose:
        settings.set("LEVEL_LOGGING", "DEBUG")
    if args.force_download:
        settings.set("FORCE_DOWNLOAD", True)


def _check_required_files(script_key: str) -> None:
    """Verifica arquivos obrigatorios que devem existir localmente.

    Estes arquivos nao estao disponiveis via SFTP e precisam ser
    copiados manualmente para o projeto.
    """
    info = SCRIPTS[script_key]
    required_files = info.get("required_files", [])

    for rf in required_files:
        local_path = Path(rf["local"])
        if not local_path.exists():
            raise FileNotFoundError(
                f"Arquivo obrigatorio nao encontrado: {rf['description']}\n"
                f"  Esperado em: {local_path}\n"
                f"  Copie manualmente para {local_path} antes de executar o script."
            )
        logger.info(f"Arquivo obrigatorio encontrado: {rf['description']} ({local_path})")


def _ensure_support_files(script_key: str) -> None:
    """Verifica e baixa arquivos de suporte via SFTP se necessario.

    Em ambiente development (SFTP_ENABLED=true), baixa automaticamente
    arquivos de suporte (climatologias) do servidor Oracle
    quando nao existem localmente.
    """
    info = SCRIPTS[script_key]
    support_files = info.get("support_files", [])

    if not support_files:
        return

    sftp_enabled = settings.get("SFTP_ENABLED", False)

    for sf in support_files:
        local_path = Path(sf["local"])

        if local_path.exists():
            logger.info(f"Arquivo de suporte encontrado: {sf['description']} ({local_path})")
            continue

        if sftp_enabled:
            logger.info(f"Baixando arquivo de suporte: {sf['description']}")
            logger.info(f"  Remoto: {sf['remote']}")
            logger.info(f"  Local:  {local_path}")

            try:
                from app.shared.sftp_client import SFTPClient

                with SFTPClient() as sftp:
                    sftp.download(sf["remote"], str(local_path))

                logger.info(f"Download concluido: {sf['description']}")
            except Exception as e:
                logger.error(f"Falha ao baixar {sf['description']}: {e}")
                logger.error("Verifique SSH_HOST, SSH_USERNAME e SSH_KEY_PATH em .secrets.toml")
                raise
        else:
            logger.warning(f"Arquivo de suporte NAO encontrado: {sf['description']}")
            logger.warning(f"  Esperado em: {local_path}")
            logger.warning(f"  SFTP desabilitado. Opcoes:")
            logger.warning(f"    1. Copie manualmente de: {sf['remote']}")
            logger.warning(f"    2. Ative SFTP_ENABLED=true no settings e configure .secrets.toml")


def run_script(script_key: str) -> None:
    """Executa um script pelo identificador."""
    info = SCRIPTS[script_key]
    module_path = info["module"]

    logger.info(f"Executando {script_key}: {info['description']}")

    # Verifica arquivos obrigatorios (sem SFTP) e de suporte (com SFTP)
    _check_required_files(script_key)
    _ensure_support_files(script_key)

    try:
        module = importlib.import_module(module_path)
        if hasattr(module, "main"):
            module.main()
        else:
            logger.warning(f"Modulo {module_path} nao possui funcao main()")
    except Exception as e:
        logger.error(f"Erro ao executar {script_key}: {e}")
        raise


def main(argv: list[str] | None = None) -> None:
    """Entry point do CLI."""
    args = parse_args(argv)

    if args.list:
        list_scripts()
        return

    if args.clear_cache:
        from app.common.cache_manager import clear_all_cache

        clear_all_cache()
        logger.info("Cache limpo.")
        if not args.script and not args.all:
            return

    _apply_overrides(args)

    if args.all:
        logger.info("Executando todos os scripts habilitados...")
        for key, info in SCRIPTS.items():
            flag = info["setting_flag"]
            if settings.get(flag, True):
                run_script(key)
            else:
                logger.info(f"Pulando {key} (desabilitado)")
        return

    if args.script:
        run_script(args.script)
        return

    # Nenhuma opcao: mostra help
    parse_args(["--help"])


if __name__ == "__main__":
    main()
