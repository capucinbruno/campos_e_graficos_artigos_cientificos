"""CLI para execucao de scripts meteorologicos."""

# Bibliotecas padrão
import argparse
import importlib
import sys
from datetime import datetime
from pathlib import Path

# Módulos locais
from app.shared.error_handler import friendly_errors
from app.shared.logger import get_logger
from app.shared.settings_factory import get_settings

logger = get_logger(__name__)
settings = get_settings()


def _build_scripts_dict() -> dict:
    """Constroi dicionario SCRIPTS usando paths do settings."""
    return {
        's00': {
            'module': 'scripts.s00_plotagem_vento_eraa5',
            'description': 'Vento 100m + MSLP (ERA5)',
            'setting_flag': 'RUN_S00',
            'support_files': [
                {
                    'local': settings.FILE_CLIMATOLOGIA_VENTO100M,
                    'remote': settings.REMOTE_CLIMATOLOGIA_VENTO100M,
                    'description': 'Climatologia vento 100m',
                },
            ],
        },
        's01': {
            'module': 'scripts.s01_geop250_anom',
            'description': 'Anomalia Geopotencial 250hPa',
            'setting_flag': 'RUN_S01',
            'support_files': [
                {
                    'local': settings.FILE_CLIMATOLOGIA_GEOP250,
                    'remote': settings.REMOTE_CLIMATOLOGIA_GEOP250,
                    'description': 'Climatologia geop250 1991-2020',
                },
            ],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's02': {
            'module': 'scripts.s02_chi200_anom',
            'description': 'Anomalia CHI200 (Velocity Potential)',
            'setting_flag': 'RUN_S02',
            'support_files': [
                {
                    'local': settings.FILE_CLIMATOLOGIA_UWND250,
                    'remote': settings.REMOTE_CLIMATOLOGIA_UWND250,
                    'description': 'Climatologia uwnd250 1995-2024',
                },
                {
                    'local': settings.FILE_CLIMATOLOGIA_VWND250,
                    'remote': settings.REMOTE_CLIMATOLOGIA_VWND250,
                    'description': 'Climatologia vwnd250 1995-2024',
                },
            ],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's03': {
            'module': 'scripts.s03_psi200_anom',
            'description': 'Anomalia PSI200 (Streamfunction)',
            'setting_flag': 'RUN_S03',
            'support_files': [
                {
                    'local': settings.FILE_CLIMATOLOGIA_UWND250,
                    'remote': settings.REMOTE_CLIMATOLOGIA_UWND250,
                    'description': 'Climatologia uwnd250 1995-2024',
                },
                {
                    'local': settings.FILE_CLIMATOLOGIA_VWND250,
                    'remote': settings.REMOTE_CLIMATOLOGIA_VWND250,
                    'description': 'Climatologia vwnd250 1995-2024',
                },
            ],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's04': {
            'module': 'scripts.s04_fluxo_rossby_wave',
            'description': 'Rossby Wave Activity Flux (TN2001)',
            'setting_flag': 'RUN_S04',
            'support_files': [
                {
                    'local': settings.FILE_CLIMATOLOGIA_GEOP250,
                    'remote': settings.REMOTE_CLIMATOLOGIA_GEOP250,
                    'description': 'Climatologia geop250 1995-2024',
                },
                {
                    'local': settings.FILE_CLIMATOLOGIA_UWND250,
                    'remote': settings.REMOTE_CLIMATOLOGIA_UWND250,
                    'description': 'Climatologia uwnd250 1995-2024',
                },
                {
                    'local': settings.FILE_CLIMATOLOGIA_VWND250,
                    'remote': settings.REMOTE_CLIMATOLOGIA_VWND250,
                    'description': 'Climatologia vwnd250 1995-2024',
                },
            ],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's05': {
            'module': 'scripts.s05_olr_anom',
            'description': 'Anomalia OLR (PSL/NOAA)',
            'setting_flag': 'RUN_S05',
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's06': {
            'module': 'scripts.s06_olr_wind250_anom',
            'description': 'Anomalia OLR + Vento 250hPa (streamlines)',
            'setting_flag': 'RUN_S06',
            'support_files': [
                {
                    'local': settings.FILE_CLIMATOLOGIA_UWND250,
                    'remote': settings.REMOTE_CLIMATOLOGIA_UWND250,
                    'description': 'Climatologia uwnd250 1995-2024',
                },
                {
                    'local': settings.FILE_CLIMATOLOGIA_VWND250,
                    'remote': settings.REMOTE_CLIMATOLOGIA_VWND250,
                    'description': 'Climatologia vwnd250 1995-2024',
                },
            ],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
    }


SCRIPTS = _build_scripts_dict()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Configura e retorna argumentos CLI."""
    parser = argparse.ArgumentParser(
        prog='run_script',
        description='Executa scripts de campos observados ERA5',
    )

    parser.add_argument(
        'script',
        nargs='?',
        choices=list(SCRIPTS.keys()),
        help='Script a executar (ex: s00, s01)',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='Lista scripts disponiveis',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Executa todos os scripts habilitados',
    )
    parser.add_argument(
        '--data-inicial',
        type=str,
        help='Data inicial (YYYY-MM-DD), sobrescreve settings',
    )
    parser.add_argument(
        '--data-final',
        type=str,
        help='Data final (YYYY-MM-DD), sobrescreve settings',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Ativa logging DEBUG',
    )
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Forca re-download dos dados',
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Limpa cache antes de executar',
    )

    return parser.parse_args(argv)


def list_scripts() -> None:
    """Exibe scripts disponiveis com cores ANSI."""
    sftp_enabled = settings.get('SFTP_ENABLED', False)

    # Cores ANSI
    BOLD = '\033[1m'
    DIM = '\033[2m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    CYAN = '\033[36m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    RESET = '\033[0m'

    print(f'\n{BOLD}Scripts disponiveis:{RESET}')
    print('-' * 70)
    for key, info in SCRIPTS.items():
        flag = info['setting_flag']
        enabled = settings.get(flag, True)
        if enabled:
            status = f'{GREEN}ON{RESET}'
        else:
            status = f'{RED}OFF{RESET}'
        print(f'  {BOLD}{CYAN}{key}{RESET}    {info["description"]:40s} [{status}]')

        # Mostra arquivos de suporte (SFTP) e seu status
        for sf in info.get('support_files', []):
            local_exists = Path(sf['local']).exists()
            if local_exists:
                sf_status = f'{GREEN}OK{RESET}'
            elif sftp_enabled:
                sf_status = f'{YELLOW}SFTP: sera baixado automaticamente{RESET}'
            else:
                sf_status = f'{RED}FALTANDO - copie manualmente ou ative SFTP{RESET}'
            print(f'         {DIM}Requisito:{RESET} {sf["description"]} [{sf_status}]')

        # Mostra arquivos obrigatorios (sem SFTP) e seu status
        for rf in info.get('required_files', []):
            local_exists = Path(rf['local']).exists()
            if local_exists:
                rf_status = f'{GREEN}OK{RESET}'
            else:
                rf_status = f'{RED}FALTANDO - copie manualmente para {rf["local"]}{RESET}'
            print(f'         {DIM}Requisito:{RESET} {rf["description"]} [{rf_status}]')

    print()
    sftp_color = GREEN if sftp_enabled else YELLOW
    print(
        f'  SFTP_ENABLED = {sftp_color}{sftp_enabled}{RESET}  |  Ambiente: {MAGENTA}{settings.current_env}{RESET}'
    )
    print()
    print(f'{BOLD}Comandos:{RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s00{RESET}')
    print(f'    {DIM}Baixa dados ERA5 (vento 100m + MSLP) via API CDS, salva .nc em Entrada/{RESET}')
    print(f'    {DIM}e gera mapas de vento e graficos diarios em Saida/s00_VENTO_EOLICAS_SEMOP/{RESET}')
    print(
        f'    {YELLOW}Requisito: climatologia 1991-2020 (baixada via SFTP ou copiada manualmente){RESET}'
    )
    print()
    print(f'  {GREEN}uv run python run_script.py s01{RESET}')
    print(f'    {DIM}Baixa geopotencial 250hPa via API CDS, salva .grb em dados/{RESET}')
    print(f'    {DIM}e gera mapas de anomalia em Saida/s01_GEOP250/{RESET}')
    print(
        f'    {YELLOW}Requisito: climatologia 1991-2020 (baixada via SFTP ou copiada manualmente){RESET}'
    )
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s02{RESET}')
    print(f'    {DIM}Baixa u/v 250hPa via API CDS, salva .nc em dados/{RESET}')
    print(f'    {DIM}e gera mapas de anomalia CHI200 em Saida/s02_CHI200/{RESET}')
    print(
        f'    {YELLOW}Requisito: climatologias uwnd250/vwnd250 (baixadas via SFTP ou copiadas manualmente){RESET}'
    )
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s05{RESET}')
    print(f'    {DIM}Baixa anomalia OLR do PSL/NOAA, salva .nc em dados/{RESET}')
    print(f'    {DIM}e gera mapas de anomalia OLR em Saida/s05_OLR_ANOM/{RESET}')
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s06{RESET}')
    print(f'    {DIM}Baixa anomalia OLR (PSL/NOAA) + vento 250 hPa (ERA5/CDS){RESET}')
    print(f'    {DIM}e gera mapas de anomalia OLR com streamlines em Saida/s06_OLR_WIND250_ANOM/{RESET}')
    print(
        f'    {YELLOW}Requisito: climatologias uwnd250/vwnd250 (baixadas via SFTP ou copiadas manualmente){RESET}'
    )
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s00 --verbose{RESET}')
    print(f'    {DIM}Executa com logging DEBUG (mostra detalhes de download e processamento){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s00 --force-download{RESET}')
    print(f'    {DIM}Forca re-download dos dados mesmo que o .nc/.grb ja exista em Entrada/{RESET}')
    print()
    print(
        f'  {GREEN}uv run python run_script.py s00 --data-inicial 2026-03-01 --data-final 2026-03-12{RESET}'
    )
    print(f'    {DIM}Sobrescreve as datas do settings.local.toml para este periodo{RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py --all{RESET}')
    print(
        f'    {DIM}Executa todos os scripts habilitados (RUN_S00=true, RUN_S01=true no settings){RESET}'
    )
    print()
    print(f'  {GREEN}uv run python run_script.py --clear-cache{RESET}')
    print(
        f'    {DIM}Limpa cache de execucoes anteriores (forca reprocessamento na proxima execucao){RESET}'
    )
    print()


def _validate_date(date_str: str) -> str:
    """Valida formato de data YYYY-MM-DD."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return date_str
    except ValueError:
        logger.error(f'Data invalida: {date_str}. Use formato YYYY-MM-DD.')
        sys.exit(1)


def _apply_overrides(args: argparse.Namespace) -> None:
    """Aplica overrides do CLI nas settings."""
    if args.data_inicial:
        settings.set('DATA_INICIAL', _validate_date(args.data_inicial))
    if args.data_final:
        settings.set('DATA_FINAL', _validate_date(args.data_final))
    if args.verbose:
        settings.set('LEVEL_LOGGING', 'DEBUG')
    if args.force_download:
        settings.set('FORCE_DOWNLOAD', True)


def _check_required_files(script_key: str) -> None:
    """Verifica arquivos obrigatorios que devem existir localmente.

    Estes arquivos nao estao disponiveis via SFTP e precisam ser
    copiados manualmente para o projeto.
    """
    info = SCRIPTS[script_key]
    required_files = info.get('required_files', [])

    for rf in required_files:
        local_path = Path(rf['local'])
        if not local_path.exists():
            raise FileNotFoundError(
                f'Arquivo obrigatorio nao encontrado: {rf["description"]}\n'
                f'  Esperado em: {local_path}\n'
                f'  Copie manualmente para {local_path} antes de executar o script.'
            )
        logger.info(f'Arquivo obrigatorio encontrado: {rf["description"]} ({local_path})')


def _download_via_sftp(sf: dict, local_path: Path) -> None:
    """Baixa um arquivo de suporte via SFTP."""
    # Módulos locais
    from app.shared.sftp_client import SFTPClient

    logger.info(f'Baixando arquivo de suporte: {sf["description"]}')
    logger.info(f'  Remoto: {sf["remote"]}')
    logger.info(f'  Local:  {local_path}')

    with SFTPClient() as sftp:
        sftp.download(sf['remote'], str(local_path))

    logger.info(f'Download concluido: {sf["description"]}')


def _validate_nc_file(path: Path) -> bool:
    """Testa se um arquivo NetCDF/GRIB pode ser aberto pelo xarray."""
    suffix = path.suffix.lower()
    if suffix not in {'.nc', '.nc4', '.grb', '.grib'}:
        return True  # nao e NetCDF/GRIB, pula validacao

    try:
        # Bibliotecas de terceiros
        import xarray as xr

        engine = 'cfgrib' if suffix in {'.grb', '.grib'} else 'netcdf4'
        with xr.open_dataset(path, engine=engine):
            pass
        return True
    except Exception:
        return False


def _ensure_support_files(script_key: str) -> None:
    """Verifica, baixa e valida arquivos de suporte.

    Fluxo para cada arquivo:
    1. Se nao existe localmente → baixa via SFTP (se habilitado)
    2. Se existe → valida integridade (.nc/.grb)
    3. Se corrompido + SFTP → apaga e re-baixa 1x
    4. Se corrompido sem SFTP ou re-download falhar → erro claro
    """
    info = SCRIPTS[script_key]
    support_files = info.get('support_files', [])

    if not support_files:
        return

    sftp_enabled = settings.get('SFTP_ENABLED', False)

    for sf in support_files:
        local_path = Path(sf['local'])

        # --- Arquivo nao existe: tenta baixar ---
        if not local_path.exists():
            if sftp_enabled:
                try:
                    _download_via_sftp(sf, local_path)
                except Exception as e:
                    raise RuntimeError(
                        f'Falha ao baixar {sf["description"]} via SFTP: {e}\n'
                        f'  Remoto: {sf["remote"]}\n'
                        f'  Verifique SSH_HOST, SSH_USERNAME e SSH_KEY_PATH em .secrets.toml'
                    ) from None
            else:
                logger.warning(f'Arquivo de suporte NAO encontrado: {sf["description"]}')
                logger.warning(f'  Esperado em: {local_path}')
                logger.warning('  SFTP desabilitado. Opcoes:')
                logger.warning(f'    1. Copie manualmente de: {sf["remote"]}')
                logger.warning('    2. Ative SFTP_ENABLED=true no settings e configure .secrets.toml')
                continue

        # --- Arquivo existe: valida integridade ---
        if not _validate_nc_file(local_path):
            logger.warning(f'Arquivo corrompido: {sf["description"]} ({local_path})')

            if sftp_enabled:
                # Tentativa unica: apaga e re-baixa
                logger.info('Apagando arquivo corrompido e re-baixando via SFTP...')
                local_path.unlink()

                try:
                    _download_via_sftp(sf, local_path)
                except Exception as e:
                    raise RuntimeError(
                        f'Re-download falhou para {sf["description"]}: {e}\n'
                        f'  O arquivo no servidor tambem pode estar corrompido.\n'
                        f'  Verifique o arquivo remoto: {sf["remote"]}'
                    ) from None

                # Valida o re-download
                if not _validate_nc_file(local_path):
                    raise RuntimeError(
                        f'Arquivo de suporte corrompido mesmo apos re-download: {sf["description"]}\n'
                        f'  Local:  {local_path}\n'
                        f'  Remoto: {sf["remote"]}\n'
                        f'  O arquivo no servidor Oracle esta corrompido.\n'
                        f'  Gere uma nova climatologia ou copie uma versao valida.'
                    )
                logger.info(f'Re-download validado com sucesso: {sf["description"]}')
            else:
                raise RuntimeError(
                    f'Arquivo de suporte corrompido: {sf["description"]}\n'
                    f'  Local: {local_path}\n'
                    f'  O arquivo nao pode ser lido como NetCDF.\n'
                    f'  Delete-o e copie novamente de: {sf["remote"]}\n'
                    f'  Ou ative SFTP_ENABLED=true para re-download automatico.'
                )
        else:
            logger.info(f'Arquivo de suporte validado: {sf["description"]} ({local_path})')


def run_script(script_key: str) -> None:
    """Executa um script pelo identificador."""
    info = SCRIPTS[script_key]
    module_path = info['module']

    logger.info(f'Executando {script_key}: {info["description"]}')

    # Verifica arquivos obrigatorios (sem SFTP) e de suporte (com SFTP)
    _check_required_files(script_key)
    _ensure_support_files(script_key)

    try:
        module = importlib.import_module(module_path)
        if hasattr(module, 'main'):
            module.main()
        else:
            logger.warning(f'Modulo {module_path} nao possui funcao main()')
    except Exception as e:
        logger.error(f'Erro ao executar {script_key}: {e}')
        raise


@friendly_errors
def main(argv: list[str] | None = None) -> None:
    """Entry point do CLI."""
    args = parse_args(argv)

    if args.list:
        list_scripts()
        return

    if args.clear_cache:
        # Módulos locais
        from app.common.cache_manager import clear_all_cache

        clear_all_cache()
        logger.info('Cache limpo.')
        if not args.script and not args.all:
            return

    _apply_overrides(args)

    if args.all:
        logger.info('Executando todos os scripts habilitados...')
        for key, info in SCRIPTS.items():
            flag = info['setting_flag']
            if settings.get(flag, True):
                run_script(key)
            else:
                logger.info(f'Pulando {key} (desabilitado)')
        return

    if args.script:
        run_script(args.script)
        return

    # Nenhuma opcao: mostra help
    parse_args(['--help'])


if __name__ == '__main__':
    main()
