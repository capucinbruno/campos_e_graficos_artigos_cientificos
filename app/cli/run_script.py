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
            'module': 'scripts.s00_geop850_anom',
            'description': 'Anomalia Geopotencial 850hPa (ERA5/GDAS + PSL)',
            'setting_flag': 'RUN_S00',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's01': {
            'module': 'scripts.s01_geop500_anom',
            'description': 'Anomalia Geopotencial 500hPa (ERA5/GDAS + PSL)',
            'setting_flag': 'RUN_S01',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's02': {
            'module': 'scripts.s02_geop250_anom',
            'description': 'Anomalia Geopotencial 250hPa (ERA5/GDAS + PSL)',
            'setting_flag': 'RUN_S02',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's03': {
            'module': 'scripts.s03_chi200_anom',
            'description': 'Anomalia CHI200 (Velocity Potential)',
            'setting_flag': 'RUN_S03',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's04': {
            'module': 'scripts.s04_psi200_anom',
            'description': 'Anomalia PSI200 (Streamfunction)',
            'setting_flag': 'RUN_S04',
            'support_files': [],
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
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's06': {
            'module': 'scripts.s06_olr_wind_250_850_anom',
            'description': 'Anomalia OLR + Vento 250 e 850 hPa (streamlines)',
            'setting_flag': 'RUN_S06',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's07': {
            'module': 'scripts.s07_fluxo_rossby_wave_geop250',
            'description': 'Rossby Wave Activity Flux + Anomalia Geopotencial 250hPa',
            'setting_flag': 'RUN_S07',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's08': {
            'module': 'scripts.s08_fluxo_rossby_wave_olr',
            'description': 'Rossby Wave Activity Flux + Anomalia OLR',
            'setting_flag': 'RUN_S08',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's09': {
            'module': 'scripts.s09_fluxo_rossby_wave_ssta',
            'description': 'Rossby Wave Activity Flux + Anomalia TSM',
            'setting_flag': 'RUN_S09',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's10': {
            'module': 'scripts.s10_fluxo_rossby_wave_tmp850',
            'description': 'Rossby Wave Activity Flux + Anomalia Temperatura 850hPa',
            'setting_flag': 'RUN_S10',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's11': {
            'module': 'scripts.s11_ssta_todas_areas',
            'description': 'Anomalia TSM - todas as areas (OISSTv2/NOAA)',
            'setting_flag': 'RUN_S11',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
                {
                    'local': 'Entrada/blue_marble.png',
                    'description': 'Fundo Blue Marble para mapas de TSM',
                },
            ],
        },
        's12': {
            'module': 'scripts.s12_ssta_todas_areas_vento850_anom',
            'description': 'Anomalia TSM + Vento 850 hPa - todas as areas (OISSTv2/NOAA + ERA5/GDAS/PSL)',
            'setting_flag': 'RUN_S12',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
                {
                    'local': 'Entrada/blue_marble.png',
                    'description': 'Fundo Blue Marble para mapas de TSM',
                },
            ],
        },
        's13': {
            'module': 'scripts.s13_sst_todas_areas',
            'description': 'Media de TSM (OISSTv2/NOAA)',
            'setting_flag': 'RUN_S13',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's14': {
            'module': 'scripts.s14_ssta_todas_areas_vento850_anom_olr_anom',
            'description': 'Anomalia TSM + Vento 850 hPa (streamlines) + Anomalia OLR',
            'setting_flag': 'RUN_S14',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's15': {
            'module': 'scripts.s15_chi200_psi200_anom',
            'description': 'CHI200 (shaded + vento div.) + PSI200 (contour)',
            'setting_flag': 'RUN_S15',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's16': {
            'module': 'scripts.s16_wnd250_zonal_anom_div',
            'description': 'Anomalia Vento Zonal 250hPa (shaded) + Vento Divergente 200hPa (quiver)',
            'setting_flag': 'RUN_S16',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's17': {
            'module': 'scripts.s17_wnd250_meridional',
            'description': 'Anomalia Vento Meridional 250hPa (shaded)',
            'setting_flag': 'RUN_S17',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's18': {
            'module': 'scripts.s18_tmp850_wnd850_anom',
            'description': 'Anomalia Temperatura 850hPa (shaded) + Vento 850hPa (quiver)',
            'setting_flag': 'RUN_S18',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's19': {
            'module': 'scripts.s19_tmp850_wnd850_geop500_anom',
            'description': 'Anomalia Temperatura 850hPa (shaded) + Vento 850hPa (quiver) + Geopotencial 500hPa (contorno)',
            'setting_flag': 'RUN_S19',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's20': {
            'module': 'scripts.s20_olr_wnd850_geop500_anom',
            'description': 'Anomalia OLR (shaded) + Vento 850hPa (quiver) + Geopotencial 500hPa (contorno)',
            'setting_flag': 'RUN_S20',
            'support_files': [],
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
        '--force-rerun',
        action='store_true',
        help='Invalida o cache do(s) script(s) antes de executar (forca reprocessamento)',
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
    print(f'    {DIM}Baixa geopotencial 850hPa via ERA5/GDAS + climatologia PSL, salva .nc em dados/{RESET}')
    print(f'    {DIM}e gera mapas de anomalia em Saida/s00_GEOP850/{RESET}')
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s01{RESET}')
    print(f'    {DIM}Baixa geopotencial 500hPa via ERA5/GDAS + climatologia PSL, salva .nc em dados/{RESET}')
    print(f'    {DIM}e gera mapas de anomalia em Saida/s01_GEOP500/{RESET}')
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s02{RESET}')
    print(f'    {DIM}Baixa geopotencial 250hPa via ERA5/GDAS + climatologia PSL, salva .nc em dados/{RESET}')
    print(f'    {DIM}e gera mapas de anomalia em Saida/s02_GEOP250/{RESET}')
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s03{RESET}')
    print(f'    {DIM}Baixa u/v 200hPa via ERA5/GDAS + climatologia PSL, salva .nc em dados/{RESET}')
    print(f'    {DIM}e gera mapas de anomalia CHI200 em Saida/s03_CHI200/{RESET}')
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s04{RESET}')
    print(f'    {DIM}Baixa u/v 200hPa via ERA5/GDAS + climatologia PSL, salva .nc em dados/{RESET}')
    print(f'    {DIM}e gera mapas de anomalia PSI200 em Saida/s04_PSI200/{RESET}')
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s05{RESET}')
    print(f'    {DIM}Baixa anomalia OLR do PSL/NOAA, salva .nc em dados/{RESET}')
    print(f'    {DIM}e gera mapas de anomalia OLR em Saida/s05_OLR_ANOM/{RESET}')
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s06{RESET}')
    print(f'    {DIM}Baixa anomalia OLR do PSL/NOAA + vento ERA5/GDAS, salva .nc em dados/{RESET}')
    print(f'    {DIM}e gera mapas OLR + streamlines em Saida/s06_OLR_WIND/{RESET}')
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s07{RESET}')
    print(f'    {DIM}Calcula Rossby Wave Activity Flux (TN2001) sobre anomalia de geopotencial 250hPa{RESET}')
    print(f'    {DIM}e gera mapas WAF em Saida/s07_ROSSBY_WAF_GEOP250/{RESET}')
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s08{RESET}')
    print(f'    {DIM}Calcula Rossby Wave Activity Flux (TN2001) sobre anomalia de OLR{RESET}')
    print(f'    {DIM}e gera mapas WAF em Saida/s08_ROSSBY_WAF_OLR/{RESET}')
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s09{RESET}')
    print(f'    {DIM}Calcula Rossby Wave Activity Flux (TN2001) sobre anomalia de TSM{RESET}')
    print(f'    {DIM}e gera mapas WAF em Saida/s09_ROSSBY_WAF_SSTA/{RESET}')
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s10{RESET}')
    print(f'    {DIM}Calcula Rossby Wave Activity Flux (TN2001) sobre anomalia de temperatura 850hPa{RESET}')
    print(f'    {DIM}e gera mapas WAF em Saida/s10_ROSSBY_WAF_TMP850/{RESET}')
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s11{RESET}')
    print(f'    {DIM}Baixa anomalia de TSM do PSL/NOAA (OISSTv2 0.25 grau, por ano){RESET}')
    print(f'    {DIM}e gera mapas de anomalia SSTA em Saida/s11_SSTA/{RESET}')
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print(f'    {YELLOW}Requisito: blue_marble.png (deve estar em Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s12{RESET}')
    print(f'    {DIM}Baixa anomalia de TSM (OISSTv2) + vento anomalo 850 hPa (ERA5/GDAS + PSL){RESET}')
    print(f'    {DIM}e gera mapas SSTA + vetores em Saida/s12_SSTA_WND850/{RESET}')
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print(f'    {YELLOW}Requisito: blue_marble.png (deve estar em Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s13{RESET}')
    print(f'    {DIM}Baixa SST media diaria (OISSTv2) e gera mapas de media de TSM{RESET}')
    print(f'    {DIM}em Saida/s13_SST_MEDIA/{RESET}')
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s14{RESET}')
    print(f'    {DIM}Anomalia de TSM (OISSTv2) + streamlines vento anomalo 850 hPa + isolinhas OLR negativo{RESET}')
    print(f'    {DIM}em Saida/s14_SSTA_VENTO850/{RESET}')
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s15{RESET}')
    print(f'    {DIM}CHI200 shaded + vento divergente + PSI200 em isolinhas azul escuro{RESET}')
    print(f'    {DIM}em Saida/s15_CHI200_PSI200/{RESET}')
    print(f'    {YELLOW}Requisito: legenda mapa Atlantico (copie manualmente para Entrada/){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s00 --verbose{RESET}')
    print(f'    {DIM}Executa com logging DEBUG (mostra detalhes de download e processamento){RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s00 --force-download{RESET}')
    print(f'    {DIM}Forca re-download dos dados mesmo que o .nc/.grb ja exista em Entrada/{RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s00 --force-rerun{RESET}')
    print(
        f'    {DIM}Invalida o cache do s00 e forca o reprocessamento (sem mexer no cache dos outros){RESET}'
    )
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


def run_script(script_key: str, force_rerun: bool = False) -> None:
    """Executa um script pelo identificador.

    Args:
        script_key: Identificador do script (ex: 's00').
        force_rerun: Se True, invalida o cache do script antes de executar,
            forcando o reprocessamento sem afetar o cache dos demais.
    """
    info = SCRIPTS[script_key]
    module_path = info['module']

    logger.info(f'Executando {script_key}: {info["description"]}')

    # --force-rerun: apaga o metadado de cache deste script para forcar reprocessamento
    if force_rerun:
        # Módulos locais
        from app.common.cache_manager import invalidate_cache

        invalidate_cache(script_key)

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
                run_script(key, force_rerun=args.force_rerun)
            else:
                logger.info(f'Pulando {key} (desabilitado)')
        return

    if args.script:
        run_script(args.script, force_rerun=args.force_rerun)
        return

    # Nenhuma opcao: mostra help
    parse_args(['--help'])


if __name__ == '__main__':
    main()
