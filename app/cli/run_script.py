"""CLI para execucao de scripts meteorologicos."""

# Bibliotecas padrão
import argparse
import importlib
import os
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
                {
                    'local': 'Entrada/sst.day.mean.ltm.1991-2020.nc',
                    'description': 'Climatologia diaria OISST (LTM 1991-2020) - base da anomalia de TSM',
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
                {
                    'local': 'Entrada/sst.day.mean.ltm.1991-2020.nc',
                    'description': 'Climatologia diaria OISST (LTM 1991-2020) - base da anomalia de TSM',
                },
                {
                    'local': 'Entrada/pdo_eof/EOF1.csv',
                    'description': 'EOF1 para o indice PDO na area global (igual ao s24)',
                },
            ],
        },
        's13': {
            'module': 'scripts.s13_sst_todas_areas_vento_medio_1000',
            'description': 'Media de TSM (OISSTv2/NOAA) + Vento medio 1000 hPa',
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
                {
                    'local': 'Entrada/sst.day.mean.ltm.1991-2020.nc',
                    'description': 'Climatologia diaria OISST (LTM 1991-2020) - base da anomalia de TSM',
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
        's21': {
            'module': 'scripts.s21_wnd850_anom_chi200_anom',
            'description': 'Anomalia CHI200 (shaded) + Streamlines vento 850hPa (dimgray) + Vento divergente 200hPa (chi<0, ±20°)',
            'setting_flag': 'RUN_S21',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's22': {
            'module': 'scripts.s22_wnd850_anom_olr_anom_div',
            'description': 'Anomalia OLR (shaded) + Streamlines vento 850hPa (black) + Vento divergente 200hPa (chi<0, ±20°)',
            'setting_flag': 'RUN_S22',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's23': {
            'module': 'scripts.s23_sst_todas_areas_correntes_marinhas',
            'description': 'Media de TSM (OISSTv2) + Correntes Marinhas de Superficie (CMEMS)',
            'setting_flag': 'RUN_S23',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's24': {
            'module': 'scripts.s24_tsm_global_todos_indices',
            'description': 'Anomalia TSM Global + Indices (ENSO, IOD, AMO, TNA, TSA, SAD, PDO)',
            'setting_flag': 'RUN_S24',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/pdo_eof/EOF1.csv',
                    'description': 'EOF1 para calculo do indice PDO',
                },
                {
                    'local': 'Entrada/sst.day.mean.ltm.1991-2020.nc',
                    'description': 'Climatologia diaria OISST (LTM 1991-2020) - base da anomalia de TSM',
                },
            ],
        },
        's25': {
            'module': 'scripts.s25_olr_anom_div_fluxo_rossby_wave',
            'description': 'OLR shaded + Geop250 contornos + WAF + Vento Divergente chi200<0',
            'setting_flag': 'RUN_S25',
            'support_files': [],
            'required_files': [],
        },
        's26': {
            'module': 'scripts.s26_chi200_anom_div_fluxo_rossby_wave',
            'description': 'chi200 shaded + Geop250 contornos + WAF + Vento Divergente chi200<0',
            'setting_flag': 'RUN_S26',
            'support_files': [],
            'required_files': [],
        },
        's27': {
            'module': 'scripts.s27_hovmoller_enso',
            'description': 'Hovmoller TSM + U850 (ENSO belt 5S-5N, 160E-80W)',
            'setting_flag': 'RUN_S27',
            'support_files': [],
            'required_files': [],
        },
        's28': {
            'module': 'scripts.s28_hovmoller_iod',
            'description': 'Hovmoller TSM + U850 (IOD belt 5S-5N, 50E-103E)',
            'setting_flag': 'RUN_S28',
            'support_files': [],
            'required_files': [],
        },
        's29': {
            'module': 'scripts.s29_walker_cell_mundo',
            'description': 'Celula de Walker: anomalia omega (30S-30N, ERA5 vs PSL clim)',
            'setting_flag': 'RUN_S29',
            'support_files': [],
            'required_files': [],
        },
        's30': {
            'module': 'scripts.s30_sst_todas_areas',
            'description': 'Media de TSM (OISSTv2/NOAA) por area, sem vento e sem OLR',
            'setting_flag': 'RUN_S30',
            'support_files': [],
            'required_files': [
                {
                    'local': 'Entrada/legenda_atlantic.png',
                    'description': 'Legenda mapa Atlantico',
                },
            ],
        },
        's31': {
            'module': 'scripts.s31_chi200_intrasazonal',
            'description': 'CHI200 intrasazonal (MJO) - mapas de pentada + Hovmoller (metodo CPC)',
            'setting_flag': 'RUN_S31',
            'support_files': [],
            'required_files': [],
        },
        's32': {
            'module': 'scripts.s32_olr_intrasazonal',
            'description': 'OLR intrasazonal (MJO) - mapas de pentada + Hovmoller (metodo CPC)',
            'setting_flag': 'RUN_S32',
            'support_files': [],
            'required_files': [],
        },
        's33': {
            'module': 'scripts.s33_homoller_free_ssta_wnd_zonal850',
            'description': 'Hovmoller TSM + U850 (caixa livre lat/lon via settings)',
            'setting_flag': 'RUN_S33',
            'support_files': [],
            'required_files': [],
        },
        's34': {
            'module': 'scripts.s34_wave_guide_rossby_wave',
            'description': 'Guia de onda de Rossby - Ks 200hPa + Z200 anom + WAF (Hoskins/Ambrizzi)',
            'setting_flag': 'RUN_S34',
            'support_files': [],
            'required_files': [],
        },
        's35': {
            'module': 'scripts.s35_aao_index',
            'description': 'Indice AAO (Antarctic Oscillation) - observado + previsoes multi-modelo (EOF propria Z700)',
            'setting_flag': 'RUN_S35',
            'support_files': [],
            'required_files': [],
        },
        's36': {
            'module': 'scripts.s36_mjo_index',
            'description': 'Indice MJO (diagrama de fase RMM1xRMM2) - observado + previsoes (EOF propria OLR+U850+U200)',
            'setting_flag': 'RUN_S36',
            'support_files': [],
            'required_files': [],
        },
        's37': {
            'module': 'scripts.s37_cold_front_track',
            'description': 'Frequencia de frentes frias por localidade (heatmap, ERA5 superficie)',
            'setting_flag': 'RUN_S37',
            'support_files': [],
            'required_files': [],
        },
        's38': {
            'module': 'scripts.s38_globo_midia_ben_noll',
            'description': 'Globo 3D animado (MP4) estilo Ben Noll: voo da camera + evolucao temporal (reanalise/forecast)',
            'setting_flag': 'RUN_S38',
            'support_files': [],
            'required_files': [],
        },
        's39': {
            'module': 'scripts.s39_globo_midia_guillaume',
            'description': 'Globo 3D animado (MP4) estilo Guillaume: voo da camera + evolucao temporal (reanalise/forecast)',
            'setting_flag': 'RUN_S39',
            'support_files': [],
            'required_files': [],
        },
        's40': {
            'module': 'scripts.s40_globo_midia_estatico',
            'description': 'Globo 3D ESTATICO (PNGs) estilo Guillaume: diario/media movel/pentadas/media total (reanalise/forecast)',
            'setting_flag': 'RUN_S40',
            'support_files': [],
            'required_files': [],
        },
        's41': {
            'module': 'scripts.s41_globo_padrao_google_earth',
            'description': 'Globo 3D animado (MP4) padrao Google Earth: copia do s39 com projecao ampliada (reanalise/forecast)',
            'setting_flag': 'RUN_S41',
            'support_files': [],
            'required_files': [],
        },
        's42': {
            'module': 'scripts.s42_globo_the_weather_channel',
            'description': 'Globo 3D animado (MP4) estilo The Weather Channel: copia do s41, em evolucao (reanalise/forecast)',
            'setting_flag': 'RUN_S42',
            'support_files': [],
            'required_files': [],
        },
        's43': {
            'module': 'scripts.s43_globo_the_weather_channel_2D',
            'description': 'Mapa 2D plano (MP4) estilo The Weather Channel: irmao do s42, sem globo (reanalise/forecast)',
            'setting_flag': 'RUN_S43',
            'support_files': [],
            'required_files': [],
        },
        's44': {
            'module': 'scripts.s44_globo_inclinado_3D',
            'description': 'Globo 3D INCLINADO ("deitado", MP4): copia do s39 com enquadramento inclinado (reanalise/forecast)',
            'setting_flag': 'RUN_S44',
            'support_files': [],
            'required_files': [],
        },
        's45': {
            'module': 'scripts.s45_globo_midia_2D',
            'description': 'Mapa global 2D (MP4) estilo s39: mesmas variaveis/features, Pacifico ao centro, sem voo de camera (reanalise/forecast)',
            'setting_flag': 'RUN_S45',
            'support_files': [],
            'required_files': [],
        },
        's46': {
            'module': 'scripts.s46_globo_inclinado_3D_chuva_modelos',
            'description': 'Globo 3D INCLINADO de CHUVA (MP4): copia do s44 com config propria (scripts/config_local/) p/ acumulados de precipitacao.',
            'setting_flag': 'RUN_S46',
            'support_files': [],
            'required_files': [],
        },
        's47': {
            'module': 'scripts.s47_globo_inclinado_3D_rajadas',
            'description': 'Globo 3D INCLINADO de RAJADAS (MP4): copia do s46 p/ rajada de vento a 10 m (10fg), so ECMWF-HRES, + PNMM opcional.',
            'setting_flag': 'RUN_S47',
            'support_files': [],
            'required_files': [],
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
    print(f'  {GREEN}uv run python run_script.py s32{RESET}')
    print(f'    {DIM}Baixa anomalia OLR do PSL/NOAA (CPC Blended 2.5°), salva .nc em dados/{RESET}')
    print(f'    {DIM}e gera mapas de pentada + Hovmoller OLR intrasazonal em Saida/s32_OLR_INTRASAZONAL/{RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s33{RESET}')
    print(f'    {DIM}Hovmoller TSM (OISSTv2) + vento zonal 850 hPa de uma caixa livre (lat/lon){RESET}')
    print(f'    {DIM}definida no settings.local.toml, + mapa SSTA/vento. Saida/s33_HOVMOLLER_CAIXA_LIVRE/{RESET}')
    print(f'    {YELLOW}Configure HOV_FREE_LAT_MIN/MAX e HOV_FREE_LON_MIN/MAX no settings.local.toml{RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s34{RESET}')
    print(f'    {DIM}Baixa u/v + geopotencial 200 hPa (ERA5/GDAS); calcula o numero de onda{RESET}')
    print(f'    {DIM}estacionario Ks (Hoskins & Ambrizzi 1993) + anomalia Z200 + WAF (TN2001);{RESET}')
    print(f'    {DIM}+ mapas de Fonte de Onda de Rossby (RWS); gera mapas por area + 2 Hovmollers{RESET}')
    print(f'    {DIM}de v\'200 em Saida/s34_WAVEGUIDE_ROSSBY/{RESET}')
    print(f'    {YELLOW}Opcional: LST_AREAS_S34 e WGUIDE_* (faixas dos Hovmollers) no settings.local.toml{RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s37{RESET}')
    print(f'    {DIM}Baixa ERA5 superficie 6-horaria (u10/v10/t2m/msl); detecta passagens de{RESET}')
    print(f'    {DIM}frente fria por cidade (Simmonds + guarda termica) e gera o heatmap de{RESET}')
    print(f'    {DIM}frequencia por localidade em Saida/s37_FRENTES_FRIAS/{RESET}')
    print(f'    {YELLOW}Opcional: LOCALIDADES e FRENTE_DV_MIN/DTEMP_MIN/REFRACTORY_H no settings.local.toml{RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s38{RESET}')
    print(f'    {DIM}Gera video MP4 de globo 3D animado (estilo Ben Noll): a camera voa em torno{RESET}')
    print(f'    {DIM}do eixo enquanto a variavel evolui no tempo. Modo reanalise ou forecast.{RESET}')
    print(f'    {DIM}Saida em Saida/s38_GLOBO_MIDIA/{RESET}')
    print(f'    {YELLOW}Config: VARIAVEL_GLOBO_3D, GLOBO_3D_MODO e GLOBO_3D_* no settings.local.toml{RESET}')
    print()
    print(f'  {GREEN}uv run python run_script.py s39{RESET}')
    print(f'    {DIM}Gera video MP4 de globo 3D animado (estilo Guillaume): a camera voa em torno{RESET}')
    print(f'    {DIM}do eixo enquanto a variavel evolui no tempo. Modo reanalise ou forecast.{RESET}')
    print(f'    {DIM}Saida em Saida/s39_GLOBO_MIDIA/{RESET}')
    print(f'    {YELLOW}Config: VARIAVEL_GLOBO_3D, GLOBO_3D_MODO e GLOBO_3D_* no settings.local.toml{RESET}')
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
    """Aplica overrides do CLI nas settings. Alem de setar o valor, marca o SINAL `AMPERE_<KEY>` no
    ambiente p/ que a config dedicada do script (aplicar_config_header/aplicar_config_script) NAO
    sobrescreva o override da CLI -- precedencia CLI/env > config do script."""
    def _set(key: str, val) -> None:
        settings.set(key, val)
        os.environ[f'AMPERE_{key}'] = str(val)   # sinal de override (o valor real ja foi via settings.set)
    if args.data_inicial:
        _set('DATA_INICIAL', _validate_date(args.data_inicial))
    if args.data_final:
        _set('DATA_FINAL', _validate_date(args.data_final))
    if args.verbose:
        _set('LEVEL_LOGGING', 'DEBUG')
    if args.force_download:
        _set('FORCE_DOWNLOAD', True)


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
