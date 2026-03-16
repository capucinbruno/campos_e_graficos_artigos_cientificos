"""
Description:

Leitura dos parâmetros do prompt.

Author:           @Palin
Created:          2021-03-29
Copyright:        (c) Ampere Consultoria Ltda
"""

# Bibliotecas padrão
import sys

try:
    # Bibliotecas padrão
    import argparse
    from datetime import datetime
except ImportError as error:
    print(error)
    print(f"error.name: {error.name}")
    print(f"error.path: {error.path}")
    sys.exit(0)


def get_data_validada() -> datetime:
    """Este método pega os dados passados pelo o usuário no prompt e valida os parâmetros.
    Verifica se a data passada pelo usuário é válida.
    Returns:
        data_da_rodada [datetime]: retorna a data (datetime) (AAAA-MM-DD)
    """
    lst_args = leitura_parametros_prompt()

    if lst_args.DATA_DA_RODADA:
        try:
            ano, mes, dia = str(lst_args.DATA_DA_RODADA).split("-")
            data_da_rodada = datetime(int(ano), int(mes), int(dia))
        except ValueError:
            print("+" * 80)
            print("DATA INVALIDA! Formato requerido: AAAA-MM-DD!")
            print("+" * 80)
            exit()
    else:
        # Obs: redundância de segurança - retorno sempre em datetime
        data_da_rodada = datetime.now()

    if lst_args.HORA_DA_RODADA:
        try:
            hora_da_rodada = str(lst_args.HORA_DA_RODADA)
        except ValueError:
            print("+" * 80)
            print("DATA INVALIDA! ERRO 001! Formato requerido: AAAA-MM-DD!")
            print("+" * 80)
            exit()
    else:
        hora_da_rodada = "00"

    return data_da_rodada, hora_da_rodada


def leitura_parametros_prompt():
    """Entrada de parâmetros pelo prompt do script.
        Caso o usuário digitar python3 main.py -h (MOSTRARÁ O HELP)
        Como capturar as entradas do usuário na programação:
            args_user = parametros_prompt()

    Returns:
        args_user: List: argumentos que o usuário passou.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Leitura dos parametros por prompt",
        epilog="python3 main.py -h. Uso: python3 main.py -data 2021-05-24",
    )
    parser.add_argument("-v", "--version", action="version", version="%(prog)s vs. 1.0")
    parser.add_argument(
        "-data",
        action="store",
        dest="DATA_DA_RODADA",
        type=str,
        default=f"{datetime.now().year}-{datetime.now().month}-{datetime.now().day}",
        help="Ex: python main.py -data 2021-10-21  - A data default é a data de hoje.",
    )
    parser.add_argument(
        "-hora",
        action="store",
        dest="HORA_DA_RODADA",
        type=str,
        default="00",
        help="Ex: python main.py -hora '00'  - A hora default é 00",
    )
    return parser.parse_args()
