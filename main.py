# -*- coding: utf-8 -*-
"""
Description: Script principal do projeto campos_observados_era5.
Responsável por orquestrar a execução dos scripts de processamento e plotagem.
No momento, executa apenas o script s00 de vento ERA5.

Author:           Bruno Capucin
Created:          2026-03-13
Project:          campos_observados_era5
"""

try:
    # Bibliotecas padrão
    import os
    import signal
    import sys

    # Módulos locais
    from app.common.cache_manager import invalidate_cache
    from app.common.connections import connect_server_with_key
    from app.common.logger import LoggerService
    from app.config import logger, settings

    # Scripts do projeto
    from scripts.s00_plotagem_vento_eraa5 import main as s00_plotagem_vento_eraa5

except ImportError as error:
    print(error)
    print(f"error.name: {getattr(error, 'name', None)}")
    print(f"error.path: {getattr(error, 'path', None)}")
    sys.exit(1)


def run_script_with_logger(script_num: str, script_func, sftp=None):
    """
    Executa um script com logger dedicado.

    Args:
        script_num: Identificador do script (ex.: "s00")
        script_func: Função principal do script
        sftp: Cliente SFTP opcional
    """
    script_logger = LoggerService().get_module_logger(script_num)
    script_logger.info("🎬 Iniciando execução do script %s...", script_num.upper())

    try:
        try:
            script_func(sftp=sftp)
        except TypeError:
            script_func()

        script_logger.info("✅ Script %s concluído com sucesso!", script_num.upper())

    except Exception as e:
        script_logger.error("❌ Erro no script %s: %s", script_num.upper(), e)
        script_logger.exception("Detalhes do erro:")
        raise


def signal_handler(sig, frame):
    """
    Handler para capturar Ctrl+C (SIGINT).
    """
    import multiprocessing

    if multiprocessing.current_process().name == "MainProcess":
        logger.warning("")
        logger.warning("=" * 80)
        logger.warning("⚠️ INTERRUPÇÃO PELO USUÁRIO (Ctrl+C)")
        logger.warning("=" * 80)
        logger.warning("🛑 Execução interrompida pelo usuário")
        logger.warning("💡 O script atual foi cancelado")
        logger.warning("=" * 80)

    sys.exit(0)


def main():
    """
    Função principal do projeto.

    Argumentos suportados:
        --force-rerun       Invalida o cache do(s) script(s) executado(s)
        --clear-cache       Remove todo o diretório de cache
        --script s00        Executa apenas o script informado
        --force-download    Força re-download de arquivos, quando aplicável
    """
    signal.signal(signal.SIGINT, signal_handler)

    force_rerun_all = "--force-rerun" in sys.argv
    clear_all_cache = "--clear-cache" in sys.argv
    force_download = "--force-download" in sys.argv

    if force_download:
        os.environ["FORCE_DOWNLOAD"] = "true"
        logger.info("🔄 Modo --force-download ativado")

    run_specific_script = None
    if "--script" in sys.argv:
        try:
            script_index = sys.argv.index("--script")
            run_specific_script = sys.argv[script_index + 1].lower()

            if not run_specific_script.startswith("s"):
                run_specific_script = f"s{run_specific_script}"

            logger.info(
                "🎯 Modo script específico: executando apenas %s",
                run_specific_script.upper(),
            )
        except (IndexError, ValueError):
            logger.error("❌ Erro: --script requer um argumento (ex.: --script s00)")
            sys.exit(1)

    elif len(sys.argv) > 1 and sys.argv[-1] not in [
        "--force-download",
        "--clear-cache",
        "--force-rerun",
    ]:
        last_arg = sys.argv[-1].lower()
        if last_arg.isdigit() or (last_arg.startswith("s") and last_arg[1:].isdigit()):
            run_specific_script = last_arg if last_arg.startswith("s") else f"s{last_arg}"
            logger.info(
                "🎯 Modo script específico: executando apenas %s",
                run_specific_script.upper(),
            )

    if clear_all_cache:
        logger.info("🗑️ Removendo todo o cache...")
        import shutil

        cache_dir = "logs/cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            logger.info("✅ Cache removido: %s", cache_dir)
        else:
            logger.info("⚠️ Diretório de cache não existe")

    if force_rerun_all:
        logger.info("🔄 Modo --force-rerun ativado")
        invalidate_cache("s00")
        logger.info("✅ Cache do script S00 invalidado")

    for arg in sys.argv:
        if arg.startswith("--force-rerun-s"):
            script_num = arg.replace("--force-rerun-", "")
            logger.info("🔄 Invalidando cache do script: %s", script_num)
            invalidate_cache(script_num)

    logger.info("=" * 100)
    logger.info("🚀 INICIANDO PROCESSAMENTO CAMPOS_OBSERVADOS_ERA5")
    logger.info("=" * 100)
    logger.info("📅 Período: %s a %s", settings.DATA_INICIAL, settings.DATA_FINAL)
    logger.info("🌍 Ambiente: %s", settings.current_env.upper())

    sftp = None
    ssh = None

    if settings.current_env == "development":
        logger.info("🔌 Conectando ao servidor via SSH/SFTP...")
        error, ssh = connect_server_with_key()

        if error:
            logger.error("❌ Erro na conexão SSH: %s", error)
            logger.error("Verifique SSH_KEY_PATH, SSH_HOST e SSH_USERNAME em settings")
            return

        sftp = ssh.open_sftp()
        logger.info("✅ Conexão SFTP estabelecida com sucesso")
    else:
        logger.info("📂 Modo production: acessando arquivos localmente")

    logger.info("=" * 100)

    scripts_map = {
        "s00": s00_plotagem_vento_eraa5,
    }

    if run_specific_script:
        if run_specific_script in scripts_map:
            logger.info("🎯 Executando script específico: %s", run_specific_script.upper())
            try:
                run_script_with_logger(
                    run_specific_script,
                    scripts_map[run_specific_script],
                    sftp,
                )
            except Exception as e:
                logger.error("❌ Erro ao executar %s: %s", run_specific_script.upper(), e)
                raise
        else:
            logger.error("❌ Script '%s' não encontrado!", run_specific_script)
            logger.error("Scripts disponíveis: s00")
            sys.exit(1)

        if sftp:
            sftp.close()
            logger.info("🔌 Conexão SFTP fechada")
        if ssh:
            ssh.close()
            logger.info("🔌 Conexão SSH fechada")

        logger.info("=" * 100)
        logger.info("✅ PROCESSAMENTO CONCLUÍDO!")
        logger.info("=" * 100)
        return

    try:
        if getattr(settings, "RUN_S00", True):
            run_script_with_logger("s00", s00_plotagem_vento_eraa5, sftp)

    except Exception as e:
        logger.error("❌ Erro durante execução: %s", e)
        logger.exception("Detalhes do erro:")

    finally:
        if sftp:
            sftp.close()
            logger.info("🔌 Conexão SFTP fechada")
        if ssh:
            ssh.close()
            logger.info("🔌 Conexão SSH fechada")

    logger.info("=" * 100)
    logger.info("✅ PROCESSAMENTO CONCLUÍDO!")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()