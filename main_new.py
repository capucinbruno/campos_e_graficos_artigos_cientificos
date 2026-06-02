import json
from datetime import datetime
from pathlib import Path

from app.common.display import tabela
from app.common.logger import get_logger
from app.src import ReanaliseAPI

logger = get_logger('reanalise-api')

params_agr = json.loads(Path('Entrada/media_dados.json').read_text())
params_ano = json.loads(Path('Entrada/anomalia_dados.json').read_text())

logger.info('=' * 72)
logger.info('EXECUÇÃO INICIADA — %s', datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
logger.info('=' * 72)

try:
    with ReanaliseAPI() as api:
        tabela('Datasets', api.listar_datasets())
        tabela('Variáveis', api.listar_variaveis())
        tabela('Tipos de agregação', api.listar_tipos_agregacao())

        # ── Agregação temporal ────────────────────────────────────────────────
        logger.info('AGREGAÇÃO TEMPORAL')
        for chave, valor in params_agr.items():
            logger.info('  %-30s %s', chave + ':', valor)

        ds_agr = api.agregacao_temporal(**params_agr)
        print(ds_agr)

        # ── Anomalia ─────────────────────────────────────────────────────────
        logger.info('ANOMALIA')
        for chave, valor in params_ano.items():
            logger.info('  %-30s %s', chave + ':', valor)

        ds_ano = api.anomalia(**params_ano)
        print(ds_ano)

except Exception as exc:
    logger.error('=' * 72)
    logger.error('ERRO: %s — %s', type(exc).__name__, exc)
    logger.error('=' * 72, exc_info=True)
    raise


logger.info('=' * 72)
logger.info('EXECUÇÃO ENCERRADA — %s', datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
logger.info('=' * 72)
