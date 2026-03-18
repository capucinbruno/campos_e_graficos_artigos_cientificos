"""
Helper para processamento paralelo de tarefas com controle de recursos.

Features:
- Processamento paralelo com multiprocessing
- Controle de batch size (processar N itens por vez)
- Progress tracking com logging
- Tratamento de erros individual por tarefa
- Suporte a funções com múltiplos argumentos

Usage:
    from app.common.parallel_helper import parallel_process
    from app.common.logger import get_logger
    
    logger = get_logger("myscript")
    
    # Lista de itens para processar
    areas = ["area1", "area2", "area3", ...]
    
    # Função que processa cada item
    def process_area(area, data, settings):
        # ... processamento ...
        return result
    
    # Processar em paralelo
    results = parallel_process(
        items=areas,
        process_func=process_area,
        func_args=(data, settings),  # Argumentos extras para a função
        batch_size=4,                # Processar 4 por vez
        logger=logger,
        description="Gerando mapas"
    )
"""

import os
import time
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable, Any, Tuple, Optional, Literal
from functools import partial


def _process_item_wrapper(args):
    """
    Wrapper interno para processar um item.

    Retorna: (index, success, result_or_error)
    """
    import traceback
    index, item, func, func_args, func_kwargs = args
    try:
        if func_args and func_kwargs:
            result = func(item, *func_args, **func_kwargs)
        elif func_args:
            result = func(item, *func_args)
        elif func_kwargs:
            result = func(item, **func_kwargs)
        else:
            result = func(item)
        return (index, True, result)
    except Exception as e:
        # Capturar traceback completo para debugging
        error_details = {
            'error_type': type(e).__name__,
            'error_msg': str(e),
            'traceback': traceback.format_exc()
        }
        return (index, False, error_details)


def parallel_process(
    items: List[Any],
    process_func: Callable,
    func_args: Tuple = (),
    func_kwargs: dict = None,
    batch_size: int = None,
    max_workers: int = None,
    logger=None,
    description: str = "Processando itens",
    log_each_item: bool = True,
    use_threads: bool = True,  # Novo: usar threads em vez de processos
) -> List[Tuple[bool, Any]]:
    """
    Processa uma lista de itens em paralelo com controle de batch.
    
    Args:
        items: Lista de itens para processar
        process_func: Função que processa cada item (recebe item como primeiro arg)
        func_args: Argumentos extras para passar à função (tupla)
        func_kwargs: Argumentos keyword extras para passar à função (dict)
        batch_size: Quantidade de itens a processar por vez (None = todos de uma vez)
        max_workers: Número máximo de workers (None = cpu_count())
        logger: Logger para output (None = print)
        description: Descrição da tarefa para logs
        log_each_item: Se True, loga cada item processado
        use_threads: Se True, usa ThreadPoolExecutor (permite nested functions).
                     Se False, usa multiprocessing.Pool (mais rápido, requer funções top-level)

    Returns:
        Lista de tuplas (success, result) na mesma ordem dos items
        - success: True se processou com sucesso, False se erro
        - result: Resultado ou mensagem de erro
    
    Example:
        >>> def generate_plot(area, data, output_dir):
        ...     plt.figure()
        ...     plt.plot(data[area])
        ...     plt.savefig(f"{output_dir}/{area}.png")
        ...     return f"{area}.png"
        
        >>> areas = ["area1", "area2", "area3", "area4"]
        >>> results = parallel_process(
        ...     items=areas,
        ...     process_func=generate_plot,
        ...     func_args=(data, "/output"),
        ...     batch_size=2,
        ...     logger=logger,
        ...     description="Gerando mapas"
        ... )
    """
    if func_kwargs is None:
        func_kwargs = {}
    
    if not items:
        if logger:
            logger.warning("⚠️  Lista de itens vazia, nada para processar")
        return []
    
    # Determinar número de workers
    if max_workers is None:
        max_workers = cpu_count()
    
    # Determinar batch size
    if batch_size is None:
        batch_size = len(items)
    else:
        batch_size = min(batch_size, len(items))
    
    # Limitar workers ao batch size
    workers = min(max_workers, batch_size)
    
    total_items = len(items)
    
    # Log inicial
    if logger:
        logger.info(f"🚀 {description}")
        logger.info(f"   📊 Total de itens: {total_items}")
        logger.info(f"   🔢 Batch size: {batch_size}")
        logger.info(f"   👷 Workers: {workers}")
        logger.info(f"   🔄 Batches: {(total_items + batch_size - 1) // batch_size}")
    else:
        print(f"🚀 {description}: {total_items} itens, batch={batch_size}, workers={workers}")
    
    # Resultados (inicializar com None para manter ordem)
    results = [None] * total_items
    successful = 0
    failed = 0
    
    # Processar em batches
    start_time = time.time()
    
    for batch_num, batch_start in enumerate(range(0, total_items, batch_size), 1):
        batch_end = min(batch_start + batch_size, total_items)
        batch_items = items[batch_start:batch_end]
        batch_count = len(batch_items)
        
        batch_start_time = time.time()
        
        if logger:
            logger.info(f"\n📦 Batch {batch_num}: Processando itens {batch_start+1}-{batch_end} ({batch_count} itens)")
        
        # Preparar argumentos para o pool
        pool_args = [
            (batch_start + i, item, process_func, func_args, func_kwargs)
            for i, item in enumerate(batch_items)
        ]
        
        # Processar batch em paralelo
        if use_threads:
            # Usar ThreadPoolExecutor (permite nested functions, GIL pode ser limitação)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                batch_results = list(executor.map(lambda args: _process_item_wrapper(args), pool_args))
        else:
            # Usar multiprocessing.Pool (mais rápido, mas requer funções top-level)
            with Pool(processes=workers) as pool:
                batch_results = pool.map(_process_item_wrapper, pool_args)
        
        # Processar resultados do batch
        for index, success, result in batch_results:
            results[index] = (success, result)

            if success:
                successful += 1
                if log_each_item and logger:
                    logger.info(f"   ✅ [{index+1}/{total_items}] {items[index]}")
            else:
                failed += 1
                if logger:
                    # Se result é um dict com traceback, logar detalhes completos
                    if isinstance(result, dict) and 'traceback' in result:
                        logger.error(f"   ❌ [{index+1}/{total_items}] Erro: {result['error_type']}: {result['error_msg']}")
                        logger.error(f"   Traceback:\n{result['traceback']}")
                    else:
                        logger.error(f"   ❌ [{index+1}/{total_items}] {items[index]}: {result}")
        
        batch_elapsed = time.time() - batch_start_time
        
        if logger:
            logger.info(f"   ⏱️  Batch {batch_num} concluído em {batch_elapsed:.2f}s")
    
    # Log final
    elapsed = time.time() - start_time
    
    if logger:
        logger.info(f"\n✅ {description} concluído!")
        logger.info(f"   ⏱️  Tempo total: {elapsed:.2f}s ({elapsed/60:.2f} min)")
        logger.info(f"   📊 Sucesso: {successful}/{total_items}")
        if failed > 0:
            logger.info(f"   ❌ Falhas: {failed}/{total_items}")
        logger.info(f"   ⚡ Velocidade média: {elapsed/total_items:.2f}s por item")
    else:
        print(f"✅ Concluído em {elapsed:.2f}s: {successful} ok, {failed} falhas")
    
    return results


def get_recommended_batch_size(total_items: int, memory_intensive: bool = False) -> int:
    """
    Calcula batch size recomendado baseado no número de CPUs e intensidade de memória.
    Respeita a configuração PARALLEL_WORKERS do settings.local.toml.

    Args:
        total_items: Número total de itens a processar
        memory_intensive: True se a tarefa usa muita memória (ex: plots grandes)

    Returns:
        Batch size recomendado

    Example:
        >>> batch = get_recommended_batch_size(25, memory_intensive=True)
        >>> # Em máquina com 8 CPUs: retorna 4 (auto) ou 8 (max)
    """
    from app.shared.settings_factory import settings

    cpus = cpu_count()

    # Ler configuração de workers
    parallel_workers = getattr(settings, 'PARALLEL_WORKERS', 'auto')

    if isinstance(parallel_workers, str):
        if parallel_workers.lower() == 'max':
            # Usar todos os CPUs disponíveis
            recommended = cpus
        elif parallel_workers.lower() == 'auto':
            # Modo automático: metade para tarefas intensivas, todos para tarefas leves
            if memory_intensive:
                recommended = max(2, cpus // 2)
            else:
                recommended = cpus
        else:
            # Se for string mas não é 'max' ou 'auto', tentar converter para int
            try:
                recommended = int(parallel_workers)
            except ValueError:
                # Fallback para auto
                recommended = max(2, cpus // 2) if memory_intensive else cpus
    elif isinstance(parallel_workers, int):
        # Valor fixo configurado
        recommended = parallel_workers
    else:
        # Fallback para comportamento padrão
        recommended = max(2, cpus // 2) if memory_intensive else cpus

    # Garantir valores válidos
    recommended = max(1, min(recommended, cpus))

    # Nunca exceder o total de itens
    return min(recommended, total_items)
