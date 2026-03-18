---
description: Convencoes de teste do projeto campos_observados_era5
paths: ["tests/**/*.py"]
---

# Testing - Campos Observados ERA5

## Framework

- **pytest** >= 8.3.5
- **Comando**: `uv run pytest tests/ -v`

## Padroes

### Import Tests (rede de seguranca)

Valida que todos os modulos importam sem erro:

```python
@pytest.mark.parametrize('module_path', ALL_MODULES)
def test_import_module(module_path):
    mod = importlib.import_module(module_path)
    assert mod is not None
```

### Smoke Tests (contrato de API)

Valida que funcoes-chave existem e sao callable:

```python
def test_script_has_main():
    mod = importlib.import_module('Scripts.s00_plotagem_vento_eraa5')
    assert hasattr(mod, 'main')
    assert callable(mod.main)
```

### Nomenclatura

- Prefixo `test_` obrigatorio
- Funcoes descritivas: `test_<modulo>_has_main`, `test_<feature>_importa`
- Sem classes de teste (funcoes soltas)

## Ao Criar Novos Scripts

1. Adicionar modulo em import tests
2. Adicionar smoke test se modulo exporta `main()` ou funcao publica
3. Manter padrao `importlib.import_module()` para consistencia
4. Testar com periodo curto (1-2 dias) para validar sem baixar muitos dados:

```bash
uv run python run_script.py sNN --data-inicial 2026-03-10 --data-final 2026-03-11
```
