#!/bin/bash
# =============================================================================
# Setup automatizado do projeto campos_observados_era5
# =============================================================================

set -e

echo "========================================"
echo "  Setup - Campos Observados ERA5"
echo "========================================"

# Funcao auxiliar: pergunta se quer substituir arquivo existente
# Uso: copy_with_confirm <origem> <destino> <descricao>
copy_with_confirm() {
    local origem="$1"
    local destino="$2"
    local descricao="$3"

    if [ ! -f "$destino" ]; then
        cp "$origem" "$destino"
        echo "  $descricao criado a partir de $origem"
        return 0  # arquivo novo
    else
        echo "  $descricao ja existe."
        read -p "  Deseja substituir? (s/N): " resposta
        if [[ "$resposta" =~ ^[sS]$ ]]; then
            cp "$origem" "$destino"
            echo "  $descricao substituido!"
            return 0  # arquivo substituido
        else
            echo "  Mantendo $descricao atual."
            return 1  # arquivo mantido
        fi
    fi
}

# --- 1. Verifica se UV esta instalado ---
if ! command -v uv &> /dev/null; then
    echo "[1/6] Instalando UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "  UV instalado com sucesso!"
else
    echo "[1/6] UV ja esta instalado: $(uv --version)"
fi

# --- 2. Selecionar ambiente ---
echo ""
echo "[2/6] Selecione o ambiente:"
echo ""
echo "  1) development  - Sua maquina local (SFTP habilitado, logging DEBUG)"
echo "  2) production   - Servidor Oracle (acesso local, logging INFO)"
echo "  3) qa           - Testes (sem SFTP, logging DEBUG)"
echo ""
read -p "  Escolha (1/2/3) [1]: " env_choice

case "$env_choice" in
    2)
        selected_env="production"
        ;;
    3)
        selected_env="qa"
        ;;
    *)
        selected_env="development"
        ;;
esac

echo "  Ambiente selecionado: $selected_env"

# --- 3. Copia arquivos de configuracao ---
echo ""
echo "[3/6] Configurando arquivos locais..."

# Cria .env com o ambiente selecionado
if [ ! -f .env ]; then
    echo "ENV_FOR_DYNACONF=$selected_env" > .env
    echo "  .env criado com ENV_FOR_DYNACONF=$selected_env"
else
    echo "  .env ja existe (ENV_FOR_DYNACONF=$(grep ENV_FOR_DYNACONF .env | cut -d= -f2))"
    read -p "  Deseja substituir com ambiente '$selected_env'? (s/N): " resposta
    if [[ "$resposta" =~ ^[sS]$ ]]; then
        echo "ENV_FOR_DYNACONF=$selected_env" > .env
        echo "  .env atualizado para $selected_env"
    else
        echo "  Mantendo .env atual."
    fi
fi

# VSCode settings (copia sem perguntar)
if [ ! -f .vscode/settings.json ]; then
    cp .vscode/settings_exemplo.json .vscode/settings.json
    echo "  .vscode/settings.json criado a partir do template"
else
    echo "  .vscode/settings.json ja existe"
fi

copy_with_confirm "settings.local.example.toml" "settings.local.toml" "settings.local.toml" || true

secrets_changed=0
copy_with_confirm "app/settings/.secrets_example.toml" "app/settings/.secrets.toml" ".secrets.toml" && secrets_changed=1 || true

if [ "$secrets_changed" -eq 1 ]; then
    echo ""
    echo "  ============================================================"
    echo "  IMPORTANTE: Preencha app/settings/.secrets.toml com sua"
    echo "  chave do Copernicus CDS (KEY_CDS)!"
    echo ""
    echo "  Obtenha sua chave em: https://cds.climate.copernicus.eu/"
    echo "  ============================================================"
    echo ""
fi

# --- 4. Cria diretorios necessarios ---
echo "[4/6] Criando diretorios..."
mkdir -p dados Entrada/arquivos_nc Saida logs

# --- 5. Instala dependencias ---
echo "[5/6] Instalando dependencias com UV..."
uv sync

# --- 6. Instala dependencias de desenvolvimento (opcional) ---
echo "[6/6] Instalando dependencias de desenvolvimento..."
uv sync --extra dev

echo ""
echo "========================================"
echo "  Setup concluido!"
echo "========================================"
echo ""
echo "  Ambiente: $selected_env"
echo ""
echo "Proximos passos:"
echo "  1. Preencha app/settings/.secrets.toml com sua KEY_CDS"
echo "  2. Ajuste settings.local.toml com suas datas e flags de scripts"
echo "  3. Execute de uma das formas:"
echo ""
echo "     uv run python run_script.py --list"
echo ""
echo "     ou"
echo ""
echo "     source .venv/bin/activate"
echo "     python run_script.py --list"
echo ""
