#!/bin/bash
# Init file do terminal VS Code: carrega .bashrc (pyenv, aliases...) e ativa o venv do projeto.
# Usado via terminal.integrated.profiles.linux com --init-file.
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../.venv/bin/activate"
[ -f "$VENV" ] && source "$VENV"
