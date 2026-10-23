#!/usr/bin/env bash
# Installe le RAG et le serveur MCP, construit l'index et prépare la config d'un agent.
#
#   ./scripts/install.sh                         installation et index (incrémental)
#   ./scripts/install.sh --rebuild               supprime l'index avant de le reconstruire
#   ./scripts/install.sh --workspace ~/fpga/proj dossier de projet donné au serveur MCP
#   ./scripts/install.sh --serve                 lance aussi `fpga_rag serve` en tâche de fond
#
# Le script peut être relancé autant de fois que nécessaire : le venv est réutilisé et
# l'indexation ne recalcule que les documents modifiés.
#
# Variables lues (toutes facultatives) :
#   FPGA_RAG_HOME, FPGA_RAG_DB, FPGA_RAG_BACKEND, FPGA_RAG_MODEL   voir rag/README.fr.md
#   FPGA_RAG_EMBED_BATCH    morceaux calculés ensemble (défaut 4, baisser si l'indexation est tuée)
#   FPGA_RAG_EMBED_THREADS  threads ONNX (défaut : tous les cœurs)
#   HF_TOKEN                évite les limites de débit au téléchargement du modèle

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv"
WORKSPACE="$HOME"
REBUILD=0
SERVE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --rebuild)   REBUILD=1 ;;
        --serve)     SERVE=1 ;;
        --workspace) WORKSPACE="$(cd "${2:?--workspace attend un dossier}" && pwd)"; shift ;;
        -h|--help)   sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "option inconnue : $1 (voir --help)" >&2; exit 2 ;;
    esac
    shift
done

etape() { printf '\n\033[1;34m== %s\033[0m\n' "$1"; }
stop()  { printf '\033[1;31mERREUR : %s\033[0m\n' "$1" >&2; exit 1; }

etape "Python"
command -v python3 >/dev/null || stop "python3 introuvable"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
    || stop "Python 3.10 ou plus requis, trouvé : $(python3 --version)"
python3 -c 'import venv, ensurepip' 2>/dev/null \
    || stop "module venv absent (Debian, Ubuntu : sudo apt install python3-venv)"
python3 --version

etape "Environnement virtuel"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -e "$REPO/rag[all]" -e "$REPO/mcp"
echo "$VENV : fpga_rag et vivado_mcp installés"

# L'index local vit hors du dépôt (défaut du paquet : ~/.local/share/fpga-rag) :
# un `git add -A` ne peut pas le ramasser, et index/index.db reste celui du dépôt.
export FPGA_RAG_HOME="${FPGA_RAG_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/fpga-rag}"
export FPGA_RAG_DB="${FPGA_RAG_DB:-$FPGA_RAG_HOME/index.db}"
export FPGA_RAG_EMBED_BATCH="${FPGA_RAG_EMBED_BATCH:-4}"

etape "Index ($FPGA_RAG_DB)"
if [ "$REBUILD" -eq 1 ]; then
    "$PY" -m fpga_rag reset --yes
fi
echo "premier lancement : téléchargement du modèle (130 Mo) puis 2 à 3 minutes de calcul sur CPU"
set +e
"$PY" -m fpga_rag index-repo
rc=$?
set -e
if [ "$rc" -eq 137 ]; then
    stop "indexation tuée par manque de mémoire : relancer avec FPGA_RAG_EMBED_BATCH=1"
elif [ "$rc" -ne 0 ]; then
    stop "échec de l'indexation (code $rc). Index construit avec un autre modèle ? relancer avec --rebuild"
fi
"$PY" -m fpga_rag status

etape "Contrôle"
"$PY" -m fpga_rag search "contraindre une horloge de 100 MHz" -k 2 --max-chars 200
"$PY" -m vivado_mcp --workspace "$WORKSPACE" --selftest >/dev/null \
    && echo "serveur MCP : démarrage vérifié (workspace $WORKSPACE)" \
    || echo "serveur MCP : le selftest a échoué, lancer $PY -m vivado_mcp --workspace $WORKSPACE --selftest"

# Config de l'agent. Hermes ne transmet pas l'environnement du shell aux serveurs
# stdio : tout ce dont le serveur a besoin est écrit dans le bloc `env`.
SNIPPET="$REPO/hermes-mcp.local.yaml"
{
    echo "# Généré par scripts/install.sh : à coller sous mcp_servers: dans ~/.hermes/config.yaml"
    echo "mcp_servers:"
    echo "  vivado-vitis:"
    echo "    command: $PY"
    echo "    args: [\"-m\", \"vivado_mcp\", \"--workspace\", \"$WORKSPACE\"]"
    echo "    env:"
    echo "      FPGA_RAG_HOME: \"$FPGA_RAG_HOME\""
    echo "      FPGA_RAG_DB: \"$FPGA_RAG_DB\""
    [ -n "${FPGA_RAG_BACKEND:-}" ] && echo "      FPGA_RAG_BACKEND: \"$FPGA_RAG_BACKEND\""
    [ -n "${FPGA_RAG_MODEL:-}" ]   && echo "      FPGA_RAG_MODEL: \"$FPGA_RAG_MODEL\""
    [ -n "${FPGA_MCP_TOOL_ROOT:-}" ] && echo "      FPGA_MCP_TOOL_ROOT: \"$FPGA_MCP_TOOL_ROOT\""
    echo "      FPGA_MCP_TIMEOUT: \"3600\""
    echo "    enabled: true"
} > "$SNIPPET"

if [ "$SERVE" -eq 1 ]; then
    etape "Service OpenAI-compatible"
    PORT="${FPGA_RAG_PORT:-8098}"
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "déjà actif sur le port $PORT"
    else
        nohup "$PY" -m fpga_rag serve > "$FPGA_RAG_HOME/serve.log" 2>&1 &
        echo $! > "$FPGA_RAG_HOME/serve.pid"
        for _ in $(seq 1 30); do
            curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
            sleep 0.5
        done
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
            || stop "le service ne répond pas, voir $FPGA_RAG_HOME/serve.log"
        echo "http://127.0.0.1:$PORT/v1  (arrêt : kill \$(cat $FPGA_RAG_HOME/serve.pid))"
    fi
fi

etape "Terminé"
cat <<EOF
Brancher Hermes (ou tout client MCP) :
  1. copier le bloc de $SNIPPET dans ~/.hermes/config.yaml
  2. dans une session Hermes : /reload-mcp
  3. demander par exemple « que signifie Synth 8-327 ? » : la réponse doit citer docs/

Utiliser le RAG depuis un shell :
  source $VENV/bin/activate
  export FPGA_RAG_DB="$FPGA_RAG_DB"
  python -m fpga_rag search "reset_run synth_1"
EOF
