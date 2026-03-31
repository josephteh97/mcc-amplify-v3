#!/usr/bin/env bash
# =============================================================================
# run.sh — MCC-Amplify-v2 Pipeline Launcher (Linux / WSL)
#
# Usage:
#   ./script/run.sh <floor_plan.pdf> [--page 0] [--context project_context.json]
#   ./script/run.sh --setup          (install deps + seed memory, first run only)
#   ./script/run.sh --frontend       (start Vite web UI only)
#   ./script/run.sh --check          (verify all dependencies)
#
# Environment (set in .env or export before running):
#   WINDOWS_REVIT_SERVER   default: http://localhost:5000
#   REVIT_SERVER_API_KEY   default: my-revit-key-2023
#   REVIT_CLIENT_PATH      default: ~/Documents/mcc-amplify-ai
# =============================================================================

set -euo pipefail

# ── Resolve project root (works when called from any directory) ───────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

banner() { echo -e "\n${CYAN}${BOLD}══════════════════════════════════════${RESET}"; \
           echo -e "${CYAN}${BOLD}  $1${RESET}"; \
           echo -e "${CYAN}${BOLD}══════════════════════════════════════${RESET}"; }
ok()     { echo -e "  ${GREEN}✓${RESET}  $1"; }
warn()   { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
err()    { echo -e "  ${RED}✗${RESET}  $1"; }
info()   { echo -e "  ${CYAN}→${RESET}  $1"; }

# ── Load .env if present ──────────────────────────────────────────────────────
if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
    ok "Loaded .env"
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
export WINDOWS_REVIT_SERVER="${WINDOWS_REVIT_SERVER:-http://localhost:5000}"
export REVIT_SERVER_API_KEY="${REVIT_SERVER_API_KEY:-my-revit-key-2023}"
export REVIT_CLIENT_PATH="${REVIT_CLIENT_PATH:-$HOME/Documents/mcc-amplify-ai}"

# =============================================================================
# MODE: --check
# =============================================================================
check_deps() {
    banner "Dependency Check"

    local all_ok=true

    # Python
    if command -v python3 &>/dev/null; then
        PY_VER=$(python3 --version 2>&1)
        ok "Python: $PY_VER"
    else
        err "Python3 not found — install Python 3.10+"; all_ok=false
    fi

    # pip packages
    for pkg in pdf2image PIL requests; do
        if python3 -c "import $pkg" 2>/dev/null; then
            ok "Python package: $pkg"
        else
            warn "Python package missing: $pkg  (run: ./script/run.sh --setup)"
            all_ok=false
        fi
    done

    # poppler (pdf2image backend)
    if command -v pdftoppm &>/dev/null; then
        ok "poppler-utils (pdftoppm)"
    else
        warn "poppler-utils missing — run: sudo apt install poppler-utils"
        all_ok=false
    fi

    # Ollama
    if command -v ollama &>/dev/null; then
        ok "Ollama: $(ollama --version 2>/dev/null || echo 'installed')"
        if ollama list 2>/dev/null | grep -q "SEA-LION\|sea-lion\|Gemma-SEA-LION"; then
            ok "SEA-LION model: present"
        else
            warn "SEA-LION model not found — run: ollama pull aisingapore/Gemma-SEA-LION-v4-4B-VL:latest"
        fi
    else
        warn "Ollama not found — detection agents will fail without it"
    fi

    # Revit server reachability
    info "Checking Revit service at $WINDOWS_REVIT_SERVER ..."
    if curl -sf --max-time 3 "$WINDOWS_REVIT_SERVER/health" | grep -q "healthy\|status" 2>/dev/null; then
        ok "Revit service: reachable"
    else
        warn "Revit service: not reachable at $WINDOWS_REVIT_SERVER"
        warn "On Windows: run  revit_server\\csharp_service\\build.bat  then  run.bat"
    fi

    # Node / npm (for frontend)
    if command -v npm &>/dev/null; then
        ok "Node/npm: $(npm --version)"
    else
        warn "npm not found — frontend won't start"
    fi

    echo ""
    if $all_ok; then
        ok "All core dependencies satisfied."
    else
        warn "Some dependencies missing — pipeline may run in degraded mode."
    fi
}

# =============================================================================
# MODE: --setup
# =============================================================================
setup() {
    banner "First-Run Setup"

    info "Installing Python dependencies ..."
    pip3 install -q -r "$ROOT/requirements.txt"
    ok "requirements.txt installed"

    # Ensure sqlite3 is available (stdlib, always present)
    python3 -c "import sqlite3" && ok "sqlite3 available"

    # Seed agent memories from v1 DfMA rules
    info "Seeding agent memories ..."
    cd "$ROOT"
    python3 seed_memory.py
    ok "Memory seeded: validation/memory.sqlite + translator/memory.sqlite"

    echo ""
    ok "Setup complete. Run the pipeline with:"
    echo -e "     ${BOLD}./script/run.sh path/to/floor_plan.pdf${RESET}"
}

# =============================================================================
# MODE: --frontend
# =============================================================================
start_frontend() {
    banner "Starting Frontend"
    FRONTEND_DIR="$ROOT/frontend"
    if [[ ! -d "$FRONTEND_DIR" ]]; then
        err "frontend/ directory not found at $FRONTEND_DIR"; exit 1
    fi
    if [[ ! -f "$FRONTEND_DIR/package.json" ]]; then
        err "package.json not found — frontend may not be installed"; exit 1
    fi
    info "Installing npm packages ..."
    cd "$FRONTEND_DIR" && npm install --silent
    ok "Starting Vite dev server ..."
    exec npm run dev
}

# =============================================================================
# MODE: run pipeline
# =============================================================================
run_pipeline() {
    local PDF_PATH="$1"; shift
    local EXTRA_ARGS=("$@")

    banner "MCC-Amplify-v2 Pipeline"
    info "PDF:    $PDF_PATH"
    info "Server: $WINDOWS_REVIT_SERVER"
    info "Args:   ${EXTRA_ARGS[*]:-none}"
    echo ""

    if [[ ! -f "$PDF_PATH" ]]; then
        err "PDF not found: $PDF_PATH"; exit 1
    fi

    # Create output directory
    mkdir -p "$ROOT/data/models/rvt"

    # Run
    cd "$ROOT"
    python3 controller.py "$PDF_PATH" "${EXTRA_ARGS[@]}"
}

# =============================================================================
# Entry point
# =============================================================================
if [[ $# -eq 0 ]]; then
    echo ""
    echo -e "${BOLD}MCC-Amplify-v2 Pipeline Runner${RESET}"
    echo ""
    echo "  Usage:"
    echo "    ./script/run.sh <floor_plan.pdf> [--page 0] [--context ctx.json] [--verbose]"
    echo "    ./script/run.sh --setup      Install deps + seed memory (first run)"
    echo "    ./script/run.sh --check      Verify all dependencies"
    echo "    ./script/run.sh --frontend   Start web UI at http://localhost:5173"
    echo ""
    echo "  Environment (.env or export):"
    echo "    WINDOWS_REVIT_SERVER   $WINDOWS_REVIT_SERVER"
    echo "    REVIT_SERVER_API_KEY   $REVIT_SERVER_API_KEY"
    echo "    REVIT_CLIENT_PATH      $REVIT_CLIENT_PATH"
    echo ""
    exit 0
fi

case "$1" in
    --setup)    setup ;;
    --check)    check_deps ;;
    --frontend) start_frontend ;;
    --*)
        err "Unknown option: $1"
        echo "  Run without arguments to see usage."
        exit 1
        ;;
    *)
        run_pipeline "$@"
        ;;
esac
