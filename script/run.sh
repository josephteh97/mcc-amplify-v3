#!/usr/bin/env bash
# run.sh — MCC-Amplify-v2 Pipeline Launcher (Linux / WSL)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

banner() { echo -e "\n${CYAN}${BOLD}══════════════════════════════════════${RESET}"; echo -e "${CYAN}${BOLD}  $1${RESET}"; echo -e "${CYAN}${BOLD}══════════════════════════════════════${RESET}"; }
ok()     { echo -e "  ${GREEN}✓${RESET}  $1"; }
warn()   { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
err()    { echo -e "  ${RED}✗${RESET}  $1"; }
info()   { echo -e "  ${CYAN}→${RESET}  $1"; }

# status table row: srow <COLOR_VAR> <label> <value>
srow() { printf "  ${!1}%-26s${RESET} %s\n" "$2" "$3"; }

# kill any process holding the given TCP port
free_port() {
    local pid
    pid=$(lsof -ti tcp:"$1" 2>/dev/null) || true
    if [[ -n "$pid" ]]; then
        info "Freeing port $1 (PID $pid)…"
        kill "$pid" 2>/dev/null || true
        sleep 1
    fi
}

if [[ -f "$ROOT/.env" ]]; then
    set -a; source "$ROOT/.env"; set +a
    ok "Loaded .env"
fi

export WINDOWS_REVIT_SERVER="${WINDOWS_REVIT_SERVER:-http://localhost:5000}"
export REVIT_SERVER_API_KEY="${REVIT_SERVER_API_KEY:-my-revit-key-2023}"

# =============================================================================
# MODE: --check
# =============================================================================
check_deps() {
    banner "Dependency Check"
    local all_ok=true

    if command -v python3 &>/dev/null; then
        ok "Python: $(python3 --version 2>&1)"
    else
        err "Python3 not found — install Python 3.10+"; all_ok=false
    fi

    for pkg in pdf2image PIL requests; do
        if python3 -c "import $pkg" 2>/dev/null; then
            ok "Python package: $pkg"
        else
            warn "Python package missing: $pkg  (run: ./script/run.sh --setup)"; all_ok=false
        fi
    done

    if command -v pdftoppm &>/dev/null; then
        ok "poppler-utils (pdftoppm)"
    else
        warn "poppler-utils missing — run: sudo apt install poppler-utils"; all_ok=false
    fi

    if command -v ollama &>/dev/null; then
        ok "Ollama: $(ollama --version 2>/dev/null || echo 'installed')"
        if ollama list 2>/dev/null | grep -qi "sea-lion\|Gemma-SEA"; then
            ok "SEA-LION model: present"
        else
            warn "SEA-LION model not found — run: ollama pull aisingapore/Gemma-SEA-LION-v4-4B-VL:latest"
        fi
    else
        warn "Ollama not found — detection agents will fail without it"
    fi

    if curl -sf --max-time 3 -H "X-API-Key: ${REVIT_SERVER_API_KEY}" \
            "${WINDOWS_REVIT_SERVER}/health" 2>/dev/null | grep -q "status"; then
        ok "Revit service: reachable"
    else
        warn "Revit service: not reachable at $WINDOWS_REVIT_SERVER"
        warn "On Windows: run  revit_server\\csharp_service\\build.bat"
    fi

    command -v npm &>/dev/null && ok "Node/npm: $(npm --version)" || { warn "npm not found — frontend won't start"; all_ok=false; }

    echo ""
    $all_ok && ok "All core dependencies satisfied." || warn "Some dependencies missing — pipeline may run in degraded mode."
}

# =============================================================================
# MODE: --setup
# =============================================================================
setup() {
    banner "First-Run Setup"
    info "Installing Python dependencies ..."
    pip3 install -q -r "$ROOT/requirements.txt"
    ok "Python deps installed"
    info "Seeding agent memories ..."
    cd "$ROOT" && python3 backend/seed_memory.py
    ok "Memory seeded"
    echo ""
    ok "Setup complete. Run the pipeline with:"
    echo -e "     ${BOLD}./script/run.sh path/to/floor_plan.pdf${RESET}"
}

# =============================================================================
# MODE: --frontend  — start backend API + Vite together
# =============================================================================
start_frontend() {
    banner "Starting MCC-Amplify-v2 Web App"
    FRONTEND_DIR="$ROOT/frontend"

    [[ -d "$FRONTEND_DIR" ]]              || { err "frontend/ not found at $FRONTEND_DIR"; exit 1; }
    [[ -f "$FRONTEND_DIR/package.json" ]] || { err "package.json missing — run: ./script/run.sh --setup"; exit 1; }

    cleanup() {
        echo -e "\n  ${CYAN}Goodbye! Amplify v2 stopped.${RESET}"
        [[ -n "${BACKEND_PID:-}" ]] && kill "$BACKEND_PID" 2>/dev/null
        [[ -n "${VITE_PID:-}"    ]] && kill "$VITE_PID"    2>/dev/null
        wait 2>/dev/null; exit 0
    }
    trap cleanup SIGINT SIGTERM

    free_port 8000
    info "Starting API backend  →  http://localhost:8000"
    cd "$ROOT" && python3 backend/server.py &
    BACKEND_PID=$!

    for ((i=1; i<=10; i++)); do
        sleep 1
        curl -sf http://localhost:8000/health >/dev/null 2>&1 && { ok "Backend API:        http://localhost:8000"; break; }
    done

    # ── System Status ──────────────────────────────────────────────────────────
    echo ""
    echo -e "  ${BOLD}System Status${RESET}"
    printf "  %-26s %s\n" "─────────────────────────" "──────────────────────────────────"

    srow GREEN "Python" "$(python3 --version 2>&1)"

    if command -v ollama &>/dev/null; then
        local ollama_out
        ollama_out=$(ollama list 2>/dev/null)
        OLLAMA_MODELS=$(echo "$ollama_out" | tail -n +2 | awk '{print $1}' | tr '\n' '  ')
        srow GREEN "Ollama" "${OLLAMA_MODELS:-no models loaded}"
        if echo "$ollama_out" | grep -qi "sea-lion\|Gemma-SEA"; then
            srow GREEN  "SEA-LION vision model" "ready"
        else
            srow YELLOW "SEA-LION vision model" "not found — run: ollama pull aisingapore/Gemma-SEA-LION-v4-4B-VL:latest"
        fi
    else
        srow YELLOW "Ollama" "not found — detection agents will fail"
    fi

    [[ -f "$ROOT/grid-detection-agent/agent.py" ]] \
        && srow GREEN  "Grid detection agent"   "ready" \
        || srow YELLOW "Grid detection agent"   "missing — check grid-detection-agent/agent.py"
    [[ -f "$ROOT/yolo_detection_agents/weights/column-detect.pt" ]] \
        && srow GREEN  "Column detection agent" "ready" \
        || srow YELLOW "Column detection agent" "missing — copy column-detect.pt to yolo_detection_agents/weights/"

    REVIT_BODY=$(curl -sf --max-time 3 \
        -H "X-API-Key: ${REVIT_SERVER_API_KEY}" \
        "${WINDOWS_REVIT_SERVER}/health" 2>/dev/null) || true
    if [[ -n "$REVIT_BODY" ]]; then
        REV_DETAIL=$(echo "$REVIT_BODY" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    print(d.get('status','ok')+'  (revit_initialized='+str(d.get('revit_initialized',False))+')')
except: print('connected')
" 2>/dev/null || echo "connected")
        printf "  ${GREEN}%-26s${RESET} %s  [%s]\n" "Revit server" "$REV_DETAIL" "$WINDOWS_REVIT_SERVER"
    else
        srow YELLOW "Revit server" "offline — start Revit on Windows then run build.bat"
    fi

    # Single python3 call for both SQLite counts
    read -r VAL_ROWS TR_ROWS < <(python3 - "$ROOT" <<'PYEOF'
import sys, sqlite3, pathlib
root = pathlib.Path(sys.argv[1])
def count(db, tbl, noun):
    try: return f"{sqlite3.connect(db).execute(f'SELECT COUNT(*) FROM {tbl}').fetchone()[0]} {noun}"
    except: return "not found"
print(count(root/'validation/memory.sqlite', 'conflict_resolutions', 'rules seeded'))
print(count(root/'translator/memory.sqlite', 'api_success_patterns', 'patterns seeded'))
PYEOF
)
    srow GREEN "Validation memory" "$VAL_ROWS"
    srow GREEN "Translator memory" "$TR_ROWS"
    printf "  %-26s %s\n" "─────────────────────────" "──────────────────────────────────"
    echo ""

    [[ -d "$FRONTEND_DIR/node_modules" ]] || { info "Installing npm packages…"; cd "$FRONTEND_DIR" && npm install --silent; }

    free_port 5173
    info "Starting frontend     →  http://localhost:5173"
    cd "$FRONTEND_DIR" && npm run dev &
    VITE_PID=$!
    sleep 2

    echo ""
    echo -e "  ${BOLD}${GREEN}Amplify v2 is ready — open your browser${RESET}"
    echo -e "  ${GREEN}  http://localhost:5173${RESET}"
    echo -e "  Press Ctrl+C to stop."
    echo ""
    wait "$BACKEND_PID" "$VITE_PID"
}

# =============================================================================
# MODE: run pipeline
# =============================================================================
run_pipeline() {
    local PDF_PATH="$1"; shift
    banner "MCC-Amplify-v2 Pipeline"
    info "PDF:    $PDF_PATH"
    info "Server: $WINDOWS_REVIT_SERVER"
    [[ -f "$PDF_PATH" ]] || { err "PDF not found: $PDF_PATH"; exit 1; }
    mkdir -p "$ROOT/data/models/rvt"
    cd "$ROOT" && python3 backend/controller.py "$PDF_PATH" "$@"
}

# =============================================================================
# Entry point
# =============================================================================
if [[ $# -eq 0 ]]; then start_frontend; exit 0; fi

case "$1" in
    --setup)    setup ;;
    --check)    check_deps ;;
    --frontend) start_frontend ;;
    --*) err "Unknown option: $1"; exit 1 ;;
    *)   run_pipeline "$@" ;;
esac
