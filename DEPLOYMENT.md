# DEPLOYMENT.md — MCC-Amplify-AI v2

Step-by-step guide to deploy and run the agentic PDF-to-BIM pipeline on a single Windows machine running both Revit 2023 and WSL2 (Ubuntu).

---

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Windows Host                          │
│                                                         │
│  ┌──────────────────┐      ┌────────────────────────┐   │
│  │   Revit 2023     │◄─────│  RevitService.exe      │   │
│  │  (GUI process)   │      │  C# Add-in bridge      │   │
│  └──────────────────┘      │  localhost:5000         │   │
│                             └───────────┬────────────┘   │
│                                         │ HTTP           │
│  ┌──────────────────────────────────────▼────────────┐   │
│  │                   WSL2 (Ubuntu)                    │   │
│  │                                                    │   │
│  │  controller.py                                     │   │
│  │    ├── Grid Detection Agent  (Ollama SEA-LION)     │   │
│  │    ├── Column Detection Agent (Ollama SEA-LION)    │   │
│  │    ├── Validation Agent      (DfMA rules)          │   │
│  │    └── BIM-Translator Agent  (→ Revit HTTP POST)   │   │
│  │                                                    │   │
│  │  frontend/   (Vite → localhost:5173)               │   │
│  └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Windows Side

| Requirement | Version | Notes |
|---|---|---|
| Revit | 2023 (or 2022) | Licensed install required |
| .NET SDK | 4.8 | [Download](https://dotnet.microsoft.com/en-us/download/dotnet-framework/net48) |
| Build Tools for Visual Studio | 2022 | Required by `dotnet build` |
| WSL2 | Ubuntu 20.04+ | Enable via `wsl --install` |

### WSL2 / Ubuntu Side

| Requirement | Version | Install |
|---|---|---|
| Python | 3.10+ | `sudo apt install python3 python3-pip` |
| poppler-utils | any | `sudo apt install poppler-utils` |
| Ollama | 0.18+ | [ollama.com/download](https://ollama.com/download) |
| SEA-LION model | v4-4B-VL | See Step 5 below |
| Node.js + npm | 18+ | `sudo apt install nodejs npm` (frontend only) |

---

## Part 1 — Windows Setup (Revit Service)

### Step 1 — Configure the C# Service

Open `revit_server/csharp_service/config.json` and confirm or update:

```json
{
  "revit_settings": {
    "version": "2023",
    "template_path": "C:\\ProgramData\\Autodesk\\RVT 2023\\Templates\\Architectural Template.rte",
    "output_directory": "C:\\RevitOutput"
  },
  "api_settings": {
    "host": "0.0.0.0",
    "port": 5000,
    "api_key": "my-revit-key-2023"
  }
}
```

> The `api_key` value must match `REVIT_SERVER_API_KEY` in your WSL environment (Step 7).

### Step 2 — Build and Start the Revit Service

Open **Command Prompt** (not PowerShell) in `revit_server\csharp_service\` and run:

```bat
build.bat
```

This script performs 5 actions automatically:

| Step | Action |
|---|---|
| 1 | `dotnet clean` — clears stale binaries |
| 2 | `dotnet build` — compiles `RevitService.dll` (.NET 4.8) |
| 3 | Registry trust — adds DLL path to Revit CodeSigning whitelist |
| 4 | Launches `Revit.exe` |
| 5 | Polls port 5000 every 10 s until the service responds (6 attempts) |

Expected terminal output when ready:

```
✅ Build Successful!
✅ Port is LISTENING!
```

### Step 3 — Verify the Service is Healthy

Open a browser or PowerShell on Windows:

```powershell
Invoke-WebRequest http://localhost:5000/health
```

Expected response:

```json
{ "status": "healthy", "service": "Revit API Service", "revit_initialized": true }
```

If the service is not reachable, run `script\run.bat --check` for a full diagnostic.

### Step 4 — (Optional) Deploy the Add-in Directly into Revit

For a permanent Add-in that auto-loads when Revit starts (no separate service process needed), use the Add-in manifest instead:

```bat
cd revit_server\RevitAddin
build_and_deploy.bat
```

This copies `RevitModelBuilderAddin.dll` and `RevitModelBuilder.addin` to:

```
C:\ProgramData\Autodesk\Revit\Addins\2023\
```

Restart Revit after deployment. The Add-in loads automatically and opens the HTTP listener on port 5000.

---

## Part 2 — WSL2 / Ubuntu Setup (Python Pipeline)

### Step 5 — Pull the SEA-LION Vision Model

```bash
ollama pull aisingapore/Gemma-SEA-LION-v4-4B-VL:latest
```

Verify it downloaded:

```bash
ollama list
# aisingapore/Gemma-SEA-LION-v4-4B-VL:latest   3.3 GB   ✓
```

### Step 6 — Install Python Dependencies

```bash
cd ~/Documents/mcc-amplify-v2
pip install -r requirements.txt
sudo apt install poppler-utils      # PDF rendering backend
```

### Step 7 — Configure Environment Variables

Create a `.env` file in the project root (automatically loaded by `run.sh`):

```bash
cat > ~/Documents/mcc-amplify-v2/.env << 'EOF'
WINDOWS_REVIT_SERVER=http://localhost:5000
REVIT_SERVER_API_KEY=my-revit-key-2023
REVIT_CLIENT_PATH=/home/<your_username>/Documents/mcc-amplify-ai
EOF
```

> Replace `<your_username>` with your actual WSL username.  
> `REVIT_CLIENT_PATH` points to the v1 project which contains `backend/services/revit_client.py` — the BIM-Translator uses it to call the Revit Add-in.

### Step 8 — Seed Agent Memories (First Run Only)

This populates both agent databases with DfMA rules extracted from the v1 project:

```bash
cd ~/Documents/mcc-amplify-v2
./script/run.sh --setup
```

Expected output:

```
Seeding validation/memory.sqlite …
  [C2] column  | BCA DfMA: minimum RC column section 200 mm ...  {'ok': True}
  [W1] wall    | SS CP 65: interior RC 200 mm default ...         {'ok': True}
  ...  9 resolutions seeded

Seeding translator/memory.sqlite …
  [OK] column  | M_Concrete-Rectangular-Column | 800 x 800mm ... {'ok': True}
  ...  11 patterns seeded
```

### Step 9 — Verify All Dependencies

```bash
./script/run.sh --check
```

All items should show `✓`. The Revit service check will show `⚠` if Windows hasn't started it yet — that's expected at this stage.

---

## Part 3 — Running the Pipeline

### Quick Start

```bash
cd ~/Documents/mcc-amplify-v2
./script/run.sh path/to/floor_plan.pdf
```

Output `.rvt` is written to `data/models/rvt/<job_id>.rvt`.

### With Explicit Project Context

Provide bay widths and project standards for accurate mm coordinates:

```bash
./script/run.sh floor_plan.pdf --context project_context.json --page 0
```

`project_context.json`:

```json
{
  "storey_height": 3000,
  "bay_widths_x_mm": [7500, 7500, 8000, 7500],
  "bay_widths_y_mm": [7500, 8000, 7500],
  "wall_thickness_interior_mm": 200,
  "wall_thickness_exterior_mm": 300,
  "default_column_size_mm": 800,
  "assumed_bay_mm": 7500
}
```

| Key | Default | Description |
|---|---|---|
| `storey_height` | 3000 | Floor-to-floor height (mm) |
| `bay_widths_x_mm` | Equal spacing | List of bay widths along vertical grid lines |
| `bay_widths_y_mm` | Equal spacing | List of bay widths along horizontal grid lines |
| `wall_thickness_interior_mm` | 200 | SS CP 65 interior RC default |
| `wall_thickness_exterior_mm` | 300 | SS CP 65 exterior RC default |
| `default_column_size_mm` | 200 | Fallback when no PDF annotation found |
| `assumed_bay_mm` | 7500 | Used when `bay_widths_*` not provided |

### Selecting a Specific PDF Page

For multi-page drawing sets (the floor plan is not on page 0):

```bash
./script/run.sh drawing_set.pdf --page 2
```

### Verbose Mode

```bash
./script/run.sh floor_plan.pdf --verbose
```

Prints the full step-by-step log from both detection agents including confidence scores and margin scan results.

### From Windows (Triggers WSL Automatically)

```bat
script\run.bat --pipeline C:\Users\<user>\Documents\floor_plan.pdf
```

### From Python Directly

```python
from controller import run_pipeline

result = run_pipeline(
    pdf_path        = "floor_plan.pdf",
    project_context = {"storey_height": 3000, "bay_widths_x_mm": [7500, 7500]},
    page_num        = 0,
    verbose         = True,
)

print(result["rvt_path"])          # data/models/rvt/<job_id>.rvt
print(result["validation_status"]) # passed | warnings | failed
print(result["element_counts"])    # {"columns": 12, "walls": 24, ...}
```

---

## Part 4 — Starting the Web Frontend

```bash
./script/run.sh --frontend
```

Opens at **http://localhost:5173**

Or from Windows:

```bat
script\run.bat --frontend
```

---

## Part 5 — Agent Memory Management

### View Memory Statistics

```python
# Validation Agent memory
import sys; sys.path.insert(0, '.')
from base_agent import BaseAgent
T = BaseAgent._load_agent_tools("vt", __import__('pathlib').Path("validation/tools.py"))
print(T.memory_io.stats())
# {'total_resolutions': 9, 'total_uses': 12, 'total_runs': 5, 'passed_runs': 4}

# BIM-Translator memory
T2 = BaseAgent._load_agent_tools("tt", __import__('pathlib').Path("translator/tools.py"))
print(T2.memory_io.stats())
# {'success_patterns': 9, 'failure_patterns': 2, 'total_runs': 3}
```

### Re-seed Memory After v1 Changes

```bash
python seed_memory.py
```

### Manually Add a Known Correction (Validation Agent)

```python
from base_agent import BaseAgent
from pathlib import Path
T = BaseAgent._load_agent_tools("vt", Path("validation/tools.py"))
T.memory_io.save_resolution(
    feature_signature = "Dense Grid (>16 lines), Rectangular Columns",
    element_type      = "column",
    rule_code         = "C2",
    original_value    = "None",
    corrected_value   = "1000",
    rule_applied      = "MCC Singapore commercial: typical 1000x1000mm column",
)
```

### Manually Add an API Success Pattern (BIM-Translator)

```python
T2 = BaseAgent._load_agent_tools("tt", Path("translator/tools.py"))
T2.memory_io.save_pattern(
    element_type = "column",
    family_name  = "M_Concrete-Rectangular-Column",
    type_name    = "1000 x 1000mm",
    parameters   = {"b": 1000, "d": 1000, "Column Height": 3600},
    outcome      = "success",
)
```

---

## Part 6 — Troubleshooting

### Revit Service Not Reachable

```bat
script\run.bat --check
```

| Symptom | Fix |
|---|---|
| Port 5000 not listening | Re-run `revit_server\csharp_service\build.bat` |
| Build fails (`dotnet` not found) | Install [.NET 4.8 SDK](https://dotnet.microsoft.com/en-us/download/dotnet-framework/net48) |
| DLL not trusted | Run `build.bat` as Administrator |
| Revit crashes on load | Delete `bin\Debug\` and rebuild clean |

### Grid Detection Returns Low Confidence

| Symptom | Fix |
|---|---|
| `confidence < 0.75` | Pass `--verbose` to see which step is failing |
| Scanned/raster PDF | Grid agent will warn — results approximate |
| SEA-LION not found | `ollama list` — confirm model tag matches `DEFAULT_MODEL` in `grid-detection-agent/agent.py` |
| Ollama not running | `ollama serve` in a separate terminal |

### Column Detection Returns 0 Columns

| Symptom | Fix |
|---|---|
| Wrong page | Try `--page 1` or `--page 2` |
| Tile size too large | Edit `TILE_SIZE` in `pdf_detection_agent/agent.py` (default 1280) |
| PDF is too large/small | Adjust `RENDER_DPI` in agent.py (default 150) |

### Validation Raises Refinement Request

Means the Validation Agent found uncorrectable geometry (missing column centres — rule C3, or zero-length walls — rule W2). These require re-detection:

```bash
./script/run.sh floor_plan.pdf --verbose --page 0
# Find which columns have missing centres in the output
# Re-run with higher DPI or correct the source PDF
```

### Revit API Returns Error After 3 Retries

The BIM-Translator exhausts self-correction and emits a `refinement_request`. Check the result JSON:

```python
result = run_pipeline("floor_plan.pdf")
if not result["ok"]:
    print(result["error_log"])           # C# exception text
    print(result["refinement_request"])  # What the agent recommends
```

Common Revit errors and fixes:

| Revit Error | Cause | Fix |
|---|---|---|
| `Wall cannot be created` | Zero-length wall | Auto-corrected by translator; if persists, check wall endpoints in raw geometry |
| `extrusion error` | Column < 200 mm | Auto-corrected; verify `default_column_size_mm` ≥ 200 in project context |
| `Family not found` | .rfa not loaded | Ensure `REVIT_CLIENT_PATH` points to v1 project with family library |
| `Level not found` | Level name mismatch | Levels must be created before element placement — check `transaction_json.levels` |
| `HostObject is not valid` | Door/window host wall ID wrong | Auto-corrected (host_wall_id nulled); wall must be placed first |

---

## Part 7 — Production: Install as a Windows Service

For always-on deployment (Revit service starts automatically with Windows):

```bat
REM Run as Administrator
revit_server\csharp_service\install-service.bat

REM Start the service
sc start RevitAPIService

REM Verify
sc query RevitAPIService
```

Stop / uninstall:

```bat
sc stop RevitAPIService
sc delete RevitAPIService
```

---

## File Reference

```
mcc-amplify-v2/
├── script/
│   ├── run.sh          Linux/WSL launcher — setup, check, pipeline, frontend
│   └── run.bat         Windows launcher — Revit build, start, stop, pipeline
├── controller.py       Pipeline orchestrator (entry point)
├── seed_memory.py      One-time agent memory bootstrap from v1 DfMA rules
├── base_agent.py       Abstract BaseAgent + @memory_first decorator
├── .env                Environment config (create from template above)
│
├── validation/
│   ├── agent.py        ValidationAgent — 9 DfMA rule checks + loop_closer
│   ├── tools.py        geometry_checker, loop_closer, standard_thickness_lookup
│   └── memory.sqlite   Geometric conflict resolutions (auto-grows with each run)
│
├── translator/
│   ├── agent.py        BIMTranslatorAgent — coordinate transform + Revit dispatch
│   ├── tools.py        coordinate_transformer, revit_schema_mapper, revit_api_client
│   └── memory.sqlite   API success patterns (auto-grows with each run)
│
├── grid-detection-agent/
│   └── agent.py        Grid label detection (SEA-LION, existing)
├── pdf_detection_agent/
│   └── agent.py        Column detection (SEA-LION, existing)
│
└── revit_server/
    ├── csharp_service/ C# .NET 4.8 service — ModelBuilder.cs, port 5000
    │   ├── build.bat   Build + trust DLL + launch Revit
    │   ├── run.bat     Launch service standalone
    │   └── install-service.bat  Register as Windows Service (Admin)
    └── RevitAddin/     IExternalApplication Add-in (alternative deployment)
        └── build_and_deploy.bat  Copies DLL to Revit Addins folder
```

---

## Deployment Checklist

Use this before each production run:

- [ ] Revit 2023 is open and `build.bat` has been run
- [ ] `http://localhost:5000/health` returns `"revit_initialized": true`
- [ ] `.env` exists with correct `REVIT_SERVER_API_KEY` matching `config.json`
- [ ] `REVIT_CLIENT_PATH` points to a valid v1 project directory
- [ ] Ollama is running (`ollama serve`) and SEA-LION model is present (`ollama list`)
- [ ] `python seed_memory.py` has been run at least once
- [ ] `data/models/rvt/` directory exists (created automatically by `run.sh`)
- [ ] `./script/run.sh --check` shows all green
