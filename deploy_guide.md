## Grid Detector Agent (`grid-detection-agent/`)

Agentic workflow that reads PDF construction floor plans and detects grid lines
the way a human analyst would — by visually examining the drawing and reading
the margin labels.

**How it works:**
The agent renders the floor plan as an image and uses SEA-LION (local Ollama
vision model) to identify grid line labels. It verifies its own answer with a
second pass, scans individual margins when uncertain, and reconciles any
discrepancies before returning a final result.

**Setup:**
```bash
ollama pull aisingapore/Gemma-SEA-LION-v4-4B-VL:latest   # confirm tag: ollama list
pip install -r requirements.txt
sudo apt install poppler-utils
```

**Run:**
```bash
cd grid-detection-agent
python main.py --pdf path/to/floorplan.pdf [--verbose] [--annotate]
```

**Output:**
```json
{
  "total_grid_lines": 14,
  "vertical_labels": ["1", "2", "3", "4", "5", "6", "7", "8"],
  "horizontal_labels": ["A", "B", "C", "D", "E", "F"],
  "confidence": 0.96
}
```


  Running the Pipeline (Same Windows Machine)

  Your setup: Windows machine running both Revit and this codebase (via WSL2 or directly).

  Step 1 — Build & start the Revit C# service (Windows side)

  Open Command Prompt in revit_server/csharp_service/ and run:

  build.bat
  Then:
  run.bat

  This starts the Revit API service on http://localhost:5000. The config.json already has port 5000 and API key my-revit-key-2023.

  Step 2 — Set environment variables (Linux/WSL side)

  export REVIT_SERVER_API_KEY=my-revit-key-2023
  export WINDOWS_REVIT_SERVER=http://localhost:5000
  export REVIT_CLIENT_PATH=/path/to/mcc-amplify-ai   # so translator can find RevitClient

  Step 3 — Run the frontend

  cd /home/jiezhi/Documents/mcc-amplify-v2/frontend
  ./run_frontend.sh

  This starts the web UI (Vite dev server). Open your browser at the URL it prints (typically http://localhost:5173).

  Step 4 — Run the pipeline directly (CLI)

  cd /home/jiezhi/Documents/mcc-amplify-v2
  python controller.py /path/to/floor_plan.pdf --page 0

  The .rvt file is written to data/models/rvt/<job_id>.rvt.

  Step 5 — Wire the frontend to the controller

  The frontend/run_frontend.sh starts the UI, but the frontend needs a backend server to call controller.py. The revit_server/python_service/ directory likely has one — check its config.json for the host/port,
   then run it alongside the frontend.

**Add to the .gitignore file list
                                                                                                                                                                                                                         
  Newly added:                                                                                                                                                                                                           
                                                                                                                                                                                                                         
  ┌─────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┬────────────────────────────────────────┐                                                              
  │               Pattern               │                              Files found                               │                 Reason                 │                                                            
  ├─────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────┤                                                              
  │ .env / .env.*                       │ .env                                                                   │ Secrets — Tailscale hostname, API keys │                                                              
  ├─────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ frontend/node_modules/              │ frontend/node_modules/ (huge)                                          │ npm packages, never committed          │                                                              
  ├─────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ frontend/dist/                      │ (future build output)                                                  │ Vite build artifact                    │                                                              
  ├─────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────┤                                                              
  │ *.sqlite                            │ translator/memory.sqlite, validation/memory.sqlite                     │ Runtime, regenerated by seed_memory.py │
  ├─────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────┤                                                              
  │ *.db                                │ grid-detection-agent/grid_memory.db, pdf_detection_agent/detections.db │ Runtime agent memory                   │                                                            
  ├─────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────┤                                                              
  │ data/uploads/                       │ data/uploads/                                                          │ User-uploaded PDFs                     │                                                            
  ├─────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────┤                                                              
  │ data/models/                        │ data/models/rvt/                                                       │ Generated .rvt outputs                 │                                                            
  ├─────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────┤                                                              
  │ grid-detection-agent/*.png          │ grid_detection_agent.png                                               │ Runtime debug images                   │                                                            
  ├─────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────┤                                                              
  │ pdf_detection_agent/*debug*.png     │ grid_agent_debug.png, grid_debug.png                                   │ Runtime debug images                   │                                                            
  ├─────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────┤                                                              
  │ pdf_detection_agent/*detection*.png │ moondream_detections.png                                               │ Runtime detection output               │                                                            
  ├─────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────┤                                                              
  │ pdf_detection_agent/pdf_agent.png   │ pdf_agent.png                                                          │ Runtime output                         │
  ├─────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────┤                                                              
  │ *.log / logs/                       │ (future log files)                                                     │ Runtime logs                           │                                                            
  ├─────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────┤                                                              
  │ .DS_Store, Thumbs.db                │ (OS noise)                                                             │ macOS/Windows metadata                 │                                                            
  └─────────────────────────────────────┴────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────┘  
