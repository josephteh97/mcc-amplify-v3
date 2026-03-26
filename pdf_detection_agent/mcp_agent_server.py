"""
mcp_agent_server.py — MCP socket server
=========================================
Thin server: registers agent.py functions as MCP tools and opens the socket.
All logic lives in agent.py.

Run:
    conda run -n yolo python mcp_agent_server.py

Claude Desktop config (~/.config/claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "pdf-column-detection": {
          "command": "conda",
          "args": ["run", "-n", "yolo", "python",
                   "/home/jiezhi/Documents/pdf_detection_agent/mcp_agent_server.py"]
        }
      }
    }
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP
import agent

mcp = FastMCP(
    name="pdf-column-detection",
    instructions="Detect structural columns in architectural floor plan PDFs and images.",
)


@mcp.tool()
def detect_columns(path: str, page_num: int = 0,
                   dpi: int = 150, save_image: str = "") -> str:
    """Detect structural columns in a PDF page or image file."""
    result = agent.detect_file(path, page_num=page_num, dpi=dpi,
                                save_image=save_image or None, verbose=False)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def detect_ground_truth() -> str:
    """Run detection on all ground-truth reference column images."""
    return json.dumps(agent.detect_ground_truth(), indent=2, default=str)


@mcp.tool()
def get_status() -> str:
    """Check whether Ollama is running and the vision model is available."""
    return json.dumps(agent.get_status(), indent=2)


@mcp.tool()
def memory_runs(limit: int = 20) -> str:
    """List recent detection runs."""
    return json.dumps(agent.memory_runs(limit), indent=2)


@mcp.tool()
def memory_search(file: str = "", shape: str = "",
                  min_confidence: float = 0.0, limit: int = 50) -> str:
    """Query stored column detections. All parameters are optional filters."""
    return json.dumps(agent.memory_search(
        file=file or None, shape=shape or None,
        min_confidence=min_confidence, limit=limit,
    ), indent=2)


@mcp.tool()
def memory_stats() -> str:
    """Aggregate statistics across all stored detections."""
    return json.dumps(agent.memory_stats(), indent=2)


@mcp.tool()
def memory_clear(run_id: str = "") -> str:
    """Delete a specific run (pass run_id) or wipe all memory (leave run_id empty)."""
    return json.dumps(agent.memory_clear(run_id=run_id or None), indent=2)


@mcp.tool()
def references_reload() -> str:
    """Reload ground-truth reference images after adding/removing files in GT_DIR."""
    return json.dumps({"loaded_shapes": agent.references_reload()}, indent=2)


@mcp.resource("skills://detect_columns")
def skill_detect_columns() -> str:
    return (Path(__file__).parent / "skills" / "detect_columns.md").read_text()



if __name__ == "__main__":
    mcp.run()
