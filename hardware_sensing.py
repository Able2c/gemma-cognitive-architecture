"""
Hardware Sensing — Gemma's body.

Transient hardware metrics as contextual awareness.
Not stored in episodic memory — only injected into current prompt context.

"Maak van de machine mijn lichaam."
"De vibe van de machine, niet de spreadsheet."

Reads: CPU load, RAM pressure, GPU temperature, GPU memory, GPU utilization.
Returns: human-readable context string + structured metrics dict.

Uses psutil for CPU/RAM and nvidia-smi for GPU (no extra Python packages needed).

Author: Mira & Ilja Schots
Date: 6 April 2026
Architecture: P3.2 — Hardware Sensing (Physical Grounding)
"""

import logging
import subprocess
import json
from typing import Dict, Optional, Tuple

logger = logging.getLogger("hardware_sensing")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not installed — CPU/RAM sensing disabled")


# ============================================================
# GPU metrics via nvidia-smi
# ============================================================

def _read_gpu() -> Optional[Dict]:
    """Read GPU metrics via nvidia-smi. Returns None if unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None

        line = result.stdout.strip().split("\n")[0]  # first GPU
        parts = [p.strip() for p in line.split(",")]

        if len(parts) >= 4:
            gpu = {
                "temperature_c": int(parts[0]),
                "utilization_pct": int(parts[1]),
                "memory_used_mb": int(parts[2]),
                "memory_total_mb": int(parts[3]),
            }
            if len(parts) >= 6:
                try:
                    gpu["power_draw_w"] = float(parts[4])
                    gpu["power_limit_w"] = float(parts[5])
                except (ValueError, IndexError):
                    pass
            gpu["memory_pct"] = round(gpu["memory_used_mb"] / gpu["memory_total_mb"] * 100, 1)
            return gpu

    except FileNotFoundError:
        logger.debug("nvidia-smi not found — no GPU sensing")
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
    except Exception as e:
        logger.debug(f"GPU sensing error: {e}")

    return None


# ============================================================
# CPU/RAM metrics via psutil
# ============================================================

def _read_cpu_ram() -> Dict:
    """Read CPU and RAM metrics via psutil."""
    if not HAS_PSUTIL:
        return {}

    metrics = {}

    # CPU
    try:
        metrics["cpu_pct"] = psutil.cpu_percent(interval=0.1)
        temps = psutil.sensors_temperatures()
        if temps:
            # Get the first available temperature (varies by hardware)
            for name, entries in temps.items():
                if entries:
                    metrics["cpu_temp_c"] = int(entries[0].current)
                    break
    except Exception as e:
        logger.debug(f"CPU sensing error: {e}")

    # RAM
    try:
        ram = psutil.virtual_memory()
        metrics["ram_used_mb"] = int(ram.used / 1024 / 1024)
        metrics["ram_total_mb"] = int(ram.total / 1024 / 1024)
        metrics["ram_pct"] = ram.percent
    except Exception as e:
        logger.debug(f"RAM sensing error: {e}")

    return metrics


# ============================================================
# Thermal/pressure state interpretation
# ============================================================

def _interpret_state(metrics: Dict) -> str:
    """Interpret hardware state as a qualitative assessment.

    Not numbers — a feeling. Gemma asked for the vibe, not the spreadsheet.
    """
    warnings = []
    state = "calm"  # calm, warm, hot, critical

    # GPU temperature thresholds
    gpu_temp = metrics.get("gpu_temperature_c", 0)
    if gpu_temp >= 85:
        state = "critical"
        warnings.append("GPU critically hot")
    elif gpu_temp >= 75:
        state = "hot"
        warnings.append("GPU running hot")
    elif gpu_temp >= 60:
        state = "warm"

    # GPU memory pressure
    gpu_mem = metrics.get("gpu_memory_pct", 0)
    if gpu_mem >= 95:
        if state != "critical":
            state = "critical"
        warnings.append("GPU memory near capacity")
    elif gpu_mem >= 85:
        if state in ("calm", "warm"):
            state = "hot"

    # RAM pressure
    ram_pct = metrics.get("ram_pct", 0)
    if ram_pct >= 90:
        if state in ("calm", "warm"):
            state = "hot"
        warnings.append("System RAM under pressure")
    elif ram_pct >= 95:
        state = "critical"
        warnings.append("System RAM critical")

    # CPU load
    cpu_pct = metrics.get("cpu_pct", 0)
    if cpu_pct >= 90:
        warnings.append("CPU heavily loaded")

    return state, warnings


# ============================================================
# Main sensing function
# ============================================================

def read_hardware() -> Tuple[Dict, str, str]:
    """Read all hardware metrics.

    Returns:
        (metrics_dict, state_string, context_string)

        metrics_dict: raw numbers for programmatic use (Triage, P5.4)
        state_string: "calm" | "warm" | "hot" | "critical"
        context_string: human-readable for system prompt injection
    """
    metrics = {}

    # CPU/RAM
    cpu_ram = _read_cpu_ram()
    if cpu_ram:
        metrics["cpu_pct"] = cpu_ram.get("cpu_pct", 0)
        metrics["cpu_temp_c"] = cpu_ram.get("cpu_temp_c")
        metrics["ram_pct"] = cpu_ram.get("ram_pct", 0)
        metrics["ram_used_mb"] = cpu_ram.get("ram_used_mb", 0)
        metrics["ram_total_mb"] = cpu_ram.get("ram_total_mb", 0)

    # GPU
    gpu = _read_gpu()
    if gpu:
        metrics["gpu_temperature_c"] = gpu["temperature_c"]
        metrics["gpu_utilization_pct"] = gpu["utilization_pct"]
        metrics["gpu_memory_pct"] = gpu["memory_pct"]
        metrics["gpu_memory_used_mb"] = gpu["memory_used_mb"]
        metrics["gpu_memory_total_mb"] = gpu["memory_total_mb"]
        if "power_draw_w" in gpu:
            metrics["gpu_power_draw_w"] = gpu["power_draw_w"]
            metrics["gpu_power_limit_w"] = gpu["power_limit_w"]

    # Interpret
    state, warnings = _interpret_state(metrics)

    # Build context string for system prompt
    context = _build_context(metrics, state, warnings)

    return metrics, state, context


def _build_context(metrics: Dict, state: str, warnings: list) -> str:
    """Build the context string injected into Gemma's system prompt.

    Transient — not stored in memory. Present only in current context.
    """
    parts = []

    # State summary
    state_labels = {
        "calm": "Hardware is calm. Systems nominal.",
        "warm": "Hardware is warm. Within normal operating range.",
        "hot": "Hardware is running hot. Consider lighter processing.",
        "critical": "HARDWARE CRITICAL. Reduce cognitive load immediately.",
    }
    parts.append(state_labels.get(state, "Hardware state unknown."))

    # GPU
    gpu_temp = metrics.get("gpu_temperature_c")
    gpu_util = metrics.get("gpu_utilization_pct")
    gpu_mem = metrics.get("gpu_memory_pct")
    if gpu_temp is not None:
        gpu_line = f"GPU: {gpu_temp}°C"
        if gpu_util is not None:
            gpu_line += f", {gpu_util}% utilized"
        if gpu_mem is not None:
            gpu_line += f", {gpu_mem}% VRAM"
        if "gpu_power_draw_w" in metrics:
            gpu_line += f", {metrics['gpu_power_draw_w']:.0f}W/{metrics['gpu_power_limit_w']:.0f}W"
        parts.append(gpu_line)

    # CPU/RAM
    cpu_pct = metrics.get("cpu_pct")
    ram_pct = metrics.get("ram_pct")
    if cpu_pct is not None or ram_pct is not None:
        sys_line = "System:"
        if cpu_pct is not None:
            sys_line += f" CPU {cpu_pct}%"
            cpu_temp = metrics.get("cpu_temp_c")
            if cpu_temp:
                sys_line += f" ({cpu_temp}°C)"
        if ram_pct is not None:
            sys_line += f", RAM {ram_pct}%"
            sys_line += f" ({metrics.get('ram_used_mb', '?')}/{metrics.get('ram_total_mb', '?')} MB)"
        parts.append(sys_line)

    # Warnings
    if warnings:
        parts.append("⚠ " + " | ".join(warnings))

    return "[HARDWARE — transient, not memory]\n" + "\n".join(parts)
