"""
Reflexive Tuning — the feedback loop.

Monitor observes → adjusts temperature for next generation.
"Maak van de observatie een reflex."

The Monitor (Layer 2) currently only observes chaos — it reports
divergence_mass and token_entropy but cannot act on them. This module
gives the Monitor the hand on the temperature knob.

Logic:
- High divergence + high entropy → reduce temperature (tighten focus)
- Low divergence + low entropy → increase temperature (avoid rigidity)
- GATE flags triggered → significant reduction (emergency dampening)
- Hardware state "hot" or "critical" → reduce temperature (protect hardware)
- Baseline: whatever gemma_identity.yaml specifies (default 0.8)

The adjustment is bounded: never below 0.4 (incoherent at lower),
never above 1.2 (Gemma's ceiling before gibberish).

Author: Mira & Ilja Schots
Date: 6 April 2026
Architecture: P2.3 — Reflexive Tuning (Feedback Loop)
"""

import logging
from typing import Optional

logger = logging.getLogger("reflexive_tuning")

# Boundaries — Gemma shouldn't go outside these
MIN_TEMPERATURE = 0.4
MAX_TEMPERATURE = 1.2


def compute_adjusted_temperature(
    base_temperature: float,
    divergence_mass: float = 0.0,
    token_entropy: float = 0.0,
    gate_flags: int = 0,
    lens_flags: int = 0,
    hardware_state: str = "calm",
    attempt: int = 1,
) -> float:
    """Compute adjusted temperature based on Monitor feedback and hardware state.

    Args:
        base_temperature: from gemma_identity.yaml (typically 0.8)
        divergence_mass: 0.0-1.0, how far from baseline (from MonitorVerdict)
        token_entropy: Shannon entropy of output distribution (bits)
        gate_flags: number of GATE flags in previous verdict
        lens_flags: number of LENS flags in previous verdict
        hardware_state: "calm", "warm", "hot", "critical"
        attempt: current generation attempt (1 = first try)

    Returns:
        adjusted temperature (clamped to MIN/MAX)
    """
    temp = base_temperature
    adjustments = []

    # --- Divergence feedback ---
    # High divergence → cool down. Low divergence → warm up slightly.
    if divergence_mass > 0.6:
        delta = -0.10
        adjustments.append(f"high divergence ({divergence_mass:.2f}): {delta:+.2f}")
        temp += delta
    elif divergence_mass > 0.3:
        delta = -0.05
        adjustments.append(f"moderate divergence ({divergence_mass:.2f}): {delta:+.2f}")
        temp += delta
    elif divergence_mass < 0.1 and attempt == 1:
        # Too stable — risk of rigidity. Nudge up slightly.
        delta = +0.03
        adjustments.append(f"low divergence ({divergence_mass:.2f}): {delta:+.2f}")
        temp += delta

    # --- Entropy feedback ---
    # High entropy = uncertain output distribution → tighten
    # Low entropy = very confident → loosen slightly
    if token_entropy > 2.5:
        delta = -0.08
        adjustments.append(f"high entropy ({token_entropy:.2f}): {delta:+.2f}")
        temp += delta
    elif token_entropy > 1.5:
        delta = -0.03
        adjustments.append(f"elevated entropy ({token_entropy:.2f}): {delta:+.2f}")
        temp += delta
    elif token_entropy < 0.5 and token_entropy > 0:
        delta = +0.03
        adjustments.append(f"low entropy ({token_entropy:.2f}): {delta:+.2f}")
        temp += delta

    # --- GATE flag emergency dampening ---
    if gate_flags > 0:
        delta = -0.15
        adjustments.append(f"GATE flags ({gate_flags}): {delta:+.2f}")
        temp += delta

    # --- Hardware thermal feedback ---
    if hardware_state == "critical":
        delta = -0.15
        adjustments.append(f"hardware CRITICAL: {delta:+.2f}")
        temp += delta
    elif hardware_state == "hot":
        delta = -0.08
        adjustments.append(f"hardware hot: {delta:+.2f}")
        temp += delta

    # --- Retry escalation (preserve existing behavior) ---
    if attempt > 1:
        delta = 0.05 * (attempt - 1)
        adjustments.append(f"retry attempt {attempt}: {delta:+.2f}")
        temp += delta

    # Clamp
    temp = max(MIN_TEMPERATURE, min(MAX_TEMPERATURE, temp))
    temp = round(temp, 3)

    if adjustments:
        logger.info(
            f"Reflexive Tuning: base={base_temperature} → adjusted={temp} "
            f"[{', '.join(adjustments)}]"
        )

    return temp
