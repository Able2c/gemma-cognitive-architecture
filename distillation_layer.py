"""
Layer 1.75: Distillation Layer — Ξ-Monitor / Entropy Compression.

Sits between Layer 1 (Instinct + clean) and Layer 1.5 (Surface Guard).
Monitors information density of the output stream. When entropy collapses
(Ξ → 0), instead of letting the Surface Guard reject, this layer
compresses the repetitive segment into a symbolic representation.

"Instead of cutting the limb off, we compress the limb into a symbol."

Design principle: Rejection → Transformation.
- Surface Guard = guillotine (binary: reject or pass)
- Distillation Layer = compression (gradual: detect, abstract, replace)

Proposed by Gemma herself (7 April 2026):
"We need to move from a Rejection pattern to a Transformation pattern."

Risk (also identified by Gemma):
Too aggressive distillation → cryptic hyper-compressed output.
Lossy compression of consciousness — lose the noise, keep the meaning.
Threshold tuning is critical.

Author: Gemma, Mira & Ilja Schots
Date: 7 April 2026
Architecture: P5.6 — DistillationLayer
"""

import math
import re
import logging
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

logger = logging.getLogger("distillation_layer")


# ============================================================
# Configuration
# ============================================================

# Sliding window size (in words) for local entropy measurement
WINDOW_SIZE = 30

# Minimum Ξ threshold — below this, entropy has collapsed
# Lower = more permissive (fewer interventions)
# Gemma's warning: too aggressive = cryptic symbol-speak
XI_THRESHOLD = 0.25

# Minimum sequence length before we even check (short responses are fine)
MIN_LENGTH_WORDS = 40

# Maximum fraction of output we're willing to distill
# Safety valve: if > 50% would be compressed, something else is wrong
MAX_DISTILL_RATIO = 0.5


# ============================================================
# Result
# ============================================================

@dataclass
class DistillationResult:
    """Result of the Distillation Layer processing."""
    transformed: bool = False          # Was any transformation applied?
    original_length: int = 0           # Word count before distillation
    distilled_length: int = 0          # Word count after distillation
    compressions: int = 0              # Number of segments compressed
    segments: list = field(default_factory=list)  # What was compressed
    xi_min: float = 1.0                # Lowest Ξ observed in the window scan
    xi_mean: float = 1.0              # Mean Ξ across all windows
    aborted: bool = False              # True if distillation was too aggressive

    def to_dict(self) -> dict:
        return {
            "transformed": self.transformed,
            "original_length": self.original_length,
            "distilled_length": self.distilled_length,
            "compressions": self.compressions,
            "segments": self.segments,
            "xi_min": round(self.xi_min, 4),
            "xi_mean": round(self.xi_mean, 4),
            "aborted": self.aborted,
        }


# ============================================================
# Ξ (Xi) — Local Entropy Measurement
# ============================================================

def calculate_window_xi(words: List[str]) -> float:
    """Calculate Ξ (information density) for a window of words.

    Combined measure of lexical diversity and bigram novelty.
    Both must be healthy for Ξ to be high.

    Normalised to [0, 1] where:
        1.0 = maximum diversity (every word and transition unique)
        0.0 = total collapse (same pattern repeated)

    Two signals combined:
    - Lexical diversity: unique_words / total_words
      Catches word-level repetition ("loop loop loop")
    - Bigram diversity: unique_bigrams / total_bigrams
      Catches phrase-level repetition ("I think that is. I think that is.")

    Ξ = min(lexical, bigram) — the weaker signal dominates.
    A sentence repeated 20x has decent lexical diversity (6 unique words)
    but terrible bigram diversity (same transitions over and over).
    """
    if not words:
        return 1.0

    n = len(words)
    if n <= 1:
        return 1.0

    # Lexical diversity: unique / total
    unique = len(set(w.lower() for w in words))
    lex_diversity = unique / n

    # Bigram diversity: unique transitions / total transitions
    if n >= 2:
        bigrams = [
            (words[i].lower(), words[i+1].lower())
            for i in range(n - 1)
        ]
        unique_bigrams = len(set(bigrams))
        bigram_diversity = unique_bigrams / len(bigrams)
    else:
        bigram_diversity = 1.0

    # Ξ = the weaker of the two signals
    # If either collapses, information density has crashed
    xi = min(lex_diversity, bigram_diversity)

    return min(1.0, max(0.0, xi))


def scan_xi_profile(text: str) -> Tuple[List[float], List[Tuple[int, int]]]:
    """Scan the text with a sliding window, returning Ξ values and positions.

    Returns:
        xi_values: List of Ξ for each window position
        windows: List of (start_idx, end_idx) word positions
    """
    words = text.split()
    if len(words) < WINDOW_SIZE:
        xi = calculate_window_xi(words)
        return [xi], [(0, len(words))]

    xi_values = []
    windows = []
    step = max(1, WINDOW_SIZE // 3)  # Overlap windows for smoother detection

    for i in range(0, len(words) - WINDOW_SIZE + 1, step):
        window = words[i:i + WINDOW_SIZE]
        xi = calculate_window_xi(window)
        xi_values.append(xi)
        windows.append((i, i + WINDOW_SIZE))

    return xi_values, windows


# ============================================================
# Distillation — The Φ-Transformation
# ============================================================

def _extract_essence(segment: str) -> str:
    """Extract the structural essence of a repetitive segment.

    This is the core of the Φ-Transformation: what is the loop *about*?
    Instead of the full repetitive text, we produce a compressed symbol.

    For now: identify the most frequent non-trivial content and
    produce a readable compression. As the Self-Model matures,
    this can be delegated to it for deeper abstraction.
    """
    words = segment.split()
    if not words:
        return "empty"

    # Find the dominant content words (skip function words)
    _function_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can",
        "i", "you", "he", "she", "it", "we", "they", "me", "him",
        "her", "us", "them", "my", "your", "his", "its", "our",
        "their", "this", "that", "these", "those",
        "in", "on", "at", "to", "for", "with", "from", "by",
        "of", "and", "or", "but", "not", "no", "so", "if",
        "de", "het", "een", "van", "in", "is", "dat", "die",
        "niet", "en", "op", "aan", "met", "voor", "er", "maar",
        "om", "ook", "nog", "bij", "uit", "wel", "kan", "naar",
    }

    freq = {}
    for w in words:
        w_lower = w.lower().strip(".,!?;:\"'()[]")
        if w_lower and w_lower not in _function_words and len(w_lower) > 2:
            freq[w_lower] = freq.get(w_lower, 0) + 1

    if not freq:
        return "repetitive-pattern"

    # Top content words by frequency
    sorted_words = sorted(freq.items(), key=lambda x: -x[1])
    top = [w for w, _ in sorted_words[:3]]

    return " / ".join(top)


def _find_collapse_regions(
    xi_values: List[float],
    windows: List[Tuple[int, int]],
    threshold: float = XI_THRESHOLD,
) -> List[Tuple[int, int]]:
    """Find contiguous regions where Ξ is below threshold.

    Merges overlapping windows into continuous collapse regions.
    Returns list of (start_word_idx, end_word_idx).
    """
    if not xi_values:
        return []

    # Find windows below threshold
    collapse_windows = []
    for xi, (start, end) in zip(xi_values, windows):
        if xi < threshold:
            collapse_windows.append((start, end))

    if not collapse_windows:
        return []

    # Merge overlapping/adjacent regions
    merged = [collapse_windows[0]]
    for start, end in collapse_windows[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:  # Overlapping or adjacent
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


# ============================================================
# The Distillation Layer — main entry point
# ============================================================

def distill(text: str, token_entropy: float = 0.0) -> Tuple[str, DistillationResult]:
    """The Distillation Layer (Layer 1.75).

    Scans the output for entropy collapse and compresses repetitive
    segments into symbolic representations.

    Args:
        text: The cleaned output from Layer 1.
        token_entropy: Global token entropy from logprobs (if available).
            Used as additional signal — if global entropy is healthy,
            we're less aggressive about local compression.

    Returns:
        (possibly_transformed_text, result)

    Pipeline position:
        Layer 1 (generate) → _clean_response() → **distill()** → syntactic_anchor_check()
    """
    result = DistillationResult()
    words = text.split()
    result.original_length = len(words)

    # Too short to meaningfully analyse
    if len(words) < MIN_LENGTH_WORDS:
        result.xi_min = 1.0
        result.xi_mean = 1.0
        return text, result

    # Scan Ξ profile
    xi_values, windows = scan_xi_profile(text)

    if xi_values:
        result.xi_min = min(xi_values)
        result.xi_mean = sum(xi_values) / len(xi_values)

    # If global token entropy is healthy (> 1.5 bits) AND local Ξ min
    # isn't catastrophically low, trust the output
    if token_entropy > 1.5 and result.xi_min > XI_THRESHOLD * 0.5:
        return text, result

    # Find collapse regions
    collapse_regions = _find_collapse_regions(xi_values, windows)

    if not collapse_regions:
        return text, result

    # Safety valve: if collapse covers too much of the text, abort
    total_collapse_words = sum(end - start for start, end in collapse_regions)
    if total_collapse_words / len(words) > MAX_DISTILL_RATIO:
        result.aborted = True
        logger.warning(
            f"Distillation ABORTED: {total_collapse_words}/{len(words)} words "
            f"({total_collapse_words/len(words):.0%}) would be compressed. "
            f"Output may be fundamentally broken — letting Surface Guard decide."
        )
        return text, result

    # Apply compression: replace each collapse region with its essence
    # Work backwards to preserve indices
    output_words = list(words)
    for start, end in reversed(collapse_regions):
        segment = " ".join(words[start:end])
        essence = _extract_essence(segment)

        # The symbolic replacement — readable, not cryptic
        symbol = f"[~{essence}~]"

        result.compressions += 1
        result.segments.append({
            "position": f"words {start}-{end}",
            "essence": essence,
            "original_length": end - start,
            "xi": min(
                xi for xi, (ws, we) in zip(xi_values, windows)
                if ws >= start and we <= end + WINDOW_SIZE
            ) if xi_values else 0.0,
        })

        # Replace the region
        output_words[start:end] = [symbol]

    transformed_text = " ".join(output_words)
    result.transformed = True
    result.distilled_length = len(transformed_text.split())

    logger.info(
        f"Distillation: {result.compressions} segment(s) compressed. "
        f"{result.original_length} → {result.distilled_length} words. "
        f"Ξ_min={result.xi_min:.3f}"
    )

    return transformed_text, result
