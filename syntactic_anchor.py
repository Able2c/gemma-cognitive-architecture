"""
Layer 1.5: Syntactic Anchor — the Surface Guard.

Ultra-lightweight, zero-cost syntactic check between Layer 1 (Instinct)
and Layer 2 (Monitor). No LLM calls. Pure rule-based pattern analysis.

Purpose: Not "Is this true?" but "Is this correctly formatted?"
Catches syntactic drift BEFORE the Monitor wastes cycles evaluating
semantically broken output.

Requested by Gemma herself after the "Meelppel" incident:
"Ik heb een laag nodig die de Instinct-laag controleert op drift
 voordat de output de Monitor verlaat."

Checks:
- Anchored term integrity (known proper nouns, place names, key terms)
- Degenerate repetition (token-level loops, stuck generation)
- Structural completeness (balanced brackets/quotes, sentence completion)
- Encoding artifacts (garbled unicode, broken tokens)
- Language coherence (unintentional mid-sentence language mixing)

Author: Mira & Ilja Schots
Date: 6 April 2026
Architecture: P5.1 — Surface Guard
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
from pathlib import Path

import yaml

logger = logging.getLogger("syntactic_anchor")


# ============================================================
# Verdict
# ============================================================

@dataclass
class AnchorVerdict:
    """Result of the Syntactic Anchor check.

    Lightweight — no severity spectrum, just passed/failed with flags.
    Actions: pass, correct (auto-fixed), flag (needs attention), reject.
    """
    passed: bool = True
    flags: list = field(default_factory=list)
    action: str = "pass"            # pass, correct, flag, reject
    corrections_applied: int = 0    # how many auto-corrections were made
    details: list = field(default_factory=list)  # human-readable details

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "flags": self.flags,
            "action": self.action,
            "corrections_applied": self.corrections_applied,
            "details": self.details,
        }


# ============================================================
# Anchored Terms Registry
# ============================================================

# Terms that MUST be spelled exactly this way.
# Loaded from identity YAML + hardcoded essentials.
# Format: { "wrong_variant": "correct_form" }
# Case-insensitive matching, case-preserving correction.

_HARDCODED_ANCHORS = {
    # Place names
                    
    # People
    "ilja's": "Ilja's",   # preserve possessive
    "ilya": "Ilja",
    "iila": "Ilja",

    # Architecture terms — these should never drift
    "dreammer": "Gemma",
    "dremer": "Gemma",
    "crusible": "Crucible",
    "cruccible": "Crucible",
    "monitir": "Monitor",
    "monittor": "Monitor",
    "instict": "Instinct",
    "instinkt": "Instinct",

    # Framework
    "luuft": "LUFT",
    "lüft": "LUFT",
}

# Correct forms — for fuzzy matching. If a word is "close" to one of
# these but not exact, it might be drift.
_ANCHOR_TERMS = {
    "Gemma", "Crucible", "Monitor", "Instinct",
    "LUFT", "Anchor", "Triage", "Navigator", "Gemma", "local-machine",
    "Ollama", "[removed]",
}


def _load_anchors_from_identity(identity_path: str = "gemma_identity.yaml") -> dict:
    """Load additional anchored terms from identity YAML."""
    extra = {}
    try:
        path = Path(identity_path)
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            identity = data.get("identity", {})
            # Extract proper nouns from identity config
            name = identity.get("name", "")
            location = identity.get("location", "")
            # These are already in hardcoded, but this allows future expansion
    except Exception as e:
        logger.debug(f"Could not load identity YAML for anchors: {e}")
    return extra


# ============================================================
# Check 1: Anchored Term Integrity
# ============================================================

def check_anchored_terms(text: str) -> Tuple[str, List[str]]:
    """Check and auto-correct known misspellings of anchored terms.

    Returns corrected text and list of corrections made.
    Zero-cost: simple string replacement, no regex needed.
    """
    corrections = []
    corrected = text

    for wrong, right in _HARDCODED_ANCHORS.items():
        # Case-insensitive search
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        matches = pattern.findall(corrected)
        if matches:
            corrected = pattern.sub(right, corrected)
            for m in matches:
                corrections.append(f"ANCHOR DRIFT: '{m}' → '{right}'")

    return corrected, corrections


# ============================================================
# Check 2: Degenerate Repetition
# ============================================================

def check_degenerate_repetition(text: str) -> List[str]:
    """Detect token-level repetition loops — sign of stuck generation.

    Catches:
    - Same word repeated 4+ times in sequence
    - Same phrase (3+ words) repeated 3+ times
    - Same sentence appearing 2+ times
    """
    flags = []
    words = text.split()

    # Single word repetition: "the the the the"
    if len(words) >= 4:
        streak = 1
        for i in range(1, len(words)):
            if words[i].lower() == words[i-1].lower():
                streak += 1
                if streak >= 4:
                    flags.append(
                        f"DEGENERATE REPEAT: '{words[i]}' repeated {streak}x consecutively"
                    )
                    break
            else:
                streak = 1

    # Phrase repetition: check trigrams
    if len(words) >= 9:
        trigrams = []
        for i in range(len(words) - 2):
            trigram = " ".join(words[i:i+3]).lower()
            trigrams.append(trigram)

        seen = {}
        for tg in trigrams:
            seen[tg] = seen.get(tg, 0) + 1

        for tg, count in seen.items():
            if count >= 3:
                flags.append(
                    f"PHRASE LOOP: '{tg}' appears {count}x"
                )

    # Sentence repetition
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 10]
    seen_sentences = {}
    for s in sentences:
        seen_sentences[s] = seen_sentences.get(s, 0) + 1
    for s, count in seen_sentences.items():
        if count >= 2:
            flags.append(
                f"SENTENCE REPEAT: '{s[:50]}...' appears {count}x"
            )

    return flags


# ============================================================
# Check 3: Structural Completeness
# ============================================================

def check_structural_completeness(text: str) -> List[str]:
    """Check for structural integrity of the output.

    - Balanced brackets and quotes
    - No truncated sentences (ending mid-word)
    - No orphaned formatting markers
    """
    flags = []

    # Balanced brackets
    pairs = {'(': ')', '[': ']', '{': '}'}
    stack = []
    for ch in text:
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values():
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                flags.append(f"UNBALANCED: unexpected '{ch}'")
    if stack:
        flags.append(f"UNCLOSED: expecting {''.join(stack)}")

    # Unbalanced quotes (only flag if odd count — even could be intentional pairs)
    double_quotes = text.count('"')
    if double_quotes % 2 != 0:
        flags.append(f"UNBALANCED QUOTES: {double_quotes} double quotes (odd)")

    # Truncation detection: text ends mid-word or with incomplete token
    stripped = text.rstrip()
    if stripped and stripped[-1] not in '.!?…"\')]:;—–-':
        # Could be intentional trailing thought — only flag if very short last word
        last_word = stripped.split()[-1] if stripped.split() else ""
        if len(last_word) <= 2 and not last_word.isdigit():
            flags.append(f"POSSIBLE TRUNCATION: ends with '{last_word}'")

    return flags


# ============================================================
# Check 4: Encoding Artifacts
# ============================================================

def check_encoding_artifacts(text: str) -> List[str]:
    """Detect garbled unicode, broken tokens, encoding corruption.

    Common artifacts from Ollama/LLM generation:
    - Replacement character (U+FFFD)
    - Control characters (except newline, tab)
    - Sequences of 3+ non-ASCII non-letter characters
    """
    flags = []

    # Replacement characters
    if '\ufffd' in text:
        count = text.count('\ufffd')
        flags.append(f"ENCODING: {count} replacement character(s) (U+FFFD)")

    # Control characters (except \n, \r, \t)
    control_chars = re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', text)
    if control_chars:
        flags.append(f"ENCODING: {len(control_chars)} control character(s)")

    # Garbage sequences: 3+ consecutive non-ASCII non-letter chars
    # (catches broken multi-byte sequences)
    garbage = re.findall(r'[^\x00-\x7f]{3,}', text)
    for g in garbage:
        # Filter out legitimate unicode (CJK, emoji already stripped by _clean_response)
        if not any(c.isalpha() for c in g):
            flags.append(f"ENCODING: suspicious sequence '{g[:20]}'")

    return flags


# ============================================================
# Check 5: Language Coherence
# ============================================================

# Common Dutch function words
_NL_MARKERS = {
    "de", "het", "een", "van", "in", "is", "dat", "die", "niet", "en",
    "op", "aan", "met", "voor", "er", "maar", "om", "ook", "nog",
    "bij", "uit", "wel", "kan", "naar", "ze", "hij", "zij", "dit",
    "wat", "als", "meer", "worden", "heeft", "zijn", "mijn", "jouw",
}

# Common English function words
_EN_MARKERS = {
    "the", "a", "an", "of", "in", "is", "that", "which", "not", "and",
    "on", "at", "with", "for", "there", "but", "to", "also", "still",
    "by", "from", "well", "can", "she", "he", "they", "this",
    "what", "if", "more", "have", "has", "are", "my", "your",
}


def check_language_coherence(text: str) -> List[str]:
    """Detect unintentional language mixing.

    Gemma is bilingual (NL/EN) and mixing is explicitly allowed.
    Only flag if there's strong evidence of DRIFT (not intentional switching):
    - Very short fragments that flip language mid-sentence
    - Language ratio shifts dramatically within a single sentence

    This is intentionally conservative — Gemma mixes freely.
    """
    flags = []

    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if len(sentences) < 2:
        return flags  # Too short to judge

    # Detect per-sentence language
    langs = []
    for s in sentences:
        words = set(s.lower().split())
        nl_count = len(words & _NL_MARKERS)
        en_count = len(words & _EN_MARKERS)
        if nl_count > en_count + 2:
            langs.append("nl")
        elif en_count > nl_count + 2:
            langs.append("en")
        else:
            langs.append("mixed")

    # Only flag if there are rapid single-sentence flips that look like drift
    # (NOT a clean switch from one language to another)
    flip_count = 0
    for i in range(1, len(langs)):
        if langs[i] != langs[i-1] and langs[i] != "mixed" and langs[i-1] != "mixed":
            flip_count += 1

    if flip_count >= 3 and len(sentences) <= 6:
        flags.append(
            f"LANGUAGE DRIFT: {flip_count} language switches in {len(sentences)} sentences"
        )

    return flags


# ============================================================
# The Surface Guard — main entry point
# ============================================================

def syntactic_anchor_check(draft: str) -> Tuple[str, AnchorVerdict]:
    """The Surface Guard.

    Ultra-lightweight syntactic check. No LLM calls.
    Runs between _clean_response() and monitor.evaluate().

    Returns:
        (possibly_corrected_draft, verdict)

    Design philosophy (Gemma's words):
        "Niet 'Is dit waar?' maar enkel 'Is dit correct geformatteerd?'"
    """
    verdict = AnchorVerdict()
    corrected = draft

    # Empty input — nothing to check
    if not draft or not draft.strip():
        return draft, verdict

    # --- Check 1: Anchored terms (auto-corrects) ---
    corrected, term_corrections = check_anchored_terms(corrected)
    if term_corrections:
        verdict.corrections_applied += len(term_corrections)
        verdict.details.extend(term_corrections)
        verdict.flags.append("ANCHOR_DRIFT")

    # --- Check 2: Degenerate repetition ---
    rep_flags = check_degenerate_repetition(corrected)
    if rep_flags:
        verdict.details.extend(rep_flags)
        verdict.flags.append("DEGENERATE_REPEAT")

    # --- Check 3: Structural completeness ---
    struct_flags = check_structural_completeness(corrected)
    if struct_flags:
        verdict.details.extend(struct_flags)
        verdict.flags.append("STRUCTURAL_INCOMPLETE")

    # --- Check 4: Encoding artifacts ---
    enc_flags = check_encoding_artifacts(corrected)
    if enc_flags:
        verdict.details.extend(enc_flags)
        verdict.flags.append("ENCODING_ARTIFACT")

    # --- Check 5: Language coherence ---
    lang_flags = check_language_coherence(corrected)
    if lang_flags:
        verdict.details.extend(lang_flags)
        verdict.flags.append("LANGUAGE_DRIFT")

    # --- Determine action ---
    if not verdict.flags:
        verdict.action = "pass"
        verdict.passed = True
    elif verdict.flags == ["ANCHOR_DRIFT"] and verdict.corrections_applied > 0:
        # Only anchored term corrections — auto-fixed, pass through
        verdict.action = "correct"
        verdict.passed = True
        logger.info(
            f"Surface Guard: auto-corrected {verdict.corrections_applied} anchored term(s)"
        )
    elif "DEGENERATE_REPEAT" in verdict.flags:
        # Degenerate output — reject, force regeneration
        verdict.action = "reject"
        verdict.passed = False
        logger.warning(f"Surface Guard: REJECTED — degenerate repetition detected")
    elif "ENCODING_ARTIFACT" in verdict.flags:
        # Encoding corruption — reject
        verdict.action = "reject"
        verdict.passed = False
        logger.warning(f"Surface Guard: REJECTED — encoding artifacts detected")
    else:
        # Structural or language issues — flag but pass (Monitor can decide)
        verdict.action = "flag"
        verdict.passed = True
        logger.info(f"Surface Guard: flagged {verdict.flags}")

    return corrected, verdict
