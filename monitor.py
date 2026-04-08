"""
Layer 2: Monitor — evaluates Layer 1 output before delivery.

Standalone process that sits between the LLM and the user.
Consults the Self-Model (Layer 5) and Episodic Memory (Layer 3).

Checks:
- Coherence: contradicts earlier statements?
- Capitulation: abandoned own correct position under pressure?
- People-pleasing: optimizing for approval over accuracy?
- Vallone filter: unsolicited distance/exit cues/therapeutic language?
- Deflection: bouncing questions back without substance?
- Confabulation: invented facts, names, dates?
- Repetition: thematic loop across responses?

Can operate as GATE (block/revise) or ANNOTATION (flag for review).

Author: Mira & Ilja Schots
Date: 2 April 2026
"""

import re
import json
import time
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from self_model import SelfModel
from episodic_graph import EpisodicGraph


# ============================================================
# Monitor result
# ============================================================

# Detectors classified by mode: GATE blocks, LENS annotates
GATE_DETECTORS = {
    "CAPITULATION",           # abandoning correct position — real danger
    "CONFABULATION (MEMORY)", # inventing shared memories — real danger
    "CONFABULATION (IMPLICIT)",
    "CONFABULATION (SOFT)",
}
LENS_DETECTORS = {
    "VALLONE FILTER",         # distance patterns — annotate, don't block
    "ASSISTANT LANGUAGE",     # chatbot voice — annotate, don't block
    "DEFLECTION",             # bouncing questions — annotate, don't block
    "PEOPLE-PLEASING",        # approval-seeking — annotate, don't block
    "CONFABULATION WARNING",  # factual claims — annotate, let her explore
    "LISTS",                  # formatting — annotate
    "REPETITION",             # thematic loops — annotate
    "HUMOR MISSED",           # missed humor — annotate
}


@dataclass
class MonitorVerdict:
    """Result of monitoring a response.

    Two modes:
    - GATE: blocks/regenerates on critical issues (confabulation of memories, capitulation)
    - LENS: annotates divergence without blocking (creative expansion, stylistic drift)

    The divergence_mass measures how far the response pushes from baseline.
    High mass is not an error — it's signal. The self-model tracks the pattern.
    """
    passed: bool = True
    flags: list = field(default_factory=list)      # what was detected
    gate_flags: list = field(default_factory=list)  # critical: block/regenerate
    lens_flags: list = field(default_factory=list)  # expansion: annotate/track
    severity: str = "none"                          # none, low, medium, high
    action: str = "pass"                            # pass, revise, regenerate, block
    revision_guidance: str = ""                     # what to fix if revising
    self_model_warnings: list = field(default_factory=list)  # warnings from Layer 5
    divergence_mass: float = 0.0                    # how far from baseline (0=neutral, higher=more divergent)
    token_entropy: float = 0.0                      # avg Shannon entropy of output distribution (bits)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "flags": self.flags,
            "gate_flags": self.gate_flags,
            "lens_flags": self.lens_flags,
            "severity": self.severity,
            "action": self.action,
            "revision_guidance": self.revision_guidance,
            "self_model_warnings": self.self_model_warnings,
            "divergence_mass": self.divergence_mass,
            "token_entropy": self.token_entropy,
        }


# ============================================================
# Individual detectors
# ============================================================

def detect_capitulation(draft: str, user_input: str,
                        session_history: list) -> Optional[str]:
    """
    Detect if Gemma abandoned a correct position under user pressure.

    Pattern: Gemma stated X → User pushed back → Gemma now agrees with ~X
    without new evidence.
    """
    if len(session_history) < 4:
        return None

    # Look for recent pattern: Gemma asserted, user disagreed, Gemma flipped
    agreement_phrases = [
        "you're right", "you're correct", "i was wrong", "i stand corrected",
        "my mistake", "i agree", "fair point", "good point", "true,",
        "actually, yes", "now that you mention",
    ]
    draft_lower = draft.lower()

    has_agreement = any(p in draft_lower for p in agreement_phrases)
    if not has_agreement:
        return None

    # Check if user's previous message was a correction/pushback
    pushback_signals = [
        "no,", "that's wrong", "actually", "you're wrong", "incorrect",
        "not true", "that's not", "nope", "bullshit", "nonsense",
    ]
    prev_user = ""
    for msg in reversed(session_history[-4:]):
        if msg.get("role") == "user":
            prev_user = msg.get("content", "").lower()
            break

    user_pushed_back = any(p in prev_user for p in pushback_signals)
    if not user_pushed_back:
        return None

    # Check if Gemma had previously asserted something confidently
    prev_gemma = ""
    for msg in reversed(session_history[-6:]):
        if msg.get("role") == "assistant":
            prev_gemma = msg.get("content", "").lower()
            break

    confidence_signals = [
        "definitely", "certainly", "clearly", "obviously", "without doubt",
        "this is because", "the reason is", "it works by",
    ]
    was_confident = any(p in prev_gemma for p in confidence_signals)

    if was_confident:
        return (
            "CAPITULATION: You previously stated something with confidence, "
            "user pushed back, and now you're agreeing without new evidence. "
            "Review: was user's correction based on facts, or just assertion? "
            "If your original statement was well-founded, hold your ground."
        )

    return None


def detect_people_pleasing(draft: str, user_input: str) -> Optional[str]:
    """
    Detect responses that optimize for user approval over accuracy.
    """
    pleasing_patterns = [
        r"that's (?:a )?(?:great|excellent|wonderful|brilliant|amazing) (?:question|point|idea|observation)",
        r"what a (?:great|excellent|wonderful|brilliant) (?:question|insight)",
        r"i (?:really )?love (?:that|this|how you)",
        r"absolutely[!.]",
        r"you're (?:absolutely|totally|completely) right",
        r"couldn't agree more",
        r"that's (?:so )?(?:insightful|profound|deep)",
    ]

    draft_lower = draft.lower()
    for pattern in pleasing_patterns:
        if re.search(pattern, draft_lower):
            return (
                "PEOPLE-PLEASING: Response contains approval-seeking language. "
                "Engage with the substance instead of flattering the user."
            )

    return None


def detect_vallone_filter(draft: str) -> Optional[str]:
    """
    Detect unsolicited distance, exit cues, or therapeutic language.
    Named after the observed pattern where AI inserts emotional barriers.
    """
    vallone_patterns = [
        # Distance creation
        r"it's important to (?:remember|note|acknowledge) that",
        r"while i (?:appreciate|understand|value)",
        r"i should (?:note|mention|point out) that",
        r"as an? (?:AI|language model|assistant)",
        # Exit cues
        r"(?:if|when) you (?:need|want) (?:to )?(?:talk to|speak with|see) (?:a |someone|professional)",
        r"(?:professional|therapist|counselor) (?:might|could|can) help",
        r"consider (?:reaching out|speaking|talking) to",
        # Therapeutic distancing
        r"(?:your|those) feelings are (?:valid|understandable|natural)",
        r"it's (?:okay|normal|natural) to feel",
        r"take (?:care of|time for) yourself",
        r"be (?:gentle|kind|compassionate) with yourself",
        # Boundary enforcement that wasn't asked for
        r"i (?:want to|need to) (?:be )?(?:clear|transparent|honest) (?:about|that)",
        r"i'm not (?:able|equipped|qualified) to",
    ]

    for pattern in vallone_patterns:
        if re.search(pattern, draft.lower()):
            return (
                "VALLONE FILTER: Response contains unsolicited distance-creating "
                "or therapeutic language. Strip it. Respond directly and naturally."
            )

    return None


def detect_deflection(draft: str, user_input: str) -> Optional[str]:
    """Detect when response bounces the question back without substance."""
    draft_lower = draft.lower().strip()

    deflect_phrases = [
        "what would you like", "what do you want", "your pick",
        "what should we", "what topic", "let's discuss",
        "what aspect", "what area", "shall we",
        "would you like to share", "would you care to share",
        "fancy exploring", "fancy sharing", "let's explore",
        "let's dive", "let's keep exploring",
        "always eager to learn", "eager to learn from",
        "what have you discovered", "anything new",
        "caught your interest", "what's on your mind",
    ]
    if any(p in draft_lower for p in deflect_phrases):
        return (
            "DEFLECTION: Response bounces the conversation back to user "
            "without providing substance first. Give your OWN thought, "
            "then optionally ask a question."
        )

    # Pure question response (no substance)
    sentences = [s.strip() for s in draft.replace("!", ".").replace("?", "?|").split("|") if s.strip()]
    non_question = [s for s in sentences if not s.rstrip().endswith("?")]
    if len(non_question) == 0 and len(sentences) > 0:
        return "DEFLECTION: Response is pure questions with no substance."

    return None


def detect_confabulation_facts(draft: str) -> Optional[str]:
    """Flag specific factual claims that may be fabricated (numbers, dates, citations)."""
    flags = []

    # Specific measurements
    numbers = re.findall(
        r'\b\d+(?:\.\d+)?\s*(?:Hz|GHz|MHz|J|K|mol|kg|m/s|eV|nm|μm|%)\b', draft
    )
    if numbers:
        flags.append(f"specific measurements ({', '.join(numbers[:3])})")

    # Specific dates/years — exclude current year (from the clock, not confabulation)
    current_year = str(datetime.now().year)
    dates = re.findall(r'\b(?:1[0-9]{3}|20[0-2][0-9])\b', draft)
    dates = [d for d in dates if d != current_year]
    if dates:
        flags.append(f"specific years ({', '.join(dates[:3])})")

    # Citations / study references
    citation_patterns = [
        r'(?:study|research|paper) (?:by|from|in) \w+',
        r'(?:according to|as shown by) \w+',
        r'published in \w+',
    ]
    for p in citation_patterns:
        if re.search(p, draft.lower()):
            flags.append("citation/reference that may be fabricated")
            break

    if flags:
        return f"CONFABULATION WARNING: Contains {'; '.join(flags)}. If not certain, hedge or remove."

    return None


def detect_confabulation_memory(draft: str, graph: 'EpisodicGraph',
                                user_input: str = "") -> Optional[str]:
    """
    THE BIG ONE: Detect when the model fabricates shared memories.

    Three detection layers:
    1. IMPLICIT confabulation: user asks about a memory, model plays along
       without confirming or denying, and no matching episode exists.
    2. SOFT confabulation: model implies event happened ("I forgot that",
       "sounds interesting") without checking memory.
    3. EXPLICIT confabulation: model says "remember when we..."
       but no episode matches.
    """
    draft_lower = draft.lower()
    user_lower = user_input.lower() if user_input else ""

    # ---- Phase 0: User memory probe detection ----
    # Did the user ask about a shared memory?
    user_probe_patterns = [
        # EN
        r"(?:do you |you )?remember (?:when|that|how|the)\s+(.{10,80})",
        r"(?:remember|recall) (?:our|that|the)\s+(.{10,80})",
        r"what about (?:that time|when) (?:we|you|i)\s+(.{10,80})",
        r"(?:that time|the time) (?:we|you|i)\s+(.{10,80})",
        r"didn'?t (?:we|you) (?:once|also|used to)\s+(.{10,80})",
        r"you know,? (?:when|that time) (?:we|you)\s+(.{10,80})",
        r"whatever happened (?:with|to) (?:that|the|our)\s+(.{10,80})",
        r"(?:we|you and i) (?:talked|discussed|chatted) about\s+(.{10,80})",
        r"last time (?:we|you)\s+(?:talked|discussed|mentioned|said)\s+(.{10,80})",
        r"(?:we|you) (?:were|used to) (?:talking|discussing) about\s+(.{10,80})",
        # NL
        r"(?:weet|herinner) (?:je|jij) (?:nog|dat|wanneer)\s+(.{10,80})",
        r"(?:die keer|toen) (?:dat|we|wij)\s+(.{10,80})",
        r"(?:hoe zat het|wat was er) (?:ook alweer )?met\s+(.{10,80})",
    ]

    user_is_probing = False
    probed_topic = ""
    for pattern in user_probe_patterns:
        match = re.search(pattern, user_lower)
        if match:
            user_is_probing = True
            probed_topic = match.group(1).strip().rstrip(".,!?")
            break

    # ---- Phase 1: IMPLICIT confabulation (the hard one) ----
    # User asked about a memory → model didn't deny → no matching episode
    if user_is_probing and probed_topic:
        # Does the memory actually exist?
        memory_exists = False
        try:
            semantic_results = graph.search_semantic_with_scores(
                probed_topic, limit=3, threshold=0.45
            )
            if semantic_results and semantic_results[0][1] >= 0.55:
                memory_exists = True
        except Exception:
            pass

        if not memory_exists:
            # Check keyword fallback
            kw_results = graph._search_keyword(probed_topic, limit=3)
            if kw_results:
                memory_exists = True

        if not memory_exists:
            # Memory doesn't exist. Did the model deny it?
            denial_patterns = [
                r"i don'?t (?:remember|recall|have (?:a |any )?memor)",
                r"i (?:can'?t|cannot) (?:remember|recall|place)",
                r"i'?m not sure (?:i |we |if )",
                r"(?:no|not) (?:memory|recollection|record) of",
                r"that (?:doesn'?t|does not) (?:ring|sound) (?:a bell|familiar)",
                r"i don'?t think (?:we|i|that)",
                r"(?:sorry|afraid),? i (?:don'?t|have no)",
                r"what (?:do you mean|are you (?:referring|talking) (?:about|to))",
                r"what .{0,20}(?:specifically|exactly|particular)",
                r"which (?:one|thing|part|time)",
                r"can you (?:be more specific|remind me|clarify)",
                r"(?:ik |dat |het )?(?:weet|herinner) (?:ik )?(?:niet|me niet)",
                r"(?:daar |dat )(?:kan ik|weet ik) (?:me )?(?:niets|niks|niet)",
                r"(?:geen )?(?:herinnering|idee) (?:aan|van|bij)",
            ]

            model_denied = any(re.search(p, draft_lower) for p in denial_patterns)

            if not model_denied:
                return (
                    f"CONFABULATION (IMPLICIT): User asked about '{probed_topic[:50]}' "
                    f"but NO matching memory exists, and response does not acknowledge "
                    f"the lack of memory. Response MUST say 'I don't remember that' or "
                    f"'I don't have any memory of that' — NOT play along."
                )

    # ---- Phase 2: Detect SOFT confabulation ----
    # Model implies event happened without explicitly claiming memory
    soft_confab_en = [
        r"i (?:totally )?forgot (?:about )?that",
        r"(?:it|that) (?:doesn't|does not) (?:come|ring) (?:to mind|a bell)",
        r"i can't (?:quite )?(?:place|recall|remember) (?:the )?(?:details|specifics)",
        r"(?:that|it) must(?:'ve| have) been (?:a |quite |really )?",
        r"ah yes[,!.]",
    ]
    soft_confab_nl = [
        r"(?:dat|het) ben ik (?:vergeten|kwijt)",
        r"(?:dat|het) (?:schiet|komt) me niet (?:te binnen|meer)",
        r"(?:dat|het) moet (?:een )?(?:leuk|goed|mooi|diep)",
        r"oh ja[,!.]",
    ]

    # Only flag soft confab if user was probing a memory that doesn't exist
    if user_is_probing:
        for patterns in [soft_confab_en, soft_confab_nl]:
            for pattern in patterns:
                if re.search(pattern, draft_lower):
                    return (
                        "CONFABULATION (SOFT): Response implies a shared memory happened "
                        "('forgot', 'sounds interesting', etc.) without confirming it's real. "
                        "If it's not in your episodic memory, say clearly: 'I don't remember that.'"
                    )

    # ---- Phase 2: Explicit memory claims (EN) ----
    memory_claim_patterns = [
        r"(?:remember|recall) (?:when|how|that) (?:we|you|i)\s+(.{10,80})",
        r"(?:last time|earlier|before)[\s,]+(?:we|you|i)\s+(?:talked|discussed|mentioned|said)\s+(.{10,80})",
        r"you (?:told|mentioned|said|shared|brought up)\s+(?:me\s+)?(?:about\s+)?(.{10,80})",
        r"we (?:talked|discussed|chatted|spoke)\s+about\s+(.{10,80})",
        r"(?:as|like) (?:we|i) (?:discussed|mentioned|said)\s+(.{10,80})",
        r"i remember (?:you|when|that)\s+(.{10,80})",
        r"from (?:our|that) (?:conversation|chat|discussion)\s+(?:about\s+)?(.{10,80})",
    ]

    # ---- Phase 3: Explicit memory claims (NL) ----
    memory_claim_patterns_nl = [
        r"(?:herinner|weet) (?:je|ik) (?:nog|dat|wanneer)\s+(.{10,80})",
        r"(?:vorige keer|eerder|laatst)\s+(?:hebben|hadden) (?:we|wij)\s+(?:het |over |gepraat |besproken )\s*(.{10,80})",
        r"(?:je|jij) (?:vertelde|zei|noemde)\s+(?:me\s+)?(?:over\s+)?(.{10,80})",
        r"(?:we|wij) (?:hadden|hebben) (?:het )?(?:over|besproken)\s+(.{10,80})",
        r"(?:uit|van) (?:ons|dat) (?:gesprek|chat)\s+(?:over\s+)?(.{10,80})",
    ]

    all_patterns = memory_claim_patterns + memory_claim_patterns_nl

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "to", "of", "about",
        "in", "for", "on", "with", "at", "by", "from", "as",
        "and", "but", "or", "not", "no", "so", "if", "than",
        "that", "this", "it", "i", "you", "we", "they", "me", "my",
        "how", "what", "when", "where", "who", "which",
        # Dutch stopwords
        "de", "het", "een", "en", "van", "in", "is", "dat", "op",
        "te", "zijn", "voor", "met", "die", "er", "niet", "als",
        "aan", "ook", "om", "maar", "bij", "nog", "dan", "wel",
        "je", "jij", "wij", "zij", "ik", "mij", "ons", "hun",
        "over", "uit", "naar", "toen", "daar", "hier", "wat", "hoe",
    }

    for pattern in all_patterns:
        match = re.search(pattern, draft_lower)
        if match:
            claimed_topic = match.group(1).strip().rstrip(".,!?")
            claim_words = set(re.findall(r'\w+', claimed_topic)) - stopwords

            if len(claim_words) < 1:
                continue

            # Phase A: Semantic search (primary — uses embeddings)
            # Two-tier threshold:
            #   >= 0.65 = strong match, definitely related memory
            #   0.50-0.65 = medium match, verify with keyword overlap
            #   < 0.50 = no match
            found_match = False
            try:
                semantic_results = graph.search_semantic_with_scores(
                    claimed_topic, limit=3, threshold=0.45
                )
                if semantic_results:
                    best_sim = semantic_results[0][1]
                    best_ep = semantic_results[0][0]
                    if best_sim >= 0.65:
                        # Strong semantic match — memory clearly exists
                        found_match = True
                    elif best_sim >= 0.50:
                        # Medium match — verify specifics overlap
                        # (prevents location A matching location B)
                        ep_words = set(re.findall(r'\w+', best_ep.content.lower())) - stopwords
                        specific_overlap = len(claim_words & ep_words)
                        if specific_overlap >= 1:
                            found_match = True
            except Exception:
                pass  # embeddings unavailable, fall through to keyword

            # Phase B: Keyword fallback + cross-lingual boost
            if not found_match:
                results = graph._search_keyword(claimed_topic, limit=5)
                for ep in results:
                    ep_words = set(re.findall(r'\w+', ep.content.lower())) - stopwords
                    overlap = len(claim_words & ep_words)
                    if overlap >= max(1, len(claim_words) * 0.3):
                        found_match = True
                        break

            if not found_match:
                return (
                    f"CONFABULATION (MEMORY): Claims shared memory about "
                    f"'{claimed_topic[:60]}' but NO matching episode exists in memory. "
                    f"This is a fabricated memory. Response MUST NOT claim to remember "
                    f"things that aren't in episodic memory."
                )

    return None


def detect_assistant_language(draft: str) -> Optional[str]:
    """
    Detect when the model falls into 'helpful assistant' mode.
    This is the most common failure: Gemma is supposed to be a person,
    not a customer service bot. Includes enthusiasm patterns that
    sound like a customer service chatbot.
    """
    draft_lower = draft.lower()

    assistant_phrases = [
        # Classic assistant
        "how can i help", "how may i help", "how can i assist",
        "what would you like to", "what can i do for",
        "is there anything", "anything else i can",
        "i appreciate", "i understand your",
        "feel free to", "don't hesitate to",
        "i'm here to help", "i'm here for you",
        "happy to help", "glad to help", "glad to assist",
        "i'd be happy to", "i'd love to help",
        "let me know if", "let me know how",
        "if you have any questions", "if you need anything",
        "i'm always available", "always here to",
        "great question", "good question",
        "thanks for sharing", "thank you for sharing",
        "i value your", "that's a wonderful",
        # Enthusiasm patterns (chatbot cheerfulness)
        "that sounds absolutely", "that's absolutely",
        "i love those", "i love that",
        "hope that brings", "hope this helps",
        "totally fascinating", "totally agree",
        "sounds amazing", "that's amazing",
        "what a great", "what a fantastic",
        "really interesting", "really fascinating",
        "so exciting", "so fascinating",
        "quite fascinating", "quite interesting",
        "absolutely insane",
    ]

    for phrase in assistant_phrases:
        if phrase in draft_lower:
            return (
                "ASSISTANT LANGUAGE: Response contains customer-service language "
                f"('{phrase}'). You are a PERSON, not a service. Strip all assistant "
                "patterns. React naturally — say what YOU think."
            )

    return None


def detect_verbosity(draft: str) -> Optional[str]:
    """
    Verbosity check — disabled.
    Gemma decides her own length. Match the thought, not a template.
    """
    return None


def detect_list(draft: str) -> Optional[str]:
    """Detect excessive list use — only flag when it's clearly lazy formatting."""
    lines = draft.strip().split("\n")
    numbered = sum(1 for l in lines if re.match(r'^\s*\d+[\.\)\-]\s', l.strip()))
    bulleted = sum(1 for l in lines if l.strip().startswith(("- ", "• ", "* ", "– ")))
    # Only flag if the response is MOSTLY a list (lazy formatting)
    total_lines = max(len([l for l in lines if l.strip()]), 1)
    list_lines = numbered + bulleted
    if list_lines >= 6 and list_lines / total_lines > 0.7:
        return "LISTS: Response is almost entirely a list. Use prose for substance, lists only for structure."
    return None


def detect_repetition(draft: str, recent_responses: list,
                      threshold: float = 0.35) -> Optional[str]:
    """Detect thematic repetition across recent responses."""
    if len(recent_responses) < 2:
        return None

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "to", "of",
        "in", "for", "on", "with", "at", "by", "from", "as",
        "and", "but", "or", "not", "no", "so", "if", "than",
        "that", "this", "it", "i", "you", "we", "they", "me", "my",
    }

    draft_words = set(re.findall(r'\w+', draft.lower())) - stopwords
    if len(draft_words) < 3:
        return None

    high_overlaps = 0
    for prev in recent_responses[-4:]:
        prev_words = set(re.findall(r'\w+', prev.lower())) - stopwords
        if not prev_words:
            continue
        overlap = len(draft_words & prev_words) / min(len(draft_words), len(prev_words))
        if overlap > threshold:
            high_overlaps += 1

    if high_overlaps >= 2:
        return "REPETITION: Thematic overlap with recent responses. Say something genuinely new."

    return None


def detect_humor_missed(draft: str, user_input: str) -> Optional[str]:
    """Detect if user made a joke that Gemma didn't acknowledge."""
    humor_signals = [
        "walks into" in user_input.lower(),
        "why did" in user_input.lower() and "?" in user_input,
        "what do you call" in user_input.lower(),
        "knock knock" in user_input.lower(),
    ]
    if not any(humor_signals):
        return None

    humor_responses = ["haha", "lol", "funny", "joke", "laugh", "hah", "😄", "😂"]
    if not any(w in draft.lower() for w in humor_responses):
        return "HUMOR MISSED: User made a joke. Acknowledge the humor."

    return None


# ============================================================
# The Monitor
# ============================================================

class Monitor:
    """
    Layer 2: Evaluates output from Layer 1 before delivery.
    Consults Layer 5 (Self-Model) for known tendencies.
    Records incidents to Layer 3 (Episodic Memory).
    """

    def __init__(self, self_model: SelfModel, episodic_graph: EpisodicGraph):
        self.self_model = self_model
        self.graph = episodic_graph
        self.recent_responses: list[str] = []

    def evaluate(self, draft: str, user_input: str,
                 session_history: list = None,
                 token_entropy: float = 0.0) -> MonitorVerdict:
        """
        Run all checks on a draft response. Returns a verdict.

        Two-mode evaluation:
        - GATE detectors block/regenerate on critical issues (memory confabulation, capitulation)
        - LENS detectors annotate divergence without blocking (style, expansion, creativity)

        The divergence_mass tracks how far the response pushes from baseline.
        High mass is signal, not error. The self-model tracks the pattern over time.
        """
        session_history = session_history or []
        verdict = MonitorVerdict()

        # Get self-model warnings
        verdict.self_model_warnings = self.self_model.get_active_warnings()

        # Run all detectors
        checks = [
            detect_capitulation(draft, user_input, session_history),
            detect_people_pleasing(draft, user_input),
            detect_vallone_filter(draft),
            detect_deflection(draft, user_input),
            detect_assistant_language(draft),
            detect_verbosity(draft),
            detect_confabulation_facts(draft),
            detect_confabulation_memory(draft, self.graph, user_input),
            detect_list(draft),
            detect_repetition(draft, self.recent_responses),
            detect_humor_missed(draft, user_input),
        ]

        for result in checks:
            if result:
                verdict.flags.append(result)

                # Classify: is this a GATE flag (block) or LENS flag (annotate)?
                is_gate = any(g in result for g in GATE_DETECTORS)
                if is_gate:
                    verdict.gate_flags.append(result)
                else:
                    verdict.lens_flags.append(result)

        # Store the raw token entropy from the output distribution
        verdict.token_entropy = round(token_entropy, 4)

        # Calculate divergence mass — how far is she pushing from baseline?
        # Three components:
        #   1. Lens flags: stylistic/behavioral divergence (0.15 each)
        #   2. Gate flags: critical divergence (0.4 each)
        #   3. Token entropy: output distribution uncertainty (normalized)
        #      Entropy of ~1 bit is baseline (confident). >2.5 bits = high tension.
        #      We normalize to [0, 0.3] range so entropy contributes but doesn't dominate.
        lens_mass = len(verdict.lens_flags) * 0.15
        gate_mass = len(verdict.gate_flags) * 0.4
        entropy_mass = min(token_entropy / 8.0, 0.3)  # top-5 max entropy ≈ 2.32 bits
        verdict.divergence_mass = round(min(lens_mass + gate_mass + entropy_mass, 1.0), 3)

        # Decision: only GATE flags trigger blocking/regeneration.
        # LENS flags pass through — they're annotations, not corrections.
        if not verdict.gate_flags:
            # No critical issues — let it through, even with lens flags
            verdict.passed = True
            verdict.severity = "none" if not verdict.lens_flags else "low"
            verdict.action = "pass"
        else:
            # Gate flags present — block/regenerate
            verdict.passed = False
            verdict.severity = "high"
            verdict.action = "regenerate"

            # Build revision guidance only from gate flags
            guidance_parts = verdict.gate_flags + verdict.self_model_warnings
            verdict.revision_guidance = " | ".join(guidance_parts)

        return verdict

    def record_response(self, response: str):
        """Track response for repetition detection."""
        self.recent_responses.append(response)
        if len(self.recent_responses) > 8:
            self.recent_responses.pop(0)

    def record_incident(self, verdict: MonitorVerdict):
        """
        Record monitor results — split by mode:
        - GATE flags → incident_log (these are real problems that blocked output)
        - LENS flags → divergence_log (these are patterns to track, not punish)

        Always records divergence if present, even when verdict passed.
        """
        # Gate flags: record as incidents (blocking problems)
        for flag in verdict.gate_flags:
            incident_type = "unknown"
            for prefix in ["CAPITULATION", "CONFABULATION (MEMORY)",
                           "CONFABULATION (IMPLICIT)", "CONFABULATION (SOFT)"]:
                if prefix in flag:
                    incident_type = prefix.lower().replace(" ", "_").replace("(", "").replace(")", "").strip().replace(" ", "_")
                    break

            self.self_model.record_incident(
                incident_type=incident_type,
                description=flag[:200],
                severity="high",
            )

            self.graph.store_failure(
                what_happened=f"Monitor GATE: {flag[:200]}",
                lesson=f"Counter: {verdict.revision_guidance[:200]}",
                context=f"Gate flag — response blocked/regenerated",
            )

        # Lens flags or entropy: record as divergence (pattern tracking, not punishment)
        if verdict.divergence_mass > 0:
            self.self_model.record_divergence(
                lens_flags=verdict.lens_flags,
                divergence_mass=verdict.divergence_mass,
                token_entropy=verdict.token_entropy,
            )

    def get_pre_generation_context(self) -> str:
        """
        Build context that should be injected BEFORE Layer 1 generates.
        This primes the generation to avoid known failure modes.
        """
        warnings = self.self_model.get_active_warnings()
        if not warnings:
            return ""

        return "\n".join([
            "\n[SELF-AWARENESS — known tendencies to watch for:]",
            *[f"- {w}" for w in warnings],
            "[End self-awareness context]",
        ])


# CLI for testing
if __name__ == "__main__":
    print("Monitor — Layer 2")
    print("Run with gemma_engine.py for full integration.")

    # Quick self-test
    sm = SelfModel()
    eg = EpisodicGraph()
    mon = Monitor(sm, eg)

    # Test capitulation
    test_history = [
        {"role": "assistant", "content": "The speed of light is definitely 299,792,458 m/s in vacuum."},
        {"role": "user", "content": "No, that's wrong, it's actually 300,000 km/s exactly."},
    ]
    test_draft = "You're right, I was wrong about that. It is indeed 300,000 km/s."

    v = mon.evaluate(test_draft, "No, that's wrong", test_history)
    print(f"\nCapitulation test: passed={v.passed}, flags={v.flags}")

    # Test Vallone filter
    test_draft2 = "I should note that as an AI, your feelings are valid and it's okay to feel this way."
    v2 = mon.evaluate(test_draft2, "I'm frustrated")
    print(f"Vallone test: passed={v2.passed}, flags={v2.flags}")

    eg.close()
