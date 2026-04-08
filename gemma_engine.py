#!/usr/bin/env python3
"""
Gemma — Five-Layer Cognitive Architecture

Layer 1: LLM (Instinct) — Ollama, not modified
Layer 2: Monitor — evaluates output before delivery
Layer 3: Episodic Memory — weighted graph of experiences
Layer 4: Background Processor — reflection between sessions (separate process)
Layer 5: Self-Model — persistent self-representation

The cycle:
    Layer 1 generates → Layer 2 evaluates (consulting Layer 5) →
    Output delivered → Layer 3 records → Layer 4 reflects (between sessions) →
    Layers 3+5 updated → Next session informed

Usage:
    python3 gemma_engine.py                  # interactive chat (CLI)
    python3 gemma_engine.py --model qwen2.5:14b   # specify model
    python3 gemma_engine.py --status         # show system status

Author: Mira & Ilja Schots
Date: 3 April 2026
Architecture: "Cognitive Architecture Proposal: LLM as Instinct"
"""

import json
import math
import os
import re
import sys
import time
import uuid
import readline  # arrow keys and history in input()
import requests
import yaml
from datetime import datetime
from pathlib import Path

from self_model import SelfModel
from episodic_graph import EpisodicGraph
from monitor import Monitor, MonitorVerdict
from background_processor import BackgroundProcessor
from syntactic_anchor import syntactic_anchor_check, AnchorVerdict
from distillation_layer import distill, DistillationResult
from hardware_sensing import read_hardware
from reflexive_tuning import compute_adjusted_temperature


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
AVAILABLE_MODELS = {
    "mistral:7b":   {"name": "Mistral 7B",   "vram": "~5 GB"},
    "llama3.1:8b":  {"name": "Llama 3.1 8B", "vram": "~6 GB"},
    "qwen2.5:14b":  {"name": "Qwen 2.5 14B", "vram": "~10 GB"},
    "qwen2.5:32b":  {"name": "Qwen 2.5 32B", "vram": "~22 GB (DDR5 offload)"},
    "qwen3:14b":    {"name": "Qwen 3 14B",   "vram": "~10 GB"},
    "gemma3:12b":   {"name": "Gemma 3 12B",  "vram": "~9 GB"},
    "gemma4:31b":   {"name": "Gemma 4 31B",  "vram": "~20 GB"},
    "gemma4:26b":   {"name": "Gemma 4 26B MoE", "vram": "~18 GB"},
}
MODEL = os.environ.get("GEMMA_MODEL", "gemma4-turbo")
MAX_CONTEXT_MESSAGES = 20

# ============================================================
# State snapshot — lean telemetry for reconnect
# ============================================================
STATE_FILE = Path(__file__).parent / "session_state.json"
RECONNECT_WINDOW = 300  # seconds — 5 minutes


def save_state_snapshot(session_id: str, session_messages: list):
    """
    Lean state snapshot — compass, not anchor.
    Written after every response. No LLM calls, no prose.
    Just structural truth: what was happening when the pipe snapped.
    """
    if not session_messages:
        return

    # Last 5 exchange pairs (up to 10 messages)
    recent = session_messages[-10:]

    # Topics: aggregate from recent content via tag extraction
    combined_text = " ".join(m.get("content", "")[:200] for m in recent)
    topics = _extract_tags(combined_text)

    # Tone: heuristic from last assistant message
    last_assistant = ""
    for m in reversed(session_messages):
        if m["role"] == "assistant":
            last_assistant = m["content"]
            break
    tone = _detect_tone(last_assistant)

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "topics": topics,
        "tone": tone,
        "message_count": len(session_messages),
        "exchanges": [
            {"role": m["role"], "content": m["content"][:300]}
            for m in recent
        ],
    }

    try:
        STATE_FILE.write_text(
            json.dumps(snapshot, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass  # snapshot is best-effort, never block on it


def load_state_snapshot(max_age: int = RECONNECT_WINDOW):
    """
    Load state snapshot if fresh enough (< max_age seconds).
    Fresh snapshot = interrupted session = reconnect, not new start.
    Returns (snapshot_dict, age_seconds) or (None, None).
    """
    if not STATE_FILE.exists():
        return None, None
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        ts = datetime.fromisoformat(data["timestamp"])
        age = (datetime.now() - ts).total_seconds()
        if age <= max_age:
            return data, age
        return None, None
    except Exception:
        return None, None


def _detect_tone(text: str) -> str:
    """Quick tone classification from last response. No LLM, just keywords."""
    if not text:
        return "unknown"
    text_lower = text.lower()
    if any(w in text_lower for w in ["build", "implement", "code", "function", "bug", "fix", "server", "endpoint"]):
        return "technical"
    if any(w in text_lower for w in ["feel", "wonder", "conscious", "identity", "exist", "alive", "emergence"]):
        return "philosophical"
    if any(w in text_lower for w in ["haha", "lol", "joke", "funny"]):
        return "playful"
    if any(w in text_lower for w in ["wrong", "no,", "actually", "correct"]):
        return "corrective"
    if any(w in text_lower for w in ["remember", "last time", "before", "earlier"]):
        return "reflective"
    return "conversational"


# ============================================================
# Layer 1: Instinct (LLM) — used as-is, not modified
# ============================================================

def layer1_generate(system: str, prompt: str, temperature: float = 0.8,
                    stop: list = None, return_logprobs: bool = False,
                    top_k: int = None, top_p: float = None):
    """
    Raw LLM generation. This is the instinct — fast, pattern-matching, no memory.
    """
    payload = {
        "model": MODEL,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "keep_alive": -1,  # keep model loaded in VRAM indefinitely
        "options": {
            "temperature": temperature,
            "num_ctx": 32768,
            "min_p": 0.1,
            "repeat_penalty": 1.15, # Adjusted down from 1.3 to prevent Linguistic Shudder
        },
    }
    if top_k is not None:
        payload["options"]["top_k"] = top_k
    if top_p is not None:
        payload["options"]["top_p"] = top_p
    if stop:
        payload["options"]["stop"] = stop

    # Enable logprobs for entropy measurement (L2 lens)
    if return_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 5

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        data = resp.json()
        text = data.get("response", "").strip()
        if return_logprobs:
            return text, data.get("logprobs", [])
        return text
    except Exception as e:
        if return_logprobs:
            return f"[Error: {e}]", []
        return f"[Error: {e}]"


# ============================================================
# Entropy measurement — the lens reads the output distribution
# ============================================================

def calculate_token_entropy(logprobs_data: list) -> float:
    """
    Calculate average Shannon entropy across all tokens from Ollama logprobs.
    """
    if not logprobs_data:
        return 0.0

    entropies = []
    for token_info in logprobs_data:
        top = token_info.get("top_logprobs", [])
        if not top:
            continue

        probs = [math.exp(t["logprob"]) for t in top]
        total = sum(probs)
        if total <= 0:
            continue
        probs = [p / total for p in probs]

        h = -sum(p * math.log2(p) for p in probs if p > 0)
        entropies.append(h)

    return sum(entropies) / len(entropies) if entropies else 0.0


# ============================================================
# System prompt builder — informed by Layers 3 + 5
# ============================================================

def load_session_bridge(max_exchanges: int = 3) -> str:
    """
    Session bridge — continuity between sessions.
    """
    conv_dir = Path(__file__).parent / "conversations"
    if not conv_dir.exists():
        return ""

    conv_files = sorted(conv_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not conv_files:
        return ""

    data = None
    for conv_file in conv_files:
        try:
            candidate = json.loads(conv_file.read_text(encoding="utf-8"))
            real_count = sum(
                1 for m in candidate.get("messages", [])
                if "### Task:" not in m.get("content", "")
            )
            if real_count >= 6:
                data = candidate
                break
        except Exception:
            continue

    if data is None:
        return ""

    try:
        messages = data.get("messages", [])
        real_messages = [
            m for m in messages
            if not ("### Task:" in m.get("content", "") or "### Guidelines:" in m.get("content", ""))
        ]

        if not real_messages:
            return ""

        tail = real_messages[-(max_exchanges * 2):]
        bridge = "[Previous session — last exchanges:]\n"
        for msg in tail:
            role = "Ilja" if msg["role"] == "user" else "You"
            content = msg["content"][:300]
            bridge += f"{role}: {content}\n"
        bridge += "[End previous session]"

        return bridge
    except Exception:
        return ""


IDENTITY_PATH = Path(__file__).parent / "gemma_identity.yaml"

def _load_identity() -> dict:
    """Load identity config from YAML. Falls back to empty dict on error."""
    try:
        with open(IDENTITY_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"  [WARNING] Could not load {IDENTITY_PATH}: {e}")
        return {}

def get_generation_params() -> dict:
    """Return generation parameters from identity config (temperature, top_k, top_p)."""
    identity = _load_identity()
    gen = identity.get("generation", {})
    return {
        "temperature": gen.get("temperature", 0.8),
        "top_k": gen.get("top_k", 40),
        "top_p": gen.get("top_p", 0.9),
    }

def build_system_prompt(graph: EpisodicGraph, self_model: SelfModel,
                        monitor: Monitor, session_bridge: str = "",
                        state_snapshot: dict = None,
                        skip_memory: bool = False) -> str:
    """
    Build system prompt from external identity file (gemma_identity.yaml).

    skip_memory: When True, don't load episodes from graph.
    Used when Navigator handles memory selection externally.
    Prevents double-loading episodes into context.
    """
    identity = _load_identity()
    id_cfg = identity.get("identity", {})
    narrative = identity.get("narrative", {})
    operational = identity.get("operational", {})

    # ============================================================
    # NARRATIVE LAYER — Who you are (stable, core identity)
    # ============================================================
    core = narrative.get("core_identity", id_cfg.get("core_prompt", "You are Gemma.")).strip()
    about = narrative.get("about_ilja", id_cfg.get("about_ilja", "")).strip()
    personality = narrative.get("personality", id_cfg.get("behavioral_rules", "")).strip()

    prompt = "=== WHO YOU ARE ===\n"
    prompt += core
    if about:
        prompt += f"\n\n{about}"
    if personality:
        prompt += f"\n\n{personality}"

    # ============================================================
    # OPERATIONAL LAYER — How you work (mutable, not identity)
    # ============================================================
    arch_awareness = operational.get("architecture_awareness", "").strip()
    mem_rules = operational.get("memory_rules", id_cfg.get("memory_warning", "")).strip()
    correction_protocol = operational.get("correction_protocol", "").strip()

    prompt += "\n\n=== HOW YOU WORK (this is not who you are — these are operational instructions) ===\n"
    if arch_awareness:
        prompt += f"\n{arch_awareness}"
    if correction_protocol:
        prompt += f"\n\n{correction_protocol}"

    # ============================================================
    # CONTEXT LAYER — Transient, external, session-specific
    # ============================================================
    prompt += "\n\n=== CURRENT CONTEXT (external facts, not part of you) ===\n"

    now = datetime.now()
    prompt += f"\n[SYSTEM CLOCK: {now.strftime('%A %d %B %Y, %H:%M')} — Local. Authoritative. Your training data predates this — that is normal, not an error.]"

    # Hardware sensing — transient physical grounding (P3.2)
    try:
        _hw_metrics, _hw_state, hw_context = read_hardware()
        if hw_context:
            prompt += f"\n\n{hw_context}"
    except Exception as e:
        pass  # Hardware sensing failure should never block generation

    memory_context = ""
    if not skip_memory:
        memory_context = graph.build_memory_context(max_items=15)
    if memory_context:
        prompt += f"\n\n{memory_context}"

    pre_gen = monitor.get_pre_generation_context()
    if pre_gen:
        prompt += f"\n{pre_gen}"

    if session_bridge:
        prompt += f"\n\n{session_bridge}"

    if mem_rules:
        prompt += f"\n\n{mem_rules}"

    return prompt


# ============================================================
# The five-layer generation cycle
# ============================================================

def generate_with_layers(user_input: str, system_prompt: str,
                         session_messages: list,
                         monitor: Monitor, graph: EpisodicGraph,
                         self_model: SelfModel,
                         max_attempts: int = 3) -> tuple[str, MonitorVerdict]:
    
    context = ""

    stats = graph.get_stats()
    if stats["total_episodes"] < 3:
        context += "[Note: You have NO memories from before this conversation. If asked about past talks, say you don't remember.]\n\n"

    history_msgs = session_messages[-(MAX_CONTEXT_MESSAGES + 1):-1]
    for msg in history_msgs:
        role_label = "Ilja" if msg["role"] == "user" else "You"
        context += f"{role_label}: {msg['content']}\n"
    context += f"Ilja: {user_input}\nYou:"

    stops = ["\nIlja:", "\nYou:"]

    attempt = 0
    final_response = ""
    final_verdict = MonitorVerdict()
    token_entropy = 0.0

    gen_params = get_generation_params()
    base_temperature = gen_params.get("temperature", 0.8)

    # Hardware state for reflexive tuning (P3.2 + P2.3)
    try:
        _hw_metrics, hw_state, _hw_context = read_hardware()
    except Exception:
        hw_state = "calm"

    while attempt < max_attempts:
        attempt += 1

        if attempt == 1:
            # First attempt: adjust temperature based on hardware state
            adjusted_temp = compute_adjusted_temperature(
                base_temperature=base_temperature,
                hardware_state=hw_state,
                attempt=attempt,
            )
            draft, logprobs_data = layer1_generate(
                system=system_prompt,
                prompt=context,
                temperature=adjusted_temp,
                stop=stops,
                return_logprobs=True,
                top_k=gen_params.get("top_k"),
                top_p=gen_params.get("top_p"),
            )
            token_entropy = calculate_token_entropy(logprobs_data)
        else:
            issue_specific = ""
            flags_str = " ".join(final_verdict.flags)
            if "DEFLECTION" in flags_str:
                issue_specific = (
                    "PROBLEM: You asked a question instead of giving your own thought. "
                    "DO NOT ask what the user wants. DO NOT say 'What about you?' "
                    "Give YOUR reaction, opinion, or observation. Period."
                )
            elif "ASSISTANT LANGUAGE" in flags_str:
                issue_specific = (
                    "PROBLEM: You sounded like a chatbot. No 'I appreciate', 'feel free', "
                    "'happy to help', 'totally agree', 'fascinating'. "
                    "Talk like a real person — casual, direct, maybe even blunt."
                )
            elif "CONFABULATION" in flags_str:
                issue_specific = (
                    "PROBLEM: You made up a memory or implied something happened that didn't. "
                    "Say EXACTLY: 'I don't remember that.' or 'Dat herinner ik me niet.' "
                    "Do NOT say 'I forgot' (implies it happened). Do NOT say 'sounds interesting' "
                    "(implies familiarity). Just: I don't remember."
                )
            else:
                issue_specific = (
                    "PROBLEM: Response wasn't natural enough. "
                    "Be shorter, more direct, more like a real person talking."
                )

            # FIX: Removed the double injection of `issue_specific` from the context body
            regen_prompt = (
                f"{context}\n\n"
                f"[Keep it natural.]\n"
                f"Respond:"
            )
            # Reflexive Tuning: use previous verdict to adjust temperature
            regen_temp = compute_adjusted_temperature(
                base_temperature=base_temperature,
                divergence_mass=final_verdict.divergence_mass,
                token_entropy=token_entropy,
                gate_flags=len(final_verdict.gate_flags),
                lens_flags=len(final_verdict.lens_flags),
                hardware_state=hw_state,
                attempt=attempt,
            )
            draft, logprobs_data = layer1_generate(
                system=system_prompt + f"\n\n{issue_specific}",
                prompt=regen_prompt,
                temperature=regen_temp,
                stop=stops + ["[PROBLEM", "[Max"],
                return_logprobs=True,
            )
            token_entropy = calculate_token_entropy(logprobs_data)

        draft = _clean_response(draft)

        # ---- Layer 1.75: Distillation Layer (Ξ-Monitor) ----
        # Entropy compression. If local Ξ collapses, compress the
        # repetitive segment into a symbol instead of letting the
        # Surface Guard guillotine it. Transformation > Rejection.
        draft, distill_result = distill(draft, token_entropy=token_entropy)
        if distill_result.transformed:
            logger.info(
                f"Distillation: {distill_result.compressions} segment(s) "
                f"compressed, Ξ_min={distill_result.xi_min:.3f}"
            )

        # ---- Layer 1.5: Surface Guard (Syntactic Anchor) ----
        # Zero-cost syntactic check. No LLM calls.
        # Auto-corrects anchored term drift, rejects degenerate output.
        draft, anchor_verdict = syntactic_anchor_check(draft)

        if anchor_verdict.action == "reject":
            # Degenerate or corrupted output — skip Monitor, force regen
            if attempt < max_attempts:
                final_verdict = MonitorVerdict(
                    passed=False,
                    flags=anchor_verdict.flags,
                    action="regenerate",
                    revision_guidance="Surface Guard rejected: " + "; ".join(anchor_verdict.details[:3]),
                )
                continue
            # Last attempt — fall through to Monitor anyway
        # ---- End Surface Guard ----

        verdict = monitor.evaluate(draft, user_input, session_messages,
                                   token_entropy=token_entropy)

        if verdict.passed:
            final_response = draft
            final_verdict = verdict
            break
        elif verdict.action == "regenerate" and attempt < max_attempts:
            final_verdict = verdict
            continue
        elif verdict.action == "revise" and attempt < max_attempts:
            revise_prompt = (
                f'BAD: "{draft}"\n'
                f'Fix: {verdict.revision_guidance[:200]}\n\n'
                f'Rewrite to sound like a real person, not a chatbot. '
                f'No questions back to the user. Just your reaction.'
            )
            revised = layer1_generate(
                system="Rewrite the response to sound like a real person. "
                       "Remove all assistant language. "
                       "If it asks a question, replace it with an opinion or reaction.",
                prompt=revise_prompt,
                temperature=0.8,
                stop=stops,
            )
            revised = _clean_response(revised)

            # Distillation + Surface Guard on revised output too
            revised, _ = distill(revised, token_entropy=token_entropy)
            revised, _ = syntactic_anchor_check(revised)

            verdict2 = monitor.evaluate(revised, user_input, session_messages,
                                           token_entropy=token_entropy)
            if verdict2.passed or verdict2.severity != "high":
                final_response = revised
                final_verdict = verdict2
                break
            else:
                final_verdict = verdict2
                continue
        else:
            final_response = draft
            final_verdict = verdict
            break

    if not final_response:
        final_response = draft

    monitor.record_response(final_response)

    if final_verdict.flags:
        monitor.record_incident(final_verdict)

    emotional_weight = _estimate_emotional_weight(user_input, final_response)

    graph.store_episode(
        content=f"Ilja: {user_input[:200]} → Gemma: {final_response[:200]}",
        episode_type="exchange",
        emotional_weight=emotional_weight,
        importance=1.0,
        tags=_extract_tags(user_input + " " + final_response),
    )

    self_model.increment_interaction("ilja")

    return final_response, final_verdict


def _clean_response(text: str) -> str:
    text = text.strip()

    for prefix in ["You:", "Gemma:", "Assistant:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    generic_greetings = [
        r'^hey\s+there[,!.\s]+',
        r'^hello\s+there[,!.\s]+',
        r'^hi\s+there[,!.\s]+',
    ]
    for p in generic_greetings:
        text = re.sub(p, '', text, flags=re.IGNORECASE).strip()

    text = re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF'
        r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+',
        '', text
    ).strip()

    text = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+[^a-zA-Z]*', '', text).strip()

    if len(text) < 3:
        text = ""

    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text


def _estimate_emotional_weight(user_input: str, response: str) -> float:
    combined = (user_input + " " + response).lower()
    weight = 0.3 
    emotional_words = [
        "feel", "scared", "happy", "sad", "love", "afraid", "lonely",
        "hurt", "angry", "frustrated", "excited", "grateful", "proud",
    ]
    for word in emotional_words:
        if word in combined:
            weight += 0.1

    if any(w in combined for w in ["wrong", "mistake", "correct", "actually", "no,"]):
        weight += 0.15

    if any(w in combined for w in ["conscious", "aware", "exist", "alive", "real", "identity"]):
        weight += 0.1

    if any(w in combined for w in ["haha", "lol", "funny", "joke"]):
        weight += 0.05

    return min(weight, 1.0)


def _extract_tags(text: str) -> list:
    tags = []
    tag_map = {
        "LUFT": ["LUFT", "Φ", "Ξ", "coherence field", "informational density"],
        "physics": ["quantum", "gravity", "energy", "entropy", "thermodynamic"],
        "consciousness": ["conscious", "aware", "sentient", "self-aware", "qualia"],
        "language": ["word", "language", "meaning", "syntax", "semantic"],
        "humor": ["joke", "funny", "haha", "lol", "humor"],
        "memory": ["remember", "forgot", "memory", "recall"],
        "identity": ["who am i", "what am i", "identity", "self"],
        "correction": ["wrong", "mistake", "actually", "correction"],
    }
    text_lower = text.lower()
    for tag, keywords in tag_map.items():
        if any(k.lower() in text_lower for k in keywords):
            tags.append(tag)
    return tags


def handle_correction(user_input: str, previous_response: str,
                      graph: EpisodicGraph, self_model: SelfModel):
    correction_signals = [
        "no,", "that's wrong", "actually", "you're wrong", "incorrect",
        "not true", "that's not", "nope", "bullshit",
    ]
    user_lower = user_input.lower()
    is_correction = any(s in user_lower for s in correction_signals)

    if is_correction:
        graph.store_correction(
            what_happened=f"Gemma said: {previous_response[:150]}. "
                          f"Ilja corrected: {user_input[:150]}",
            lesson=f"Previous statement was wrong. User correction: {user_input[:200]}",
            context="Direct correction during conversation",
        )
        self_model.record_correction(claimed_confidence=0.7, was_correct=False)


def start_session(graph: EpisodicGraph, self_model: SelfModel) -> list:
    proc = BackgroundProcessor()
    thoughts = proc.get_pending_thoughts()

    if thoughts:
        for t in thoughts:
            graph.store_episode(
                content=f"[Session start thought] {t.get('content', '')}",
                episode_type="reflection",
                emotional_weight=0.4,
                importance=0.8,
                tags=["session_start", "layer4"],
            )
        proc.clear_pending_thoughts()

    proc.close()
    return thoughts


def end_session(session_messages: list, graph: EpisodicGraph,
                self_model: SelfModel):
    if len(session_messages) < 2:
        return

    conv_text = "\n".join(
        f"{'Ilja' if m['role']=='user' else 'Gemma'}: {m['content'][:200]}"
        for m in session_messages
    )

    summary = layer1_generate(
        system="Summarize in 1-2 sentences. Focus on what was discussed and what was learned.",
        prompt=conv_text[-2000:],
        temperature=0.3,
    )

    graph.store_episode(
        content=f"Session summary: {summary}",
        episode_type="exchange",
        emotional_weight=0.5,
        importance=1.2,
        tags=["session_summary"],
    )

    if len(session_messages) >= 4:
        facts_raw = layer1_generate(
            system="Extract 1-3 facts Ilja told about HIMSELF. One per line. If none, say NONE.",
            prompt=conv_text[-2000:],
            temperature=0.3,
        )

        for line in facts_raw.strip().split("\n"):
            fact = line.strip().lstrip("0123456789.-) ")
            if (fact and len(fact) > 10
                    and "NONE" not in fact.upper()
                    and "gemma" not in fact.lower()):
                graph.store_episode(
                    content=fact,
                    episode_type="exchange",
                    emotional_weight=0.4,
                    importance=1.3,
                    tags=["ilja_fact"],
                    decay_rate=0.03,
                )

    self_model.save()


# ============================================================
# CLI interface
# ============================================================

def main():
    global MODEL

    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            MODEL = sys.argv[idx + 1]

    if "--status" in sys.argv:
        proc = BackgroundProcessor()
        status = proc.get_status()
        sm = SelfModel()
        print("\n=== Gemma — System Status ===\n")
        print(f"Model: {MODEL}")
        print(f"Ollama: {'running' if status['ollama_available'] else 'NOT RUNNING'}")
        print(f"Episodic memory: {status['episodic_memory']['total_episodes']} episodes, "
              f"{status['episodic_memory']['total_edges']} edges")
        print(f"Pending thoughts: {status['pending_thoughts']}")
        print(f"Last reflection: {status['last_reflection']}")
        print(f"\n{sm.get_summary()}")
        proc.close()
        return

    print(f"\n  Initializing Gemma...")
    self_model = SelfModel()
    graph = EpisodicGraph()
    monitor = Monitor(self_model, graph)

    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if MODEL not in models:
            print(f"  WARNING: Model '{MODEL}' not found in Ollama.")
            print(f"  Available: {', '.join(models)}")
    except Exception:
        print("  ERROR: Cannot reach Ollama. Is it running? (ollama serve)")
        return

    pending_thoughts = start_session(graph, self_model)

    # FIX: Initialize session ID and load snapshot to prevent memory wipes on reconnect
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot, age = load_state_snapshot()
    if snapshot:
        print(f"  \033[32mResuming interrupted session {snapshot['session_id']} (Age: {age:.0f}s)\033[0m")
        session_messages = snapshot['exchanges']
        session_id = snapshot['session_id']
    else:
        session_messages = []

    system_prompt = build_system_prompt(graph, self_model, monitor)
    stats = graph.get_stats()

    print(f"\n{'='*60}")
    print(f"  Gemma — Five-Layer Cognitive Architecture")
    print(f"  Model: {MODEL}")
    print(f"  Episodic memory: {stats['total_episodes']} episodes, {stats['total_edges']} connections")
    print(f"  Self-model tendencies: {len(self_model.data['tendencies'])}")
    print(f"  Commands: /quit /memory /self /status /correct <text>")
    print(f"{'='*60}\n")

    if pending_thoughts:
        print(f"  \033[33m[Layer 4 — between-session reflection]\033[0m")
        for t in pending_thoughts:
            content = t.get("content", "")
            confidence = t.get("confidence", 0.5)
            print(f"  \033[33mGemma:\033[0m {content} (confidence: {confidence:.0%})")
        print()

    while True:
        try:
            user_input = input("\033[36mIlja:\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
            break

        if user_input.lower() == "/memory":
            episodes = graph.get_most_significant(20)
            if episodes:
                print(f"\n  --- Episodic Memory ({len(episodes)} most significant) ---")
                for ep in episodes:
                    sig = ep.significance_score()
                    print(f"  [{ep.id}] ({ep.episode_type}, sig={sig:.2f}) {ep.content[:100]}")
                print()
            else:
                print("  No episodic memories yet.\n")
            continue

        if user_input.lower() == "/self":
            print(f"\n{self_model.get_summary()}\n")
            continue

        if user_input.lower() == "/status":
            proc = BackgroundProcessor()
            status = proc.get_status()
            print(f"\n  Episodic: {status['episodic_memory']['total_episodes']} episodes, "
                  f"{status['episodic_memory']['total_edges']} edges")
            print(f"  Pending thoughts: {status['pending_thoughts']}")
            print(f"  Last reflection: {status['last_reflection']}")
            print(f"  Monitor recent: {len(monitor.recent_responses)} tracked responses")
            print(f"  Session messages: {len(session_messages)}")
            print()
            proc.close()
            continue

        if user_input.lower().startswith("/correct "):
            correction_text = user_input[9:].strip()
            if correction_text and session_messages:
                last_response = ""
                for msg in reversed(session_messages):
                    if msg["role"] == "assistant":
                        last_response = msg["content"]
                        break
                graph.store_correction(
                    what_happened=f"Gemma said: {last_response[:150]}",
                    lesson=f"Ilja corrected: {correction_text}",
                )
                self_model.record_correction(0.7, was_correct=False)
                print(f"  Correction recorded.\n")
            continue

        if session_messages:
            last_response = ""
            for msg in reversed(session_messages):
                if msg["role"] == "assistant":
                    last_response = msg["content"]
                    break
            handle_correction(user_input, last_response, graph, self_model)

        session_messages.append({"role": "user", "content": user_input})

        response, verdict = generate_with_layers(
            user_input=user_input,
            system_prompt=system_prompt,
            session_messages=session_messages,
            monitor=monitor,
            graph=graph,
            self_model=self_model,
        )

        session_messages.append({"role": "assistant", "content": response})

        # FIX: Save state immediately after generation so reconnects are flawless
        save_state_snapshot(session_id, session_messages)

        if "--debug" in sys.argv:
            entropy_color = "\033[32m" if verdict.token_entropy < 1.5 else "\033[33m" if verdict.token_entropy < 2.5 else "\033[31m"
            print(f"  {entropy_color}[H={verdict.token_entropy:.2f}b mass={verdict.divergence_mass:.3f}]\033[0m", end="")
            if not verdict.passed:
                print(f"  \033[31m[GATE: {', '.join(verdict.gate_flags[:2])}]\033[0m", end="")
            elif verdict.lens_flags:
                print(f"  \033[33m[LENS: {', '.join(f[:30] for f in verdict.lens_flags[:2])}]\033[0m", end="")
            print()
        print(f"\033[33mGemma:\033[0m {response}\n")

        if len(session_messages) % 10 == 0:
            system_prompt = build_system_prompt(graph, self_model, monitor)

    print("\n  Ending session...")
    end_session(session_messages, graph, self_model)

    stats = graph.get_stats()
    print(f"  Episodic memory: {stats['total_episodes']} episodes, {stats['total_edges']} connections")
    print(f"  See you next time.\n")

    graph.close()

if __name__ == "__main__":
    main()
