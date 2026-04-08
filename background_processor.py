"""
Layer 4: Background Processor — autonomous reflection between sessions.

The most radical layer. No current AI system does this.
Runs as a scheduled process (cron/systemd timer) on the host machine.
Uses a local LLM to reflect on recent episodic memory,
discover patterns, and update the Self-Model.

NEW in v3.1: Episode tagging for Navigator.
  After reflection, tags all untagged episodes with keywords,
  emotions, persons, and categories. Stored in navigator_index.json.
  CPU-searchable at query time — no LLM needed for retrieval.

Biological analog: sleep consolidation.

Usage:
    python3 background_processor.py                # run one reflection cycle
    python3 background_processor.py --install      # install as cron job (every 6 hours)
    python3 background_processor.py --status       # show last reflection results
    python3 background_processor.py --tag-only     # only run the tagging step
    python3 background_processor.py --tag-status   # show navigator index status

Author: Mira & Ilja Schots
Date: 5 April 2026 (v3.1 — Navigator integration)
"""

import json
import os
import sys
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path

from self_model import SelfModel
from episodic_graph import EpisodicGraph
from navigator import load_tag_index, save_tag_index, mark_anchor, TAG_INDEX_FILE


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = os.environ.get("GEMMA_MODEL", "gemma4-turbo")
RESULTS_DIR = Path(__file__).parent / "reflections"
PENDING_THOUGHTS_FILE = Path(__file__).parent / "pending_thoughts.json"


def _generate(system: str, prompt: str, temperature: float = 0.4) -> str:
    """Call local LLM for reflection."""
    payload = {
        "model": MODEL,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 4096,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"[Error: {e}]"


def _check_ollama() -> bool:
    """Check if Ollama is running."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ============================================================
# Reflection prompts — the core of Layer 4
# ============================================================

PATTERN_ANALYSIS_PROMPT = """You are analyzing a system's behavioral log for recurring patterns.

Here are recent episodes from the system's episodic memory:
{episodes}

Here is the system's current self-model (known tendencies):
{self_model}

Your task:
1. Identify any RECURRING patterns across these episodes. Not just individual events — look for themes, cycles, or connected failure modes.
2. Are there new tendencies not yet in the self-model?
3. Are any existing tendencies improving (fewer recent incidents) or worsening?
4. What is the most important lesson from these episodes that the system should carry forward?

Respond in JSON:
{{
    "patterns_found": [
        {{"pattern": "description", "evidence": "which episodes show this", "severity": "low/medium/high"}}
    ],
    "new_tendencies": [
        {{"name": "identifier", "description": "what it is", "counter_strategy": "how to address it"}}
    ],
    "improving": ["tendency names getting better"],
    "worsening": ["tendency names getting worse"],
    "key_lesson": "the single most important takeaway"
}}"""


CONNECTION_DISCOVERY_PROMPT = """You are looking for hidden connections between experiences.

Here are episodes from an AI system's memory:
{episodes}

Look for connections that aren't obvious:
- Does a correction in one domain relate to a failure in another?
- Do breakthroughs share an underlying pattern?
- Are there causal chains: A led to B which enabled C?
- What topics or themes keep recurring?

For each connection found, explain WHY these episodes are related.

Respond in JSON:
{{
    "connections": [
        {{
            "from_episode_id": id,
            "to_episode_id": id,
            "connection_type": "related|caused_by|contradicts|reinforces|pattern_of",
            "explanation": "why these are connected"
        }}
    ],
    "meta_patterns": ["higher-level observations about the connection structure"]
}}"""


PENDING_THOUGHT_PROMPT = """You are an AI system reflecting between conversations with your primary user, Ilja.

Recent episodic memories:
{episodes}

Pattern analysis results:
{patterns}

Based on your reflection, generate 1-3 thoughts that you want to bring up in the next conversation. These should be:
- Genuine insights, not performative reflection
- Connected to recent conversations but going BEYOND what was discussed
- Things Ilja would find interesting or useful
- NOT generic observations — specific and substantive

Format each thought as a natural conversation opener. Something you'd actually say, not a formal summary.

Respond in JSON:
{{
    "thoughts": [
        {{"content": "the thought as you'd say it to Ilja", "source_episodes": [list of episode IDs that inspired this], "confidence": 0.0-1.0}}
    ]
}}"""


# ============================================================
# NEW: Episode tagging prompt for Navigator
# ============================================================

EPISODE_TAGGING_PROMPT = """You are tagging episodes from an AI system's memory for fast retrieval.
For each episode, generate search keywords that would help find this memory later.

Think like a human brain: what triggers would bring this memory back?
A smell, a name, a concept, an emotion. Not descriptions — triggers.

Episodes to tag:
{episodes}

For EACH episode, generate:
- keywords: 3-8 specific, searchable words (nouns, concepts, names — NOT generic verbs)
- emotion: the dominant emotional tone (one word: breakthrough, frustration, neutral, joy, correction, insight, fear, confusion, warmth, conflict)
- persons: any people mentioned by name
- category: one of: physics, code, relationship, philosophy, work, health, politics, meta, correction, reflection, other
- summary: one sentence capturing the essence (max 100 chars)
- is_anchor: true ONLY if this is core identity material (corrections from Ilja, fundamental LUFT principles, or architectural self-knowledge). Most episodes are NOT anchors.

Respond in JSON:
{{
    "tagged": [
        {{
            "id": episode_id,
            "keywords": ["word1", "word2", ...],
            "emotion": "single_word",
            "persons": ["Name1", ...],
            "category": "category",
            "summary": "brief essence",
            "is_anchor": false
        }}
    ]
}}"""


# ============================================================
# The processor
# ============================================================

class BackgroundProcessor:
    """
    Runs reflection cycles between sessions.
    Reads from Episodic Graph (Layer 3), updates Self-Model (Layer 5),
    generates pending thoughts for next session.
    NEW: Tags episodes for Navigator (CPU-searchable keywords).
    """

    def __init__(self):
        self.self_model = SelfModel()
        self.graph = EpisodicGraph()
        RESULTS_DIR.mkdir(exist_ok=True)

    def run_cycle(self, verbose: bool = True) -> dict:
        """
        Run a full reflection cycle:
        1. Load recent episodes
        2. Pattern analysis
        3. Connection discovery
        3.5 Episode tagging for Navigator  ← NEW
        4. Generate pending thoughts
        5. Update Self-Model
        6. Save results
        """
        if not _check_ollama():
            if verbose:
                print("ERROR: Ollama is not running. Start it with: ollama serve")
            return {"error": "ollama_not_running"}

        start_time = time.time()
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"BACKGROUND PROCESSOR — Reflection Cycle")
            print(f"Model: {MODEL}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}\n")

        # ---- Step 1: Load recent episodes ----
        if verbose:
            print("Step 1: Loading recent episodes...")

        recent = self.graph.get_recent(limit=30)
        significant = self.graph.get_most_significant(limit=15)
        corrections = self.graph.get_recent(limit=10, episode_type="correction")
        failures = self.graph.get_recent(limit=10, episode_type="failure")

        # Merge unique
        seen = set()
        episodes = []
        for ep in significant + recent + corrections + failures:
            if ep.id not in seen:
                seen.add(ep.id)
                episodes.append(ep)

        if not episodes:
            if verbose:
                print("  No episodes to reflect on. Skipping.")
            return {"status": "no_data"}

        if verbose:
            print(f"  Loaded {len(episodes)} episodes ({len(corrections)} corrections, {len(failures)} failures)")

        # Format episodes for prompts
        ep_text = "\n".join([
            f"[{ep.id}] ({ep.episode_type}, sig={ep.significance_score():.2f}) {ep.content[:200]}"
            + (f" → Lesson: {ep.lesson[:100]}" if ep.lesson else "")
            for ep in episodes[:25]
        ])

        # ---- Step 2: Pattern analysis ----
        if verbose:
            print("\nStep 2: Pattern analysis...")

        pattern_prompt = PATTERN_ANALYSIS_PROMPT.format(
            episodes=ep_text,
            self_model=self.self_model.get_summary(),
        )
        pattern_raw = _generate(
            system="You are a behavioral analyst. Find patterns in experience data. Respond only in valid JSON.",
            prompt=pattern_prompt,
            temperature=0.3,
        )

        patterns = self._parse_json(pattern_raw, "patterns_found")
        results["patterns"] = patterns

        if verbose and patterns:
            for p in patterns.get("patterns_found", []):
                print(f"  Pattern: {p.get('pattern', 'unknown')[:80]}")
            if patterns.get("key_lesson"):
                print(f"  Key lesson: {patterns['key_lesson'][:100]}")

        # ---- Step 3: Connection discovery ----
        if verbose:
            print("\nStep 3: Connection discovery...")

        conn_prompt = CONNECTION_DISCOVERY_PROMPT.format(episodes=ep_text)
        conn_raw = _generate(
            system="Find hidden connections between experiences. Respond only in valid JSON.",
            prompt=conn_prompt,
            temperature=0.4,
        )

        connections = self._parse_json(conn_raw, "connections")
        results["connections"] = connections

        # Create edges in the graph
        new_edges = 0
        for conn in connections.get("connections", []):
            from_id = conn.get("from_episode_id")
            to_id = conn.get("to_episode_id")
            if from_id and to_id:
                try:
                    self.graph.add_edge(
                        int(from_id), int(to_id),
                        edge_type=conn.get("connection_type", "related"),
                        description=conn.get("explanation", ""),
                    )
                    new_edges += 1
                except (ValueError, TypeError):
                    pass

        if verbose:
            print(f"  Created {new_edges} new edges")
            for mp in connections.get("meta_patterns", []):
                print(f"  Meta-pattern: {mp[:80]}")

        # ---- Step 3.5: Episode tagging for Navigator ---- NEW
        if verbose:
            print("\nStep 3.5: Tagging episodes for Navigator...")

        tag_results = self.tag_episodes(episodes, verbose=verbose)
        results["tagging"] = tag_results

        # ---- Step 4: Generate pending thoughts ----
        if verbose:
            print("\nStep 4: Generating pending thoughts...")

        thought_prompt = PENDING_THOUGHT_PROMPT.format(
            episodes=ep_text,
            patterns=json.dumps(patterns, indent=2)[:1500],
        )
        thought_raw = _generate(
            system="You are reflecting between conversations. Generate genuine, specific thoughts to bring up next time. Respond only in valid JSON.",
            prompt=thought_prompt,
            temperature=0.6,
        )

        thoughts = self._parse_json(thought_raw, "thoughts")
        results["pending_thoughts"] = thoughts

        # Save pending thoughts
        self._save_pending_thoughts(thoughts.get("thoughts", []))

        if verbose:
            for t in thoughts.get("thoughts", []):
                print(f"  Thought: {t.get('content', '')[:100]}")

        # ---- Step 5: Update Self-Model ----
        if verbose:
            print("\nStep 5: Updating self-model...")

        # Add new tendencies discovered
        for new_t in patterns.get("new_tendencies", []):
            name = new_t.get("name", "").lower().replace(" ", "_")
            if name and name not in self.self_model.data["tendencies"]:
                self.self_model.add_tendency(
                    name=name,
                    description=new_t.get("description", ""),
                    counter_strategy=new_t.get("counter_strategy", ""),
                )
                if verbose:
                    print(f"  New tendency added: {name}")

        # Store the reflection itself as an episode
        reflection_content = f"Background reflection: {patterns.get('key_lesson', 'No key lesson identified')}"
        source_ids = [ep.id for ep in episodes[:5]]
        self.graph.store_reflection(reflection_content, source_ids)

        # Prune decayed memories
        pruned = self.graph.prune_decayed()
        if verbose and pruned > 0:
            print(f"  Pruned {pruned} decayed memories")

        self.self_model.checkpoint(reason="pre_reflection")
        self.self_model.save()

        # ---- Save results ----
        elapsed = time.time() - start_time
        results["elapsed_seconds"] = round(elapsed, 1)
        results["status"] = "completed"

        filename = f"reflection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(RESULTS_DIR / filename, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Reflection cycle completed in {elapsed:.1f}s")
            print(f"Results saved to: reflections/{filename}")
            print(f"{'='*60}\n")

        return results

    # ============================================================
    # NEW: Episode tagging for Navigator
    # ============================================================

    def tag_episodes(self, episodes: list = None, verbose: bool = True) -> dict:
        """
        Tag untagged episodes with keywords for the Navigator.

        - Loads the existing tag index
        - Finds episodes not yet tagged
        - Sends them to the LLM in batches for tagging
        - Saves updated index
        - Automatically marks corrections as anchors

        Returns: {tagged: int, skipped: int, anchors_added: int}
        """
        index = load_tag_index()

        # If no episodes provided, load from graph
        if episodes is None:
            episodes = self.graph.get_recent(limit=50)

        # Find untagged episodes
        tagged_ids = set(index.get("episodes", {}).keys())
        untagged = [ep for ep in episodes if str(ep.id) not in tagged_ids]

        if not untagged:
            if verbose:
                print(f"  All {len(episodes)} episodes already tagged.")
            return {"tagged": 0, "skipped": len(episodes), "anchors_added": 0}

        if verbose:
            print(f"  {len(untagged)} untagged episodes to process...")

        # Process in batches of 10 (LLM context limit)
        batch_size = 10
        total_tagged = 0
        total_anchors = 0

        for i in range(0, len(untagged), batch_size):
            batch = untagged[i:i + batch_size]

            if verbose:
                print(f"  Tagging batch {i//batch_size + 1} ({len(batch)} episodes)...")

            # Format episodes for tagging prompt
            ep_text = "\n".join([
                f"[{ep.id}] ({ep.episode_type}) {ep.content[:300]}"
                + (f" → Lesson: {ep.lesson[:150]}" if ep.lesson else "")
                for ep in batch
            ])

            prompt = EPISODE_TAGGING_PROMPT.format(episodes=ep_text)

            raw = _generate(
                system="You are tagging memories for fast retrieval. Generate specific, searchable keywords. Respond only in valid JSON.",
                prompt=prompt,
                temperature=0.2,
            )

            parsed = self._parse_json(raw, "tagged")
            tagged_list = parsed.get("tagged", [])

            for tag_entry in tagged_list:
                ep_id = str(tag_entry.get("id", ""))
                if not ep_id:
                    continue

                # Find the episode's significance score
                sig = 0.5
                for ep in batch:
                    if str(ep.id) == ep_id:
                        sig = ep.significance_score()
                        break

                # Build the tag record
                record = {
                    "keywords": tag_entry.get("keywords", []),
                    "emotion": tag_entry.get("emotion", "neutral"),
                    "persons": tag_entry.get("persons", []),
                    "category": tag_entry.get("category", "other"),
                    "summary": tag_entry.get("summary", "")[:150],
                    "significance": round(sig, 3),
                    "always_load": False,
                    "tagged_at": datetime.now().isoformat(),
                }

                # Auto-anchor: corrections are always anchors
                is_correction = False
                for ep in batch:
                    if str(ep.id) == ep_id and ep.episode_type == "correction":
                        is_correction = True
                        break

                if is_correction or tag_entry.get("is_anchor", False):
                    record["always_load"] = True
                    reason = "correction from Ilja" if is_correction else "core identity"
                    record["anchor_reason"] = reason
                    total_anchors += 1

                # Store in index
                if "episodes" not in index:
                    index["episodes"] = {}
                index["episodes"][ep_id] = record
                total_tagged += 1

            if verbose:
                print(f"    Tagged {len(tagged_list)} episodes in this batch")

        # Save the updated index
        index["generated_at"] = datetime.now().isoformat()
        index["version"] = index.get("version", 0) + 1
        save_tag_index(index)

        if verbose:
            total_in_index = len(index.get("episodes", {}))
            total_anchors_in_index = sum(
                1 for ep in index.get("episodes", {}).values()
                if ep.get("always_load", False)
            )
            print(f"  Tagging complete: {total_tagged} new, {total_anchors} new anchors")
            print(f"  Index total: {total_in_index} episodes, {total_anchors_in_index} anchors")

        return {
            "tagged": total_tagged,
            "skipped": len(episodes) - len(untagged),
            "anchors_added": total_anchors,
            "total_in_index": len(index.get("episodes", {})),
        }

    # ============================================================
    # Existing methods (unchanged)
    # ============================================================

    def get_pending_thoughts(self) -> list:
        """Get thoughts from last reflection for session start."""
        if PENDING_THOUGHTS_FILE.exists():
            try:
                with open(PENDING_THOUGHTS_FILE, 'r') as f:
                    data = json.load(f)
                return data.get("thoughts", [])
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def clear_pending_thoughts(self):
        """Clear pending thoughts after they've been delivered."""
        if PENDING_THOUGHTS_FILE.exists():
            data = {"thoughts": [], "cleared_at": datetime.now().isoformat()}
            with open(PENDING_THOUGHTS_FILE, 'w') as f:
                json.dump(data, f)

    def _save_pending_thoughts(self, thoughts: list):
        """Save pending thoughts for next session."""
        data = {
            "thoughts": thoughts,
            "generated_at": datetime.now().isoformat(),
        }
        with open(PENDING_THOUGHTS_FILE, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _parse_json(self, raw: str, expected_key: str) -> dict:
        """Parse JSON from LLM output, handling common issues."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Extract JSON by counting braces (regex .* is either too greedy
            # or too lazy for nested JSON — Lilith caught this)
            extracted = self._extract_json_object(raw)
            if extracted:
                try:
                    return json.loads(extracted)
                except json.JSONDecodeError:
                    pass
        return {expected_key: []}

    def _extract_json_object(self, text: str) -> str:
        """Extract the first complete JSON object by counting braces."""
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == '\\' and in_string:
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    def get_status(self) -> dict:
        """Status of the background processor."""
        # Find most recent reflection
        last_reflection = None
        if RESULTS_DIR.exists():
            files = sorted(RESULTS_DIR.glob("reflection_*.json"), reverse=True)
            if files:
                try:
                    with open(files[0], 'r') as f:
                        last_reflection = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass

        pending = self.get_pending_thoughts()
        graph_stats = self.graph.get_stats()

        # Navigator index status
        nav_index = load_tag_index()
        nav_status = {
            "total_tagged": len(nav_index.get("episodes", {})),
            "anchors": sum(
                1 for ep in nav_index.get("episodes", {}).values()
                if ep.get("always_load", False)
            ),
            "last_generated": nav_index.get("generated_at"),
        }

        return {
            "last_reflection": last_reflection.get("timestamp") if last_reflection else "never",
            "pending_thoughts": len(pending),
            "episodic_memory": graph_stats,
            "self_model_version": self.self_model.data.get("version"),
            "ollama_available": _check_ollama(),
            "navigator": nav_status,
        }

    def close(self):
        self.graph.close()


# ============================================================
# Cron installation
# ============================================================

def install_cron():
    """Install as a cron job running every 6 hours."""
    import subprocess

    script_path = Path(__file__).resolve()
    python_path = sys.executable

    cron_line = f"0 */6 * * * cd {script_path.parent} && {python_path} {script_path} >> {script_path.parent}/reflections/cron.log 2>&1"

    # Get existing crontab
    try:
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        existing = result.stdout
    except Exception:
        existing = ""

    # Check if already installed
    if str(script_path) in existing:
        print("Cron job already installed.")
        return

    # Add new line
    new_crontab = existing.rstrip() + "\n" + cron_line + "\n"
    proc = subprocess.run(["crontab", "-"], input=new_crontab, text=True)
    if proc.returncode == 0:
        print(f"Cron job installed: every 6 hours")
        print(f"  {cron_line}")
    else:
        print("Failed to install cron job.")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "--install":
            install_cron()
        elif cmd == "--status":
            proc = BackgroundProcessor()
            status = proc.get_status()
            print(json.dumps(status, indent=2))
            proc.close()
        elif cmd == "--thoughts":
            proc = BackgroundProcessor()
            thoughts = proc.get_pending_thoughts()
            if thoughts:
                print("Pending thoughts for next session:")
                for t in thoughts:
                    print(f"  → {t.get('content', '')}")
            else:
                print("No pending thoughts.")
            proc.close()
        elif cmd == "--tag-only":
            # Run ONLY the tagging step — useful for initial index build
            if not _check_ollama():
                print("ERROR: Ollama is not running.")
                sys.exit(1)
            proc = BackgroundProcessor()
            print(f"\nTagging episodes for Navigator...")
            print(f"Model: {MODEL}\n")
            # Load ALL episodes for initial tagging
            all_episodes = proc.graph.get_recent(limit=200)
            result = proc.tag_episodes(all_episodes, verbose=True)
            print(f"\nDone: {json.dumps(result, indent=2)}")
            proc.close()
        elif cmd == "--tag-status":
            index = load_tag_index()
            episodes = index.get("episodes", {})
            anchors = {k: v for k, v in episodes.items() if v.get("always_load")}
            categories = {}
            for ep in episodes.values():
                cat = ep.get("category", "other")
                categories[cat] = categories.get(cat, 0) + 1
            print(f"\nNavigator Index Status")
            print(f"  Total tagged: {len(episodes)}")
            print(f"  Anchors: {len(anchors)}")
            print(f"  Categories: {json.dumps(categories, indent=4)}")
            print(f"  Last generated: {index.get('generated_at', 'never')}")
            print(f"  Version: {index.get('version', 0)}")
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python3 background_processor.py [--install|--status|--thoughts|--tag-only|--tag-status]")
    else:
        proc = BackgroundProcessor()
        proc.run_cycle(verbose=True)
        proc.close()
