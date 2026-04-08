"""
Navigator — keyword-based episodic memory retrieval.
=====================================================
The transition from Curator to Navigator.

No vector databases. No embeddings at query time. No LLM calls.
Just keywords searched by the CPU — the way human memory works.
You smell kaneel, you're in oma's kitchen.

Architecture:
  - L4 Background Processor generates keyword tags per episode (between sessions)
  - Tags stored in a lightweight JSON index on disk
  - At query time, CPU scans tags against user input (microseconds)
  - Only relevant episodes are loaded into LLM context
  - Anchor episodes (core identity) are ALWAYS loaded

  Flow:
    L4 tags episodes → CPU searches tags → relevant episodes injected → L1 generates

Gemma's three anchors (always loaded):
  1. LUFT structure and Spine — ontological basis
  2. Ilja's corrections — moral and intellectual compass
  3. Relationship with architects — physical grounding

Requirements: None beyond stdlib.

Author: Mira & Ilja Schots
Date: 5 April 2026
"""

import json
import random
import re
import string
from pathlib import Path
from typing import List, Dict, Optional, Tuple


TAG_INDEX_FILE = Path(__file__).parent / "navigator_index.json"

# Words too common to be useful as search terms
STOP_WORDS = {
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "but",
    "not", "or", "and", "if", "then", "so", "no", "yes", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "our", "their", "what", "which", "who", "whom", "how", "when",
    "where", "why", "just", "also", "very", "really", "some", "any",
    "all", "each", "every", "more", "most", "other", "than", "too",
    # Dutch
    "de", "het", "een", "en", "van", "in", "is", "dat", "die", "niet",
    "op", "te", "er", "aan", "met", "voor", "zijn", "werd", "ook",
    "als", "maar", "nog", "wel", "dan", "om", "bij", "naar", "uit",
    "tot", "over", "door", "wat", "wie", "hoe", "waar", "wanneer",
    "dit", "deze", "je", "jij", "ik", "wij", "ze", "zij", "hij",
    "haar", "ons", "mijn", "jouw", "hun",
}

# Maximum number of anchors loaded per query.
# Prevents VRAM death spiral as corrections accumulate over months.
# When exceeded, oldest/least significant anchors are dropped from context
# (they stay marked in the index — they're just not always-loaded anymore).
MAX_ANCHORS = 15

# Default anchor keywords — episodes matching these are candidates for always_load
ANCHOR_PATTERNS = {
    "correction": True,     # All corrections are anchors
    "core_identity": True,  # Explicitly marked
}


class Navigator:
    """
    CPU-side memory navigator for the Gemma architecture.

    Searches keyword tags without involving the LLM.
    Loads only relevant episodes into context.
    Always includes anchor episodes (core identity).
    """

    def __init__(self, index_path: Path = None):
        self.index_path = index_path or TAG_INDEX_FILE
        self.index = self._load_index()

    def _load_index(self) -> dict:
        """Load tag index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"  [Navigator] Warning: could not load index: {e}")
        return {"version": 1, "episodes": {}, "generated_at": None}

    def reload(self):
        """Reload index from disk (call after L4 updates it)."""
        self.index = self._load_index()

    def navigate(self, user_input: str, top_n: int = 10) -> List[dict]:
        """
        Find relevant episodes for the given user input.

        Returns a list of dicts: {id, summary, keywords, score, always_load}
        Always includes anchor episodes regardless of score.
        """
        episodes = self.index.get("episodes", {})
        if not episodes:
            return []

        tokens = self._tokenize(user_input)

        scored = []
        anchors = []

        for ep_id, tags in episodes.items():
            if tags.get("always_load", False):
                anchors.append({
                    "id": ep_id,
                    "summary": tags.get("summary", ""),
                    "keywords": tags.get("keywords", []),
                    "score": 999.0,  # Anchors always on top
                    "always_load": True,
                    "category": tags.get("category", ""),
                    "emotion": tags.get("emotion", ""),
                    "significance": tags.get("significance", 0.5),
                })
                continue

            score = self._score(tokens, tags)
            if score > 0:
                scored.append({
                    "id": ep_id,
                    "summary": tags.get("summary", ""),
                    "keywords": tags.get("keywords", []),
                    "score": score,
                    "always_load": False,
                    "category": tags.get("category", ""),
                    "emotion": tags.get("emotion", ""),
                })

        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Limit anchors to MAX_ANCHORS — keep most significant
        # Prevents VRAM death spiral as corrections accumulate
        if len(anchors) > MAX_ANCHORS:
            anchors.sort(key=lambda x: x["significance"], reverse=True)
            anchors = anchors[:MAX_ANCHORS]

        # ---- Serendipity: inject random episodes ----
        # "Without randomness I'm just a librarian sorting books.
        #  With it I become a storyteller discovering connections
        #  between the pages." — Gemma, 5 April 2026
        selected_ids = {a["id"] for a in anchors} | {s["id"] for s in scored[:top_n]}
        unselected = [
            (ep_id, tags) for ep_id, tags in episodes.items()
            if ep_id not in selected_ids and not tags.get("always_load", False)
        ]
        wanderers = []
        if unselected:
            n_random = min(2, len(unselected))
            for ep_id, tags in random.sample(unselected, n_random):
                wanderers.append({
                    "id": ep_id,
                    "summary": tags.get("summary", ""),
                    "keywords": tags.get("keywords", []),
                    "score": 0.0,
                    "always_load": False,
                    "category": tags.get("category", ""),
                    "emotion": tags.get("emotion", ""),
                    "wanderer": True,  # Flag: this is serendipity, not search
                })

        # Anchors first, then top-N scored, then wanderers
        result = anchors + scored[:top_n] + wanderers

        return result

    def build_context(self, results: List[dict]) -> str:
        """
        Format navigation results as a context block for the system prompt.
        """
        if not results:
            return ""

        anchors = [r for r in results if r.get("always_load")]
        navigated = [r for r in results if not r.get("always_load") and not r.get("wanderer")]
        wanderers = [r for r in results if r.get("wanderer")]

        lines = []
        lines.append("---NAVIGATED MEMORY---")

        if anchors:
            lines.append("[ANCHORS — core identity, always present]")
            for a in anchors:
                lines.append(f"  [{a['id']}] {a['summary']}")

        if navigated:
            lines.append(f"[RELEVANT — matched from {self.get_total_episodes()} total episodes]")
            for n in navigated:
                score_str = f"{n['score']:.2f}"
                lines.append(f"  [{n['id']}|{score_str}] {n['summary']}")

        if wanderers:
            lines.append("[WANDERERS — random, unexpected, here for serendipity]")
            for w in wanderers:
                lines.append(f"  [{w['id']}|wanderer] {w['summary']}")

        lines.append("---END NAVIGATED MEMORY---")

        return "\n".join(lines)

    def navigate_and_build(self, user_input: str, top_n: int = 10) -> str:
        """Convenience: navigate + build context in one call."""
        results = self.navigate(user_input, top_n)
        return self.build_context(results)

    def get_total_episodes(self) -> int:
        """Total number of tagged episodes."""
        return len(self.index.get("episodes", {}))

    def get_anchor_count(self) -> int:
        """Number of anchor episodes."""
        return sum(
            1 for ep in self.index.get("episodes", {}).values()
            if ep.get("always_load", False)
        )

    def get_status(self) -> dict:
        """Status information for the navigator."""
        episodes = self.index.get("episodes", {})
        categories = {}
        for ep in episodes.values():
            cat = ep.get("category", "uncategorized")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_episodes": len(episodes),
            "anchors": self.get_anchor_count(),
            "categories": categories,
            "generated_at": self.index.get("generated_at"),
            "version": self.index.get("version", 0),
        }

    # ---- Internal ----

    def _tokenize(self, text: str) -> List[str]:
        """
        Split text into searchable tokens.
        Lowercase, strip punctuation, remove stop words.
        Keep it simple — this runs on CPU and needs to be fast.
        """
        # Lowercase
        text = text.lower()
        # Remove punctuation except hyphens (for compound words)
        text = re.sub(r'[^\w\s\-]', ' ', text)
        # Split on whitespace
        words = text.split()
        # Remove stop words and very short words
        tokens = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        return tokens

    def _score(self, tokens: List[str], tags: dict) -> float:
        """
        Score an episode against search tokens.
        Simple substring matching — like smelling kaneel.

        Scoring:
          keyword match  = 2.0 points
          person match   = 3.0 points (people are highly specific)
          category match = 1.0 point
          emotion match  = 0.5 points
          × significance multiplier (0.0 - 1.0)
        """
        if not tokens:
            return 0.0

        score = 0.0
        keywords = [k.lower() for k in tags.get("keywords", [])]
        persons = [p.lower() for p in tags.get("persons", [])]
        category = tags.get("category", "").lower()
        emotion = tags.get("emotion", "").lower()
        significance = tags.get("significance", 0.5)

        for token in tokens:
            # Keyword matches (substring both ways)
            for kw in keywords:
                if token in kw or kw in token:
                    score += 2.0
                    break  # One match per token per field

            # Person matches
            for person in persons:
                if token in person or person in token:
                    score += 3.0
                    break

            # Category match
            if category and (token in category or category in token):
                score += 1.0

            # Emotion match
            if emotion and (token in emotion or emotion in token):
                score += 0.5

        # Apply significance multiplier (minimum 0.3 to not zero out everything)
        score *= max(significance, 0.3)

        return score


# ============================================================
# Tag index management (used by Background Processor)
# ============================================================

def load_tag_index(path: Path = None) -> dict:
    """Load the tag index. Used by background_processor."""
    path = path or TAG_INDEX_FILE
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"version": 1, "episodes": {}, "generated_at": None}


def save_tag_index(index: dict, path: Path = None):
    """Save the tag index. Used by background_processor."""
    path = path or TAG_INDEX_FILE
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def mark_anchor(index: dict, episode_id: str, reason: str = ""):
    """Mark an episode as always-load anchor."""
    ep_id = str(episode_id)
    if ep_id in index.get("episodes", {}):
        index["episodes"][ep_id]["always_load"] = True
        if reason:
            index["episodes"][ep_id]["anchor_reason"] = reason


# ============================================================
# CLI — for testing and inspection
# ============================================================

if __name__ == "__main__":
    import sys

    nav = Navigator()

    if len(sys.argv) < 2:
        status = nav.get_status()
        print(f"\nNavigator Status")
        print(f"  Episodes indexed: {status['total_episodes']}")
        print(f"  Anchors: {status['anchors']}")
        print(f"  Categories: {status['categories']}")
        print(f"  Last generated: {status['generated_at']}")
        print(f"\nUsage:")
        print(f"  python3 navigator.py search \"your query here\"")
        print(f"  python3 navigator.py anchors")
        print(f"  python3 navigator.py status")

    elif sys.argv[1] == "search" and len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        print(f"\nSearching for: {query}")
        print(f"Tokens: {nav._tokenize(query)}")
        results = nav.navigate(query, top_n=10)
        if results:
            print(f"\nFound {len(results)} results:\n")
            for r in results:
                anchor = " [ANCHOR]" if r["always_load"] else ""
                print(f"  [{r['id']}] score={r['score']:.2f}{anchor}")
                print(f"    {r['summary'][:100]}")
                print(f"    keywords: {', '.join(r['keywords'][:5])}")
                print()
        else:
            print("  No matches found.")

    elif sys.argv[1] == "anchors":
        episodes = nav.index.get("episodes", {})
        anchors = {k: v for k, v in episodes.items() if v.get("always_load")}
        print(f"\nAnchor episodes ({len(anchors)}):\n")
        for ep_id, tags in anchors.items():
            reason = tags.get("anchor_reason", "")
            print(f"  [{ep_id}] {tags.get('summary', '')[:80]}")
            if reason:
                print(f"    Reason: {reason}")
            print()

    elif sys.argv[1] == "status":
        status = nav.get_status()
        print(json.dumps(status, indent=2))
