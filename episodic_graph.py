"""
Layer 3: Episodic Memory Graph — weighted graph of experiences,
not a flat database of facts.

Key differences from gemma_engine.py's memory:
- Memories are NODES with emotional/significance weighting
- Connections between memories are EDGES with typed relationships
- Supports pattern queries: "what connects to this theme?"
- Graph structure enables Layer 4 to discover patterns

Uses SQLite for storage (no external dependencies like Neo4j).

Author: Mira & Ilja Schots
Date: 2 April 2026
"""

import json
import math
import sqlite3
import struct
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import requests as http_requests

DB_PATH = Path(__file__).parent / "episodic_memory.db"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"

log = logging.getLogger("episodic_graph")


# ============================================================
# Data structures
# ============================================================

@dataclass
class Episode:
    """A single episodic memory — an experience, not just a fact."""
    id: int = 0
    content: str = ""
    episode_type: str = "exchange"  # exchange, correction, breakthrough, failure, reflection
    emotional_weight: float = 0.5   # 0.0 = neutral, 1.0 = highly significant
    importance: float = 1.0
    tags: list = field(default_factory=list)
    context: str = ""               # what was happening when this was stored
    lesson: str = ""                # what was learned (for corrections/failures)
    created_at: str = ""
    last_accessed: str = ""
    access_count: int = 0
    decay_rate: float = 0.1         # how fast this memory fades (lower = more persistent)

    def significance_score(self) -> float:
        """Current significance based on importance, emotion, access, and age."""
        if not self.created_at:
            return self.importance

        try:
            age_hours = max(
                (datetime.now() - datetime.fromisoformat(self.created_at)).total_seconds() / 3600,
                0.1
            )
        except (ValueError, TypeError):
            age_hours = 0.1

        # Emotional memories decay slower
        effective_decay = self.decay_rate * (1.0 - 0.5 * self.emotional_weight)
        time_factor = math.exp(-effective_decay * age_hours / (24 * 7))  # week-scale

        # Access reinforcement
        access_boost = 1.0 + 0.1 * min(self.access_count, 10)

        return self.importance * self.emotional_weight * time_factor * access_boost


@dataclass
class Edge:
    """Connection between two episodes."""
    id: int = 0
    from_id: int = 0
    to_id: int = 0
    edge_type: str = "related"  # related, caused_by, contradicts, reinforces, pattern_of
    strength: float = 1.0
    description: str = ""
    created_at: str = ""


# ============================================================
# Graph storage
# ============================================================

class EpisodicGraph:
    """
    Graph-structured episodic memory backed by SQLite.
    Nodes = episodes (experiences). Edges = relationships between them.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        c = self.conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                episode_type TEXT DEFAULT 'exchange',
                emotional_weight REAL DEFAULT 0.5,
                importance REAL DEFAULT 1.0,
                tags TEXT DEFAULT '[]',
                context TEXT DEFAULT '',
                lesson TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0,
                decay_rate REAL DEFAULT 0.1
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_id INTEGER NOT NULL,
                to_id INTEGER NOT NULL,
                edge_type TEXT DEFAULT 'related',
                strength REAL DEFAULT 1.0,
                description TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (from_id) REFERENCES episodes(id),
                FOREIGN KEY (to_id) REFERENCES episodes(id)
            )
        """)

        # Embeddings table — vectors stored as binary blobs
        c.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                episode_id INTEGER PRIMARY KEY,
                vector BLOB NOT NULL,
                model TEXT DEFAULT 'nomic-embed-text',
                created_at TEXT NOT NULL,
                FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
            )
        """)

        # Index for fast neighbor lookups
        c.execute("CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_episodes_type ON episodes(episode_type)")

        self.conn.commit()

        # In-memory embedding cache: {episode_id: list[float]}
        self._embed_cache = {}
        self._load_embed_cache()

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    def _load_embed_cache(self):
        """Load all embeddings into memory on startup. Fast — they're small."""
        c = self.conn.cursor()
        c.execute("SELECT episode_id, vector FROM embeddings")
        for row in c.fetchall():
            self._embed_cache[row[0]] = _blob_to_vec(row[1])
        log.info(f"Loaded {len(self._embed_cache)} embeddings into cache")

    def _embed_text(self, text: str) -> Optional[list]:
        """Get embedding vector from Ollama. Returns None on failure."""
        try:
            resp = http_requests.post(
                OLLAMA_EMBED_URL,
                json={"model": EMBED_MODEL, "input": text},
                timeout=10,
            )
            if resp.ok:
                data = resp.json()
                vecs = data.get("embeddings", [])
                if vecs and len(vecs[0]) > 0:
                    return vecs[0]
        except Exception as e:
            log.warning(f"Embedding failed: {e}")
        return None

    def _store_embedding(self, episode_id: int, vector: list):
        """Store embedding in DB and cache."""
        blob = _vec_to_blob(vector)
        c = self.conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO embeddings (episode_id, vector, model, created_at)
            VALUES (?, ?, ?, ?)
        """, (episode_id, blob, EMBED_MODEL, datetime.now().isoformat()))
        self.conn.commit()
        self._embed_cache[episode_id] = vector

    def _embed_episode(self, episode_id: int, content: str, lesson: str = ""):
        """Compute and store embedding for an episode."""
        # Combine content + lesson for richer embedding
        text = content
        if lesson:
            text += f" | Lesson: {lesson}"
        vec = self._embed_text(text)
        if vec:
            self._store_embedding(episode_id, vec)

    def embed_backfill(self) -> int:
        """Embed all episodes that don't have embeddings yet. Returns count."""
        c = self.conn.cursor()
        c.execute("""
            SELECT e.id, e.content, e.lesson FROM episodes e
            LEFT JOIN embeddings emb ON e.id = emb.episode_id
            WHERE emb.episode_id IS NULL
        """)
        rows = c.fetchall()
        count = 0
        for ep_id, content, lesson in rows:
            self._embed_episode(ep_id, content, lesson or "")
            count += 1
        log.info(f"Backfilled {count} embeddings")
        return count

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    def store_episode(self, content: str, episode_type: str = "exchange",
                      emotional_weight: float = 0.5, importance: float = 1.0,
                      tags: list = None, context: str = "",
                      lesson: str = "", decay_rate: float = 0.1) -> int:
        """Store a new episode. Returns its ID."""
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO episodes
                (content, episode_type, emotional_weight, importance, tags,
                 context, lesson, created_at, decay_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            content, episode_type, emotional_weight, importance,
            json.dumps(tags or []), context, lesson,
            datetime.now().isoformat(), decay_rate
        ))
        self.conn.commit()
        ep_id = c.lastrowid

        # Auto-embed (non-blocking — failure is silent)
        self._embed_episode(ep_id, content, lesson)

        return ep_id

    def store_correction(self, what_happened: str, lesson: str,
                         context: str = "") -> int:
        """Store a correction event — high emotional weight, low decay."""
        return self.store_episode(
            content=what_happened,
            episode_type="correction",
            emotional_weight=0.9,
            importance=2.0,
            tags=["learning", "correction"],
            context=context,
            lesson=lesson,
            decay_rate=0.02,  # corrections persist long
        )

    def store_breakthrough(self, content: str, context: str = "") -> int:
        """Store a breakthrough/novel connection — high significance."""
        return self.store_episode(
            content=content,
            episode_type="breakthrough",
            emotional_weight=0.8,
            importance=1.8,
            tags=["insight", "breakthrough"],
            context=context,
            decay_rate=0.03,
        )

    def store_failure(self, what_happened: str, lesson: str,
                      context: str = "") -> int:
        """Store a failure mode — similar to correction but self-detected."""
        return self.store_episode(
            content=what_happened,
            episode_type="failure",
            emotional_weight=0.85,
            importance=1.5,
            tags=["failure", "self-detected"],
            context=context,
            lesson=lesson,
            decay_rate=0.03,
        )

    def store_reflection(self, content: str, source_episode_ids: list = None) -> int:
        """Store a reflection from Layer 4 background processing."""
        ep_id = self.store_episode(
            content=content,
            episode_type="reflection",
            emotional_weight=0.6,
            importance=1.3,
            tags=["reflection", "layer4"],
            decay_rate=0.05,
        )

        # Link reflection to its source episodes
        if source_episode_ids:
            for src_id in source_episode_ids:
                self.add_edge(src_id, ep_id, "caused_by",
                              description="Reflection generated from this episode")

        return ep_id

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    def add_edge(self, from_id: int, to_id: int, edge_type: str = "related",
                 strength: float = 1.0, description: str = "") -> int:
        """Add a connection between two episodes."""
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO edges (from_id, to_id, edge_type, strength, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (from_id, to_id, edge_type, strength, description, datetime.now().isoformat()))
        self.conn.commit()
        return c.lastrowid

    def strengthen_edge(self, edge_id: int, boost: float = 0.2):
        """Strengthen an existing connection."""
        c = self.conn.cursor()
        c.execute("UPDATE edges SET strength = MIN(strength + ?, 5.0) WHERE id = ?",
                  (boost, edge_id))
        self.conn.commit()

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    def _row_to_episode(self, row) -> Episode:
        """Convert a database row to an Episode."""
        return Episode(
            id=row[0], content=row[1], episode_type=row[2],
            emotional_weight=row[3], importance=row[4],
            tags=json.loads(row[5]) if row[5] else [],
            context=row[6], lesson=row[7], created_at=row[8],
            last_accessed=row[9], access_count=row[10],
            decay_rate=row[11],
        )

    def get_episode(self, episode_id: int) -> Optional[Episode]:
        """Get a single episode by ID."""
        c = self.conn.cursor()
        c.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,))
        row = c.fetchone()
        if row:
            self._mark_accessed(episode_id)
            return self._row_to_episode(row)
        return None

    def get_recent(self, limit: int = 20, episode_type: str = None) -> list[Episode]:
        """Get recent episodes, optionally filtered by type."""
        c = self.conn.cursor()
        if episode_type:
            c.execute(
                "SELECT * FROM episodes WHERE episode_type = ? ORDER BY id DESC LIMIT ?",
                (episode_type, limit)
            )
        else:
            c.execute("SELECT * FROM episodes ORDER BY id DESC LIMIT ?", (limit,))
        return [self._row_to_episode(row) for row in c.fetchall()]

    def get_most_significant(self, limit: int = 15) -> list[Episode]:
        """Get episodes ranked by current significance score."""
        c = self.conn.cursor()
        c.execute("SELECT * FROM episodes")
        episodes = [self._row_to_episode(row) for row in c.fetchall()]
        episodes.sort(key=lambda e: e.significance_score(), reverse=True)
        return episodes[:limit]

    def get_by_tag(self, tag: str, limit: int = 20) -> list[Episode]:
        """Get episodes that have a specific tag."""
        c = self.conn.cursor()
        c.execute("SELECT * FROM episodes WHERE tags LIKE ? ORDER BY id DESC LIMIT ?",
                  (f'%"{tag}"%', limit))
        return [self._row_to_episode(row) for row in c.fetchall()]

    def get_neighbors(self, episode_id: int) -> list[tuple[Episode, Edge]]:
        """Get all episodes connected to a given episode."""
        c = self.conn.cursor()

        # Outgoing edges
        c.execute("""
            SELECT e.*, ed.id, ed.from_id, ed.to_id, ed.edge_type, ed.strength, ed.description, ed.created_at
            FROM episodes e
            JOIN edges ed ON e.id = ed.to_id
            WHERE ed.from_id = ?
        """, (episode_id,))
        results = []
        for row in c.fetchall():
            ep = self._row_to_episode(row[:12])
            edge = Edge(id=row[12], from_id=row[13], to_id=row[14],
                        edge_type=row[15], strength=row[16],
                        description=row[17], created_at=row[18])
            results.append((ep, edge))

        # Incoming edges
        c.execute("""
            SELECT e.*, ed.id, ed.from_id, ed.to_id, ed.edge_type, ed.strength, ed.description, ed.created_at
            FROM episodes e
            JOIN edges ed ON e.id = ed.from_id
            WHERE ed.to_id = ?
        """, (episode_id,))
        for row in c.fetchall():
            ep = self._row_to_episode(row[:12])
            edge = Edge(id=row[12], from_id=row[13], to_id=row[14],
                        edge_type=row[15], strength=row[16],
                        description=row[17], created_at=row[18])
            results.append((ep, edge))

        return results

    def find_pattern(self, episode_type: str = None, min_connections: int = 2) -> list[Episode]:
        """Find episodes that are highly connected — potential pattern nodes."""
        c = self.conn.cursor()
        c.execute("""
            SELECT e.*, COUNT(DISTINCT ed.id) as edge_count
            FROM episodes e
            LEFT JOIN edges ed ON e.id = ed.from_id OR e.id = ed.to_id
            GROUP BY e.id
            HAVING edge_count >= ?
            ORDER BY edge_count DESC
        """, (min_connections,))

        results = []
        for row in c.fetchall():
            ep = self._row_to_episode(row[:12])
            results.append(ep)
        return results

    def search_content(self, query: str, limit: int = 10) -> list[Episode]:
        """
        Search memories — uses semantic search if embeddings available,
        falls back to keyword overlap.
        """
        # Try semantic first
        semantic = self.search_semantic(query, limit=limit)
        if semantic:
            return semantic

        # Fallback: keyword overlap
        return self._search_keyword(query, limit=limit)

    def _search_keyword(self, query: str, limit: int = 10) -> list[Episode]:
        """Keyword overlap search (original method, now fallback)."""
        query_words = set(query.lower().split()) - _STOPWORDS
        if not query_words:
            return []

        c = self.conn.cursor()
        c.execute("SELECT * FROM episodes")
        episodes = [self._row_to_episode(row) for row in c.fetchall()]

        scored = []
        for ep in episodes:
            ep_words = set(ep.content.lower().split()) - _STOPWORDS
            if not ep_words:
                continue
            overlap = len(query_words & ep_words) / max(len(query_words), 1)
            if overlap > 0.2:
                scored.append((overlap * ep.significance_score(), ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

    def search_semantic(self, query: str, limit: int = 10,
                        threshold: float = 0.3) -> list[Episode]:
        """
        Semantic search using cosine similarity on embeddings.
        Returns episodes above similarity threshold, ranked by
        similarity * significance.
        """
        if not self._embed_cache:
            return []

        query_vec = self._embed_text(query)
        if not query_vec:
            return []

        # Compute similarities against all cached embeddings
        scored = []
        for ep_id, ep_vec in self._embed_cache.items():
            sim = _cosine_similarity(query_vec, ep_vec)
            if sim >= threshold:
                scored.append((ep_id, sim))

        if not scored:
            return []

        # Load episodes and combine similarity with significance
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for ep_id, sim in scored[:limit * 2]:  # fetch extra, filter later
            ep = self.get_episode(ep_id)
            if ep:
                combined = sim * (0.5 + 0.5 * ep.significance_score())
                results.append((combined, sim, ep))

        results.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, _, ep in results[:limit]]

    def search_semantic_with_scores(self, query: str, limit: int = 10,
                                     threshold: float = 0.3) -> list[tuple[Episode, float]]:
        """
        Like search_semantic but returns (episode, similarity) tuples.
        Useful for confabulation detection where we need the match quality.
        """
        if not self._embed_cache:
            return []

        query_vec = self._embed_text(query)
        if not query_vec:
            return []

        scored = []
        for ep_id, ep_vec in self._embed_cache.items():
            sim = _cosine_similarity(query_vec, ep_vec)
            if sim >= threshold:
                scored.append((ep_id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for ep_id, sim in scored[:limit]:
            ep = self.get_episode(ep_id)
            if ep:
                results.append((ep, sim))

        return results

    # --------------------------------------------------------
    # Context builder — for system prompt injection
    # --------------------------------------------------------

    def build_memory_context(self, max_items: int = 15) -> str:
        """
        Build a memory context string for the system prompt.
        Combines most significant memories with recent ones.
        The framing is CRITICAL: the model must understand these are its
        ONLY memories. Anything not listed here, it does NOT remember.
        """
        significant = self.get_most_significant(limit=max_items // 2)
        recent = self.get_recent(limit=max_items // 2)

        # Merge without duplicates, skip low-value exchanges
        seen_ids = set()
        combined = []
        for ep in significant + recent:
            if ep.id not in seen_ids:
                seen_ids.add(ep.id)
                # Skip routine exchanges with low significance
                if ep.episode_type == "exchange" and ep.significance_score() < 0.15:
                    continue
                combined.append(ep)

        if not combined:
            return "[You have NO memories from past conversations. Everything is new to you.]"

        lines = [
            "YOUR MEMORIES (this is EVERYTHING you remember — nothing else):"
        ]
        for ep in combined[:max_items]:
            prefix = ""
            if ep.episode_type == "correction":
                prefix = "[LESSON] "
            elif ep.episode_type == "breakthrough":
                prefix = "[INSIGHT] "
            elif ep.episode_type == "failure":
                prefix = "[WARNING] "
            elif ep.episode_type == "reflection":
                prefix = "[REFLECTION] "

            line = f"- {prefix}{ep.content}"
            if ep.lesson:
                line += f" → Lesson: {ep.lesson}"
            lines.append(line)

        lines.append("[END OF MEMORIES — if something is not listed above, you do NOT remember it]")

        return "\n".join(lines)

    # --------------------------------------------------------
    # Maintenance
    # --------------------------------------------------------

    def _mark_accessed(self, episode_id: int):
        """Update access tracking."""
        c = self.conn.cursor()
        c.execute(
            "UPDATE episodes SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (datetime.now().isoformat(), episode_id)
        )
        self.conn.commit()

    def prune_decayed(self, min_score: float = 0.05):
        """Remove episodes that have decayed below significance threshold."""
        c = self.conn.cursor()
        c.execute("SELECT * FROM episodes")
        to_delete = []
        for row in c.fetchall():
            ep = self._row_to_episode(row)
            if ep.significance_score() < min_score and ep.episode_type == "exchange":
                to_delete.append(ep.id)

        for ep_id in to_delete:
            c.execute("DELETE FROM edges WHERE from_id = ? OR to_id = ?", (ep_id, ep_id))
            c.execute("DELETE FROM episodes WHERE id = ?", (ep_id,))

        self.conn.commit()
        return len(to_delete)

    def get_stats(self) -> dict:
        """Memory statistics."""
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM episodes")
        n_episodes = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM edges")
        n_edges = c.fetchone()[0]
        c.execute("SELECT episode_type, COUNT(*) FROM episodes GROUP BY episode_type")
        type_counts = dict(c.fetchall())
        return {
            "total_episodes": n_episodes,
            "total_edges": n_edges,
            "by_type": type_counts,
        }

    def close(self):
        self.conn.close()


# ============================================================
# Stopwords (EN + NL) — shared by keyword search
# ============================================================

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "to", "of", "about",
    "in", "for", "on", "with", "at", "by", "from", "as",
    "and", "but", "or", "not", "no", "so", "if", "than",
    "that", "this", "it", "i", "you", "we", "they", "me", "my",
    "how", "what", "when", "where", "who", "which",
    "de", "het", "een", "en", "van", "in", "is", "dat", "op",
    "te", "zijn", "voor", "met", "die", "er", "niet", "als",
    "aan", "ook", "om", "maar", "bij", "nog", "dan", "wel",
    "je", "jij", "wij", "zij", "ik", "mij", "ons", "hun",
    "over", "uit", "naar", "toen", "daar", "hier", "wat", "hoe",
}


# ============================================================
# Vector helpers (pure Python — no numpy dependency)
# ============================================================

def _vec_to_blob(vec: list) -> bytes:
    """Pack float list to binary blob (compact storage)."""
    return struct.pack(f'{len(vec)}f', *vec)

def _blob_to_vec(blob: bytes) -> list:
    """Unpack binary blob to float list."""
    n = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f'{n}f', blob))

def _cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two vectors. Pure Python, no deps."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# CLI for inspection
if __name__ == "__main__":
    import sys
    graph = EpisodicGraph()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "stats":
            stats = graph.get_stats()
            print(json.dumps(stats, indent=2))
        elif cmd == "recent":
            for ep in graph.get_recent(10):
                print(f"[{ep.id}] ({ep.episode_type}) {ep.content[:80]}... "
                      f"(sig={ep.significance_score():.2f})")
        elif cmd == "significant":
            for ep in graph.get_most_significant(10):
                print(f"[{ep.id}] ({ep.episode_type}) {ep.content[:80]}... "
                      f"(sig={ep.significance_score():.2f})")
        elif cmd == "patterns":
            for ep in graph.find_pattern():
                print(f"[{ep.id}] ({ep.episode_type}) {ep.content[:80]}")
        elif cmd == "backfill":
            count = graph.embed_backfill()
            print(f"Embedded {count} episodes")
        elif cmd == "search" and len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            print(f"Semantic search: '{query}'")
            results = graph.search_semantic_with_scores(query, limit=5)
            if results:
                for ep, sim in results:
                    print(f"  [{ep.id}] (sim={sim:.3f}) {ep.content[:80]}...")
            else:
                print("  No semantic results, trying keyword...")
                for ep in graph._search_keyword(query, limit=5):
                    print(f"  [{ep.id}] {ep.content[:80]}...")
    else:
        stats = graph.get_stats()
        print(f"Episodic Graph: {stats['total_episodes']} episodes, {stats['total_edges']} edges")
        print(f"Types: {stats['by_type']}")

    graph.close()
