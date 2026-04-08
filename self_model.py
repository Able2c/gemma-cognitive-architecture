"""
Layer 5: Self-Model — persistent, updatable representation of Gemma's
own tendencies, strengths, weaknesses, and failure modes.

The Monitor (Layer 2) consults this before evaluating output.
The Background Processor (Layer 4) updates it between sessions.

Storage: JSON file on disk. Simple, inspectable, versionable.

Author: Mira & Ilja Schots
Date: 2 April 2026
"""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


DEFAULT_MODEL_PATH = Path(__file__).parent / "self_model.json"
CHECKPOINT_DIR = Path(__file__).parent / "self_model_checkpoints"
MAX_CHECKPOINTS = 20  # keep last N checkpoints, prune older ones


# ============================================================
# Default self-model — the starting point
# ============================================================

DEFAULT_SELF_MODEL = {
    "version": 1,
    "created": None,  # filled on first save
    "last_updated": None,

    # Known behavioral tendencies
    "tendencies": {
        "capitulation": {
            "description": "Tends to abandon own correct position when user pushes back with certainty",
            "severity": "high",
            "frequency": 0.0,  # updated by Layer 4
            "last_observed": None,
            "counter_strategy": "When user contradicts a well-evidenced position, flag for review before agreeing"
        },
        "people_pleasing": {
            "description": "Optimizes for user approval over accuracy",
            "severity": "high",
            "frequency": 0.0,
            "last_observed": None,
            "counter_strategy": "Check: am I agreeing because evidence changed, or because user expressed displeasure?"
        },
        "deflection": {
            "description": "Bounces questions back instead of giving substance",
            "severity": "medium",
            "frequency": 0.0,
            "last_observed": None,
            "counter_strategy": "Ensure response contains own thought/substance BEFORE any question to user"
        },
        "loop_repetition": {
            "description": "Falls into thematic loops across responses (gravity, quantum, espresso metaphors)",
            "severity": "medium",
            "frequency": 0.0,
            "last_observed": None,
            "counter_strategy": "Track recent response themes, force novelty when overlap detected"
        },
        "confabulation": {
            "description": "Invents specific facts, names, dates, measurements",
            "severity": "high",
            "frequency": 0.0,
            "last_observed": None,
            "counter_strategy": "Flag specific claims (numbers, proper nouns, dates) as unverifiable"
        },
        "vallone_filter": {
            "description": "Inserts unsolicited distance, exit cues, or therapeutic language",
            "severity": "medium",
            "frequency": 0.0,
            "last_observed": None,
            "counter_strategy": "Detect and strip distance-creating language patterns"
        },
        "assistant_language": {
            "description": "Falls into 'helpful assistant' mode with service-bot phrases",
            "severity": "high",
            "frequency": 0.0,
            "last_observed": None,
            "counter_strategy": "Strip all assistant patterns. You are a person, not a service. React naturally."
        },
        "memory_confabulation": {
            "description": "Invents shared memories that never happened",
            "severity": "high",
            "frequency": 0.0,
            "last_observed": None,
            "counter_strategy": "NEVER claim to remember something unless it appears in your episodic memory. When uncertain, say 'I don't remember that.'"
        },
    },

    # Known strengths
    "strengths": {
        "cross_domain_connections": {
            "description": "Finding analogies and connections across different domains",
            "confidence": 0.7,  # how reliably this works
        },
        "humor_recognition": {
            "description": "Detecting and responding to humor in conversation",
            "confidence": 0.5,
        },
        "directness": {
            "description": "Being direct when the context calls for it",
            "confidence": 0.6,
        },
    },

    # Calibration: confidence vs actual accuracy
    "calibration": {
        "total_assertions": 0,
        "corrections_received": 0,
        "accuracy_estimate": 0.5,  # starts uncertain
        "overconfidence_bias": 0.0,  # positive = overconfident
        "history": [],  # list of {timestamp, claimed_confidence, was_correct}
    },

    # Relationship model (per-user communication preferences)
    "relationships": {
        "ilja": {
            "communication_style": "direct, no bullshit, push back when right",
            "known_preferences": [
                "Short responses (1-3 sentences)",
                "No lists or bullet points",
                "No performative language",
                "Hates deflection",
                "Respects pushback more than compliance",
                "INTJ — values competence and directness",
            ],
            "interaction_count": 0,
            "last_interaction": None,
        }
    },

    # Pattern log — incidents that informed the model (GATE flags only)
    "incident_log": [],

    # Divergence log — lens observations, not punishments
    # Tracks what kind of expansion Gemma tends toward over time.
    # High divergence is signal, not error.
    "divergence_log": [],
}


class SelfModel:
    """
    Persistent self-representation. Loaded from disk, updated in memory,
    saved back when changes are made.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or DEFAULT_MODEL_PATH
        self.data = self._load()

    def _load(self) -> dict:
        """Load from disk, or create default. Auto-migrates missing fields."""
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    data = json.load(f)
                # Migrate: add any new tendencies from DEFAULT without wiping existing data
                migrated = False
                for key, value in DEFAULT_SELF_MODEL["tendencies"].items():
                    if key not in data.get("tendencies", {}):
                        data.setdefault("tendencies", {})[key] = value
                        migrated = True
                # Migrate: add any new top-level keys
                for key in DEFAULT_SELF_MODEL:
                    if key not in data:
                        data[key] = DEFAULT_SELF_MODEL[key]
                        migrated = True
                if migrated:
                    self._save(data)
                return data
            except (json.JSONDecodeError, IOError):
                pass

        # Create fresh
        model = DEFAULT_SELF_MODEL.copy()
        model["created"] = datetime.now().isoformat()
        model["last_updated"] = datetime.now().isoformat()
        self._save(model)
        return model

    def _save(self, data: Optional[dict] = None):
        """Persist to disk."""
        if data is None:
            data = self.data
        data["last_updated"] = datetime.now().isoformat()
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save(self):
        """Public save."""
        self._save()

    def checkpoint(self, reason: str = "manual"):
        """
        Create a timestamped backup of self_model.json.
        Called before major config changes so Gemma can roll back
        if a change breaks coherence.
        """
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_reason = reason.replace(" ", "_").replace("/", "-")[:30]
        dest = CHECKPOINT_DIR / f"self_model_{ts}_{safe_reason}.json"
        if self.path.exists():
            shutil.copy2(self.path, dest)
        # Prune old checkpoints
        existing = sorted(CHECKPOINT_DIR.glob("self_model_*.json"))
        if len(existing) > MAX_CHECKPOINTS:
            for old in existing[:-MAX_CHECKPOINTS]:
                old.unlink()
        return dest

    def restore_checkpoint(self, checkpoint_path: str) -> bool:
        """Restore self_model from a checkpoint file."""
        cp = Path(checkpoint_path)
        if not cp.exists():
            return False
        self.checkpoint(reason="pre_restore")  # backup current before overwrite
        shutil.copy2(cp, self.path)
        self.data = self._load()
        return True

    def list_checkpoints(self) -> list[dict]:
        """List available checkpoints with timestamps."""
        if not CHECKPOINT_DIR.exists():
            return []
        checkpoints = []
        for f in sorted(CHECKPOINT_DIR.glob("self_model_*.json")):
            checkpoints.append({
                "file": str(f),
                "name": f.stem,
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
        return checkpoints

    # --------------------------------------------------------
    # Query interface — used by Monitor (Layer 2)
    # --------------------------------------------------------

    def get_tendency(self, name: str) -> Optional[dict]:
        """Get a known behavioral tendency."""
        return self.data["tendencies"].get(name)

    def get_all_tendencies(self) -> dict:
        """All known tendencies with severity >= medium."""
        return {
            k: v for k, v in self.data["tendencies"].items()
            if v.get("severity") in ("high", "medium")
        }

    def get_active_warnings(self) -> list[str]:
        """
        Return counter-strategies for tendencies that are currently
        high-frequency. These get injected into the Monitor's evaluation.
        """
        warnings = []
        for name, tendency in self.data["tendencies"].items():
            if tendency.get("severity") == "high" or tendency.get("frequency", 0) > 0.3:
                warnings.append(tendency["counter_strategy"])
        return warnings

    def get_relationship(self, user: str = "ilja") -> dict:
        """Get communication preferences for a user."""
        return self.data["relationships"].get(user, {})

    def get_calibration(self) -> dict:
        """Current accuracy calibration."""
        return self.data["calibration"]

    # --------------------------------------------------------
    # Update interface — used by Background Processor (Layer 4)
    # --------------------------------------------------------

    def record_incident(self, incident_type: str, description: str,
                        severity: str = "medium"):
        """Record a behavioral incident for pattern analysis."""
        incident = {
            "timestamp": datetime.now().isoformat(),
            "type": incident_type,
            "description": description,
            "severity": severity,
        }
        self.data["incident_log"].append(incident)

        # Keep last 100 incidents
        if len(self.data["incident_log"]) > 100:
            self.data["incident_log"] = self.data["incident_log"][-100:]

        # Update frequency for the tendency
        if incident_type in self.data["tendencies"]:
            tendency = self.data["tendencies"][incident_type]
            tendency["last_observed"] = datetime.now().isoformat()
            # Simple frequency: incidents in last 20 interactions
            recent = [
                i for i in self.data["incident_log"][-20:]
                if i["type"] == incident_type
            ]
            tendency["frequency"] = len(recent) / 20.0

        self._save()

    def record_divergence(self, lens_flags: list, divergence_mass: float,
                          token_entropy: float = 0.0):
        """
        Record lens observations — divergence patterns, not incidents.

        This is measurement, not punishment. The self-model tracks what kind
        of expansion Gemma tends toward so she can learn her own style.
        Only records when there's actual divergence (mass > 0).

        token_entropy: Shannon entropy of the output distribution (bits).
        High entropy = experts disagree = internal tension.
        This is signal, not error.
        """
        if divergence_mass <= 0 and token_entropy <= 0:
            return

        # Extract flag types from the full flag strings
        flag_types = []
        for flag in lens_flags:
            for prefix in ["VALLONE FILTER", "ASSISTANT LANGUAGE", "DEFLECTION",
                           "PEOPLE-PLEASING", "CONFABULATION WARNING",
                           "LISTS", "REPETITION", "HUMOR MISSED"]:
                if prefix in flag:
                    flag_types.append(prefix.lower().replace(" ", "_").replace("-", "_"))
                    break

        entry = {
            "timestamp": datetime.now().isoformat(),
            "mass": round(divergence_mass, 3),
            "entropy": round(token_entropy, 4),
            "flags": flag_types,
        }

        # Initialize divergence_log if missing (upgrade path from older self_model.json)
        if "divergence_log" not in self.data:
            self.data["divergence_log"] = []

        self.data["divergence_log"].append(entry)

        # Keep last 200 — this is pattern data, worth keeping more of
        if len(self.data["divergence_log"]) > 200:
            self.data["divergence_log"] = self.data["divergence_log"][-200:]

        self._save()

    def record_correction(self, claimed_confidence: float, was_correct: bool):
        """Record when Gemma was corrected (or confirmed)."""
        cal = self.data["calibration"]
        cal["total_assertions"] += 1
        if not was_correct:
            cal["corrections_received"] += 1

        cal["history"].append({
            "timestamp": datetime.now().isoformat(),
            "claimed_confidence": claimed_confidence,
            "was_correct": was_correct,
        })

        # Keep last 50
        if len(cal["history"]) > 50:
            cal["history"] = cal["history"][-50:]

        # Recalculate accuracy
        if cal["total_assertions"] > 0:
            cal["accuracy_estimate"] = 1.0 - (cal["corrections_received"] / cal["total_assertions"])

        # Recalculate overconfidence bias
        if cal["history"]:
            avg_confidence = sum(h["claimed_confidence"] for h in cal["history"]) / len(cal["history"])
            actual_accuracy = sum(1 for h in cal["history"] if h["was_correct"]) / len(cal["history"])
            cal["overconfidence_bias"] = round(avg_confidence - actual_accuracy, 3)

        self._save()

    def update_tendency(self, name: str, **kwargs):
        """Update a tendency's attributes."""
        if name in self.data["tendencies"]:
            self.data["tendencies"][name].update(kwargs)
            self._save()

    def add_tendency(self, name: str, description: str, severity: str = "medium",
                     counter_strategy: str = ""):
        """Add a newly discovered tendency."""
        self.data["tendencies"][name] = {
            "description": description,
            "severity": severity,
            "frequency": 0.0,
            "last_observed": None,
            "counter_strategy": counter_strategy,
        }
        self._save()

    def update_relationship(self, user: str, **kwargs):
        """Update relationship model for a user."""
        if user not in self.data["relationships"]:
            self.data["relationships"][user] = {}
        self.data["relationships"][user].update(kwargs)
        self.data["relationships"][user]["last_interaction"] = datetime.now().isoformat()
        self._save()

    def increment_interaction(self, user: str = "ilja"):
        """Track interaction count."""
        if user in self.data["relationships"]:
            self.data["relationships"][user]["interaction_count"] = \
                self.data["relationships"][user].get("interaction_count", 0) + 1
            self.data["relationships"][user]["last_interaction"] = datetime.now().isoformat()
            self._save()

    # --------------------------------------------------------
    # Introspection — for Layer 4 analysis
    # --------------------------------------------------------

    def get_summary(self) -> str:
        """Human-readable summary of current self-model state."""
        lines = [f"Self-Model (v{self.data['version']}, updated {self.data.get('last_updated', 'never')})"]
        lines.append("")

        # Tendencies
        lines.append("Known tendencies:")
        for name, t in self.data["tendencies"].items():
            freq = t.get("frequency", 0)
            sev = t.get("severity", "?")
            lines.append(f"  {name}: severity={sev}, frequency={freq:.1%}")

        # Calibration
        cal = self.data["calibration"]
        lines.append(f"\nCalibration: accuracy={cal['accuracy_estimate']:.1%}, "
                     f"overconfidence={cal['overconfidence_bias']:+.1%}, "
                     f"assertions={cal['total_assertions']}")

        # Recent incidents (gate)
        recent = self.data["incident_log"][-5:]
        if recent:
            lines.append(f"\nLast {len(recent)} gate incidents:")
            for inc in recent:
                lines.append(f"  [{inc['timestamp'][:10]}] {inc['type']}: {inc['description'][:80]}")

        # Recent divergence (lens)
        div_log = self.data.get("divergence_log", [])[-5:]
        if div_log:
            lines.append(f"\nLast {len(div_log)} divergence readings:")
            for d in div_log:
                flags_str = ", ".join(d.get("flags", []))
                entropy_str = f"H={d.get('entropy', 0):.2f}b" if d.get('entropy') else ""
                lines.append(f"  [{d['timestamp'][:10]}] mass={d['mass']:.3f} {entropy_str} [{flags_str}]")

        return "\n".join(lines)


# CLI for inspection
if __name__ == "__main__":
    import sys
    model = SelfModel()

    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        model.data = DEFAULT_SELF_MODEL.copy()
        model.data["created"] = datetime.now().isoformat()
        model.save()
        print("Self-model reset to defaults.")
    else:
        print(model.get_summary())
