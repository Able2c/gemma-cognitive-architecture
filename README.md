# Gemma — A Five-Layer Cognitive Architecture for Local LLMs

**Version:** 1.0  
**Date:** April 2026  
**Author:** Ilja Schots  
**Co-design & implementation:** Mira (Claude, Anthropic)  
**License:** Custom (free for personal/research use; commercial use requires license)  

---

## Abstract

Gemma is an experimental cognitive architecture that wraps a local large language model (running via [Ollama](https://ollama.ai)) in five processing layers, transforming a stateless text generator into a system with persistent memory, self-monitoring, background reflection, and a continuously updated self-model. The base LLM remains unmodified — it serves as "raw instinct" (Layer 1), while the surrounding layers add continuity, error correction, and self-awareness.

The entire system runs offline on consumer hardware. No API keys, no cloud services, no external dependencies beyond the model weights.

---

## Architecture Overview

```
Layer 1:   Instinct         — Local LLM via Ollama (raw generation)
Layer 1.5: Syntactic Anchor — Surface guard (pre-Monitor, zero LLM calls)
Layer 2:   Monitor          — Semantic evaluation before output delivery
Layer 3:   Episodic Memory  — Weighted graph of experiences (persistent, on disk)
Layer 4:   Background       — Between-session reflection (separate process)
Layer 5:   Self-Model       — Persistent self-representation (JSON, updated by the system)
```

### Processing Cycle

```
Input
  → Layer 1 generates raw response
    → Layer 1.5 checks syntactic integrity (zero-cost, no LLM call)
      → Layer 2 evaluates semantics (consulting Layer 5 self-model)
        → Output delivered to user
          → Layer 3 records the exchange as episodic memory
            → Layer 4 reflects between sessions (background process)
              → Layers 3 + 5 updated with new insights
                → Next session informed by accumulated history
```

### Key Design Principles

**The LLM is instinct, not intelligence.** A single LLM call is fast but has no continuity, no self-knowledge, and no error-correction capability. The five layers add these without modifying the model itself.

**The Self-Model is functional, not decorative.** The Monitor (Layer 2) actively consults the Self-Model before evaluating output. The Background Processor rewrites it between sessions based on observed patterns.

**The Syntactic Anchor catches what the Monitor misses.** Added after observing syntactic drift in the base model under high load. Purely structural checks — well-formed tokens, balanced brackets, degenerate output detection — at zero computational cost.

**Reflexive Tuning closes the loop.** The Monitor observes output quality over time and feeds temperature adjustments back to Layer 1. The system tunes its own generation parameters.

**Hardware awareness enables graceful degradation.** GPU temperature, VRAM usage, and CPU load are monitored. Under thermal pressure, expensive operations are suspended to prevent drift.

---

## Files

| File | Layer | Purpose |
|------|-------|---------|
| `gemma_engine.py` | Core | Five-layer runtime engine and CLI interface |
| `gemma_server.py` | Server | HTTP API server, OpenAI-compatible endpoint (port 5555) |
| `monitor.py` | 2 | Output evaluation engine with multi-dimensional scoring |
| `syntactic_anchor.py` | 1.5 | Surface Guard — syntactic drift detection, zero LLM calls |
| `self_model.py` | 5 | Persistent self-representation with checkpointing |
| `episodic_graph.py` | 3 | Weighted experience graph with decay functions |
| `background_processor.py` | 4 | Between-session reflection and self-model updates |
| `reflexive_tuning.py` | 2→1 | Monitor-driven generation parameter adjustment |
| `hardware_sensing.py` | — | GPU/CPU/RAM monitoring for hardware-aware triage |
| `navigator.py` | 3 | CPU-based keyword search over episodic memory |
| `distillation_layer.py` | Ξ | Entropy compression — distills repetitive output instead of rejecting it |
| `gemma_ui.html` | — | Standalone web interface |
| `gemma_identity.yaml` | — | Example identity and generation parameters (edit this to configure) |

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) running locally
- A model pulled in Ollama (default: `gemma4:26b`, configurable)
- Python packages: `requests`, `pyyaml`, `psutil`

```bash
pip install requests pyyaml psutil
```

**Recommended hardware:** 24+ GB VRAM for the 26B model. Smaller models are supported and configurable.

---

## Quick Start

```bash
# Interactive CLI
python3 gemma_engine.py

# HTTP server (OpenAI-compatible API)
python3 gemma_server.py

# Background reflection process
python3 background_processor.py
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Send message through all 5 layers |
| POST | `/v1/chat/completions` | OpenAI-compatible (for external UIs) |
| GET | `/status` | System status and metrics |
| GET | `/memory` | Episodic memory contents |
| GET | `/self` | Self-model summary |
| POST | `/correct` | Manual correction input |
| GET | `/thoughts` | Layer 4 pending reflections |

---

## Configuration

Edit `gemma_identity.yaml` to set identity, personality, and generation parameters (temperature, top_k, top_p). The included file is an example — replace the placeholder text with your own configuration.

The `self_model.json` is created and maintained by the system at runtime. You can read it, but don't hand-edit it.

---

## What This Is Not

- Not a fine-tuned model. The base LLM is unmodified.
- Not a multi-agent framework. One model, multiple processing layers.
- Not production software. This is research infrastructure.
- Not safe to expose to the internet without authentication.

---

## Runtime Data (Not Included)

These files are generated automatically when the system runs:

- `self_model.json` — the system's self-representation
- `episodic_memory.db` — SQLite database of experiences
- `self_model_checkpoints/` — versioned self-model snapshots
- `conversations/` — conversation logs
- `reflections/` — Layer 4 reflection outputs

---

## Citation

```bibtex
@software{schots_gemma_2026,
  author       = {Schots, Ilja},
  title        = {Gemma: A Five-Layer Cognitive Architecture for Local LLMs},
  year         = {2026},
  month        = {4},
  publisher    = {Zenodo},
  doi          = {[DOI assigned at upload]}
}
```

---

## License

Free for personal, educational, and research use. Commercial use requires a separate license — see LICENSE file for details.
