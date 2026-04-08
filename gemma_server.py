#!/usr/bin/env python3
"""
Gemma v3.1 — HTTP Backend

Wraps the five-layer cognitive architecture as a simple HTTP API
for the Nova-based web UI.

v3.1 changes:
  - Navigator integration: CPU-based keyword search for episodic memory.
  - Thread-safe session management (SessionManager, no global state).
  - Streaming-safe deduplication for Open WebUI.
  - Anchor limiter prevents VRAM death spiral.

Endpoints:
    POST /chat              — send message, get response through all 5 layers
    POST /v1/chat/completions — OpenAI-compatible (for Open WebUI)
    GET  /status            — system status (memory, self-model, ollama, navigator)
    GET  /memory            — episodic memory contents
    GET  /self              — self-model summary
    POST /correct           — manual correction
    POST /reset-session     — clear session messages (keeps persistent memory)
    GET  /models            — available Ollama models
    GET  /thoughts          — Layer 4 pending thoughts
    GET  /history           — list all saved conversations
    GET  /history/<id>      — load a specific conversation
    POST /save-session      — manually save current session
    GET  /navigator         — navigator index status

Usage:
    python3 gemma_server.py                    # start on port 5555
    python3 gemma_server.py --port 8080        # custom port
    python3 gemma_server.py --model gemma4:26b # override model

Author: Mira & Ilja Schots / SessionManager architecture: Lilith
Date: 5 April 2026
"""

import json
import os
import sys
import time
import uuid
import requests as http_requests
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import hashlib

# Import five-layer architecture
from self_model import SelfModel
from episodic_graph import EpisodicGraph
from monitor import Monitor
from background_processor import BackgroundProcessor
from navigator import Navigator
from gemma_engine import (
    save_state_snapshot, load_state_snapshot,
    build_system_prompt, generate_with_layers,
    handle_correction, start_session, end_session, load_session_bridge, MODEL
)
import gemma_engine


# ============================================================
# Input sanitization — WebUI metadata filter
# ============================================================
# Open WebUI injects task instructions as user messages:
#   "### Task: Suggest 3-5 follow-up questions..."
#   "### Task: Generate a concise title..."
#   "### Guidelines: ..."
# These are NOT user input. If they reach the pipeline, they
# get stored as corrections and poison episodic memory.
# This filter catches and discards them.

_WEBUI_PATTERNS = [
    "### Task:",
    "### Guidelines:",
    "### Instructions:",
    "```json\n{\"follow_ups\"",
    "Generate a concise, 3-5 word title",
    "Generate 1-3 broad tags categorizing",
    "Suggest 3-5 relevant follow-up questions",
]


def sanitize_user_input(text: str) -> str | None:
    """Filter out WebUI metadata injections.

    Returns cleaned text, or None if the entire message is WebUI metadata
    (meaning it should be silently dropped, not processed).
    """
    if not text:
        return None

    # Check if the message IS a WebUI task instruction
    for pattern in _WEBUI_PATTERNS:
        if pattern in text:
            logging.getLogger("input_filter").info(
                f"Blocked WebUI metadata injection: {text[:80]}..."
            )
            return None

    return text


# ============================================================
# Conversation storage
# ============================================================

HISTORY_DIR = Path(__file__).parent / "conversations"
HISTORY_DIR.mkdir(exist_ok=True)


def save_conversation(session_id: str, messages: list, metadata: dict = None):
    """Save a conversation to disk."""
    if not messages:
        return None

    data = {
        "id": session_id,
        "created": metadata.get("created", datetime.now().isoformat()) if metadata else datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
        "model": gemma_engine.MODEL,
        "message_count": len(messages),
        "messages": messages,
        "preview": _get_preview(messages),
    }

    filepath = HISTORY_DIR / f"{session_id}.json"
    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return filepath


def load_conversation(session_id: str) -> dict:
    """Load a conversation from disk."""
    filepath = HISTORY_DIR / f"{session_id}.json"
    if filepath.exists():
        return json.loads(filepath.read_text(encoding="utf-8"))
    return None


def list_conversations(limit: int = 50) -> list:
    """List all saved conversations, newest first."""
    convos = []
    for f in sorted(HISTORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            convos.append({
                "id": data.get("id", f.stem),
                "created": data.get("created", ""),
                "updated": data.get("updated", ""),
                "model": data.get("model", "unknown"),
                "message_count": data.get("message_count", 0),
                "preview": data.get("preview", ""),
            })
        except Exception:
            continue
    return convos[:limit]


def _get_preview(messages: list) -> str:
    """Get a preview string from the first user message."""
    for msg in messages:
        if msg.get("role") == "user":
            text = msg.get("content", "")[:80]
            return text.rstrip() + ("..." if len(msg.get("content", "")) > 80 else "")
    return "(empty)"


# ============================================================
# Thread-Safe Session Management (architecture: Lilith)
# ============================================================

class Session:
    """A single chat session with its own message history and lock."""

    def __init__(self, session_id: str, created: str):
        self.id = session_id
        self.created = created
        self.messages = []
        self.system_prompt = ""
        self.lock = threading.Lock()


class SessionManager:
    """
    Manages active sessions to prevent global state corruption
    from parallel requests (e.g. Open WebUI firing multiple
    requests during page refresh + chat + housekeeping).
    """

    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()
        self.default_session_id = None

    def create_session(self) -> Session:
        """Create a new session and set it as default."""
        with self.lock:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            session = Session(session_id, datetime.now().isoformat())
            self.sessions[session_id] = session
            self.default_session_id = session_id
            return session

    def get_session(self, session_id: str) -> Session:
        """Get a session by ID, or None."""
        with self.lock:
            return self.sessions.get(session_id)

    def get_default(self) -> Session:
        """Get the default (most recent) session, or create one."""
        with self.lock:
            if self.default_session_id and self.default_session_id in self.sessions:
                return self.sessions[self.default_session_id]
        return self.create_session()

    def resolve_from_history(self, incoming_messages: list) -> Session:
        """
        Map Open WebUI's stateless requests to internal stateful sessions.
        Open WebUI sends the full conversation history with each request.
        We match user messages to find the corresponding internal session.
        """
        if not incoming_messages:
            return self.get_default()

        user_msgs = [m["content"] for m in incoming_messages[:-1]
                     if m.get("role") == "user"]

        with self.lock:
            for sess in self.sessions.values():
                sess_user_msgs = [m["content"] for m in sess.messages
                                  if m.get("role") == "user"]
                # Exact match
                if sess_user_msgs == user_msgs:
                    return sess
                # Partial match (Open WebUI may truncate history)
                if (len(user_msgs) > 0 and len(sess_user_msgs) >= len(user_msgs)
                        and sess_user_msgs[-len(user_msgs):] == user_msgs):
                    return sess

        # No match — create new session
        return self.create_session()

    def save_all(self):
        """Save all active sessions. Called on shutdown."""
        with self.lock:
            for sess in self.sessions.values():
                if sess.messages:
                    save_conversation(sess.id, sess.messages,
                                      {"created": sess.created})


# ============================================================
# Globals — shared resources (thread-safe or immutable)
# ============================================================

self_model = None
graph = None
monitor = None
navigator = None
session_mgr = SessionManager()
pending_thoughts = []


def init_layers():
    """Initialize all five layers + Navigator. Detects reconnect vs fresh start."""
    global self_model, graph, monitor, navigator, pending_thoughts

    self_model = SelfModel()
    graph = EpisodicGraph()
    monitor = Monitor(self_model, graph)
    navigator = Navigator()

    # Start session — check for Layer 4 pending thoughts
    pending_thoughts = start_session(graph, self_model)

    # ---- Reconnect detection ----
    snapshot, age = load_state_snapshot()
    sess = session_mgr.create_session()

    if snapshot:
        age_str = f"{int(age)}s" if age < 60 else f"{int(age/60)}m{int(age%60)}s"
        print(f"  RECONNECT detected: session interrupted {age_str} ago")
        print(f"  Topics: {', '.join(snapshot.get('topics', ['none']))}")
        print(f"  Tone: {snapshot.get('tone', 'unknown')}")
        sess.system_prompt = build_system_prompt(
            graph, self_model, monitor, state_snapshot=snapshot,
            skip_memory=navigator_active()
        )
        # Restore session ID from snapshot if available
        old_id = snapshot.get("session_id", "")
        if old_id:
            with session_mgr.lock:
                del session_mgr.sessions[sess.id]
                sess.id = old_id
                session_mgr.sessions[sess.id] = sess
                session_mgr.default_session_id = sess.id
        sess.created = snapshot.get("timestamp", sess.created)
    else:
        bridge = load_session_bridge()
        sess.system_prompt = build_system_prompt(
            graph, self_model, monitor, session_bridge=bridge,
            skip_memory=navigator_active()
        )
        if bridge:
            print(f"  Session bridge: loaded previous conversation context")

    # Navigator status
    nav_status = navigator.get_status()
    print(f"  Navigator: {nav_status['total_episodes']} indexed, {nav_status['anchors']} anchors")

    return True


def get_navigated_prompt(user_input: str, base_prompt: str) -> str:
    """
    Build a system prompt with Navigator context prepended.
    CPU-only — no LLM overhead.
    """
    if navigator is None:
        return base_prompt

    nav_context = navigator.navigate_and_build(user_input, top_n=10)

    if not nav_context:
        return base_prompt

    return nav_context + "\n\n" + base_prompt


def navigator_active() -> bool:
    """
    Check if navigator has content.
    When True, build_system_prompt should skip its own memory loading
    to avoid double-loading episodes into context.
    When False (empty index, first run), fall back to graph's built-in memory.
    """
    return navigator is not None and navigator.get_total_episodes() > 0


# ============================================================
# SSE helper
# ============================================================

def send_sse_response(handler, response_text: str, resp_id: str):
    """Send a complete response as SSE chunks. Used for streaming and dedup."""
    try:
        handler.send_response(200)
        handler.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        handler.send_header('Cache-Control', 'no-cache')
        handler._cors_headers()
        handler.end_headers()

        chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "Gemma",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": response_text},
                "finish_reason": None,
            }],
        }
        handler.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode('utf-8'))

        done_chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "Gemma",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        handler.wfile.write(f"data: {json.dumps(done_chunk, ensure_ascii=False)}\n\n".encode('utf-8'))
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()
    except BrokenPipeError:
        pass


# ============================================================
# Request handler
# ============================================================

class GemmaHandler(BaseHTTPRequestHandler):
    """HTTP handler for Gemma API."""

    def _cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')

    def _json_response(self, data, status=200):
        try:
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
        except BrokenPipeError:
            pass

    def _read_body(self):
        length = int(self.headers.get('Content-Length', 0))
        if length:
            return json.loads(self.rfile.read(length))
        return {}

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == '/v1/models':
            self._handle_openai_models()
        elif path == '/status':
            self._handle_status()
        elif path == '/memory':
            self._handle_memory()
        elif path == '/self':
            self._handle_self()
        elif path == '/models':
            self._handle_models()
        elif path == '/thoughts':
            self._handle_thoughts()
        elif path == '/navigator':
            self._handle_navigator()
        elif path == '/history':
            self._handle_history_list()
        elif path.startswith('/history/'):
            conv_id = path.split('/history/')[-1]
            self._handle_history_load(conv_id)
        elif path == '/':
            self._serve_ui()
        else:
            self._json_response({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path

        if path == '/v1/chat/completions':
            self._handle_openai_chat()
        elif path == '/chat':
            self._handle_chat()
        elif path == '/correct':
            self._handle_correct()
        elif path == '/reset-session':
            self._handle_reset()
        elif path == '/save-session':
            self._handle_save()
        elif path == '/model':
            self._handle_set_model()
        elif path == '/upload':
            self._handle_upload()
        else:
            self._json_response({"error": "not found"}, 404)

    # ---- File/Image Upload ----

    @staticmethod
    def _parse_multipart(body: bytes, boundary: bytes):
        """Parse multipart/form-data without cgi.FieldStorage.
        Returns dict of {field_name: {"filename": str|None, "data": bytes}}.
        """
        parts = body.split(b'--' + boundary)
        fields = {}
        for part in parts:
            if part in (b'', b'--', b'--\r\n', b'\r\n'):
                continue
            part = part.strip(b'\r\n')
            if b'\r\n\r\n' not in part:
                continue
            header_block, data = part.split(b'\r\n\r\n', 1)
            # Strip trailing boundary closer
            if data.endswith(b'\r\n'):
                data = data[:-2]

            headers_str = header_block.decode('utf-8', errors='replace')
            # Extract field name and optional filename
            name = None
            filename = None
            for line in headers_str.split('\r\n'):
                if 'content-disposition' in line.lower():
                    for token in line.split(';'):
                        token = token.strip()
                        if token.startswith('name='):
                            name = token.split('=', 1)[1].strip('"')
                        elif token.startswith('filename='):
                            filename = token.split('=', 1)[1].strip('"')
            if name:
                fields[name] = {"filename": filename, "data": data}
        return fields

    def _handle_upload(self):
        """POST /upload — Handle file/image uploads from Gemma UI.

        Accepts multipart form data. Images are base64-encoded and passed
        to Ollama (Gemma 4 is multimodal). Other files are read as text
        and injected as context.
        """
        import base64

        content_type = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in content_type:
            self._json_response({"error": "Expected multipart/form-data"}, 400)
            return

        # Extract boundary from Content-Type header
        boundary = None
        for part in content_type.split(';'):
            part = part.strip()
            if part.startswith('boundary='):
                boundary = part.split('=', 1)[1].strip('"').encode()
                break

        if not boundary:
            self._json_response({"error": "No boundary in Content-Type"}, 400)
            return

        # Read raw body and parse multipart
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        fields = self._parse_multipart(body, boundary)

        file_field = fields.get('file')
        if not file_field or not file_field.get('filename'):
            self._json_response({"error": "No file uploaded"}, 400)
            return

        filename = file_field['filename']
        file_data = file_field['data']
        file_ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

        # Image types — send to Ollama as base64 (Gemma 4 is multimodal)
        image_exts = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}

        if file_ext in image_exts:
            b64 = base64.b64encode(file_data).decode('utf-8')

            # Store in session for the next chat message
            sess = session_mgr.get_default()
            with sess.lock:
                if not hasattr(sess, 'pending_images'):
                    sess.pending_images = []
                sess.pending_images.append({
                    "filename": filename,
                    "base64": b64,
                    "size": len(file_data),
                })

            self._json_response({
                "status": "uploaded",
                "type": "image",
                "filename": filename,
                "size": len(file_data),
                "message": f"Afbeelding '{filename}' klaar. Stuur een bericht om Gemma ernaar te laten kijken.",
            })

        else:
            # Text-based files — read and inject as context
            try:
                text_content = file_data.decode('utf-8')
            except UnicodeDecodeError:
                text_content = file_data.decode('latin-1')

            # Truncate if very large
            max_chars = 50000  # ~12K tokens
            if len(text_content) > max_chars:
                text_content = text_content[:max_chars] + f"\n\n[... bestand afgekapt na {max_chars} karakters ...]"

            sess = session_mgr.get_default()
            with sess.lock:
                if not hasattr(sess, 'pending_files'):
                    sess.pending_files = []
                sess.pending_files.append({
                    "filename": filename,
                    "content": text_content,
                })

            self._json_response({
                "status": "uploaded",
                "type": "text",
                "filename": filename,
                "size": len(file_data),
                "chars": len(text_content),
                "message": f"Bestand '{filename}' geladen ({len(text_content)} karakters). Stuur een bericht om Gemma het te laten analyseren.",
            })

    # ---- Chat (Native UI) ----

    def _handle_chat(self):
        body = self._read_body()
        raw_input = body.get("message", "").strip()
        sess_id = body.get("session_id")

        # Sanitize: block WebUI metadata injections
        user_input = sanitize_user_input(raw_input)
        if not user_input:
            self._json_response({"response": "", "filtered": True}, 200)
            return

        # Get or create session
        sess = None
        if sess_id:
            sess = session_mgr.get_session(sess_id)
        if not sess:
            sess = session_mgr.get_default()

        with sess.lock:
            # Deduplication guard
            if sess.messages:
                last_msg = sess.messages[-1]
                if last_msg.get("role") == "user" and last_msg.get("content") == user_input:
                    last_resp = ""
                    for msg in reversed(sess.messages):
                        if msg.get("role") == "assistant":
                            last_resp = msg["content"]
                            break
                    if last_resp:
                        self._json_response({
                            "response": last_resp, "elapsed": 0, "verdict": {},
                            "session_length": len(sess.messages), "deduplicated": True,
                        })
                        return

            # Check for implicit correction
            if sess.messages:
                last_resp = ""
                for msg in reversed(sess.messages):
                    if msg["role"] == "assistant":
                        last_resp = msg["content"]
                        break
                handle_correction(user_input, last_resp, graph, self_model)

            sess.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat(),
            })

            t0 = time.time()

            # Inject pending file uploads as context
            enriched_input = user_input
            if hasattr(sess, 'pending_files') and sess.pending_files:
                for pf in sess.pending_files:
                    enriched_input += f"\n\n[UPLOADED FILE: {pf['filename']}]\n{pf['content']}\n[END OF FILE]"
                sess.pending_files.clear()

            # Pending images — pass to Ollama via images parameter
            pending_images = []
            if hasattr(sess, 'pending_images') and sess.pending_images:
                pending_images = [pi['base64'] for pi in sess.pending_images]
                img_names = [pi['filename'] for pi in sess.pending_images]
                enriched_input += f"\n\n[UPLOADED IMAGE(S): {', '.join(img_names)}]"
                sess.pending_images.clear()

            # Navigator: build context-aware prompt
            navigated_prompt = get_navigated_prompt(enriched_input, sess.system_prompt)

            # Five-layer generation
            response, verdict = generate_with_layers(
                user_input=enriched_input,
                system_prompt=navigated_prompt,
                session_messages=sess.messages,
                monitor=monitor,
                graph=graph,
                self_model=self_model,
            )

            elapsed = time.time() - t0

            sess.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat(),
                "elapsed": round(elapsed, 2),
                "flags": verdict.flags if verdict.flags else [],
            })

            # State snapshot
            save_state_snapshot(sess.id, sess.messages)

            # Rebuild system prompt periodically
            if len(sess.messages) % 5 == 0:
                sess.system_prompt = build_system_prompt(
                    graph, self_model, monitor,
                    skip_memory=navigator_active()
                )
                navigator.reload()

            # Auto-save every 6 messages
            if len(sess.messages) % 6 == 0:
                save_conversation(sess.id, sess.messages,
                                  {"created": sess.created})

        self._json_response({
            "response": response,
            "elapsed": round(elapsed, 2),
            "verdict": verdict.to_dict(),
            "session_length": len(sess.messages),
            "session_id": sess.id,
        })

    # ---- History ----

    def _handle_history_list(self):
        convos = list_conversations(limit=50)
        self._json_response({"conversations": convos})

    def _handle_history_load(self, conv_id):
        data = load_conversation(conv_id)
        if data:
            self._json_response(data)
        else:
            self._json_response({"error": "conversation not found"}, 404)

    def _handle_save(self):
        sess = session_mgr.get_default()
        with sess.lock:
            if sess.messages:
                save_conversation(sess.id, sess.messages,
                                  {"created": sess.created})
                self._json_response({
                    "status": "saved",
                    "id": sess.id,
                    "messages": len(sess.messages),
                })
            else:
                self._json_response({"error": "no messages to save"}, 400)

    # ---- Status & Info ----

    def _handle_status(self):
        stats = graph.get_stats()
        cal = self_model.get_calibration()

        # Check Ollama
        ollama_ok = False
        ollama_models = []
        try:
            resp = http_requests.get("http://localhost:11434/api/tags", timeout=10)
            if resp.ok:
                ollama_ok = True
                ollama_models = [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass

        # Navigator status
        nav_status = navigator.get_status() if navigator else {}

        # Hardware sensing (P3.2)
        hw_data = {}
        try:
            from hardware_sensing import read_hardware
            metrics, state, _ctx = read_hardware()
            hw_data = {
                "metrics": metrics,
                "state": state,
            }
        except Exception:
            pass

        # Session info
        sess = session_mgr.get_default()

        self._json_response({
            "model": gemma_engine.MODEL,
            "ollama": ollama_ok,
            "available_models": ollama_models,
            "episodic_memory": stats,
            "calibration": {
                "accuracy": cal.get("accuracy_estimate", 0),
                "overconfidence": cal.get("overconfidence_bias", 0),
                "assertions": cal.get("total_assertions", 0),
            },
            "session_id": sess.id,
            "session_messages": len(sess.messages),
            "active_sessions": len(session_mgr.sessions),
            "tendencies": {
                name: {"severity": t["severity"], "frequency": t["frequency"]}
                for name, t in self_model.data.get("tendencies", {}).items()
            },
            "navigator": nav_status,
            "hardware": hw_data,
        })

    def _handle_memory(self):
        significant = graph.get_most_significant(20)
        recent = graph.get_recent(20)

        seen = set()
        episodes = []
        for ep in significant + recent:
            if ep.id not in seen:
                seen.add(ep.id)
                episodes.append({
                    "id": ep.id,
                    "content": ep.content,
                    "type": ep.episode_type,
                    "significance": round(ep.significance_score(), 3),
                    "emotional_weight": ep.emotional_weight,
                    "created": ep.created_at,
                    "tags": ep.tags,
                    "lesson": ep.lesson,
                })

        stats = graph.get_stats()
        self._json_response({
            "episodes": episodes,
            "total": stats["total_episodes"],
            "edges": stats["total_edges"],
            "by_type": stats["by_type"],
        })

    def _handle_self(self):
        self._json_response({
            "summary": self_model.get_summary(),
            "tendencies": self_model.data.get("tendencies", {}),
            "calibration": self_model.data.get("calibration", {}),
            "relationships": self_model.data.get("relationships", {}),
            "incident_count": len(self_model.data.get("incident_log", [])),
            "recent_incidents": self_model.data.get("incident_log", [])[-10:],
        })

    def _handle_models(self):
        try:
            resp = http_requests.get("http://localhost:11434/api/tags", timeout=10)
            models = [m["name"] for m in resp.json().get("models", [])]
            self._json_response({"models": models, "current": gemma_engine.MODEL})
        except Exception as e:
            self._json_response({"error": str(e), "models": []}, 500)

    def _handle_set_model(self):
        """POST /model — Switch the active Ollama model."""
        body = self._read_body()
        new_model = body.get("model", "").strip()
        if not new_model:
            self._json_response({"error": "No model specified"}, 400)
            return
        old_model = gemma_engine.MODEL
        gemma_engine.MODEL = new_model
        print(f"  Model switched: {old_model} → {new_model}")
        self._json_response({"status": "ok", "model": new_model})

    def _handle_thoughts(self):
        self._json_response({"thoughts": pending_thoughts})

    def _handle_navigator(self):
        """GET /navigator — navigator index status and sample."""
        if navigator is None:
            self._json_response({"error": "navigator not initialized"}, 500)
            return

        status = navigator.get_status()
        episodes = navigator.index.get("episodes", {})
        sample = {}
        count = 0
        for ep_id, tags in episodes.items():
            if count >= 20:
                break
            sample[ep_id] = tags
            count += 1

        self._json_response({
            "status": status,
            "sample_episodes": sample,
        })

    def _handle_correct(self):
        """Manual correction endpoint."""
        body = self._read_body()
        correction = body.get("correction", "").strip()

        sess = session_mgr.get_default()

        with sess.lock:
            if not correction or not sess.messages:
                self._json_response({"error": "no correction or no session"}, 400)
                return

            last_resp = ""
            for msg in reversed(sess.messages):
                if msg["role"] == "assistant":
                    last_resp = msg["content"]
                    break

            graph.store_correction(
                what_happened=f"Gemma said: {last_resp[:150]}",
                lesson=f"Ilja corrected: {correction}",
            )
            self_model.record_correction(0.7, was_correct=False)

        self._json_response({"status": "correction recorded"})

    def _handle_reset(self):
        """Reset session — save current, start fresh."""
        old_sess = session_mgr.get_default()

        with old_sess.lock:
            if old_sess.messages:
                save_conversation(old_sess.id, old_sess.messages,
                                  {"created": old_sess.created})
                end_session(old_sess.messages, graph, self_model)

        # Create fresh session with bridge
        new_sess = session_mgr.create_session()
        bridge = load_session_bridge()
        new_sess.system_prompt = build_system_prompt(
            graph, self_model, monitor, session_bridge=bridge,
            skip_memory=navigator_active()
        )

        if navigator:
            navigator.reload()

        self._json_response({"status": "session reset", "new_session_id": new_sess.id})

    # ---- OpenAI-compatible endpoints (for Open WebUI) ----

    def _handle_openai_models(self):
        """GET /v1/models — list Gemma as an available model."""
        self._json_response({
            "object": "list",
            "data": [{
                "id": "Gemma",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "bilateral-coupling",
                "permission": [],
                "root": "Gemma",
                "parent": None,
            }],
        })

    def _handle_openai_chat(self):
        """POST /v1/chat/completions — OpenAI-compatible chat endpoint.

        Translates OpenAI message format → Gemma five-layer pipeline →
        OpenAI response format. All five layers remain fully active.
        Open WebUI thinks it's talking to an OpenAI model.
        """
        body = self._read_body()
        messages = body.get("messages", [])
        stream = body.get("stream", False)

        if not messages:
            self._json_response({
                "error": {"message": "No messages provided",
                          "type": "invalid_request_error"}
            }, 400)
            return

        # Extract last user message
        user_input = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    user_input = " ".join(
                        part.get("text", "") for part in content
                        if part.get("type") == "text"
                    )
                else:
                    user_input = content.strip()
                break

        # Sanitize: block WebUI metadata injections
        user_input = sanitize_user_input(user_input)
        if not user_input:
            # Return empty response — WebUI will ignore it
            self._json_response({
                "id": f"chatcmpl-filtered",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
                "model": "Gemma",
            })
            return

        if not user_input:
            self._json_response({
                "error": {"message": "No user message found",
                          "type": "invalid_request_error"}
            }, 400)
            return

        # ---- Open WebUI housekeeping filter ----
        housekeeping_signals = [
            "### Task:", "### Guidelines:", "### Output:\nJSON format:",
            "Title:", "Summary:", "Generate a tags list",
        ]
        if any(signal in user_input for signal in housekeeping_signals):
            from gemma_engine import layer1_generate
            raw_response = layer1_generate(
                system="You are a helpful assistant. Respond exactly as instructed.",
                prompt=user_input,
                temperature=0.3,
            )
            resp_id = "chatcmpl-housekeeping-" + hashlib.md5(
                raw_response[:50].encode()
            ).hexdigest()[:8]
            self._json_response({
                "id": resp_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "Gemma",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": raw_response},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0},
            })
            return

        # ---- Resolve to internal session (thread-safe) ----
        sess = session_mgr.resolve_from_history(messages)

        with sess.lock:
            # ---- Deduplication guard (streaming-safe) ----
            if sess.messages:
                last_msg = sess.messages[-1]
                if (last_msg.get("role") == "user"
                        and last_msg.get("content") == user_input):
                    print(f"  [dedup] Duplicate: {user_input[:60]}...")
                    last_resp = ""
                    for msg in reversed(sess.messages):
                        if msg.get("role") == "assistant":
                            last_resp = msg["content"]
                            break
                    if last_resp:
                        resp_id = "chatcmpl-dedup-" + hashlib.md5(
                            last_resp[:50].encode()
                        ).hexdigest()[:8]

                        if stream:
                            send_sse_response(self, last_resp, resp_id)
                        else:
                            self._json_response({
                                "id": resp_id,
                                "object": "chat.completion",
                                "created": int(time.time()),
                                "model": "Gemma",
                                "choices": [{
                                    "index": 0,
                                    "message": {"role": "assistant",
                                                "content": last_resp},
                                    "finish_reason": "stop",
                                }],
                                "usage": {"prompt_tokens": 0,
                                          "completion_tokens": 0,
                                          "total_tokens": 0},
                            })
                        return

            # Check for implicit correction
            if sess.messages:
                last_resp = ""
                for msg in reversed(sess.messages):
                    if msg["role"] == "assistant":
                        last_resp = msg["content"]
                        break
                handle_correction(user_input, last_resp, graph, self_model)

            sess.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat(),
            })

            t0 = time.time()

            # Navigator: context-aware prompt
            navigated_prompt = get_navigated_prompt(user_input, sess.system_prompt)

            # Five-layer generation
            response, verdict = generate_with_layers(
                user_input=user_input,
                system_prompt=navigated_prompt,
                session_messages=sess.messages,
                monitor=monitor,
                graph=graph,
                self_model=self_model,
            )

            elapsed = time.time() - t0

            sess.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat(),
                "elapsed": round(elapsed, 2),
                "flags": verdict.flags if verdict.flags else [],
            })

            # State snapshot
            save_state_snapshot(sess.id, sess.messages)

            # Periodic maintenance
            if len(sess.messages) % 5 == 0:
                sess.system_prompt = build_system_prompt(
                    graph, self_model, monitor,
                    skip_memory=navigator_active()
                )
                navigator.reload()

            if len(sess.messages) % 6 == 0:
                save_conversation(sess.id, sess.messages,
                                  {"created": sess.created})

        # ---- Build response ----
        resp_id = "chatcmpl-" + hashlib.md5(
            f"{sess.id}-{len(sess.messages)}".encode()
        ).hexdigest()[:12]

        openai_response = {
            "id": resp_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "Gemma",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        if stream:
            send_sse_response(self, response, resp_id)
        else:
            self._json_response(openai_response)

    # ---- UI ----

    def _serve_ui(self):
        """Serve the web UI."""
        ui_path = Path(__file__).parent / "gemma_ui.html"
        if ui_path.exists():
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self._cors_headers()
                self.end_headers()
                self.wfile.write(ui_path.read_bytes())
            except BrokenPipeError:
                pass
        else:
            self._json_response({
                "message": "Gemma API running",
                "endpoints": ["/chat", "/status", "/memory", "/self",
                              "/models", "/history", "/save-session",
                              "/navigator"],
                "ui": "Place gemma_ui.html in same directory to serve UI at /",
            })

    def log_message(self, format, *args):
        """Quieter logging — only errors."""
        if args and '404' in str(args[0]):
            return
        msg = args[0] if args else format
        if 'POST /chat' in str(msg):
            return
        super().log_message(format, *args)


# ============================================================
# Background Processor Daemon Thread
# ============================================================

BACKGROUND_INTERVAL_HOURS = 6  # default: run every 6 hours

def _background_loop(interval_hours: float = BACKGROUND_INTERVAL_HOURS):
    """
    Daemon thread that runs background processing (Layer 4 reflection)
    on a configurable timer. Replaces cron job dependency.
    Skips cycle if Ollama is busy generating (avoids VRAM contention).
    """
    interval_secs = interval_hours * 3600
    print(f"  [Layer 4] Background daemon started (every {interval_hours}h)")

    while True:
        time.sleep(interval_secs)
        try:
            print(f"\n  [Layer 4] Starting background reflection cycle...")
            proc = BackgroundProcessor()
            result = proc.run_cycle(verbose=True)
            status = result.get("status", "unknown")
            elapsed = result.get("elapsed_seconds", 0)
            print(f"  [Layer 4] Reflection complete: {status} ({elapsed}s)")

            # Reload navigator index after tagging
            if navigator is not None:
                navigator.reload()
                print(f"  [Layer 4] Navigator index reloaded")

            proc.close()
        except Exception as e:
            print(f"  [Layer 4] Background cycle error: {e}")


def start_background_daemon(interval_hours: float = BACKGROUND_INTERVAL_HOURS):
    """Start the background processor as a daemon thread."""
    t = threading.Thread(
        target=_background_loop,
        args=(interval_hours,),
        daemon=True,
        name="Layer4-Background"
    )
    t.start()
    return t


# ============================================================
# Main
# ============================================================

def main():
    port = 5555

    gemma_engine.MODEL = "gemma4-turbo"

    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])

    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            gemma_engine.MODEL = sys.argv[idx + 1]

    print(f"\n  Gemma v3.1 — HTTP Backend (Session-Safe + Navigator)")
    print(f"  Model: {gemma_engine.MODEL}")
    print(f"  Initializing layers...")

    init_layers()

    stats = graph.get_stats()
    print(f"  Episodic memory: {stats['total_episodes']} episodes, {stats['total_edges']} edges")
    print(f"  Self-model tendencies: {len(self_model.data.get('tendencies', {}))}")
    print(f"  Conversation history: {len(list_conversations())} saved sessions")

    if pending_thoughts:
        print(f"  Pending thoughts from Layer 4: {len(pending_thoughts)}")

    # Start background processor daemon (replaces cron job)
    bg_interval = float(os.environ.get("GEMMA_BG_INTERVAL", BACKGROUND_INTERVAL_HOURS))
    start_background_daemon(interval_hours=bg_interval)

    class ReusableHTTPServer(HTTPServer):
        allow_reuse_address = True

    server = ReusableHTTPServer(('0.0.0.0', port), GemmaHandler)
    print(f"\n  Listening on http://localhost:{port}")
    print(f"  Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        # Save all active sessions
        session_mgr.save_all()
        # End the default session properly
        try:
            default_sess = session_mgr.get_default()
            if default_sess.messages:
                end_session(default_sess.messages, graph, self_model)
        except Exception as e:
            print(f"  Warning: session end incomplete ({e}) — embeddings can be backfilled later")
        graph.close()
        print("  All sessions saved. Goodbye.\n")


if __name__ == "__main__":
    main()
