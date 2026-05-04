"""
AIR Controls - Local API Server

Serves agent event data from the SQLite store over HTTP at localhost:7420.
The AIR Controls dashboard connects to this endpoint for real-time visibility.

Zero external dependencies - uses Python's built-in http.server.

Usage:
    air-controls serve              # Start on default port 7420
    air-controls serve --port 8080  # Custom port
    air-controls serve --db /path/to/events.db  # Custom DB path

API Endpoints:
    GET  /health                    → { "status": "ok", "agents": N, "events": N }
    GET  /events                    → { "events": [...] }
    GET  /events?agent=sales-bot    → Filtered by agent
    GET  /events?type=llm_call      → Filtered by event type
    GET  /agents                    → { "agents": [...] }
    GET  /agents/<id>/stats         → { "stats": {...} }
    POST /agents/<id>/pause         → { "status": "paused" }
    POST /agents/<id>/resume        → { "status": "active" }
    GET  /verify                    → { "valid": true/false }
    GET  /verify?agent=sales-bot    → Verify specific agent chain
"""

import json
import re
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# Add parent directory to path so we can import air_controls
# This allows running from the air-controls repo directly
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import from installed package first, then from local
try:
    from air_controls.store import EventStore
except ImportError:
    # If running from the repo, add the air-controls directory
    air_controls_path = Path(__file__).parent
    sys.path.insert(0, str(air_controls_path))
    from air_controls.store import EventStore


class CORSRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler with CORS support for the dashboard."""

    store: EventStore = None  # Set by serve() before starting

    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
        self.end_headers()

    def _json_response(self, data, status=200):
        self._set_headers(status)
        self.wfile.write(json.dumps(data, default=str).encode())

    def _error(self, status, message):
        self._json_response({"error": message}, status)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self._set_headers(204)

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        try:
            if path == "/health":
                self._handle_health()
            elif path == "/events":
                self._handle_events(params)
            elif path == "/agents":
                self._handle_agents()
            elif re.match(r"^/agents/[\w-]+/stats$", path):
                agent_id = path.split("/")[2]
                self._handle_agent_stats(agent_id)
            elif path == "/verify":
                self._handle_verify(params)
            else:
                self._error(404, f"Not found: {path}")
        except Exception as e:
            self._error(500, str(e))

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        try:
            if re.match(r"^/agents/[\w-]+/pause$", path):
                agent_id = path.split("/")[2]
                self._handle_pause(agent_id)
            elif re.match(r"^/agents/[\w-]+/resume$", path):
                agent_id = path.split("/")[2]
                self._handle_resume(agent_id)
            else:
                self._error(404, f"Not found: {path}")
        except Exception as e:
            self._error(500, str(e))

    # ── Endpoint handlers ──────────────────────────────────

    def _handle_health(self):
        agents = self.store.get_agents()
        events = self.store.get_events(limit=1)
        total_events = 0
        for agent in agents:
            stats = self.store.get_agent_stats(agent["id"])
            total_events += stats.get("total_events", 0)

        self._json_response({
            "status": "ok",
            "agents": len(agents),
            "events": total_events,
            "version": "0.1.0",
        })

    def _handle_events(self, params):
        agent_id = params.get("agent", [None])[0]
        action_type = params.get("type", [None])[0]
        limit = int(params.get("limit", [100])[0])
        offset = int(params.get("offset", [0])[0])

        events = self.store.get_events(
            agent_id=agent_id,
            action_type=action_type,
            limit=limit,
            offset=offset,
        )

        # Transform events to match dashboard expected format
        dashboard_events = []
        for evt in events:
            # Map action_type to dashboard event type IDs
            type_map = {
                "llm_call": "llm-call",
                "llm_error": "llm-call",
                "tool_use": "tool-use",
                "tool_error": "tool-use",
                "api_call": "api-request",
                "email_sent": "email",
                "file_access": "file",
                "database": "database",
                "web_search": "web-search",
                "function_call": "tool-use",
                "chain_end": "tool-use",
                "chain_error": "tool-use",
                "agent_action": "tool-use",
                "agent_finish": "tool-use",
                "retrieval": "database",
                "analysis": "llm-call",
                "decision": "llm-call",
                "error": "api-request",
                "kill_switch": "api-request",
                "resumed": "api-request",
                "session_end": "tool-use",
                "ticket_received": "api-request",
            }

            dashboard_events.append({
                "id": evt["id"],
                "agent": evt["agent_id"],
                "type": type_map.get(evt["action_type"], "tool-use"),
                "description": evt.get("human_summary") or evt.get("raw_action") or evt["action_type"],
                "timestamp": evt["timestamp"],
                "risk": evt.get("risk_score", "low"),
                "hash": evt.get("chain_hash", "")[:16],
                "tokens": evt.get("tokens_used", 0),
                "cost": evt.get("cost_usd", 0),
                "duration": evt.get("duration_ms", 0),
            })

        self._json_response({"events": dashboard_events})

    def _handle_agents(self):
        agents = self.store.get_agents()
        result = []
        for agent in agents:
            stats = self.store.get_agent_stats(agent["id"])
            result.append({
                "id": agent["id"],
                "name": agent.get("name", agent["id"]),
                "framework": agent.get("framework", "custom"),
                "status": agent.get("status", "active"),
                "total_events": stats.get("total_events", 0),
                "total_tokens": stats.get("total_tokens", 0) or 0,
                "total_cost": stats.get("total_cost", 0) or 0,
                "last_event": stats.get("last_event", ""),
            })
        self._json_response({"agents": result})

    def _handle_agent_stats(self, agent_id):
        agent = self.store.get_agent(agent_id)
        if not agent:
            self._error(404, f"Agent '{agent_id}' not found")
            return
        stats = self.store.get_agent_stats(agent_id)
        self._json_response({"stats": stats})

    def _handle_pause(self, agent_id):
        agent = self.store.get_agent(agent_id)
        if not agent:
            self._error(404, f"Agent '{agent_id}' not found")
            return
        self.store.pause_agent(agent_id)
        self.store.log_event(
            agent_id=agent_id,
            action_type="kill_switch",
            raw_action="Agent paused via dashboard",
            human_summary=f"Agent '{agent_id}' paused from dashboard",
            risk_score="medium",
        )
        self._json_response({"status": "paused", "agent_id": agent_id})

    def _handle_resume(self, agent_id):
        agent = self.store.get_agent(agent_id)
        if not agent:
            self._error(404, f"Agent '{agent_id}' not found")
            return
        self.store.resume_agent(agent_id)
        self.store.log_event(
            agent_id=agent_id,
            action_type="resumed",
            raw_action="Agent resumed via dashboard",
            human_summary=f"Agent '{agent_id}' resumed from dashboard",
        )
        self._json_response({"status": "active", "agent_id": agent_id})

    def _handle_verify(self, params):
        agent_id = params.get("agent", [None])[0]
        is_valid = self.store.verify_chain(agent_id)
        scope = agent_id or "all"
        self._json_response({
            "valid": is_valid,
            "scope": scope,
            "message": "Chain intact" if is_valid else "Chain broken - possible tampering detected",
        })

    def log_message(self, format, *args):
        """Override to use cleaner logging."""
        print(f"  [AIR Controls] {args[0]}")


def serve(db_path=None, port=7420, host="127.0.0.1"):
    """
    Start the AIR Controls API server.

    Args:
        db_path: Path to SQLite database (default: ~/.air-controls/events.db)
        port: Port to listen on (default: 7420)
        host: Host to bind to (default: 127.0.0.1)
    """
    store = EventStore(db_path)
    CORSRequestHandler.store = store

    agents = store.get_agents()
    total_events = 0
    for agent in agents:
        stats = store.get_agent_stats(agent["id"])
        total_events += stats.get("total_events", 0)

    server = HTTPServer((host, port), CORSRequestHandler)

    print("")
    print("=" * 60)
    print("  AIR Controls - API Server")
    print("=" * 60)
    print(f"  Dashboard:  http://{host}:{port}")
    print(f"  Database:   {store.db_path}")
    print(f"  Agents:     {len(agents)}")
    print(f"  Events:     {total_events}")
    print("=" * 60)
    print("")
    print("  Endpoints:")
    print(f"    GET  http://{host}:{port}/health")
    print(f"    GET  http://{host}:{port}/events")
    print(f"    GET  http://{host}:{port}/agents")
    print(f"    POST http://{host}:{port}/agents/<id>/pause")
    print(f"    POST http://{host}:{port}/agents/<id>/resume")
    print(f"    GET  http://{host}:{port}/verify")
    print("")
    print("  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        server.shutdown()
        store.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AIR Controls API Server")
    parser.add_argument("--port", type=int, default=7420)
    parser.add_argument("--db", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    serve(db_path=args.db, port=args.port, host=args.host)
