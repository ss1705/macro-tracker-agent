"""A deliberately broken MCP server for failure-mode tests (never used by the app).

It speaks enough raw JSON-RPC over stdio to pass `initialize` and `tools/list`,
advertising the same lookup_nutrition tool as the real server, then misbehaves on
`tools/call` according to the mode argument:

  crash      exit without answering (transport closes mid-operation)
  garbage    write a line that isn't JSON
  malformed  answer with a result that violates the MCP schema
  hang       never answer
"""

import json
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "crash"

TOOL = {
    "name": "lookup_nutrition",
    "description": "Look up calories and macros for a single food at a given weight.",
    "inputSchema": {
        "type": "object",
        "properties": {"food_name": {"type": "string"}, "grams": {"type": "number"}},
        "required": ["food_name", "grams"],
    },
}


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    request = json.loads(line)
    method, request_id = request.get("method"), request.get("id")
    if request_id is None:  # notification, e.g. notifications/initialized
        continue
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": request_id, "result": {
            "protocolVersion": request["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": f"faulty-{MODE}", "version": "0"},
        }})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": request_id, "result": {"tools": [TOOL]}})
    elif method == "tools/call":
        if MODE == "crash":
            sys.exit(1)
        if MODE == "garbage":
            sys.stdout.write("<html>502 Bad Gateway</html>\n")
            sys.stdout.flush()
        elif MODE == "malformed":
            send({"jsonrpc": "2.0", "id": request_id, "result": {"content": "not-a-list"}})
        elif MODE == "hang":
            time.sleep(3600)
    else:
        send({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "not found"}})
