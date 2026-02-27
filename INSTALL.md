# Install AIR Blackbox MCP Server — Step by Step

## Step 1: Copy the project to your Desktop

The `air-blackbox-mcp` folder should already be in your outputs. Copy it to Desktop:

```bash
cp -r ~/Desktop/air-blackbox-mcp ~/Desktop/air-blackbox-mcp-backup 2>/dev/null; echo "Ready"
```

## Step 2: Install the package

```bash
cd ~/Desktop/air-blackbox-mcp && pip3 install -e . --break-system-packages
```

## Step 3: Verify it works

```bash
python3 -c "from air_blackbox_mcp.server import mcp; print('AIR Blackbox MCP: OK — 10 tools loaded')"
```

You should see: `AIR Blackbox MCP: OK — 10 tools loaded`

## Step 4: Add to Claude Desktop

Run this command to add the MCP server to your Claude Desktop config:

```bash
cat > /tmp/add-mcp.py << 'ENDOFFILE'
import json
import os

config_path = os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json")

# Read existing config or create new one
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
else:
    config = {}

# Add air-blackbox server
if "mcpServers" not in config:
    config["mcpServers"] = {}

config["mcpServers"]["air-blackbox"] = {
    "command": "python3",
    "args": ["-m", "air_blackbox_mcp"]
}

# Write back
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"Updated: {config_path}")
print(f"MCP servers configured: {list(config['mcpServers'].keys())}")
ENDOFFILE
python3 /tmp/add-mcp.py
```

## Step 5: Restart Claude Desktop

Quit Claude Desktop completely (Cmd+Q), then reopen it.

## Step 6: Test it

In Claude Desktop, try:

> "Scan this code for EU AI Act compliance:
> ```python
> from langchain.agents import AgentExecutor
> from langchain_openai import ChatOpenAI
> llm = ChatOpenAI(model='gpt-4')
> ```"

You should see a compliance report with 0/6 passing and fix recommendations.

## Troubleshooting

**"Module not found" error:**
```bash
which python3
python3 -c "import air_blackbox_mcp; print('found')"
```
If python3 isn't finding the package, try using the full path:
```bash
python3 -c "import sys; print(sys.executable)"
```
Then use that path in the Claude Desktop config instead of "python3".

**Server doesn't appear in Claude Desktop:**
Make sure the config file is valid JSON:
```bash
python3 -c "import json; json.load(open(os.path.expanduser('~/Library/Application Support/Claude/claude_desktop_config.json'))); print('Config OK')" 2>&1 || echo "Config has JSON errors"
```
