#!/usr/bin/env bash
# read_doc_gen_commit — Claude Code PreToolUse hook
# When Claude is about to run `git commit`, this hook reads project docs and
# the staged diff, then returns a comprehensive commit message suggestion.
#
# Registered as a PreToolUse hook on Bash in .claude/settings.json

set -euo pipefail

# Read hook input from stdin
INPUT=$(cat)

# Extract the command being run
COMMAND=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    # PreToolUse input has tool_input with the command
    tool_input = data.get('tool_input', {})
    print(tool_input.get('command', ''))
except:
    print('')
" 2>/dev/null || echo "")

# Only act on git commit commands
if ! echo "$COMMAND" | grep -qE '^\s*git\s+commit'; then
    # Not a commit — approve silently
    echo '{"decision": "approve"}'
    exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"

# Gather context
DIFF=$(cd "$PROJECT_DIR" && git diff --cached --stat 2>/dev/null || echo "no staged changes")
DIFF_CONTENT=$(cd "$PROJECT_DIR" && git diff --cached 2>/dev/null | head -500 || echo "")
RECENT_LOG=$(cd "$PROJECT_DIR" && git log --oneline -5 2>/dev/null || echo "")

# Read key doc files (first 50 lines each to stay concise)
CLAUDE_MD=""
if [ -f "$PROJECT_DIR/CLAUDE.md" ]; then
    CLAUDE_MD=$(head -50 "$PROJECT_DIR/CLAUDE.md" 2>/dev/null || echo "")
fi

STREAMS_MD=""
if [ -f "$PROJECT_DIR/.claude/plans/STREAMS.md" ]; then
    STREAMS_MD=$(head -40 "$PROJECT_DIR/.claude/plans/STREAMS.md" 2>/dev/null || echo "")
fi

INTEGRATION_MD=""
if [ -f "$PROJECT_DIR/INTEGRATION.md" ]; then
    INTEGRATION_MD=$(head -30 "$PROJECT_DIR/INTEGRATION.md" 2>/dev/null || echo "")
fi

# Build the message for Claude
MESSAGE=$(cat <<MSGEOF
[read_doc_gen_commit hook] Generate a comprehensive commit message based on this context:

**Staged changes (stat):**
$DIFF

**Recent commits (for style reference):**
$RECENT_LOG

**Key diff content (first 500 lines):**
$DIFF_CONTENT

**Project context (CLAUDE.md header):**
$CLAUDE_MD

**Active streams (STREAMS.md header):**
$STREAMS_MD

Guidelines for the commit message:
- Lead with the type: feat/fix/docs/test/chore/refactor
- Summarize the "why" not just the "what"
- List major file groups changed
- Reference stream IDs if relevant
- Keep first line under 72 chars
- Use body for details
MSGEOF
)

# Return approval with the context message
# The message will be shown to Claude, who can use it to write a better commit
python3 -c "
import json, sys
msg = sys.stdin.read()
print(json.dumps({'decision': 'approve', 'message': msg}))
" <<< "$MESSAGE"
