#!/usr/bin/env bash
# install-skill.sh — one-command DeepSight agent-skill installer.
#
# Usage (the whole point: ONE command, no user steps):
#   curl -fsSL https://raw.githubusercontent.com/Reality-Shifting-Tech/deepsight/main/scripts/install-skill.sh | bash
#
# Installs the DeepSight skill (skill/deepsight/ in this repo) into the
# agent's skills directory, then prints how the agent should load it.
#
# Target detection (first match wins):
#   1. $DEEPSIGHT_SKILLS_DIR          — explicit override
#   2. ~/.hermes/skills               — Hermes Agent (default)
#   3. ~/.claude/skills               — Claude Code / Claude Desktop
#   4. ~/.codex/skills                — OpenAI Codex CLI
#   5. default                        — ~/.hermes/skills (created if missing)
#
# Overrides:
#   DEEPSIGHT_SKILLS_DIR  — install target root
#   DEEPSIGHT_SKILL_SRC   — local skill dir (debug/offline escape hatch)
set -euo pipefail

REPO_OWNER="Reality-Shifting-Tech"
REPO_NAME="deepsight"
REPO_BRANCH="main"
SKILL_NAME="deepsight"

# --- resolve source ---------------------------------------------------------
# Running from a checkout (script lives at <repo>/scripts/install-skill.sh):
# use the sibling skill/ dir directly. Running via curl|bash (stdin): fetch
# the skill out of the repo tarball.
SRC=""
SCRIPT_PATH="${BASH_SOURCE[0]:-}"
if [[ -n "$SCRIPT_PATH" && "$SCRIPT_PATH" != "bash" && "$SCRIPT_PATH" != "-bash" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd || true)"
  if [[ -n "$SCRIPT_DIR" && -d "$SCRIPT_DIR/../skill/$SKILL_NAME" ]]; then
    SRC="$SCRIPT_DIR/../skill/$SKILL_NAME"
  fi
fi
if [[ -n "${DEEPSIGHT_SKILL_SRC:-}" && -d "$DEEPSIGHT_SKILL_SRC" ]]; then
  SRC="$DEEPSIGHT_SKILL_SRC"
fi

TMP=""
cleanup() {
  if [[ -n "$TMP" && -d "$TMP" ]]; then
    rm -rf "$TMP"
  fi
  true
}
trap cleanup EXIT

if [[ -z "$SRC" ]]; then
  TMP="$(mktemp -d)"
  echo "→ downloading DeepSight skill ($REPO_OWNER/$REPO_NAME@$REPO_BRANCH)..."
  curl -fsSL "https://codeload.github.com/$REPO_OWNER/$REPO_NAME/tar.gz/refs/heads/$REPO_BRANCH" \
    | tar -xz -C "$TMP" --strip-components=2 "$REPO_NAME-$REPO_BRANCH/skill/$SKILL_NAME"
  SRC="$TMP/$SKILL_NAME"
fi
[[ -d "$SRC" ]] || { echo "error: skill source not found at $SRC" >&2; exit 1; }

# --- resolve target ---------------------------------------------------------
if [[ -n "${DEEPSIGHT_SKILLS_DIR:-}" ]]; then
  DEST="$DEEPSIGHT_SKILLS_DIR/$SKILL_NAME"
elif [[ -d "$HOME/.hermes/skills" ]]; then
  DEST="$HOME/.hermes/skills/$SKILL_NAME"
elif [[ -d "$HOME/.claude/skills" ]]; then
  DEST="$HOME/.claude/skills/$SKILL_NAME"
elif [[ -d "$HOME/.codex/skills" ]]; then
  DEST="$HOME/.codex/skills/$SKILL_NAME"
else
  DEST="$HOME/.hermes/skills/$SKILL_NAME"
fi

# --- install ----------------------------------------------------------------
mkdir -p "$(dirname "$DEST")"
if [[ -d "$DEST" ]]; then
  rm -rf "$DEST.old"
  mv "$DEST" "$DEST.old"
fi
cp -R "$SRC" "$DEST"
chmod -R u+rwX "$DEST"

echo "✓ DeepSight skill installed at: $DEST"
if [[ -d "$DEST.old" ]]; then
  echo "  (previous copy preserved at $DEST.old)"
fi
echo
echo "Next step for the agent — load the skill and follow it:"
echo "  Hermes:        skill_view(name='$SKILL_NAME')"
echo "  any agent:     read $DEST/SKILL.md"
echo
echo "The skill covers: compiling the vision binary, reasoning-loop wiring,"
echo "the 16-tool protocol, auxiliary.vision integration, and troubleshooting."
