#!/usr/bin/env bash
# Publish glacier-edge-arm to Codeberg and GitHub.
# Usage:
#   export CODEBERG_TOKEN=your_token
#   export GITHUB_TOKEN=your_token
#   ./scripts/publish.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${CODEBERG_TOKEN:?Set CODEBERG_TOKEN}"
: "${GITHUB_TOKEN:?Set GITHUB_TOKEN}"

CODEBERG_REPO="https://codeberg.org/cubiczan/glacier-edge-arm"
GITHUB_REPO="https://github.com/Cubiczan/glacier-edge-arm"

echo "Creating Codeberg repo (if missing)..."
curl -fsS -X POST "https://codeberg.org/api/v1/user/repos" \
  -H "Authorization: token ${CODEBERG_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"name":"glacier-edge-arm","description":"Arm-optimized edge inference for data center BESS fault detection","private":false,"auto_init":false}' \
  >/dev/null 2>&1 || true

echo "Creating GitHub repo (if missing)..."
curl -fsS -X POST "https://api.github.com/user/repos" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  -d '{"name":"glacier-edge-arm","description":"Arm-optimized edge inference for data center BESS fault detection","private":false,"auto_init":false}' \
  >/dev/null 2>&1 || true

git remote remove codeberg 2>/dev/null || true
git remote remove github 2>/dev/null || true
git remote add codeberg "${CODEBERG_REPO}.git"
git remote add github "${GITHUB_REPO}.git"

echo "Pushing to Codeberg..."
git push "https://cubiczan:${CODEBERG_TOKEN}@codeberg.org/cubiczan/glacier-edge-arm.git" main:main
git branch --set-upstream-to=codeberg/main main 2>/dev/null || true

echo "Pushing to GitHub..."
git push "https://x-access-token:${GITHUB_TOKEN}@github.com/Cubiczan/glacier-edge-arm.git" main:main

echo "Done."
echo "  Codeberg: ${CODEBERG_REPO}"
echo "  GitHub:   ${GITHUB_REPO}"
