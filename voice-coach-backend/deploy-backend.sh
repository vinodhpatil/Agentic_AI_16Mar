#!/usr/bin/env bash
# Helper to prepare the Strategic Voice Coach backend for deployment.
# This script initializes git and commits — it does NOT create a remote or push.
# It prints the exact commands you run by hand (with a YOUR_USERNAME placeholder).

set -e

echo "=== Strategic Voice Coach — backend deploy prep ==="

if [ ! -f "server.js" ]; then
  echo "Run this from inside voice-coach-backend/ (server.js not found)."
  exit 1
fi

# Safety: never commit a real .env.
if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  echo "ERROR: .env is tracked by git. Remove it before deploying."
  exit 1
fi

if [ ! -d ".git" ]; then
  git init
fi

git add server.js data.js package.json package-lock.json .env.example .gitignore railway.json deploy-backend.sh 2>/dev/null || git add .
git commit -m "Strategic Voice Coach backend" || echo "(nothing new to commit)"

echo ""
echo "=== NEXT — run these by hand (replace YOUR_USERNAME / repo name) ==="
echo "  git branch -M main"
echo "  git remote add origin https://github.com/YOUR_USERNAME/voice-coach-backend.git"
echo "  git push -u origin main"
echo ""
echo "Then on Railway (https://railway.app):"
echo "  1. New Project -> Deploy from GitHub repo -> pick voice-coach-backend"
echo "  2. Variables -> add ANTHROPIC_API_KEY = <your real key>"
echo "  3. (Railway sets PORT automatically; the server reads process.env.PORT)"
echo "  4. Settings -> Networking -> Generate Domain"
echo "  5. Verify: curl https://<your-domain>/health  ->  {\"status\":\"ok\",\"version\":\"1.0.0\"}"
echo ""
echo "Finally, paste that domain into the PWA settings modal and into"
echo "VoiceCoachApp/src/constants/index.js (API_BASE_URL)."
