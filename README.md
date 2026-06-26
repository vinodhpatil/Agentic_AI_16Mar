# Strategic Voice Coach

An AI-powered leadership coach that guides you through a **14-week strategic leadership program** by voice. Powered by the Anthropic Claude API.

This repository ships **three deployable artifacts that share one backend**:

| Artifact | Folder | What it is |
|----------|--------|-----------|
| **Backend** | `voice-coach-backend/` | Express/Node API that holds the Anthropic key and proxies Claude |
| **PWA** | `voice-coach-pwa/` | Installable web app — vanilla HTML/CSS/JS, no build step |
| **Expo app** | `VoiceCoachApp/` | React Native app for full native iPhone voice (TTS + STT) |

The coach is Socratic, direct, and references real frameworks (CARE, PREP, First Principles, 3 Horizons, SBI, and more). Responses stay **under 80 words** because this is voice.

> **iOS note:** Safari does not support speech-to-text (`SpeechRecognition`). The PWA detects this and falls back to a text box; text-to-speech still works. The Expo app uses native `expo-speech-recognition` for full voice on iPhone.

---

## Prerequisites

- **Node.js 18+** and npm
- An **Anthropic API key** — https://console.anthropic.com (you create this; it is never stored in code)
- A **GitHub account** (to deploy the backend)
- A **Railway account** — https://railway.app (free tier works) for the backend
- For the Expo app: the **Expo Go** app on your iPhone/Android, plus `npx expo`

---

## 1. Backend → Railway

The backend is the only place the Anthropic key lives.

### Run locally first
```bash
cd voice-coach-backend
cp .env.example .env          # then edit .env and paste your real ANTHROPIC_API_KEY
npm install
npm run dev                   # or: npm start
curl http://localhost:3001/health
# -> {"status":"ok","version":"1.0.0"}
```

### Deploy to Railway
```bash
cd voice-coach-backend
./deploy-backend.sh           # inits git + commits, then prints the push commands
```
Then, by hand:
1. Push to a GitHub repo (the script prints the exact `git remote add` + `git push` commands).
2. In Railway: **New Project → Deploy from GitHub repo** → select your repo.
3. **Variables** → add `ANTHROPIC_API_KEY` = your real key. (Railway provides `PORT` automatically.)
4. **Settings → Networking → Generate Domain**.
5. Verify: `curl https://<your-domain>/health` returns `{"status":"ok","version":"1.0.0"}`.

Keep that domain handy — both clients point at it.

### API
- `GET /health` → `{ status, version }`
- `POST /session/start` — body `{ weekIndex }` → `{ reply, week, title, phase }`
- `POST /chat` — body `{ weekIndex, messages }` → `{ reply }`

`messages` is the running `[{ role: "user" | "assistant", content }]` history.

---

## 2. PWA hosting

The PWA is fully static — host the `voice-coach-pwa/` folder anywhere over **HTTPS** (required for service workers, mic, and "Add to Home Screen").

Options:
- **Netlify Drop** — drag the `voice-coach-pwa/` folder onto https://app.netlify.com/drop
- **Vercel** — `vercel` from inside the folder (or import the repo, set root to `voice-coach-pwa`)
- **GitHub Pages** — push the folder contents to a `gh-pages` branch / Pages-enabled repo

After it's live:
1. Open the site, tap **⚙ Settings**, paste your **backend URL** (e.g. `https://your-backend.up.railway.app`), Save.
2. Tap the orb to start a session.

### iOS — Add to Home Screen
1. Open the hosted URL in **Safari**.
2. Tap **Share → Add to Home Screen**.
3. Launch from the home-screen icon for full-screen standalone mode.
4. On iOS you'll see a **text input row** (Safari has no speech-to-text); the coach still speaks back.

Icons are pre-generated in `voice-coach-pwa/icons/`. To regenerate:
`python3 voice-coach-pwa/icons/generate_icons.py` (requires Pillow).

---

## 3. Expo app (full native iPhone voice)

```bash
cd VoiceCoachApp
# 1) Set your backend URL:
#    edit src/constants/index.js -> API_BASE_URL = 'https://your-backend.up.railway.app'
npm install
npx expo start
```
Scan the QR code with **Expo Go** (Android) or the Camera app (iOS). Grant microphone + speech-recognition permission when prompted, then tap the orb to talk.

> `expo-speech-recognition` needs a real device (and a development/Go build that includes the plugin). The simulator has no microphone.

---

## Troubleshooting

- **"Could not reach the coach" / server unreachable** — Check the backend URL has `https://` and no trailing slash. Confirm `/health` responds. Confirm `ANTHROPIC_API_KEY` is set in Railway (a bad key returns a clear `502` "check the server ANTHROPIC_API_KEY" message, not a crash).
- **Mic does nothing on iOS Safari** — Expected; Safari lacks `SpeechRecognition`. Use the text box, or use the Expo app for native voice.
- **No audio on first load** — Browsers block autoplay until you interact. The **first orb tap unlocks audio**; tap once and the coach will speak from then on.
- **Stale PWA after an update** — The service worker caches static assets. Hard-reload, or in DevTools → Application → Service Workers → **Unregister**, then reload. The cache name (`voice-coach-v1` in `sw.js`) can be bumped to force a refresh.
- **Expo: unresolved import** — Run `npm install` inside `VoiceCoachApp/`; ensure the Expo SDK matches your Expo Go version.

---

## Security

- The Anthropic key lives **only** in the backend environment (`ANTHROPIC_API_KEY`). It is never in client code, the PWA, or the Expo app.
- `.env` is git-ignored. Only `.env.example` (empty) is committed.

---

## Next Actions (human-only — requires your accounts/secrets)

These steps need your credentials and cannot be automated for you:

- [ ] **Create an Anthropic API key** at https://console.anthropic.com.
- [ ] **Create a GitHub repo** for the backend and push it (`voice-coach-backend/deploy-backend.sh` prints the commands).
- [ ] **Connect Railway** to that repo and deploy.
- [ ] **Set `ANTHROPIC_API_KEY`** in Railway's Variables.
- [ ] **Generate a Railway domain** and confirm `/health`.
- [ ] **Host the PWA** (`voice-coach-pwa/`) over HTTPS and enter the backend URL in its Settings.
- [ ] **Set `API_BASE_URL`** in `VoiceCoachApp/src/constants/index.js`, then `npm install` + `npx expo start`.
