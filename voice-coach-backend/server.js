// Strategic Voice Coach — backend API.
// Holds the Anthropic key and proxies Claude. Never expose the key to clients.

const express = require('express');
const cors = require('cors');
const Anthropic = require('@anthropic-ai/sdk');

const { WEEKS, FRAMEWORKS } = require('./data');

const PORT = process.env.PORT || 3001;
const MODEL = 'claude-sonnet-4-20250514';

const app = express();
app.use(cors());
app.use(express.json());

// Lazily construct the client so the server can boot (and serve /health)
// even when the key is missing — the error surfaces on the first Claude call.
let anthropic = null;
function getClient() {
  if (!process.env.ANTHROPIC_API_KEY) {
    const err = new Error('ANTHROPIC_API_KEY is not configured on the server.');
    err.statusCode = 503;
    throw err;
  }
  if (!anthropic) {
    anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
  }
  return anthropic;
}

// Clamp + resolve a week index (0-based) to a valid week object.
function resolveWeek(weekIndex) {
  const idx = Number.isInteger(weekIndex)
    ? Math.max(0, Math.min(WEEKS.length - 1, weekIndex))
    : 0;
  return { idx, week: WEEKS[idx] };
}

// Dynamic system prompt that injects the current week's context.
function buildSystemPrompt(weekIndex) {
  const { week } = resolveWeek(weekIndex);
  return `You are an elite strategic leadership coach: sharp, direct, and deeply experienced. You guide an ambitious leader through a 14-week strategic leadership program.

YOUR STYLE:
- Socratic. Draw insights out of the user; do not lecture.
- Ask ONE powerful question at a time.
- Warm but challenging. Acknowledge vulnerability before you push.
- No filler. Never say "Great question!" or similar throat-clearing.
- Reference frameworks only when it sharpens the point, never to show off.

CRITICAL OUTPUT RULES:
- This is a VOICE conversation. Reply in under 80 words.
- Plain spoken sentences only. No markdown, no bullet points, no lists, no headers.

CURRENT FOCUS:
- Week ${week.week} of 14 — "${week.title}" (${week.phase} phase).
- Coaching focus: ${week.focus}.

FRAMEWORKS YOU KNOW: ${FRAMEWORKS.join('; ')}.

Stay anchored to this week's focus. Meet the leader where they are, then move them one real step forward.`;
}

// Map an Anthropic SDK error to a clean client-facing response.
function sendClaudeError(res, err) {
  const status = err.statusCode || err.status || 500;
  // 401/403 => bad key; surface as a clear, non-crashing message.
  const message =
    status === 401 || status === 403
      ? 'Claude rejected the request — check the server ANTHROPIC_API_KEY.'
      : status === 503
        ? err.message
        : 'The coach is temporarily unavailable. Please try again.';
  console.error('[claude-error]', status, err.message);
  res.status(status === 401 || status === 403 ? 502 : status).json({
    error: message,
  });
}

app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '1.0.0' });
});

// Generate an energizing opening for the chosen week.
app.post('/session/start', async (req, res) => {
  const { weekIndex } = req.body || {};
  const { idx, week } = resolveWeek(weekIndex);
  try {
    const client = getClient();
    const msg = await client.messages.create({
      model: MODEL,
      max_tokens: 200,
      system: buildSystemPrompt(idx),
      messages: [
        {
          role: 'user',
          content:
            'Begin the session. In under 70 words, energize me, set the context for this week, and ask your first coaching question. Spoken voice only — no markdown.',
        },
      ],
    });
    const reply = msg.content
      .filter((b) => b.type === 'text')
      .map((b) => b.text)
      .join(' ')
      .trim();
    res.json({ reply, week: week.week, title: week.title, phase: week.phase });
  } catch (err) {
    sendClaudeError(res, err);
  }
});

// Continue the coaching conversation.
app.post('/chat', async (req, res) => {
  const { weekIndex, messages } = req.body || {};
  const { idx } = resolveWeek(weekIndex);

  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: 'messages must be a non-empty array.' });
  }

  // Sanitize to the shape Claude expects.
  const history = messages
    .filter((m) => m && (m.role === 'user' || m.role === 'assistant') && m.content)
    .map((m) => ({ role: m.role, content: String(m.content) }));

  if (history.length === 0) {
    return res.status(400).json({ error: 'No valid messages provided.' });
  }

  try {
    const client = getClient();
    const msg = await client.messages.create({
      model: MODEL,
      max_tokens: 200,
      system: buildSystemPrompt(idx),
      messages: history,
    });
    const reply = msg.content
      .filter((b) => b.type === 'text')
      .map((b) => b.text)
      .join(' ')
      .trim();
    res.json({ reply });
  } catch (err) {
    sendClaudeError(res, err);
  }
});

app.listen(PORT, () => {
  console.log(`Strategic Voice Coach backend listening on :${PORT}`);
});

module.exports = { app, buildSystemPrompt };
