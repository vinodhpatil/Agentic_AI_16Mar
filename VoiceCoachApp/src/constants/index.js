// Shared constants for the Strategic Voice Coach app.

// ============================================================================
// CONFIGURE ME: point this at your deployed backend (no trailing slash).
// e.g. 'https://your-backend.up.railway.app'
// ============================================================================
export const API_BASE_URL = 'https://YOUR-BACKEND-URL.up.railway.app';

export const COLORS = {
  bg: '#07070d',
  panel: 'rgba(255,255,255,0.04)',
  panelBorder: 'rgba(255,255,255,0.08)',
  ink: '#f4f3ef',
  inkDim: '#8b8b96',
  accent: '#3ecfcf',
};

export const PHASE_COLORS = {
  Foundation: '#e8c547',
  Growth: '#3ecfcf',
  Mastery: '#7b5ea7',
  Integration: '#e85447',
};

// The full 14-week program. `focus` mirrors the backend coaching focus.
export const WEEKS = [
  {
    week: 1,
    title: 'Leadership Identity',
    phase: 'Foundation',
    tag: 'Audit your leadership identity; articulate a personal philosophy.',
    focus:
      'Audit current leadership identity; articulate a personal leadership philosophy',
  },
  {
    week: 2,
    title: 'Mental Models Deep Dive',
    phase: 'Foundation',
    tag: 'First Principles, Inversion, Second-Order Thinking, Circle of Competence.',
    focus:
      'Apply First Principles, Inversion, Second-Order Thinking, Opportunity Cost, Circle of Competence',
  },
  {
    week: 3,
    title: 'Vision Architecture',
    phase: 'Foundation',
    tag: 'Build a 3-horizon vision map; find your #1 constraint.',
    focus:
      'Build a 3-horizon vision map (90 days / 1 yr / 3 yr); find the #1 constraint',
  },
  {
    week: 4,
    title: 'AI-First Operating Model',
    phase: 'Foundation',
    tag: 'Audit manual tasks; design your first AI agent workflow.',
    focus:
      'Audit manual tasks; design first AI agent workflow + weekly sprint rhythm',
  },
  {
    week: 5,
    title: 'CARE: Structured Communication',
    phase: 'Growth',
    tag: 'Master CARE, PREP, and BLUF for high-signal communication.',
    focus: 'Teach CARE (Context to Answer to Rationale to Example); also PREP and BLUF',
  },
  {
    week: 6,
    title: 'Decision Architecture',
    phase: 'Growth',
    tag: 'Reversible vs irreversible decisions; criteria for recurring calls.',
    focus:
      'Reversible vs irreversible decisions; define criteria for recurring decisions',
  },
  {
    week: 7,
    title: 'Systems & Leverage',
    phase: 'Growth',
    tag: 'Map systems; find where you are the bottleneck.',
    focus: 'Map systems; find where user is the bottleneck; identify leverage points',
  },
  {
    week: 8,
    title: 'Resilience & Adaptability',
    phase: 'Growth',
    tag: 'Stress protocol; calm urgency; a 48-hour recovery framework.',
    focus: 'Stress operating protocol; calm urgency; 48-hour recovery framework',
  },
  {
    week: 9,
    title: 'Building High-Trust Teams',
    phase: 'Mastery',
    tag: 'Delegation, accelerating feedback, psychological safety.',
    focus: 'Delegation, feedback that accelerates, psychological safety',
  },
  {
    week: 10,
    title: 'Speed as Strategy',
    phase: 'Mastery',
    tag: "Sprint rhythm; compress cycles 40% with AI; Parkinson's Law.",
    focus: "Sprint rhythm; compress cycles 40% with AI; Parkinson's Law",
  },
  {
    week: 11,
    title: 'Data-Driven Leadership',
    phase: 'Mastery',
    tag: 'Define 5 metrics that matter; your personal dashboard.',
    focus: 'Define 5 metrics that matter; personal dashboard; AI insight ritual',
  },
  {
    week: 12,
    title: 'Influence at Scale',
    phase: 'Mastery',
    tag: 'Thought leadership; the Three Stories narrative framework.',
    focus: 'Thought leadership position; Three Stories narrative framework',
  },
  {
    week: 13,
    title: 'Scenario Planning',
    phase: 'Mastery',
    tag: 'Map best/base/worst futures; track signals; pre-make decisions.',
    focus: 'Map best/base/worst futures; signal tracking; pre-made decisions',
  },
  {
    week: 14,
    title: 'Integration & Launch',
    phase: 'Integration',
    tag: 'Compile your Strategic Operating Manual + 90-day plan.',
    focus: 'Compile Strategic Operating Manual + 90-day action plan',
  },
];

export const STORAGE_KEYS = {
  week: 'vc_week',
  mute: 'vc_mute',
};
