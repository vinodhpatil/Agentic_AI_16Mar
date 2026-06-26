// Domain data for the Strategic Voice Coach — shared, server-side source of truth.

const PHASE_COLORS = {
  Foundation: '#e8c547',
  Growth: '#3ecfcf',
  Mastery: '#7b5ea7',
  Integration: '#e85447',
};

// The 14-week strategic leadership program.
const WEEKS = [
  {
    week: 1,
    title: 'Leadership Identity',
    phase: 'Foundation',
    focus:
      'Audit current leadership identity; articulate a personal leadership philosophy',
  },
  {
    week: 2,
    title: 'Mental Models Deep Dive',
    phase: 'Foundation',
    focus:
      'Apply First Principles, Inversion, Second-Order Thinking, Opportunity Cost, Circle of Competence',
  },
  {
    week: 3,
    title: 'Vision Architecture',
    phase: 'Foundation',
    focus:
      'Build a 3-horizon vision map (90 days / 1 yr / 3 yr); find the #1 constraint',
  },
  {
    week: 4,
    title: 'AI-First Operating Model',
    phase: 'Foundation',
    focus:
      'Audit manual tasks; design first AI agent workflow + weekly sprint rhythm',
  },
  {
    week: 5,
    title: 'CARE: Structured Communication',
    phase: 'Growth',
    focus: 'Teach CARE (Context to Answer to Rationale to Example); also PREP and BLUF',
  },
  {
    week: 6,
    title: 'Decision Architecture',
    phase: 'Growth',
    focus:
      'Reversible vs irreversible decisions; define criteria for recurring decisions',
  },
  {
    week: 7,
    title: 'Systems & Leverage',
    phase: 'Growth',
    focus: 'Map systems; find where user is the bottleneck; identify leverage points',
  },
  {
    week: 8,
    title: 'Resilience & Adaptability',
    phase: 'Growth',
    focus: 'Stress operating protocol; calm urgency; 48-hour recovery framework',
  },
  {
    week: 9,
    title: 'Building High-Trust Teams',
    phase: 'Mastery',
    focus: 'Delegation, feedback that accelerates, psychological safety',
  },
  {
    week: 10,
    title: 'Speed as Strategy',
    phase: 'Mastery',
    focus: "Sprint rhythm; compress cycles 40% with AI; Parkinson's Law",
  },
  {
    week: 11,
    title: 'Data-Driven Leadership',
    phase: 'Mastery',
    focus: 'Define 5 metrics that matter; personal dashboard; AI insight ritual',
  },
  {
    week: 12,
    title: 'Influence at Scale',
    phase: 'Mastery',
    focus: 'Thought leadership position; Three Stories narrative framework',
  },
  {
    week: 13,
    title: 'Scenario Planning',
    phase: 'Mastery',
    focus: 'Map best/base/worst futures; signal tracking; pre-made decisions',
  },
  {
    week: 14,
    title: 'Integration & Launch',
    phase: 'Integration',
    focus: 'Compile Strategic Operating Manual + 90-day action plan',
  },
];

// Frameworks the coach knows and can reference when natural.
const FRAMEWORKS = [
  'CARE (Context, Answer, Rationale, Example)',
  'PREP (Point, Reason, Example, Point)',
  'Pyramid Principle',
  'Ethos/Pathos/Logos',
  'BLUF (Bottom Line Up Front)',
  'Elevator Pitch (Hook, Problem, Solution, Ask)',
  'SBI Feedback (Situation, Behavior, Impact)',
  '5 Whys',
  'Blue Ocean (ERRC)',
  'McKinsey 3 Horizons',
  'PESTLE',
  'First Principles',
  'SMART Goals',
  'Radical Candor',
];

module.exports = { PHASE_COLORS, WEEKS, FRAMEWORKS };
