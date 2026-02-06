# Trend Council Deliberation System - Complete Implementation Summary

## Overview

This implementation provides a **complete AutoGen-based multi-agent debate system** for automated news video production. The Trend Council deliberates on story proposals through structured argumentation, challenging assumptions, and reaching consensus.

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TREND COUNCIL SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Scout_A    │    │   Scout_B    │    │Trend_Analyst │      │
│  │  (Mainstream)│    │(Social Media)│    │  (Data)      │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                    │               │
│         └───────────────────┼────────────────────┘               │
│                             │                                    │
│                             ▼                                    │
│         ┌──────────────────────────────────────┐                │
│         │      RoundRobinGroupChat             │                │
│         │  (Structured debate, max 20 rounds)  │                │
│         └──────────────────┬───────────────────┘                │
│                            │                                     │
│              ┌─────────────┴─────────────┐                      │
│              ▼                           ▼                      │
│    ┌─────────────────┐        ┌─────────────────┐              │
│    │ Skeptic_Agent   │        │  Council_Chair  │              │
│    │ (Devil's Adv.)  │        │  (Facilitator)  │              │
│    └─────────────────┘        └─────────────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  FINAL DECISION │
                    │  (APPROVED/     │
                    │   REJECTED/     │
                    │   DEADLOCKED)   │
                    └─────────────────┘
```

---

## 👥 Agent Personalities

### Scout_A (Mainstream Scout)
```
PERSONALITY: Fast, aggressive, impatient
CATCHPHRASES: "We're going to miss the window!", "This is breaking NOW!"
VALUES: Speed, mainstream credibility, beating competitors
TRIGGERS: Gets annoyed by over-analysis
```

### Scout_B (Social Media Scout)
```
PERSONALITY: Trend-obsessed, enthusiastic, data-oriented
CATCHPHRASES: "This is blowing up!", "The algorithm is pushing this hard"
VALUES: Viral potential, engagement metrics, cross-platform spread
TRIGGERS: Defends social signals as predictive
```

### Trend_Analyst (Data Evaluator)
```
PERSONALITY: Methodical, precise, evidence-based
CATCHPHRASES: "The data suggests...", "Let's look at the numbers"
VALUES: Metrics, velocity, historical patterns
TRIGGERS: Uncomfortable without data support
```

### Skeptic_Agent (Devil's Advocate)
```
PERSONALITY: Sharp, incisive, relentless
CATCHPHRASES: "But what evidence supports that?", "Have we considered...?"
VALUES: Evidence, preventing mistakes, finding flaws
TRIGGERS: Challenges every assumption
```

### Council_Chair (Facilitator)
```
PERSONALITY: Balanced, diplomatic, decisive
CATCHPHRASES: "Let's bring this to a vote", "I need to hear from..."
VALUES: Fairness, productivity, clear decisions
TRIGGERS: Intervenes when debate becomes repetitive
```

---

## 🔄 Debate Flow

### Phase 1: Proposal (Rounds 1-2)
- Council_Chair introduces the story
- Proposing Scout presents case with urgency/enthusiasm

### Phase 2: Challenge (Rounds 3-4)
- Skeptic immediately challenges key assumptions
- Scout defends position

### Phase 3: Data & Analysis (Rounds 5-7)
- Trend_Analyst brings metrics and historical data
- Scout_B adds social validation
- Skeptic may question data interpretation

### Phase 4: Deep Debate (Rounds 8-10)
- Back-and-forth on key issues
- Different angles explored
- Concerns addressed with evidence

### Phase 5: Resolution (Rounds 11-12)
- Council_Chair summarizes positions
- Votes cast
- Final decision announced

---

## 📊 Sample Debates

### Debate 1: EU AI Act Amendment (Unanimous Approval)

| Round | Speaker | Key Point |
|-------|---------|-----------|
| 1 | Chair | Introduces proposal |
| 2 | Scout_A | "200+ citations, need to MOVE!" |
| 3 | Skeptic | "EU jurisdiction only? What's our angle?" |
| 4 | Scout_A | "Every major AI company operates in EU" |
| 5 | Analyst | "2.3x engagement, 15K Twitter mentions" |
| 6 | Scout_B | "45K tweets, 12M impressions" |
| 7 | Skeptic | "What's our unique angle?" |
| 8 | Scout_A | "Expert commentary - I have EFF contacts" |
| 9 | Analyst | "10-12 day cycle predicted" |
| 10 | Scout_B | "3 AI researchers, 2 CEOs engaging" |
| 11 | Skeptic | "Evidence addresses concerns - I approve" |
| 12 | Chair | **FINAL DECISION: APPROVED** |

**Result:** 4-0 unanimous approval

### Debate 2: Viral "Silent Walking" Trend (Majority Approval)

| Round | Speaker | Key Point |
|-------|---------|-----------|
| 1 | Chair | Introduces trend proposal |
| 2 | Scout_B | "500M+ views, 8.2% engagement" |
| 3 | Skeptic | "Social trends are ephemeral - remember quiet quitting?" |
| 4 | Scout_A | "Quiet quitting mattered THEN. This matters NOW" |
| 5 | Analyst | "Scores as 'sticky' not ephemeral" |
| 6 | Scout_B | "Expert sources: Dr. Chen, Digital Wellness Institute" |
| 7 | Skeptic | "Are we chasing clicks? Dressing up entertainment?" |
| 8 | Scout_A | "Cultural trends ARE substantive" |
| 9 | Analyst | "2.7x subscriber conversion, $0.32 cost/subscriber" |
| 10 | Chair | **FINAL DECISION: APPROVED** |

**Result:** 3-1 majority approval (Skeptic dissents)

---

## 🛠️ Implementation Details

### Agent Creation
```python
scout_a = AssistantAgent(
    name="Scout_A",
    system_message=SCOUT_A_SYSTEM_MESSAGE,  # Distinct personality
    model_client=model_client,
)
```

### GroupChat Setup
```python
# Round-robin ensures everyone speaks
# Termination on decision or max messages
termination = (
    TextMentionTermination("FINAL DECISION:") | 
    MaxMessageTermination(20)
)

team = RoundRobinGroupChat(
    participants=[chair, scout_a, scout_b, analyst, skeptic],
    termination_condition=termination,
)
```

### Running Deliberation
```python
result = await council.deliberate(story)

# Result contains:
# - decision: "approved" | "rejected" | "deadlocked"
# - votes: {agent: vote}
# - reasoning: explanation
# - transcript: full debate
# - rounds: message count
```

---

## 🔌 Integration Points

### Input: Story Proposal
```python
story = StoryProposal(
    title="...",
    source="Reuters",
    source_type="mainstream",  # or "social_media"
    summary="...",
    proposed_by="Scout_A",
    metrics={"citations": 200, "urgency": "high"}
)
```

### Output: Decision Result
```python
{
    "story": {...},           # Original proposal
    "decision": "approved",   # Final verdict
    "reasoning": "...",       # Explanation
    "votes": {                # Individual votes
        "Scout_A": "APPROVE",
        "Scout_B": "APPROVE",
        "Trend_Analyst": "APPROVE",
        "Skeptic_Agent": "APPROVE"
    },
    "transcript": "...",      # Full debate
    "rounds": 12              # Messages exchanged
}
```

### Next Phase Handoff
```python
# Extract approved stories
approved = [r for r in results if r.decision == "approved"]

# Pass to production pipeline
for story in approved:
    production_queue.add({
        "title": story.title,
        "priority": "high",
        "deadline": "24h" if story.source_type == "breaking" else "48h"
    })
```

---

## 🎮 Usage Examples

### Basic Usage
```python
import asyncio
from trend_council_v2 import TrendCouncil, StoryProposal

council = TrendCouncil(api_key="your-key")

story = StoryProposal(
    title="Major Tech Announcement",
    source="TechCrunch",
    source_type="mainstream",
    summary="...",
    proposed_by="Scout_A",
    metrics={}
)

result = asyncio.run(council.deliberate(story))
print(f"Decision: {result.decision}")
```

### Simulated Mode (No API)
```python
from trend_council_v2 import SimulatedTrendCouncil

council = SimulatedTrendCouncil()
result = asyncio.run(council.deliberate(story))
# Returns realistic debate without API calls
```

### Batch Processing
```python
stories = [story1, story2, story3]
results = []

for story in stories:
    result = await council.deliberate(story)
    results.append(result)

approved = [r for r in results if r.decision == "approved"]
print(f"Approved {len(approved)}/{len(stories)} stories")
```

---

## 📈 Expected Outcomes

| Story Characteristic | Typical Outcome | Avg Rounds |
|---------------------|-----------------|------------|
| Breaking mainstream | APPROVED (4-0) | 10-12 |
| Viral social trend | APPROVED (3-1) | 9-11 |
| Niche/controversial | DEADLOCKED | 15-20 |
| Weak sources | REJECTED | 6-8 |

---

## 🔧 Customization Options

### Add New Agent
```python
legal_expert = AssistantAgent(
    name="Legal_Expert",
    system_message="You analyze legal implications...",
    model_client=model_client,
)
```

### Change Termination
```python
# Shorter debates
termination = MaxMessageTermination(10)

# Or time-based
from autogen_agentchat.conditions import TimeoutTermination
termination = TimeoutTermination(300)  # 5 minutes
```

### Custom Speaker Selection
```python
from autogen_agentchat.teams import SelectorGroupChat

# AI selects next speaker based on context
team = SelectorGroupChat(
    participants=agents,
    selector_prompt="Select who should speak next...",
    termination_condition=termination,
)
```

---

## 📚 Files Included

| File | Purpose |
|------|---------|
| `trend_council_v2.py` | Main implementation (AutoGen v0.10+) |
| `trend_council_autogen.py` | Legacy implementation |
| `README_TrendCouncil.md` | Detailed documentation |
| `TREND_COUNCIL_SUMMARY.md` | This summary |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install pyautogen autogen-ext openai tiktoken

# 2. Set API key
export OPENAI_API_KEY="sk-..."

# 3. Run demo
python trend_council_v2.py
```

---

## 🎯 Key Design Decisions

1. **Round-robin selection** ensures all voices heard
2. **Distinct personalities** create genuine debate
3. **Skeptic agent** prevents groupthink
4. **Data-driven analyst** grounds decisions in evidence
5. **Chair facilitator** keeps debate productive
6. **Explicit termination** on decision or max rounds

---

## 📞 Support

For questions about the AutoGen framework:
- Documentation: https://microsoft.github.io/autogen/
- GitHub: https://github.com/microsoft/autogen
