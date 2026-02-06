# Trend Council Deliberation System - AutoGen Implementation

A multi-agent debate system for automated news video production where agents **DELIBERATE** and **ARGUE** about whether stories are worth covering.

## 🏛️ Council Members

| Agent | Role | Personality | LLM Config |
|-------|------|-------------|------------|
| **Scout_A** | Mainstream news scout | Fast, aggressive, wants speed | GPT-4 (creative, temp=0.9) |
| **Scout_B** | Social media scout | Trend-obsessed, viral-focused | GPT-4 (temp=0.7) |
| **Trend_Analyst** | Data evaluator | Methodical, numbers-focused | GPT-4 (temp=0.7) |
| **Skeptic_Agent** | Devil's advocate | Challenges everything | GPT-4 (temp=0.7) |
| **Council_Chair** | Facilitator | Balanced, decisive | GPT-4 (temp=0.7) |

## 🎯 Key Features

### 1. Agent Configuration
Each agent has a **distinct system message** that creates personality:

```python
SCOUT_A_SYSTEM_MESSAGE = """You are Scout_A, an aggressive news scout...
- You are FAST and AGGRESSIVE. You want to beat competitors to stories.
- You believe SPEED is everything in news. Being second is being last.
- You often say things like "We're going to miss the window!"
"""
```

### 2. GroupChat Setup
Uses **RoundRobinGroupChat** for structured debate:

```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination

termination = TextMentionTermination("FINAL DECISION:") | MaxMessageTermination(20)

team = RoundRobinGroupChat(
    participants=[chair, scout_a, scout_b, analyst, skeptic],
    termination_condition=termination,
)
```

### 3. Custom Termination Conditions
Debate ends when:
- ✅ Council_Chair makes a decision (`FINAL DECISION:` detected)
- ✅ Max messages exceeded (default 20)
- ✅ Clear consensus emerges

### 4. Debate Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  ROUND 1: Council_Chair introduces the story proposal           │
├─────────────────────────────────────────────────────────────────┤
│  ROUND 2: Scout_A/B presents the case with urgency              │
├─────────────────────────────────────────────────────────────────┤
│  ROUND 3: Skeptic challenges assumptions, demands evidence      │
├─────────────────────────────────────────────────────────────────┤
│  ROUND 4: Scouts defend their position                          │
├─────────────────────────────────────────────────────────────────┤
│  ROUND 5+: Trend_Analyst brings data; back-and-forth continues │
├─────────────────────────────────────────────────────────────────┤
│  FINAL: Council_Chair calls vote, announces decision            │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Usage Example

```python
import asyncio
from trend_council_v2 import TrendCouncil, StoryProposal

# Initialize council
council = TrendCouncil(api_key="your-openai-key")

# Create a story proposal
story = StoryProposal(
    title="EU Passes Landmark AI Act Amendment",
    source="Reuters",
    source_type="mainstream",
    summary="European Parliament passes AI Act amendment targeting foundation models...",
    proposed_by="Scout_A",
    metrics={"mainstream_citations": 200, "urgency": "high"}
)

# Run deliberation
result = asyncio.run(council.deliberate(story))

# Check decision
print(f"Decision: {result.decision}")
print(f"Votes: {result.votes}")
print(f"Transcript: {result.transcript}")
```

## 🎭 Sample Debate Transcript

### Story: EU AI Act Amendment

**[Round 1] Council_Chair:** The Trend Council is now in session...

**[Round 2] Scout_A:** This is breaking NOW! Reuters, AP, Financial Times all running with it. 200+ citations. We need to MOVE!

**[Round 3] Skeptic_Agent:** Hold on. Does every AI company really have to comply? EU jurisdiction only. And what's our angle?

**[Round 4] Scout_A:** OpenAI, Google, Microsoft ALL do business in Europe. This is about WHO CONTROLS AI. Our audience cares!

**[Round 5] Trend_Analyst:** Data shows AI regulation stories get 2.3x engagement. 15,400 Twitter mentions, growing 12%/hour.

**[Round 6] Scout_B:** #AIAct trending: 45K tweets, 12M impressions. TikTok explainer at 2.3M views.

**[Round 7] Skeptic_Agent:** Better than expected. But what's our unique angle? Every outlet is covering this.

**[Round 8] Scout_A:** Expert commentary angle - I have EFF contacts. This is a BEAT, not just a story.

**[Round 9] Trend_Analyst:** Predictive metrics: 10-12 day cycle. Compliance announcements expected within 72 hours.

**[Round 10] Scout_B:** Influencer validation: 3 AI researchers, 2 CEOs engaging. Quality signal.

**[Round 11] Skeptic_Agent:** Evidence addresses my concerns. I'll support with conditions.

**[Round 12] Council_Chair:** FINAL DECISION: APPROVED - Unanimous approval. Expert commentary angle. 24-hour production target.

## 📊 Integration: Extracting Decisions

```python
def extract_approved_stories(results: List[DeliberationResult]) -> List[StoryProposal]:
    """Filter to only approved stories for next phase."""
    return [r.story for r in results if r.decision == "approved"]

# Pass approved stories to next phase
approved = extract_approved_stories(results)
for story in approved:
    production_pipeline.add_story(story)
```

## 🔧 Simulated Mode (No API Calls)

For testing without OpenAI API:

```python
from trend_council_v2 import SimulatedTrendCouncil

council = SimulatedTrendCouncil()
result = asyncio.run(council.deliberate(story))
# Returns realistic debate transcript without API calls
```

## 📁 Files

| File | Description |
|------|-------------|
| `trend_council_autogen.py` | Original implementation (legacy AutoGen API) |
| `trend_council_v2.py` | Modern implementation (AutoGen v0.10+ API) |
| `README_TrendCouncil.md` | This documentation |

## 🚀 Running the Demo

```bash
# Install dependencies
pip install pyautogen autogen-ext openai tiktoken

# Set API key
export OPENAI_API_KEY="your-key"

# Run simulation
python trend_council_v2.py
```

## 🎨 Customization

### Add New Agent Types

```python
# Create new agent with custom personality
legal_expert = AssistantAgent(
    name="Legal_Expert",
    system_message="You analyze legal implications of stories...",
    model_client=model_client,
)

# Add to team
agent_list = [chair, scout_a, scout_b, analyst, skeptic, legal_expert]
```

### Modify Termination Conditions

```python
from autogen_agentchat.conditions import TimeoutTermination

# End after 5 minutes OR when decision made
termination = TextMentionTermination("FINAL DECISION:") | TimeoutTermination(300)
```

### Custom Speaker Selection

Instead of round-robin, use selector-based:

```python
from autogen_agentchat.teams import SelectorGroupChat

team = SelectorGroupChat(
    participants=agent_list,
    selector_prompt="Select the next speaker based on who can best address...",
    termination_condition=termination,
)
```

## 📈 Decision Patterns

| Story Type | Typical Outcome | Key Factors |
|------------|-----------------|-------------|
| **Breaking mainstream** | APPROVED (unanimous) | Source credibility, urgency |
| **Viral social trend** | APPROVED (majority) | Engagement metrics, expert validation |
| **Niche/unclear angle** | REJECTED or DEADLOCKED | Lack of differentiation |
| **Controversial** | Extended debate | Skeptic challenges heavily |

## 🔗 Next Phase Integration

```python
# After deliberation
if result.decision == "approved":
    # Pass to script writing team
    script_team = ScriptWritingTeam()
    script = await script_team.write_script(result.story)
    
    # Queue for production
    production_queue.add({
        "story": result.story,
        "script": script,
        "priority": "high" if "breaking" in result.story.source_type else "normal"
    })
```

## 📝 License

MIT License - Microsoft AutoGen Framework
