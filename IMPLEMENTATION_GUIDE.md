# Agent Swarm Implementation Guide
## Automated News Video Production System

---

## Executive Summary

This guide provides a complete, production-ready implementation of a **self-organizing Agent Swarm** for automated news video production. The system replaces traditional linear pipelines with emergent coordination, where specialized agents bid, debate, vote, and iteratively refine content.

### Key Innovations
- **Distributed Decision-Making**: Agents vote on trend importance rather than following fixed rules
- **Redundant Coverage**: Multiple scouts compete; best story wins through bidding
- **Iterative Refinement**: Scripts get rewritten by critique agents until approved
- **Autonomous Recovery**: Failed visual generation triggers alternative agents automatically

---

## Architecture Overview

```
                    ┌─────────────────┐
                    │  COORDINATOR    │ ← Orchestrator (meta-agent)
                    │    (Editor)     │   Manages consensus, resolves conflicts
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ SCOUT SWARM   │   │ COUNCIL SWARM │   │ STUDIO SWARM  │
│  (Discovery)  │   │  (Evaluation) │   │ (Production)  │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
   ┌────┴────┐         ┌───┴───┐          ┌────┴────┐
   ▼         ▼         ▼       ▼          ▼         ▼
[Wire    [Social  [Trend  [Fact   [Script  [Visual  [Voice
 Scout]  Scout]   Voter]  Check]  Squad]  Artist]  Actor]
```

---

## 📁 Deliverables Summary

| Component | File | Description |
|-----------|------|-------------|
| **Architecture** | `news_video_swarm_architecture.md` | Complete agent specifications, interaction patterns, consensus protocols |
| **CrewAI Implementation** | `scout_council_swarm.py` | Production-ready Scout + Council swarms using CrewAI |
| **AutoGen Implementation** | `trend_council_v2.py` | Deliberative debate system using AutoGen |
| **Infrastructure** | `swarm_infrastructure/` | Blackboard, Redis, monitoring, cost management |
| **Documentation** | `README.md`, `README_TrendCouncil.md` | Usage guides and examples |

---

## Phase 1: Quick Start (Week 1)

### Step 1: Set Up Infrastructure

```bash
# Navigate to infrastructure directory
cd /mnt/okcomputer/output/swarm_infrastructure

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Start services
docker-compose up -d

# Verify
 docker-compose ps
```

Services will be available at:
- **Redis**: `localhost:6379`
- **ChromaDB**: `localhost:8000`
- **Langfuse**: `localhost:3000`

### Step 2: Install Dependencies

```bash
pip install -r /mnt/okcomputer/output/requirements.txt
```

### Step 3: Run Your First Swarm

```python
import asyncio
from scout_council_swarm import StoryPipeline

async def main():
    pipeline = StoryPipeline(storage_path="./output")
    
    # Run the full pipeline
    result = await pipeline.run_full_pipeline(topic="artificial intelligence")
    
    print(f"Stories discovered: {result.total_discovered}")
    print(f"Stories approved: {result.total_approved}")
    
    # View approved stories
    for story in result.approved_stories:
        print(f"\n✅ {story.headline}")
        print(f"   Confidence: {story.confidence_score:.2f}")

asyncio.run(main())
```

---

## Phase 2: Understanding the Components

### 1. Scout Swarm (Tier 1) - News Discovery

The Scout Swarm consists of **4 specialized agents** that compete to discover stories:

| Agent | Specialty | Personality | Key Tools |
|-------|-----------|-------------|-----------|
| **Wire_Scout** | Traditional media | Veteran journalist, trusts authority | RSS, NewsAPI, GDELT |
| **Social_Scout** | Viral trends | Digital native, trend-obsessed | Twitter/X API, Reddit, TikTok |
| **Semantic_Scout** | Pattern detection | Data scientist, sees connections | NLP clustering, entity mapping |
| **Geo_Scout** | Regional news | Foreign correspondent, location-focused | Geolocation, regional sources |

**Bidding Protocol**: Each scout submits a structured pitch:

```python
{
    "story_id": "uuid",
    "headline": "Compelling headline",
    "summary": "2-3 sentence summary",
    "confidence_score": 0.92,  # 0.0 - 1.0
    "urgency_score": 0.85,
    "novelty_score": 0.78,
    "sources": [...],
    "discovered_by": "Wire_Scout"
}
```

**Conflict Resolution**: When multiple scouts find the same story, an auction resolves conflicts using weighted scoring:
- Confidence: 40%
- Urgency: 25%
- Novelty: 20%
- Source diversity: 15%

### 2. Council Swarm (Tier 2) - Trend Validation

The Council Swarm is a **deliberative body** that replaces traditional trend detection algorithms with multi-agent debate.

#### Option A: CrewAI Voting (Fast)

```python
from scout_council_swarm import StoryPipeline

# Council agents vote in parallel
# - Trend_Voter_Mainstream (media consensus)
# - Trend_Voter_Social (engagement metrics)
# - Trend_Voter_Historical (long-term significance)
# - Fact_Checker (validation with VETO power)

pipeline = StoryPipeline()
result = await pipeline.run_council_vote(stories)
```

#### Option B: AutoGen Debate (Thorough)

```python
from trend_council_v2 import TrendCouncil

# Council members debate and argue
council = TrendCouncil()

story = {
    "title": "EU passes AI Act amendment",
    "source": "Reuters",
    "summary": "European Union lawmakers approved significant amendments..."
}

result = await council.deliberate(story)
print(result.decision)  # APPROVED / REJECTED / DEADLOCKED
print(result.transcript)  # Full debate transcript
```

**Consensus Thresholds**:
| Story Type | Threshold | Description |
|------------|-----------|-------------|
| Standard | 0.70 | Regular news stories |
| Breaking | 0.60 | Fast-track for urgent news |
| High-risk | 0.80 | Sensitive topics |
| Legal review | 0.85 | Potential legal issues |

**Voting Weights**:
- Trend_Voter_B (Realist): 25%
- Fact_Checker: 25% + VETO power
- Trend_Voter_A (Optimist): 20%
- Trend_Voter_C (Skeptic): 20%
- Compliance_Guard: 10% + LEGAL_HOLD power

### 3. Studio Swarm (Tier 3) - Content Production

The Studio Swarm uses an **iterative refinement loop**:

```
Script_Writer (Draft) 
    ↓
Hook_Doctor (Critiques first 3 seconds) 
    ↓
Script_Writer (Revise) 
    ↓
Fact_Guard (Verify claims) 
    ↓
Voice_Actor + Visual_Artist (Parallel generation)
    ↓
Video_Editor (Assemble)
    ↓
Quality_Critic (Reviews final cut)
    ↓
[Pass] → Publish / [Fail] → Revise (max 3 iterations)
```

**Agent Specializations**:
- **Hook_Doctor**: Obsessive about retention psychology
- **Visual_Artist**: Uses DALL-E 3/Midjourney + stock video APIs
- **Video_Editor**: MoviePy expert, can reject assets if they don't sync

### 4. Infrastructure Layer

#### Blackboard Architecture (Shared Memory)

All agents read/write to a central **News Blackboard**:

```python
from blackboard import Blackboard, BlackboardSection

blackboard = Blackboard()

# Write story to raw feed
await blackboard.write(
    section=BlackboardSection.RAW_FEED,
    item_id="news_001",
    data={"headline": "...", "content": "..."},
    check_duplicate=True  # Prevents duplicates
)

# Move to debate queue
await blackboard.move(
    from_section=BlackboardSection.RAW_FEED,
    to_section=BlackboardSection.DEBATE_QUEUE,
    item_id="news_001"
)

# Claim for processing (distributed lock)
if await blackboard.claim("news_001", "agent_1"):
    # Exclusive access
    await process_story("news_001")
```

#### Redis Communication

```python
from message_bus import MessageBus, MessageType

bus = MessageBus()

# Subscribe to tasks
async def handle_task(message):
    print(f"Received: {message.payload}")

bus.on(MessageType.TASK_ASSIGN, handle_task)

# Send direct message
await bus.send_to_agent(
    recipient="writer_1",
    message=Message.create(
        msg_type=MessageType.TASK_ASSIGN,
        payload={"task": "write_script", "story_id": "news_001"}
    )
)
```

#### Cost Management

```python
from cost_manager import CostManager, LLMTier

cost_manager = CostManager(
    daily_budget=100.0,
    hourly_budget=20.0
)

# Track LLM usage
await cost_manager.track_usage(
    agent_id="scout_1",
    task_type="news_discovery",
    model="gpt-3.5-turbo",
    tier=LLMTier.FAST,
    prompt_tokens=200,
    completion_tokens=50,
    confidence=0.85
)

# Check budget before expensive operation
if await cost_manager.can_execute("writer_1", LLMTier.PREMIUM):
    # Safe to use GPT-4
    result = await generate_with_gpt4()
```

**LLM Tiering Strategy**:
| Tier | Models | Cost/1K tokens | Use Case |
|------|--------|----------------|----------|
| FAST | GPT-3.5, Claude Haiku | $0.0015 | Screening, initial filtering |
| BALANCED | GPT-4 Turbo, Claude Sonnet | $0.015 | Analysis, voting |
| PREMIUM | GPT-4, Claude Opus | $0.06 | Final scripts, quality control |

---

## Phase 3: Integration Patterns

### Pattern 1: Scout → Council Handoff

```python
# After scouts discover stories
scouts = [Wire_Scout(), Social_Scout(), Semantic_Scout()]
discovered = await asyncio.gather(*[s.discover(topic) for s in scouts])

# Deduplicate and bid
unique_stories = deduplicate_stories(discovered)
winning_bids = resolve_bids(unique_stories)

# Pass to council for validation
pipeline = StoryPipeline()
council_results = await pipeline.run_council_vote(winning_bids)

# Filter approved stories
approved = [s for s in council_results if s.approved]
```

### Pattern 2: Council → Studio Handoff

```python
# Council approves story
from trend_council_v2 import TrendCouncil

council = TrendCouncil()
result = await council.deliberate(story)

if result.decision == "APPROVED":
    # Write to blackboard for studio
    await blackboard.move(
        from_section=BlackboardSection.DEBATE_QUEUE,
        to_section=BlackboardSection.PRODUCTION_FLOOR,
        item_id=story.id
    )
    
    # Notify studio agents
    await message_bus.publish(
        channel=Channels.TIER_PRODUCER,
        message=Message.create(
            msg_type=MessageType.PROD_START,
            payload={"story_id": story.id}
        )
    )
```

### Pattern 3: Studio Iteration Loop

```python
max_iterations = 3
iteration = 0
script_approved = False

while not script_approved and iteration < max_iterations:
    # Generate script
    script = await script_writer.write(story)
    
    # Critique
    critiques = await asyncio.gather(
        hook_doctor.critique(script),
        fact_guard.verify(script),
        quality_critic.review(script)
    )
    
    # Check if all approve
    if all(c.approved for c in critiques):
        script_approved = True
    else:
        # Revise with feedback
        feedback = merge_critiques(critiques)
        script = await script_writer.revise(script, feedback)
        iteration += 1
```

---

## Phase 4: Production Deployment

### Docker Compose Stack

```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma
      
  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://...
      - NEXTAUTH_SECRET=...
      - SALT=...
```

### Environment Configuration

```bash
# .env file
OPENAI_API_KEY=sk-...
NEWSAPI_KEY=...
REDIS_URL=redis://localhost:6379
CHROMA_URL=http://localhost:8000
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...

# Budget controls
DAILY_BUDGET=100.0
HOURLY_BUDGET=20.0
MAX_ITERATIONS=3
```

### Monitoring Dashboard

Access Langfuse at `http://localhost:3000` to view:
- Agent decision traces
- Consensus vote patterns
- Cost breakdown by agent
- Pipeline latency metrics
- Error rates and retries

---

## Phase 5: Advanced Features

### 1. Emergent Load Balancing

During breaking news events, the Coordinator automatically:

```python
if story_queue.size > threshold:
    # Spawn temporary auxiliary scouts
    auxiliary_scouts = spawn_auxiliary_agents(count=3)
    
    # Prioritize high-confidence items
    prioritized = sort_by_confidence(story_queue)
    
    # Queue lower-priority items for tomorrow
    defer_to_batch(prioritized[low_priority])
```

### 2. Self-Optimization (Month 2+)

Agents adjust their behavior based on performance:

```python
# Feedback_Analyst writes to blackboard
feedback = {
    "scout_patterns": {
        "Wire_Scout": {"acceptance_rate": 0.72, "avg_confidence": 0.85},
        "Social_Scout": {"acceptance_rate": 0.58, "avg_confidence": 0.78}
    },
    "recommendations": [
        "Increase Social_Scout threshold from 0.6 to 0.7",
        "Weight technology topics 15% higher"
    ]
}

# Scouts read feedback and adjust
wire_scout.adjust_threshold(new_threshold=0.75)
```

### 3. Human-in-the-Loop

For sensitive stories, require human approval:

```python
if story.risk_level == "high":
    # Pause for human review
    await blackboard.move(
        to_section=BlackboardSection.HUMAN_REVIEW,
        item_id=story.id
    )
    
    # Notify human editor
    await send_notification(
        to="editor@newsroom.com",
        subject=f"High-risk story requires approval: {story.headline}"
    )
```

---

## Cost Optimization Strategies

### 1. Early Stopping

```python
# Stop processing low-confidence stories early
if confidence < 0.3:
    logger.info(f"Stopping early: confidence {confidence} < 0.3")
    return StoryResult(rejected=True, reason="low_confidence")
```

### 2. LLM Tiering

```python
# Use cheaper models for screening
if task_type == "initial_screening":
    model = "gpt-3.5-turbo"  # $0.0015/1K tokens
elif task_type == "final_script":
    model = "gpt-4"  # $0.06/1K tokens - worth it for quality
```

### 3. Batch Processing

```python
# Council reviews 5 stories at once, not 1-by-1
batch = collect_stories(count=5)
results = await council.deliberate_batch(batch)
```

### 4. Caching

```python
# Cache fact-checking results
@cache(ttl=3600)
async def check_fact(claim: str) -> FactCheckResult:
    return await fact_checker.verify(claim)
```

---

## Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Story discovery → publish | < 2 hours | End-to-end pipeline |
| Council consensus rate | > 80% | Stories reaching agreement |
| Visual generation success | > 90% | With fallback recovery |
| System availability | > 99.5% | Including failure recovery |
| Cost per video | $2-5 | Depending on complexity |
| Daily output | 10-20 videos | With load balancing |

---

## Troubleshooting

### Issue: Circular Debates

**Solution**: Set max round limits; Coordinator breaks ties

```python
# In AutoGen GroupChat
termination = MaxMessageTermination(max_messages=20) | 
              TextMentionTermination("FINAL DECISION:")
```

### Issue: Agent Collusion

**Solution**: Ensure conflicting incentives
- Scout rewards: Speed + discovery rate
- Voter rewards: Accuracy + quality

### Issue: Token Budget Exceeded

**Solution**: Implement circuit breakers

```python
if daily_spend > DAILY_BUDGET * 0.8:
    # Switch to cheaper models
    force_tier = LLMTier.FAST
    send_alert("Budget alert: 80% of daily budget consumed")
```

---

## Next Steps

1. **Week 1**: Deploy infrastructure, run Phase 1 MVP
2. **Week 2-3**: Implement Council Swarm, test debate vs voting
3. **Week 4-6**: Add Studio Swarm, implement iteration loops
4. **Month 2+**: Add self-optimization, feedback loops

### Immediate Action

**Build the "Trend Council" first**—it's the highest-value component:

```bash
# Run the Trend Council demo
python /mnt/okcomputer/output/trend_council_v2.py
```

This will show you how agents debate and reach consensus on real stories.

---

## File Reference

```
/mnt/okcomputer/output/
├── news_video_swarm_architecture.md    # Complete architecture spec
├── scout_council_swarm.py              # CrewAI implementation
├── trend_council_v2.py                 # AutoGen implementation
├── trend_council_autogen.py            # Alternative AutoGen version
├── example_usage.py                    # Usage examples
├── requirements.txt                    # Python dependencies
├── README.md                           # CrewAI README
├── README_TrendCouncil.md              # AutoGen README
├── TREND_COUNCIL_SUMMARY.md            # Council architecture
└── swarm_infrastructure/               # Infrastructure code
    ├── docker-compose.yml
    ├── src/
    │   ├── blackboard.py
    │   ├── message_bus.py
    │   ├── cost_manager.py
    │   ├── monitoring.py
    │   └── failure_recovery.py
    └── examples/
```

---

## Support & Resources

- **CrewAI Docs**: https://docs.crewai.com
- **AutoGen Docs**: https://microsoft.github.io/autogen/
- **Langfuse**: https://langfuse.com/docs

---

*This implementation provides everything needed to build a production-grade Agent Swarm for automated news video production. Start with Phase 1 and iterate based on your specific requirements.*
