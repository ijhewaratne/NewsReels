# Multi-Agent News Video Production Swarm - Infrastructure

A production-ready infrastructure for orchestrating multi-agent systems that collaboratively produce news videos. Built with Redis, ChromaDB, Langfuse, and designed for cost optimization and failure recovery.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SWARM INFRASTRUCTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Scout      │    │   Analyst    │    │   Writer     │    Agents        │
│  │  (FAST LLM)  │    │(BALANCED LLM)│    │(PREMIUM LLM) │                  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                  │
│         │                   │                   │                           │
│         └───────────────────┼───────────────────┘                           │
│                             │                                               │
│         ┌───────────────────┴───────────────────┐                          │
│         │                                       │                          │
│  ┌──────▼────────┐    ┌──────────────┐    ┌────▼─────────┐               │
│  │  Blackboard   │    │ Message Bus  │    │ Cost Manager │   Core Systems │
│  │  (Shared Mem) │    │   (Redis)    │    │              │               │
│  └──────┬────────┘    └──────┬───────┘    └──────┬───────┘               │
│         │                    │                   │                         │
│  ┌──────▼────────────────────▼───────────────────▼───────┐               │
│  │                      Redis Layer                      │               │
│  │  - Pub/Sub Channels   - State Storage   - Queues     │               │
│  └───────────────────────────────────────────────────────┘               │
│                                                                             │
│  ┌───────────────────────────────────────────────────────┐               │
│  │                    ChromaDB Layer                     │               │
│  │  - Vector Memory    - Semantic Search   - Deduplication│              │
│  └───────────────────────────────────────────────────────┘               │
│                                                                             │
│  ┌───────────────────────────────────────────────────────┐               │
│  │                   Monitoring Layer                    │               │
│  │  - Langfuse Traces  - Prometheus      - Grafana      │               │
│  └───────────────────────────────────────────────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Start Infrastructure Services

```bash
# Clone and navigate
cd swarm_infrastructure

# Copy environment file
cp .env.example .env
# Edit .env with your API keys

# Start services
docker-compose up -d

# Verify services
 docker-compose ps
```

Services will be available at:
- Redis: `localhost:6379`
- ChromaDB: `localhost:8000`
- Langfuse: `localhost:3000`
- Redis Insight (debug): `localhost:5540`

### 2. Install Python Dependencies

```bash
pip install redis chromadb langfuse sentence-transformers
```

### 3. Run Example

```bash
cd examples
python basic_usage.py
```

## Core Components

### 1. Blackboard Architecture

The Blackboard is the central shared memory where all agents read and write state.

#### Data Schema

```python
# Sections
BlackboardSection.RAW_FEED        # Incoming news items
BlackboardSection.DEBATE_QUEUE    # Items awaiting debate
BlackboardSection.CONSENSUS       # Agreed-upon stories
BlackboardSection.PRODUCTION_FLOOR # Active video production
BlackboardSection.REVIEW_QUEUE    # Completed videos
BlackboardSection.PUBLISHED       # Published content
BlackboardSection.ARCHIVE         # Historical data
BlackboardSection.AGENT_STATE     # Agent statuses
```

#### State Transitions

```
PENDING → PROCESSING → DEBATING → CONSENSUS_REACHED → PRODUCING → REVIEWING → APPROVED → PUBLISHED
   ↓          ↓           ↓              ↓               ↓            ↓          ↓
FAILED    (retry)    (no consensus)  (failed)       (failed)    (rejected)  (archive)
```

#### Duplicate Detection

```python
# Content hash-based deduplication
news_item = NewsItem(
    id="news_001",
    title="Breaking News",
    content="Content here...",
    source="NewsSource"
)
# Automatically computes content_hash for deduplication

# Write with duplicate check
success = await blackboard.write(
    section=BlackboardSection.RAW_FEED,
    item_id=news_item.id,
    data=news_item.to_dict(),
    check_duplicate=True  # Rejects if hash exists
)
```

#### Locking Mechanism

```python
# Distributed lock for exclusive access
async with blackboard.acquire_lock(item_id, agent_id):
    # Only this agent can modify the item
    await blackboard.update(section, item_id, updates)
```

### 2. Redis Communication Layer

#### Message Types

```python
MessageType.TASK_ASSIGN      # Task assignment
MessageType.TASK_COMPLETE    # Task completion
MessageType.DEBATE_INVITE    # Debate invitation
MessageType.DEBATE_VOTE      # Consensus vote
MessageType.PROD_START       # Production start
MessageType.HEARTBEAT        # Agent heartbeat
MessageType.EMERGENCY        # Emergency broadcast
```

#### Pub/Sub Channels

```python
# Broadcast channels
Channels.BROADCAST_ALL        # All agents
Channels.BROADCAST_CONTROL    # Control messages
Channels.BROADCAST_EMERGENCY  # Emergency alerts

# Tier channels
Channels.TIER_SCOUT          # Scout agents
Channels.TIER_ANALYST        # Analyst agents
Channels.TIER_DEBATER        # Debater agents
Channels.TIER_PRODUCER       # Producer agents

# Direct agent channels
Channels.agent_channel("agent_1")  # Direct to agent_1
```

#### Example Usage

```python
# Publish message
message = Message.create(
    msg_type=MessageType.TASK_ASSIGN,
    sender="analyst_1",
    recipient="writer_1",
    payload={"task": "write_script", "news_id": "news_001"}
)
await message_bus.send_to_agent("writer_1", message)

# Subscribe to messages
async def handle_task(msg: Message):
    print(f"Received task: {msg.payload}")

message_bus.on(MessageType.TASK_ASSIGN, handle_task)
```

#### Work Queues

```python
# Enqueue task
task = TaskMessage(
    task_id="task_001",
    task_type="script_generation",
    parameters={"news_id": "news_001"},
    estimated_tokens=2000
)
await message_bus.enqueue_task(Channels.QUEUE_PRODUCTION, task)

# Dequeue task
task = await message_bus.dequeue_task(Channels.QUEUE_PRODUCTION)
```

### 3. Cost Management System

#### LLM Tiering Strategy

```python
# Fast tier (GPT-3.5, Claude Haiku) - $0.0015/1K tokens
TASK_TIER_MAPPING = {
    "news_scout": LLMTier.FAST,        # Initial screening
    "content_filter": LLMTier.FAST,    # Simple filtering
    "topic_tagging": LLMTier.FAST,     # Classification
}

# Balanced tier (GPT-4 Turbo, Claude Sonnet) - $0.015/1K tokens
TASK_TIER_MAPPING = {
    "content_analysis": LLMTier.BALANCED,   # Analysis
    "debate_argument": LLMTier.BALANCED,    # Debate
    "script_draft": LLMTier.BALANCED,       # Draft scripts
}

# Premium tier (GPT-4, Claude Opus) - $0.06/1K tokens
TASK_TIER_MAPPING = {
    "script_final": LLMTier.PREMIUM,        # Final scripts
    "quality_assurance": LLMTier.PREMIUM,   # QA
}
```

#### Token Tracking

```python
# Record usage
await cost_manager.record_usage(
    agent_id="writer_1",
    task_id="script_001",
    task_type="script_final",
    model=LLMModel.GPT4,
    tokens=TokenUsage(prompt_tokens=1000, completion_tokens=500),
    duration_ms=3000,
    confidence_score=0.92
)
```

#### Budget Management

```python
# Set agent budget
budget = AgentBudget(
    agent_id="writer_1",
    daily_limit=100.0,
    hourly_limit=20.0,
    task_limit=10.0,
    alert_threshold=0.8  # Alert at 80%
)
await cost_manager.set_budget(budget)

# Check budget
status = await cost_manager.get_budget_status("writer_1")
print(f"Daily: ${status.daily_spent:.2f} / ${status.daily_limit:.2f}")
```

#### Early Stopping Rules

```python
# Automatic early stopping based on confidence
if confidence_score < 0.3:
    # Stop processing - low quality
    await cost_manager._trigger_stop(agent_id, task_id, confidence_score)
elif confidence_score > 0.9:
    # Stop processing - good enough
    await cost_manager._trigger_stop(agent_id, task_id, confidence_score)
```

### 4. Monitoring with Langfuse

#### What to Trace

```python
# Agent decisions
trace_id = monitor.start_trace(TraceType.AGENT_DECISION)
monitor.track_agent_decision(trace_id, AgentDecision(
    agent_id="analyst_1",
    agent_role="analyst",
    decision_type="coverage_decision",
    input_context={"news_item": item},
    output_decision={"should_cover": True},
    confidence=0.85,
    reasoning="High credibility source"
))

# Consensus votes
monitor.track_consensus_vote(trace_id, ConsensusVote(
    debate_id="debate_001",
    agent_id="debater_1",
    vote="for",
    confidence=0.9
))

# Script revisions
monitor.track_revision(trace_id, RevisionRecord(
    content_id="script_001",
    revision_number=2,
    revision_type="grammar",
    improvement_score=0.15
))
```

#### Dashboard Metrics

Key metrics to track:
- **Pipeline Performance**: End-to-end latency, throughput
- **Agent Decisions**: Confidence distribution, decision counts
- **Consensus Quality**: Consensus scores, vote patterns
- **Cost Tracking**: Token usage, budget utilization
- **Quality Metrics**: Revision counts, improvement scores
- **Error Rates**: Failure rates by agent/task type

### 5. Failure Recovery

#### Retry with Exponential Backoff

```python
@failure_recovery.with_retry(
    policy=RetryPolicy(
        max_retries=3,
        base_delay=1.0,
        exponential_base=2.0
    )
)
async def generate_visual(prompt: str):
    return await runway.generate(prompt)
```

#### Circuit Breaker

```python
# Create circuit breaker for external service
cb = failure_recovery.get_circuit_breaker("runway")

# Use with protection
try:
    result = await cb.call(runway.generate, prompt)
except CircuitOpenError:
    # Circuit is open, use fallback
    result = await fallback_provider.generate(prompt)
```

#### Fallback Agents

```python
# Register fallback chain
failure_recovery.register_fallback(
    "visual_designer",
    ["visual_designer_backup", "visual_designer_simple", "stock_image_agent"]
)

# Spawn fallback on failure
fallback_id = await failure_recovery.spawn_fallback_agent(
    failed_agent_id="visual_designer_1",
    agent_role="visual_designer",
    task_id="task_001",
    task_data={"prompt": "news scene"}
)
```

#### Checkpoint and Resume

```python
# Create checkpoint
await failure_recovery.create_checkpoint(
    task_id="video_001",
    stage="script_written",
    data={"script": script_content},
    agent_states={"writer": writer_state},
    ttl_hours=24
)

# Resume from checkpoint
checkpoint = await failure_recovery.resume_from_checkpoint("video_001")
if checkpoint:
    # Continue from checkpoint.stage
    await continue_production(checkpoint)
```

### 6. ChromaDB Vector Memory

#### Collections

```python
# News articles collection
await chroma.add_document(
    collection_name="news_articles",
    document_id="news_001",
    content="Article content...",
    metadata={
        "source": "TechNews",
        "topics": ["AI", "Technology"],
        "sentiment": 0.7
    }
)

# Semantic search
results = await chroma.search(
    collection_name="news_articles",
    query="artificial intelligence breakthrough",
    n_results=10
)
```

#### Agent Memory

```python
# Store agent memory
memory_id = await chroma.store_agent_memory(
    agent_id="analyst_1",
    memory_type="lesson_learned",
    content="Users engage more with positive news stories",
    importance=0.8
)

# Recall memories
memories = await chroma.recall_agent_memories(
    agent_id="analyst_1",
    query="user engagement patterns"
)
```

## Configuration

### Environment Variables

```bash
# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_password

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_AUTH_TOKEN=your_token

# Langfuse
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_key
LANGFUSE_HOST=http://localhost:3000

# Cost Management
GLOBAL_DAILY_BUDGET=1000.0
ENABLE_COST_ALERTS=true

# External APIs
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
ELEVENLABS_API_KEY=your_key
HEYGEN_API_KEY=your_key
RUNWAYML_API_KEY=your_key
```

### Docker Compose

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
      - chromadb_data:/chroma/chroma
      
  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:pass@postgres:5432/langfuse
```

## Agent Implementation Pattern

```python
class MyAgent(BaseAgent):
    def __init__(self, agent_id: str, swarm: SwarmInfrastructure):
        super().__init__(
            agent_id=agent_id,
            role="my_role",
            swarm=swarm,
            capabilities=["capability_1", "capability_2"]
        )
    
    async def _process_cycle(self):
        # Get pending tasks
        items = await self.swarm.blackboard.query_by_status(
            section=BlackboardSection.RAW_FEED,
            status=ContentStatus.PENDING
        )
        
        for item in items:
            # Claim task
            if await self.swarm.claim_task(
                agent_id=self.agent_id,
                section=BlackboardSection.RAW_FEED,
                item_id=item["id"]
            ):
                # Process with monitoring
                trace_id = self.swarm.start_trace(TraceType.AGENT_DECISION)
                
                try:
                    result = await self._process_item(item)
                    
                    # Track costs
                    await self.swarm.track_llm_usage(...)
                    
                    # Complete task
                    await self.swarm.complete_task(...)
                    
                    self.swarm.end_trace(trace_id, "success")
                    
                except Exception as e:
                    await self.swarm.fail_task(...)
                    self.swarm.end_trace(trace_id, "error")
```

## Cost Optimization Tips

1. **Use appropriate LLM tiers**: Fast models for screening, premium only for final outputs
2. **Implement early stopping**: Stop if confidence < 0.3 or > 0.9
3. **Cache embeddings**: Reuse embeddings for similar content
4. **Batch requests**: Process multiple items in single LLM call when possible
5. **Set budget alerts**: Get notified at 80% of budget
6. **Use circuit breakers**: Fail fast when services are down
7. **Checkpoint frequently**: Avoid reprocessing on failure

## Troubleshooting

### Redis Connection Issues
```bash
# Check Redis is running
docker-compose ps redis
redis-cli ping

# Check logs
docker-compose logs redis
```

### ChromaDB Connection Issues
```bash
# Check ChromaDB is running
curl http://localhost:8000/api/v1/heartbeat

# Check logs
docker-compose logs chromadb
```

### Langfuse Connection Issues
```bash
# Check Langfuse is running
curl http://localhost:3000/api/public/health

# Check database connection
docker-compose logs langfuse
```

## File Structure

```
swarm_infrastructure/
├── docker-compose.yml          # Infrastructure services
├── .env.example                # Environment template
├── README.md                   # This file
├── redis/
│   └── redis.conf             # Redis configuration
├── monitoring/
│   ├── prometheus.yml         # Prometheus config
│   └── grafana/
│       └── datasources/
│           └── datasources.yml
├── src/
│   ├── __init__.py
│   ├── blackboard.py          # Shared memory
│   ├── message_bus.py         # Communication layer
│   ├── cost_manager.py        # Budget control
│   ├── monitoring.py          # Langfuse integration
│   ├── failure_recovery.py    # Resilience
│   ├── chroma_integration.py  # Vector memory
│   └── swarm_infrastructure.py # Main integration
└── examples/
    ├── basic_usage.py         # Basic example
    └── agent_implementation.py # Agent patterns
```

## License

MIT License - See LICENSE file for details.
