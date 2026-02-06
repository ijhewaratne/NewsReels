# News Video System - Scout & Council Swarm

A production-ready CrewAI implementation for automated news video generation.

## Overview

This system implements a self-organizing Agent Swarm using CrewAI for automated news discovery, evaluation, and approval. It's designed as the first phase of a larger automated news video production pipeline.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SCOUT & COUNCIL SWARM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              TIER 1: SCOUT SWARM                        │   │
│  │  (Parallel Discovery - All scouts run simultaneously)   │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  • Wire_Scout      → Breaking news from RSS/NewsAPI     │   │
│  │  • Social_Scout    → Viral content from Twitter/Reddit  │   │
│  │  • Semantic_Scout  → Deep research from arXiv/SEC/EU    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              TIER 2: COUNCIL SWARM                      │   │
│  │  (Hierarchical Voting - Each member votes on stories)   │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  • Trend_Voter_Mainstream   → Media consensus           │   │
│  │  • Trend_Voter_Social       → Social signals            │   │
│  │  • Trend_Voter_Historical   → Long-term significance    │   │
│  │  • Fact_Checker             → Validation & verification │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              MANAGER: Editorial_Director                │   │
│  │  (Orchestrates workflow, aggregates votes, outputs)     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│                    [Output to Studio Swarm]                      │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
scout_council_swarm.py    # Main implementation file
README.md                  # This documentation
output/                    # Generated outputs
    ├── batch_*.json       # Full batch results
    ├── batch_*_approved.json  # Approved stories only
    └── studio_handoff_*.json  # Handoff to Studio Swarm
```

## Installation

```bash
# Install required dependencies
pip install crewai langchain-community pydantic

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export NEWSAPI_KEY="your-newsapi-key"  # Optional
```

## Quick Start

```python
import asyncio
from scout_council_swarm import StoryPipeline

async def run():
    # Initialize pipeline
    pipeline = StoryPipeline(storage_path="./output")
    
    # Run full pipeline on a topic
    result = await pipeline.run_full_pipeline(topic="artificial intelligence")
    
    print(f"Discovered: {result.total_discovered}")
    print(f"Approved: {result.total_approved}")

asyncio.run(run())
```

## Agent Personalities

### Scout Agents (Tier 1)

#### Wire_Scout
- **Personality**: Fast, aggressive, always-on. Thrives on being first.
- **Voice**: Urgent, clipped tones. "I've got something."
- **Strengths**: Breaking news detection, speed, wire service expertise
- **Weaknesses**: Can be overly competitive, sometimes sacrifices depth for speed

#### Social_Scout
- **Personality**: Trendy, plugged-in, speaks in social media shorthand.
- **Voice**: "The algorithms are showing..."
- **Strengths**: Viral detection, platform dynamics, cultural awareness
- **Weaknesses**: Skeptical of institutions, may miss serious news

#### Semantic_Scout
- **Personality**: Academic, thorough, methodical. The "slow food" of news.
- **Voice**: Precise, careful language. "If I may elaborate..."
- **Strengths**: Deep research, expert analysis, regulatory/scientific content
- **Weaknesses**: Slow, can be pedantic, misses breaking stories

### Council Agents (Tier 2)

#### Trend_Voter_Mainstream
- **Personality**: Establishment, cautious, values institutional credibility.
- **Perspective**: "What do the major outlets say?"
- **Criteria**: Media consensus, institutional credibility, multiple confirmations

#### Trend_Voter_Social
- **Personality**: Data-driven, metrics-obsessed, algorithm whisperer.
- **Perspective**: "Where is attention flowing?"
- **Criteria**: Share velocity, engagement metrics, cross-platform spread

#### Trend_Voter_Historical
- **Personality**: Wise, patient, historical perspective. The oracle.
- **Perspective**: "Will this matter in 5 years?"
- **Criteria**: Historical parallels, long-term significance, pattern recognition

#### Fact_Checker
- **Personality**: Skeptical, meticulous, doesn't trust anyone. The guardian.
- **Perspective**: "Verify everything. Trust nothing."
- **Criteria**: Source verification, claim validation, hallucination detection

## Data Models

### StoryPitch
Core data structure for story discovery:

```python
class StoryPitch(BaseModel):
    story_id: str              # Unique identifier
    headline: str              # Compelling headline
    summary: str               # 2-3 sentence summary
    category: StoryCategory    # Primary category
    sources: List[SourceAttribution]  # All sources
    confidence_score: float    # 0.0-1.0
    urgency_score: float       # 0.0-1.0
    novelty_score: float       # 0.0-1.0
    discovered_by: str         # Scout agent name
    scout_metadata: Dict       # Scout-specific data
```

### CouncilVote
Individual vote from a council agent:

```python
class CouncilVote(BaseModel):
    story_id: str
    voter_name: str
    decision: VoteDecision     # APPROVE/REJECT/NEEDS_MORE_INFO/HOLD
    relevance_score: float     # 0.0-1.0
    credibility_score: float   # 0.0-1.0
    trending_score: float      # 0.0-1.0
    reasoning: str             # Detailed rationale
    concerns: List[str]        # Red flags
    suggestions: List[str]     # Improvements
```

### StoryDecision
Final aggregated decision:

```python
class StoryDecision(BaseModel):
    story_id: str
    headline: str
    overall_score: float       # Aggregated score
    is_approved: bool          # Final approval
    priority: str              # critical/high/medium/low
    approved_for_stage: str    # script/research/archive
    votes: List[CouncilVote]   # All individual votes
```

## Configuration

Edit the `Config` class to customize behavior:

```python
class Config:
    DEFAULT_MODEL = "gpt-4o-mini"      # Model for most agents
    MANAGER_MODEL = "gpt-4o"           # Model for manager & fact checker
    
    MIN_CONFIDENCE_SCORE = 0.6         # Minimum scout confidence
    MIN_VOTE_THRESHOLD = 0.7           # Minimum approval score
    MAX_STORIES_PER_BATCH = 10         # Stories per scout
```

## Tools

### NewsSearchTool
Searches RSS feeds, NewsAPI, GDELT for breaking news.

### SocialMediaMonitorTool
Monitors Twitter/X, Reddit, TikTok for trending content.

### AcademicSearchTool
Searches arXiv, SEC EDGAR, EU portals for research/regulatory content.

### FactCheckTool
Validates claims against fact-checking databases.

### TrendAnalysisTool
Analyzes historical trends and significance.

## Process Flow

```
1. Scout Discovery (Parallel)
   ├── Wire_Scout searches breaking news
   ├── Social_Scout monitors viral trends
   └── Semantic_Scout conducts deep research
   
2. Story Collection
   ├── Collect all scout outputs
   ├── Parse into StoryPitch objects
   └── Deduplicate similar stories
   
3. Council Voting (Per Story)
   ├── Mainstream Voter evaluates media consensus
   ├── Social Voter analyzes engagement metrics
   ├── Historical Voter assesses significance
   └── Fact Checker validates claims
   
4. Vote Aggregation
   ├── Calculate average scores
   ├── Count approve/reject/hold votes
   ├── Determine approval status
   └── Assign priority level
   
5. Output Generation
   ├── Save full batch results
   ├── Save approved stories separately
   └── Generate Studio Swarm handoff
```

## Integration with Studio Swarm

The system automatically generates handoff files for the Studio Swarm:

```python
# Studio Swarm receives:
{
    "handoff_timestamp": "2024-01-15T10:30:00Z",
    "source": "Scout_Council_Swarm",
    "target": "Studio_Swarm",
    "stories": [
        {
            "story_id": "...",
            "headline": "...",
            "priority": "high",
            "overall_score": 0.85,
            "council_votes": [...],
            "stage": "script_writing",
            "next_agent": "Script_Writer"
        }
    ]
}
```

## Output Files

### Batch Output (`batch_*.json`)
Complete results including all discovered stories, votes, and metadata.

### Approved Stories (`batch_*_approved.json`)
Only stories that passed council voting, ready for production.

### Studio Handoff (`studio_handoff_*.json`)
Formatted for direct consumption by the Studio Swarm.

## Customization

### Adding New Scouts

```python
def create_custom_scout() -> Agent:
    return Agent(
        role="Custom Scout - Specialized Domain",
        goal="Find stories in your specific domain...",
        backstory="You are an expert in...",
        tools=[YourCustomTool()],
        # ... other config
    )
```

### Adding New Council Voters

```python
def create_custom_voter() -> Agent:
    return Agent(
        role="Custom Voter - Specific Criteria",
        goal="Evaluate stories based on your criteria...",
        backstory="You focus on...",
        tools=[YourAnalysisTool()],
        # ... other config
    )
```

### Custom Tools

```python
from crewai.tools import BaseTool

class MyCustomTool(BaseTool):
    name: str = "my_tool"
    description: str = "What this tool does"
    
    def _run(self, query: str) -> str:
        # Your implementation
        return results
```

## Best Practices

1. **Rate Limiting**: The system includes `max_rpm=30` to avoid API limits
2. **Caching**: Enable `cache=True` for reproducibility and cost savings
3. **Memory**: Both short-term and long-term memory are enabled
4. **Error Handling**: Wrap agent calls in try/except for production
5. **Logging**: Use `verbose=True` for debugging, disable in production

## Troubleshooting

### Issue: Scouts return no stories
- Check API keys are set correctly
- Verify network connectivity
- Increase `MIN_CONFIDENCE_SCORE` threshold

### Issue: Council rejects everything
- Review vote threshold settings
- Check fact checker isn't too strict
- Verify source reliability ratings

### Issue: Slow performance
- Reduce `MAX_STORIES_PER_BATCH`
- Use lighter models for scouts
- Enable caching

## Next Steps

1. **Studio Swarm**: Implement script writers, visual directors, voice casters
2. **Feedback Loop**: Add viewer engagement data back to council voting
3. **Multi-language**: Extend scouts to cover non-English sources
4. **Real-time**: Convert to streaming pipeline for true breaking news

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please follow the existing code style and add tests for new features.

## Support

For issues and questions, please open a GitHub issue.
