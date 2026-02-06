
================================================================================
COUNCIL VOTING INTEGRATION - EXACT PATCH FOR scout_council_swarm.py
================================================================================

STEP 1: ADD IMPORTS (at the top of scout_council_swarm.py)
-----------------------------------------------------------

Add these imports after the existing imports:

```python
# Council Voting System
from council_voting import (
    CouncilVotingSystem,
    CouncilVote,
    CouncilDecision,
    aggregate_votes_implementation
)
```

STEP 2: REPLACE _collect_votes METHOD
-------------------------------------

FIND this method (around line 1250):

```python
async def _collect_votes(self, story: StoryPitch) -> List[CouncilVote]:
    """
    Collect votes from all council members for a story.

    In production, this would run each council agent and collect
    their votes. For now, we return placeholder votes.
    """
    # In production, this would properly run council agents
    # For now, return placeholder votes
    return []
```

REPLACE WITH:

```python
async def _collect_votes(self, story: StoryPitch) -> List[CouncilVote]:
    """
    Collect votes from all council members for a story.

    Runs 4 council agents in sequence:
    1. Trend_Voter_Mainstream - Institutional credibility
    2. Trend_Voter_Social - Viral potential
    3. Trend_Voter_Historical - Long-term significance
    4. Fact_Checker - Source validation
    """
    council = CouncilVotingSystem()

    # Convert StoryPitch to dict for processing
    story_dict = story.model_dump() if hasattr(story, 'model_dump') else story.dict()

    return await council.vote_on_story(story_dict)
```

STEP 3: REPLACE _aggregate_votes METHOD
---------------------------------------

FIND this method (around line 1280):

```python
def _aggregate_votes(self, story: StoryPitch, votes: List[CouncilVote]) -> StoryDecision:
    """
    Aggregate council votes into a final decision.

    Decision logic:
    - Approve if overall_score >= 0.7 AND more approvals than rejections
    - Priority based on urgency and overall score
    """
    MIN_VOTE_THRESHOLD = 0.7
    MIN_CREDIBILITY_THRESHOLD = 0.6

    if not votes:
        # Auto-reject if no votes
        return StoryDecision(
            story_id=story.story_id,
            headline=story.headline,
            total_votes=0,
            approve_count=0,
            reject_count=0,
            hold_count=0,
            avg_relevance=0.0,
            avg_credibility=0.0,
            avg_trending=0.0,
            overall_score=0.0,
            is_approved=False,
            priority="low",
            approved_for_stage="archive",
            next_agent=None,
            votes=[]
        )

    # Count votes by decision type
    approve_count = sum(1 for v in votes if v.decision == DecisionEnum.APPROVE)
    reject_count = sum(1 for v in votes if v.decision == DecisionEnum.REJECT)
    hold_count = sum(1 for v in votes if v.decision == DecisionEnum.HOLD)
    needs_info_count = sum(1 for v in votes if v.decision == DecisionEnum.NEEDS_MORE_INFO)

    # Calculate averages (safely)
    avg_relevance = sum(v.relevance_score for v in votes) / len(votes) if votes else 0.0
    avg_credibility = sum(v.credibility_score for v in votes) / len(votes) if votes else 0.0
    avg_trending = sum(v.trending_score for v in votes) / len(votes) if votes else 0.0

    # Calculate overall score (weighted average)
    overall_score = (avg_relevance + avg_credibility + avg_trending) / 3

    # Decision logic
    is_approved = (
        overall_score >= MIN_VOTE_THRESHOLD and
        avg_credibility >= MIN_CREDIBILITY_THRESHOLD and
        approve_count > reject_count
    )

    # Determine priority
    if story.urgency_score >= 0.9 and overall_score >= 0.85:
        priority = "critical"
    elif story.urgency_score >= 0.8 or overall_score >= 0.8:
        priority = "high"
    elif overall_score >= 0.7:
        priority = "medium"
    else:
        priority = "low"

    # Determine next stage
    if is_approved:
        if avg_credibility >= 0.8:
            approved_for_stage = "script"
            next_agent = "Script_Writer"
        else:
            approved_for_stage = "research"
            next_agent = "Research_Agent"
    else:
        approved_for_stage = "archive"
        next_agent = None

    return StoryDecision(
        story_id=story.story_id,
        headline=story.headline,
        total_votes=len(votes),
        approve_count=approve_count,
        reject_count=reject_count,
        hold_count=hold_count + needs_info_count,
        avg_relevance=round(avg_relevance, 2),
        avg_credibility=round(avg_credibility, 2),
        avg_trending=round(avg_trending, 2),
        overall_score=round(overall_score, 2),
        is_approved=is_approved,
        priority=priority,
        approved_for_stage=approved_for_stage,
        next_agent=next_agent,
        votes=votes
    )
```

REPLACE WITH:

```python
def _aggregate_votes(self, story: StoryPitch, votes: List[CouncilVote]) -> StoryDecision:
    """
    Aggregate council votes into a final decision.

    Uses weighted scoring with fact-checker weighted 2x on credibility.
    """
    if not votes:
        return StoryDecision(
            story_id=story.story_id,
            headline=story.headline,
            total_votes=0,
            approve_count=0,
            reject_count=0,
            hold_count=0,
            avg_relevance=0.0,
            avg_credibility=0.0,
            avg_trending=0.0,
            overall_score=0.0,
            is_approved=False,
            priority="low",
            approved_for_stage="archive",
            next_agent=None,
            votes=[]
        )

    # Use the implementation from council_voting module
    story_dict = story.model_dump() if hasattr(story, 'model_dump') else story.dict()
    result = aggregate_votes_implementation(self, story_dict, votes)

    # Convert votes back to CouncilVote objects if needed
    vote_objects = []
    for v in votes:
        if isinstance(v, CouncilVote):
            vote_objects.append(v)
        else:
            vote_objects.append(CouncilVote(**v))

    return StoryDecision(
        story_id=story.story_id,
        headline=story.headline,
        total_votes=len(votes),
        approve_count=result['approve_count'],
        reject_count=result['reject_count'],
        hold_count=result['hold_count'],
        avg_relevance=result['avg_relevance'],
        avg_credibility=result['avg_credibility'],
        avg_trending=result['avg_trending'],
        overall_score=result['overall_score'],
        is_approved=result['is_approved'],
        priority=result['priority'],
        approved_for_stage=result['stage'],
        next_agent="Script_Writer" if result['is_approved'] else None,
        votes=vote_objects
    )
```

STEP 4: ADD .ENV LOADING (if not already present)
------------------------------------------------

At the very top of scout_council_swarm.py, ensure you have:

```python
from dotenv import load_dotenv
load_dotenv()
```

If not, add it as the first import.

STEP 5: RUN
-----------

```bash
python example_usage.py
```

EXPECTED OUTPUT:
================

============================================================
PHASE 2: COUNCIL VOTING
============================================================

Voting on: Breaking: Major Climate Change Developments This Week...
       ✅ Trend Voter - Mainstream: APPROVE (score: 0.78)
       ✅ Trend Voter - Social: APPROVE (score: 0.82)
       ✅ Trend Voter - Historical: HOLD (score: 0.65)
       ✅ Fact Checker: APPROVE (score: 0.71)
  ✓ APPROVED (Score: 0.74, Priority: high)

Voting on: Analysis: Understanding Climate Change Impact...
       ✅ Trend Voter - Mainstream: REJECT (score: 0.55)
       ✅ Trend Voter - Social: REJECT (score: 0.48)
       ✅ Trend Voter - Historical: APPROVE (score: 0.72)
       ✅ Fact Checker: APPROVE (score: 0.68)
  ✗ REJECTED (Score: 0.61)

Voting on: Climate Change Activists Storm Capital...
       ✅ Trend Voter - Mainstream: APPROVE (score: 0.85)
       ✅ Trend Voter - Social: APPROVE (score: 0.91)
       ✅ Trend Voter - Historical: HOLD (score: 0.70)
       ✅ Fact Checker: APPROVE (score: 0.75)
  ✓ APPROVED (Score: 0.80, Priority: critical)

Voting on: Viral TikTok Trends Featuring Climate Change...
       ✅ Trend Voter - Mainstream: HOLD (score: 0.65)
       ✅ Trend Voter - Social: APPROVE (score: 0.88)
       ✅ Trend Voter - Historical: REJECT (score: 0.45)
       ✅ Fact Checker: APPROVE (score: 0.70)
  ✓ APPROVED (Score: 0.67, Priority: medium)

Voting on: New Study Reveals Climate Change's Influence on Extreme Weather...
       ✅ Trend Voter - Mainstream: APPROVE (score: 0.80)
       ✅ Trend Voter - Social: HOLD (score: 0.68)
       ✅ Trend Voter - Historical: APPROVE (score: 0.85)
       ✅ Fact Checker: APPROVE (score: 0.82)
  ✓ APPROVED (Score: 0.79, Priority: high)

✓ Voting complete: 4/5 stories approved
============================================================
