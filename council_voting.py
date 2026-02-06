
"""
================================================================================
COUNCIL VOTING SYSTEM - FULL IMPLEMENTATION
================================================================================
Implements the Editorial Council that votes on discovered stories.

Council Members:
1. Trend_Voter_Mainstream - Institutional credibility perspective
2. Trend_Voter_Social - Viral potential and engagement analysis  
3. Trend_Voter_Historical - Long-term significance and historical context
4. Fact_Checker - Source validation and accuracy verification

Manager:
5. Editorial_Director - Aggregates votes and makes final decisions
================================================================================
"""

import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

from crewai import Agent, Task, Crew, Process

# ==============================================================================
# COUNCIL MODELS
# ==============================================================================

class CouncilDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    NEEDS_MORE_INFO = "NEEDS_MORE_INFO"
    HOLD = "HOLD"

class CouncilVote(BaseModel):
    """Individual vote from a council member"""
    model_config = ConfigDict(extra="forbid")

    council_member: str = Field(description="Name of the voting agent")
    decision: CouncilDecision = Field(description="Vote decision")

    # Scores (0.0 - 1.0)
    relevance_score: float = Field(ge=0.0, le=1.0, description="How relevant to current audience")
    credibility_score: float = Field(ge=0.0, le=1.0, description="Trustworthiness of sources")
    trending_score: float = Field(ge=0.0, le=1.0, description="Viral/engagement potential")

    # Reasoning
    reasoning: str = Field(description="Why this vote was cast")
    key_strengths: List[str] = Field(default_factory=list)
    key_concerns: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

    # Comparison
    similar_stories: List[str] = Field(default_factory=list, description="Similar recent coverage")
    competitive_advantage: Optional[str] = Field(default=None, description="Why this story wins")

    voted_at: datetime = Field(default_factory=datetime.utcnow)

# ==============================================================================
# COUNCIL AGENTS
# ==============================================================================

def create_trend_voter_mainstream() -> Agent:
    """
    Trend_Voter_Mainstream: Institutional credibility perspective.
    Evaluates: Source reputation, exclusivity, news value, editorial standards.
    """
    return Agent(
        role="Trend Voter - Mainstream Media Perspective",
        goal="""Evaluate stories through the lens of institutional journalism.

        Ask:
        - Are the sources reputable? (Tier 1: Reuters, AP, BBC, NYT, WSJ)
        - Is this exclusive or commodity news?
        - Would CNN/BBC/Reuters cover this?
        - What's the news value? (timeliness, proximity, prominence, impact)
        - Are we too late? (has it peaked?)

        SCORING:
        - relevance_score: How relevant to general news audience (0-1)
        - credibility_score: Source reliability and verification (0-1)
        - trending_score: How much traction in mainstream outlets (0-1)

        Be conservative but not gatekeeping. Quality over speed.""",

        backstory="""You are a senior assignment editor at Reuters with 20 years 
        experience. You've seen every type of story come across your desk. You 
        know that credibility is everything - one bad source can destroy a 
        newsroom's reputation forever.

        You evaluate based on:
        - SOURCE HIERARCHY: Reuters/AP > major national papers > reputable 
          outlets > emerging sources > unknown blogs
        - EXCLUSIVITY: First to report = gold, 47th outlet = skip
        - NEWS VALUES: Impact, timeliness, prominence, proximity, conflict, 
          human interest, novelty

        You are skeptical of:
        - Single-source stories
        - Anonymous sources without corroboration
        - Stories that seem too good to be true
        - Trends driven by bot activity

        You appreciate:
        - Multi-source confirmation
        - Named experts with credentials
        - Data-driven reporting
        - On-the-ground reporting vs aggregation""",

        verbose=True,
        allow_delegation=False,
        max_iter=3,
        llm="gpt-4o-mini",
    )

def create_trend_voter_social() -> Agent:
    """
    Trend_Voter_Social: Viral potential and social media analysis.
    Evaluates: Shareability, emotional resonance, platform fit, audience match.
    """
    return Agent(
        role="Trend Voter - Social Media & Viral Potential",
        goal="""Evaluate stories through the lens of shareability and engagement.

        Ask:
        - Will people share this? Why?
        - What emotion does it trigger? (anger, joy, fear, surprise, hope)
        - Is there a clear "villain" or "hero"?
        - Does it confirm or challenge audience biases?
        - Is it visually compelling? (B-roll potential)
        - Platform fit: YouTube retention? TikTok virality? Twitter discourse?

        SCORING:
        - relevance_score: Match to target demographic (0-1)
        - credibility_score: Authenticity (not astroturfed, real engagement) (0-1)
        - trending_score: Current momentum on social platforms (0-1)

        You are the voice of what WORKS on social media.""",

        backstory="""You are a social media strategist who grew accounts from 
        zero to 10M followers. You've studied thousands of viral hits and 
        understand the psychology of sharing.

        You know that people share content that:
        - SIGNALS IDENTITY: "This is who I am"
        - PROVIDES VALUE: "This helps my friends"
        - TRIGGERS EMOTION: Especially high-arousal emotions (awe, anger, 
          anxiety, excitement)
        - HAS NARRATIVE ARC: Clear setup, conflict, resolution
        - OFFERS NOVELTY: "I learned something new"

        You can spot:
        - Astroturfed campaigns (fake grassroots)
        - Bot-driven trends
        - Engagement bait
        - Stories that will burn out in 6 hours

        You look for:
        - First-mover advantage on emerging trends
        - Counter-intuitive angles
        - Human stories that scale
        - Visual gold (protest footage, before/after, satisfying moments)""",

        verbose=True,
        allow_delegation=False,
        max_iter=3,
        llm="gpt-4o-mini",
    )

def create_trend_voter_historical() -> Agent:
    """
    Trend_Voter_Historical: Long-term significance analysis.
    Evaluates: Historical context, future implications, evergreen value.
    """
    return Agent(
        role="Trend Voter - Historical Significance",
        goal="""Evaluate stories through the lens of history and future impact.

        Ask:
        - Will this matter in 5 years? 10 years?
        - Is this a turning point or incremental news?
        - What historical parallels exist?
        - Does this fit a larger trend or is it an outlier?
        - Will we look back at this as "the moment when..."?

        SCORING:
        - relevance_score: Long-term importance (0-1)
        - credibility_score: How well it fits historical patterns (0-1)
        - trending_score: Is this a paradigm shift beginning? (0-1)

        You are the long-term thinker.""",

        backstory="""You are a historian and futurist who has advised governments 
        and corporations on long-term trends. You have a mental database of 
        historical patterns and can spot inflection points before others.

        You evaluate based on:
        - INFLECTION POINTS: Is this a before/after moment?
        - PATTERN RECOGNITION: Does this rhyme with history? (1929, 1968, 1989, 
          2008, 2020)
        - SECOND-ORDER EFFECTS: What happens next? Then what?
        - SIGNAL VS NOISE: Is this a blip or a trend?
        - ARCHIVAL VALUE: Will researchers study this in 50 years?

        You are skeptical of:
        - "This changes everything" hype
        - Short-term volatility
        - Novelty without substance

        You value:
        - Structural shifts over cyclical news
        - Stories that explain "why the world is the way it is"
        - Early warnings (canaries in coal mines)
        - Counter-narratives that age well""",

        verbose=True,
        allow_delegation=False,
        max_iter=3,
        llm="gpt-4o-mini",
    )

def create_fact_checker() -> Agent:
    """
    Fact_Checker: Source validation and accuracy verification.
    Stronger model (gpt-4o) for validation tasks.
    """
    return Agent(
        role="Fact Checker - Accuracy & Verification",
        goal="""Validate every claim and source in the story.

        Check:
        - Can primary sources be verified?
        - Are quotes accurate and in context?
        - Are statistics from reliable sources?
        - Are there red flags for misinformation?
        - Is there contradictory information?
        - Are source attributions correct?

        SCORING:
        - relevance_score: Importance of accuracy for this topic (0-1)
        - credibility_score: Confidence in source reliability (0-1)
        - trending_score: Not applicable (use 0.5 or N/A focus)

        FLAG: Hallucinations, false attribution, context manipulation""",

        backstory="""You are a fact-checker at a major news organization with 
        a reputation for bulletproof accuracy. You have caught major 
        misinformation campaigns and prevented libel suits.

        Your verification process:
        1. SOURCE CHECK: Who said this? Can we verify they exist? Are they 
           in a position to know?
        2. QUOTE VERIFICATION: Are quotes accurate? In context? 
        3. DATA VALIDATION: Do numbers add up? From reputable sources?
        4. CROSS-REFERENCE: Do other sources confirm?
        5. RED FLAG CHECK: Is this too perfect? Does it fit a narrative 
           too well? (Possible fabrication)

        RED FLAGS you watch for:
        - Anonymous sources making explosive claims
        - Statistics without clear methodology
        - Quotes that sound "too good"
        - Images/video without provenance
        - Stories that confirm biases too perfectly
        - Urgency pressure ("must publish NOW")

        You never approve stories with unverified claims.""",

        verbose=True,
        allow_delegation=False,
        max_iter=3,
        llm="gpt-4o",  # Stronger model for fact checking
    )

def create_editorial_director() -> Agent:
    """
    Editorial_Director: Manager agent for final aggregation.
    Not used for individual voting, but for final decisions.
    """
    return Agent(
        role="Editorial Director - Final Authority",
        goal="""Make final publication decisions based on council votes.

        Consider:
        - Aggregate scores from all voters
        - Weight fact-checker heavily on credibility
        - Consider audience mix (mainstream + social + historical)
        - Resource allocation (is this worth production effort?)

        Output final decision with clear reasoning.""",

        backstory="""You are the Editor-in-Chief. The buck stops with you. 
        You balance journalistic integrity with business realities. You 
        understand that not every good story gets made, and not every 
        made story is perfect.

        Your decision framework:
        - APPROVE: High scores + clean fact-check + strategic fit
        - HOLD: Good story, wrong timing or needs more reporting
        - NEEDS_MORE_INFO: Promising but unverified claims
        - REJECT: Low scores, credibility issues, or resource mismatch""",

        verbose=True,
        allow_delegation=False,
        max_iter=2,
        llm="gpt-4o",
    )

# ==============================================================================
# VOTING IMPLEMENTATION
# ==============================================================================

class CouncilVotingSystem:
    """Implements the full council voting process."""

    def __init__(self):
        self.voters = [
            create_trend_voter_mainstream(),
            create_trend_voter_social(),
            create_trend_voter_historical(),
            create_fact_checker(),
        ]

    async def vote_on_story(self, story: dict) -> List[CouncilVote]:
        """
        Collect votes from all council members for a single story.

        Args:
            story: StoryPitch dictionary

        Returns:
            List of CouncilVote objects (one per voter)
        """
        votes = []

        print(f"    🗳️  Council voting on: {story['headline'][:50]}...")

        for voter in self.voters:
            try:
                vote = await self._get_vote_from_member(voter, story)
                votes.append(vote)
                print(f"       ✅ {voter.role.split(' - ')[0]}: {vote.decision} (score: {vote.relevance_score:.2f})")
            except Exception as e:
                print(f"       ⚠️  {voter.role.split(' - ')[0]} failed: {e}")
                # Add neutral vote on failure
                votes.append(CouncilVote(
                    council_member=voter.role,
                    decision=CouncilDecision.HOLD,
                    relevance_score=0.5,
                    credibility_score=0.5,
                    trending_score=0.5,
                    reasoning=f"Voting system error: {str(e)}"
                ))

        return votes

    async def _get_vote_from_member(self, voter: Agent, story: dict) -> CouncilVote:
        """Get a single vote from one council member."""

        task = Task(
            description=f"""
            Evaluate this story pitch and cast your vote:

            STORY HEADLINE: {story['headline']}
            SUMMARY: {story['summary']}
            CATEGORY: {story['category']}
            TAGS: {story.get('tags', [])}

            SCOUT SCORES:
            - Confidence: {story['confidence_score']}/1.0
            - Urgency: {story['urgency_score']}/1.0
            - Novelty: {story['novelty_score']}/1.0

            SOURCES:
            {json.dumps([s['name'] + ' (' + s.get('reliability', 'unknown') + ')' 
                        for s in story.get('sources', [])], indent=2)}

            KEY QUOTES: {story.get('key_quotes', [])}
            DISCOVERED BY: {story['discovered_by']}

            YOUR TASK:
            1. Evaluate based on your expertise and perspective
            2. Cast vote: APPROVE, REJECT, NEEDS_MORE_INFO, or HOLD
            3. Provide scores (0.0-1.0):
               - relevance_score: Relevance to audience
               - credibility_score: Trustworthiness of story
               - trending_score: Engagement/viral potential
            4. Explain your reasoning
            5. List key strengths and concerns
            6. Suggest improvements if applicable

            Output as valid JSON matching CouncilVote model.
            Be decisive - don't hedge with "maybe" votes.
            """,
            expected_output="""
            {
              "council_member": "Your name/role",
              "decision": "APPROVE|REJECT|NEEDS_MORE_INFO|HOLD",
              "relevance_score": 0.0-1.0,
              "credibility_score": 0.0-1.0,
              "trending_score": 0.0-1.0,
              "reasoning": "Detailed explanation of your vote",
              "key_strengths": ["strength 1", "strength 2"],
              "key_concerns": ["concern 1", "concern 2"],
              "suggestions": ["suggestion 1"],
              "similar_stories": ["recent similar coverage"],
              "competitive_advantage": "Why this story stands out"
            }
            """,
            agent=voter,
        )

        crew = Crew(agents=[voter], tasks=[task], verbose=False)
        result = await crew.kickoff_async()

        # Parse result
        raw_output = result.raw if hasattr(result, 'raw') else str(result)

        try:
            # Clean markdown
            clean = raw_output.replace("```json", "").replace("```", "").strip()

            # Extract JSON
            if '{' in clean and '}' in clean:
                start = clean.find('{')
                end = clean.rfind('}') + 1
                clean = clean[start:end]

            data = json.loads(clean)

            # Ensure required fields
            if 'council_member' not in data:
                data['council_member'] = voter.role
            if 'decision' not in data:
                data['decision'] = 'HOLD'

            return CouncilVote(**data)

        except Exception as e:
            print(f"       Parse error: {e}, using fallback")
            # Fallback vote
            return CouncilVote(
                council_member=voter.role,
                decision=CouncilDecision.HOLD,
                relevance_score=0.5,
                credibility_score=0.5,
                trending_score=0.5,
                reasoning=f"Could not parse vote: {str(e)[:100]}"
            )

# ==============================================================================
# INTEGRATION WITH STORY PIPELINE
# ==============================================================================

# This replaces the placeholder _collect_votes method in StoryPipeline

async def collect_votes_implementation(self, story: dict) -> List[CouncilVote]:
    """
    REAL IMPLEMENTATION: Collect votes from all council members.

    This replaces the placeholder that returned [].
    """
    council = CouncilVotingSystem()
    return await council.vote_on_story(story)

# Also need to fix _aggregate_votes to handle the real data properly

def aggregate_votes_implementation(
    self, 
    story: dict, 
    votes: List[CouncilVote]
) -> dict:
    """
    Aggregate council votes into a final decision.

    Logic:
    - Weight fact-checker higher on credibility
    - Require majority approval
    - Check minimum thresholds
    """
    if not votes:
        return {
            'is_approved': False,
            'overall_score': 0.0,
            'priority': 'low',
            'reason': 'No votes collected'
        }

    # Count votes
    approve_count = sum(1 for v in votes if v.decision == CouncilDecision.APPROVE)
    reject_count = sum(1 for v in votes if v.decision == CouncilDecision.REJECT)
    hold_count = sum(1 for v in votes if v.decision == CouncilDecision.HOLD)

    # Calculate weighted averages
    # Weight fact-checker 2x on credibility
    total_weight = len(votes) + 1  # +1 for fact-checker extra weight

    relevance_sum = sum(v.relevance_score for v in votes)
    trending_sum = sum(v.trending_score for v in votes)

    # Credibility: fact-checker counts double
    credibility_sum = 0
    for v in votes:
        weight = 1
        if 'Fact' in v.council_member or 'fact' in v.council_member.lower():
            weight = 2
        credibility_sum += v.credibility_score * weight

    avg_relevance = relevance_sum / len(votes)
    avg_credibility = credibility_sum / total_weight
    avg_trending = trending_sum / len(votes)

    # Overall score (weighted)
    overall_score = (avg_relevance * 0.3 + avg_credibility * 0.4 + avg_trending * 0.3)

    # Decision logic
    MIN_SCORE_THRESHOLD = 0.6
    MIN_CREDIBILITY = 0.5

    is_approved = (
        overall_score >= MIN_SCORE_THRESHOLD and
        avg_credibility >= MIN_CREDIBILITY and
        approve_count > reject_count and
        approve_count >= 2  # At least 2 approvals
    )

    # Determine priority
    if overall_score >= 0.85 and approve_count >= 3:
        priority = 'critical'
    elif overall_score >= 0.75:
        priority = 'high'
    elif overall_score >= 0.65:
        priority = 'medium'
    else:
        priority = 'low'

    # Determine stage
    if is_approved:
        if avg_credibility >= 0.8 and 'Fact' in [v.council_member for v in votes if v.decision == CouncilDecision.APPROVE][0] if [v for v in votes if v.decision == CouncilDecision.APPROVE] else False:
            stage = 'script'
        else:
            stage = 'research'
    else:
        stage = 'archive'

    return {
        'is_approved': is_approved,
        'overall_score': round(overall_score, 2),
        'avg_relevance': round(avg_relevance, 2),
        'avg_credibility': round(avg_credibility, 2),
        'avg_trending': round(avg_trending, 2),
        'approve_count': approve_count,
        'reject_count': reject_count,
        'hold_count': hold_count,
        'priority': priority,
        'stage': stage,
        'votes': [v.model_dump() for v in votes]
    }

# ==============================================================================
# PATCH FILE FOR INTEGRATION
# ==============================================================================

def generate_patch_instructions():
    """Instructions for integrating into existing codebase."""
    return """
INTEGRATION INSTRUCTIONS:
========================

1. SAVE THIS FILE as council_voting.py in your NewsScrape directory

2. IN scout_council_swarm.py, ADD at the top:
   from council_voting import (
       CouncilVotingSystem, 
       CouncilVote, 
       CouncilDecision,
       collect_votes_implementation,
       aggregate_votes_implementation
   )

3. REPLACE the _collect_votes method in StoryPipeline class:

   OLD:
   async def _collect_votes(self, story: StoryPitch) -> List[CouncilVote]:
       return []

   NEW:
   async def _collect_votes(self, story: StoryPitch) -> List[CouncilVote]:
       story_dict = story.model_dump() if hasattr(story, 'model_dump') else story.dict()
       return await collect_votes_implementation(self, story_dict)

4. REPLACE the _aggregate_votes method:

   OLD:
   def _aggregate_votes(self, story: StoryPitch, votes: List[CouncilVote]) -> StoryDecision:
       # ... existing placeholder logic ...

   NEW:
   def _aggregate_votes(self, story: StoryPitch, votes: List[CouncilVote]) -> StoryDecision:
       story_dict = story.model_dump() if hasattr(story, 'model_dump') else story.dict()
       result = aggregate_votes_implementation(self, story_dict, votes)

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
           votes=votes
       )

5. RUN your example_usage.py - stories should now get real votes!

EXPECTED OUTPUT:
===============

🎬 STUDIO SWARM: Processing 5 stories

Voting on: Breaking: Major Climate Change Developments This Week...
       ✅ Trend Voter - Mainstream: APPROVE (score: 0.75)
       ✅ Trend Voter - Social: APPROVE (score: 0.80)
       ✅ Trend Voter - Historical: HOLD (score: 0.60)
       ✅ Fact Checker: APPROVE (score: 0.70)
  ✓ APPROVED (Score: 0.72, Priority: high)

Voting on: Analysis: Understanding Climate Change Impact...
       ✅ Trend Voter - Mainstream: REJECT (score: 0.45)
       ✅ Trend Voter - Social: REJECT (score: 0.40)
       ✅ Trend Voter - Historical: APPROVE (score: 0.80)
       ✅ Fact Checker: APPROVE (score: 0.75)
  ✗ REJECTED (Score: 0.60)

✓ Voting complete: 3/5 stories approved
"""

print(generate_patch_instructions())