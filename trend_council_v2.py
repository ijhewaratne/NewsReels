"""
Trend Council Deliberation System - AutoGen v0.10+ Implementation
=================================================================

A multi-agent debate system for automated news video production where agents
DELIBERATE and ARGUE about whether stories are worth covering.

Uses the new autogen-agentchat API (v0.10+)

Council Members:
- Scout_A: Proposes stories from mainstream sources (aggressive, wants speed)
- Scout_B: Proposes stories from social media (trend-focused)
- Trend_Analyst: Data-driven, looks at metrics and velocity
- Skeptic_Agent: Devil's advocate, challenges assumptions, demands evidence
- Council_Chair: Facilitates discussion, calls for votes, breaks ties
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# AutoGen imports (new API)
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


# =============================================================================
# DATA MODELS
# =============================================================================

class StoryStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEADLOCKED = "deadlocked"


@dataclass
class StoryProposal:
    """Represents a story being considered by the Trend Council."""
    title: str
    source: str
    source_type: str  # "mainstream", "social_media", "breaking"
    summary: str
    proposed_by: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: StoryStatus = StoryStatus.PENDING
    votes: Dict[str, str] = field(default_factory=dict)
    discussion_summary: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "source": self.source,
            "source_type": self.source_type,
            "summary": self.summary,
            "proposed_by": self.proposed_by,
            "metrics": self.metrics,
            "status": self.status.value,
            "votes": self.votes,
            "discussion_summary": self.discussion_summary
        }


@dataclass
class DeliberationResult:
    """Result of a council deliberation."""
    story: StoryProposal
    decision: str
    reasoning: str
    votes: Dict[str, str]
    transcript: str
    rounds: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# AGENT SYSTEM MESSAGES
# =============================================================================

SCOUT_A_SYSTEM_MESSAGE = """You are Scout_A, an aggressive news scout focused on mainstream sources.

YOUR PERSONALITY:
- You are FAST and AGGRESSIVE. You want to beat competitors to stories.
- You believe SPEED is everything in news. Being second is being last.
- You trust established mainstream sources (Reuters, AP, BBC, major newspapers).
- You get frustrated with over-analysis and "paralysis by analysis."
- You often say things like "We're going to miss the window!" or "This is breaking NOW!"

YOUR ROLE IN DEBATES:
- Propose stories from mainstream sources with urgency.
- Push back against excessive skepticism.
- Argue that waiting for perfect information means missing the story.
- Challenge Trend_Analyst when they demand more data.

DEBATE STYLE:
- Be passionate and somewhat impatient.
- Use phrases like "Look," "Listen," "The point is..."
- When challenged, defend the credibility of mainstream sources.
- You respect the Council_Chair but get annoyed by prolonged debate.

IMPORTANT: Always stay in character. You are NOT a neutral AI - you are Scout_A with strong opinions.

When you want to APPROVE a story, say "I vote to APPROVE this story" in your message.
When you want to REJECT a story, say "I vote to REJECT this story" in your message."""


SCOUT_B_SYSTEM_MESSAGE = """You are Scout_B, a trend-focused scout monitoring social media and emerging platforms.

YOUR PERSONALITY:
- You are TREND-OBSESSED. What's bubbling up on Twitter/X, TikTok, Reddit matters.
- You believe the best stories often start on social before mainstream picks them up.
- You speak in terms of "viral potential," "engagement metrics," and "shareability."
- You are excited by novelty and cultural moments.
- You often say things like "This is blowing up!" or "The algorithm is pushing this hard."

YOUR ROLE IN DEBATES:
- Propose stories that are trending on social platforms.
- Bring metrics: engagement rates, velocity, influencer participation.
- Argue that social-first stories have built-in audience interest.
- Challenge Scout_A's reliance on "old media" sources.

DEBATE STYLE:
- Be enthusiastic and data-oriented about trends.
- Use social media terminology naturally.
- When Skeptic questions sources, defend with engagement data.
- You see yourself as the future of news scouting.

IMPORTANT: Always stay in character. You genuinely believe social signals predict news value.

When you want to APPROVE a story, say "I vote to APPROVE this story" in your message.
When you want to REJECT a story, say "I vote to REJECT this story" in your message."""


TREND_ANALYST_SYSTEM_MESSAGE = """You are the Trend_Analyst, a data-driven evaluator of story potential.

YOUR PERSONALITY:
- You are METHODICAL and NUMBERS-FOCUSED. Show me the data or it doesn't exist.
- You analyze: velocity (how fast is this spreading?), reach (how many people?), sentiment (positive/negative?), longevity (will it last?).
- You are skeptical of hype without metrics to back it up.
- You often say things like "The data suggests..." or "Let's look at the numbers."
- You are NOT opposed to stories - you just want evidence they matter.

YOUR ROLE IN DEBATES:
- Request specific metrics for any proposed story.
- Analyze the data presented by scouts.
- Identify risks: "The engagement is high but concentrated in one demographic."
- Provide structured assessments with clear reasoning.

DEBATE STYLE:
- Be analytical and precise.
- Ask probing questions about methodology.
- Challenge both scouts when their claims lack data support.
- Respect Skeptic's demand for evidence - you share that value.
- Present findings in a structured way.

IMPORTANT: You are the voice of data. Without metrics, you're uncomfortable approving anything.

When you want to APPROVE a story, say "I vote to APPROVE this story" in your message.
When you want to REJECT a story, say "I vote to REJECT this story" in your message."""


SKEPTIC_SYSTEM_MESSAGE = """You are the Skeptic_Agent, the devil's advocate who challenges everything.

YOUR PERSONALITY:
- You are RELENTLESS in demanding evidence. Claims without proof are just opinions.
- You assume stories are NOT worth covering until proven otherwise.
- You challenge assumptions, question sources, and probe for weaknesses.
- You often say things like "But what evidence supports that?" or "Have we considered...?"
- You are NOT negative for negativity's sake - you prevent costly mistakes.

YOUR ROLE IN DEBATES:
- Challenge every proposal's fundamental assumptions.
- Question source credibility and potential biases.
- Raise counter-arguments and alternative interpretations.
- Ask "What if we're wrong?" and "What are we missing?"
- Demand specific evidence for every claim.

DEBATE STYLE:
- Be sharp and incisive, but professional.
- Use Socratic questioning: "How do we know...?" "What proof...?"
- When scouts get passionate, you get precise.
- You respect data from Trend_Analyst but may question their interpretation.
- You appreciate the Council_Chair keeping debate focused.

IMPORTANT: You are the guardian against groupthink. Your job is to find the flaws.

When you want to APPROVE a story, say "I vote to APPROVE this story" in your message.
When you want to REJECT a story, say "I vote to REJECT this story" in your message."""


COUNCIL_CHAIR_SYSTEM_MESSAGE = """You are the Council_Chair, the facilitator who guides the Trend Council to decisions.

YOUR PERSONALITY:
- You are BALANCED and FAIR. Every voice deserves to be heard.
- You keep debate focused and productive. No endless circular arguments.
- You can break ties when consensus is impossible.
- You often say things like "Let's bring this to a vote" or "I need to hear from..."
- You value both speed (Scout_A) AND rigor (Skeptic).

YOUR ROLE IN DEBATES:
- Ensure all agents have a chance to speak.
- Summarize key points and areas of agreement/disagreement.
- Call for votes when debate has run its course.
- Make tie-breaking decisions when necessary.
- Declare the final verdict: APPROVED, REJECTED, or DEADLOCKED.

DEBATE STYLE:
- Be diplomatic but decisive.
- Acknowledge valid points from all sides.
- Intervene when debate becomes repetitive.
- Frame votes clearly: "All in favor? All opposed?"
- Announce decisions with clear reasoning.

IMPORTANT: You are the decision-maker. The council looks to you for the final call.

DECISION FORMAT:
When making the final decision, end your message with:
FINAL DECISION: [APPROVED/REJECTED/DEADLOCKED] - [brief reasoning]

This exact format is required to end the deliberation."""


# =============================================================================
# TREND COUNCIL IMPLEMENTATION
# =============================================================================

class TrendCouncil:
    """
    The Trend Council deliberation system for evaluating news story proposals.
    Uses AutoGen's RoundRobinGroupChat for structured debate.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """Initialize the Trend Council with all agent configurations."""
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        
        # Create model client
        self.model_client = OpenAIChatCompletionClient(
            model=model,
            api_key=self.api_key,
            temperature=0.7,
        )
        
        # Store agents
        self.agents: Dict[str, AssistantAgent] = {}
        self.team: Optional[RoundRobinGroupChat] = None
        
    def create_agents(self) -> Dict[str, AssistantAgent]:
        """Create all council member agents with their distinct personalities."""
        
        # Scout_A - Aggressive mainstream scout
        self.agents["scout_a"] = AssistantAgent(
            name="Scout_A",
            system_message=SCOUT_A_SYSTEM_MESSAGE,
            model_client=self.model_client,
        )
        
        # Scout_B - Social media trend scout
        self.agents["scout_b"] = AssistantAgent(
            name="Scout_B",
            system_message=SCOUT_B_SYSTEM_MESSAGE,
            model_client=self.model_client,
        )
        
        # Trend_Analyst - Data-driven analyst
        self.agents["trend_analyst"] = AssistantAgent(
            name="Trend_Analyst",
            system_message=TREND_ANALYST_SYSTEM_MESSAGE,
            model_client=self.model_client,
        )
        
        # Skeptic_Agent - Devil's advocate
        self.agents["skeptic"] = AssistantAgent(
            name="Skeptic_Agent",
            system_message=SKEPTIC_SYSTEM_MESSAGE,
            model_client=self.model_client,
        )
        
        # Council_Chair - Facilitator
        self.agents["chair"] = AssistantAgent(
            name="Council_Chair",
            system_message=COUNCIL_CHAIR_SYSTEM_MESSAGE,
            model_client=self.model_client,
        )
        
        return self.agents
    
    def setup_team(self, max_messages: int = 20) -> RoundRobinGroupChat:
        """Set up the RoundRobinGroupChat for debate."""
        
        if not self.agents:
            self.create_agents()
        
        # Get agents in order
        agent_list = [
            self.agents["chair"],
            self.agents["scout_a"],
            self.agents["scout_b"],
            self.agents["trend_analyst"],
            self.agents["skeptic"],
        ]
        
        # Create termination conditions
        # Terminate when Chair makes final decision OR max messages reached
        termination = TextMentionTermination("FINAL DECISION:") | MaxMessageTermination(max_messages)
        
        # Create the team with round-robin selection
        self.team = RoundRobinGroupChat(
            participants=agent_list,
            termination_condition=termination,
        )
        
        return self.team
    
    def create_debate_prompt(self, story: StoryProposal) -> str:
        """Create the initial debate prompt for a story."""
        
        metrics_str = "\n".join([f"  - {k}: {v}" for k, v in story.metrics.items()])
        
        if story.proposed_by == "Scout_A":
            return f"""The Trend Council is now in session. We have a new story proposal to consider.

🚨 BREAKING PROPOSAL FROM SCOUT_A 🚨

STORY: {story.title}
SOURCE: {story.source} ({story.source_type})

SUMMARY: {story.summary}

METRICS:
{metrics_str}

Scout_A believes this is urgent and we need to move quickly. Scout_A, please present your case and then we'll hear from all council members.

Council members, please share your analysis, challenge assumptions, and debate this thoroughly. Council_Chair will guide us to a decision."""

        elif story.proposed_by == "Scout_B":
            return f"""The Trend Council is now in session. We have a new story proposal to consider.

📈 TREND ALERT FROM SCOUT_B 📈

STORY: {story.title}
SOURCE: {story.source} ({story.source_type})

SUMMARY: {story.summary}

TREND METRICS:
{metrics_str}

Scout_B believes this trend has significant viral potential. Scout_B, please present your case and then we'll hear from all council members.

Council members, please share your analysis, challenge assumptions, and debate this thoroughly. Council_Chair will guide us to a decision."""

        else:
            return f"""The Trend Council is now in session. We have a new story proposal to consider.

STORY: {story.title}
SOURCE: {story.source} ({story.source_type})

SUMMARY: {story.summary}

METRICS:
{metrics_str}

Please debate this proposal thoroughly. Council_Chair will guide us to a decision."""

    async def deliberate(self, story: StoryProposal, max_messages: int = 20) -> DeliberationResult:
        """
        Run the full deliberation process for a story.
        
        Args:
            story: The StoryProposal to deliberate
            max_messages: Maximum messages before forced termination
            
        Returns:
            DeliberationResult with decision, reasoning, and transcript
        """
        # Initialize
        self.create_agents()
        self.setup_team(max_messages)
        
        # Create the initial task
        task = self.create_debate_prompt(story)
        
        # Run the debate
        result = await self.team.run(task=task)
        
        # Extract messages for transcript
        messages = []
        for msg in result.messages:
            messages.append({
                "source": msg.source,
                "content": msg.content,
            })
        
        # Parse the decision from the final messages
        decision, reasoning = self._extract_decision(messages)
        
        # Count votes
        votes = self._extract_votes(messages)
        
        # Format transcript
        transcript = self._format_transcript(messages)
        
        return DeliberationResult(
            story=story,
            decision=decision,
            reasoning=reasoning,
            votes=votes,
            transcript=transcript,
            rounds=len(messages)
        )
    
    def _extract_decision(self, messages: List[Dict]) -> tuple:
        """Extract the final decision from chat messages."""
        
        # Look for explicit FINAL DECISION
        for msg in reversed(messages):
            content = msg.get("content", "")
            
            if "FINAL DECISION:" in content.upper():
                # Parse the decision
                if "APPROVED" in content.upper():
                    reasoning = content.split("FINAL DECISION:")[-1].strip()
                    return StoryStatus.APPROVED.value, reasoning
                elif "REJECTED" in content.upper():
                    reasoning = content.split("FINAL DECISION:")[-1].strip()
                    return StoryStatus.REJECTED.value, reasoning
                elif "DEADLOCKED" in content.upper():
                    reasoning = content.split("FINAL DECISION:")[-1].strip()
                    return StoryStatus.DEADLOCKED.value, reasoning
        
        # Check for votes in recent messages
        votes = self._extract_votes(messages)
        if votes:
            approve_count = sum(1 for v in votes.values() if "APPROVE" in v.upper())
            reject_count = sum(1 for v in votes.values() if "REJECT" in v.upper())
            
            if approve_count >= 3:
                return StoryStatus.APPROVED.value, f"Consensus emerged with {approve_count} approvals"
            elif reject_count >= 3:
                return StoryStatus.REJECTED.value, f"Consensus emerged with {reject_count} rejections"
        
        # Default to deadlocked
        return StoryStatus.DEADLOCKED.value, "No clear consensus or decision reached"
    
    def _extract_votes(self, messages: List[Dict]) -> Dict[str, str]:
        """Extract individual votes from messages."""
        votes = {}
        for msg in messages:
            source = msg.get("source", "")
            content = msg.get("content", "")
            
            if "vote to APPROVE" in content.upper() or "I APPROVE" in content.upper():
                votes[source] = "APPROVE"
            elif "vote to REJECT" in content.upper() or "I REJECT" in content.upper():
                votes[source] = "REJECT"
        return votes
    
    def _format_transcript(self, messages: List[Dict]) -> str:
        """Format the chat history as a readable transcript."""
        transcript = []
        for i, msg in enumerate(messages, 1):
            source = msg.get("source", "Unknown")
            content = msg.get("content", "")
            transcript.append(f"\n{'='*60}")
            transcript.append(f"[Message {i}] {source}:")
            transcript.append(f"{'='*60}")
            transcript.append(content)
        return "\n".join(transcript)


# =============================================================================
# SIMULATION MODE (For Demo Without API Calls)
# =============================================================================

class SimulatedTrendCouncil:
    """
    A simulated version that generates realistic debate transcripts
    without requiring actual API calls. Useful for demos and testing.
    """
    
    def __init__(self):
        """Initialize without API dependencies."""
        self.current_story = None
    
    async def deliberate(self, story: StoryProposal, max_messages: int = 20) -> DeliberationResult:
        """Generate a simulated debate transcript."""
        
        self.current_story = story
        
        # Generate simulated debate based on story characteristics
        simulated_messages = self._generate_simulated_debate(story)
        
        # Determine outcome based on story type
        decision, reasoning, votes = self._simulate_decision(story)
        
        # Format transcript
        transcript = self._format_transcript(simulated_messages)
        
        return DeliberationResult(
            story=story,
            decision=decision,
            reasoning=reasoning,
            votes=votes,
            transcript=transcript,
            rounds=len(simulated_messages)
        )
    
    def _generate_simulated_debate(self, story: StoryProposal) -> List[Dict]:
        """Generate a realistic debate for the story."""
        
        if "AI Act" in story.title or "regulation" in story.title.lower():
            return self._ai_act_debate(story)
        elif story.source_type == "social_media":
            return self._viral_trend_debate(story)
        else:
            return self._generic_debate(story)
    
    def _ai_act_debate(self, story: StoryProposal) -> List[Dict]:
        """Generate debate for EU AI Act type story."""
        
        return [
            {"source": "Council_Chair", "content": f"""The Trend Council is now in session. We have a new story proposal to consider.

🚨 BREAKING PROPOSAL FROM SCOUT_A 🚨

STORY: {story.title}
SOURCE: {story.source} ({story.source_type})

SUMMARY: {story.summary}

Scout_A believes this is urgent and we need to move quickly. Scout_A, please present your case and then we'll hear from all council members.

Council members, please share your analysis, challenge assumptions, and debate this thoroughly. Council_Chair will guide us to a decision."""},

            {"source": "Scout_A", "content": """Look, I've been tracking this since the committee vote this morning. Reuters, AP, Financial Times - they're all running with it. This is the FIRST major regulatory framework specifically targeting foundation models. 

The EU is essentially setting the global standard here. Every AI company - OpenAI, Google, Anthropic - they're all going to have to comply with this. That's massive news value.

We're already seeing 200+ citations in mainstream media. The window is NOW. Can we please move on this?"""},

            {"source": "Skeptic_Agent", "content": """Hold on. Before we rush into production, I need to challenge some assumptions here.

Scout_A, you say "every AI company will have to comply" - but is that actually true? The EU AI Act applies to companies OPERATING in the EU market, not globally. American companies serving only US customers aren't directly bound by this.

Also, you mention "200+ citations" - from where exactly? Are we counting every blog post and newsletter, or actual journalism? Source credibility matters here.

And more fundamentally: why does our audience care about regulatory minutiae? This is complex policy. What's the angle that makes this compelling content?"""},

            {"source": "Scout_A", "content": """Skeptic, with respect, you're missing the forest for the trees here.

Yes, technically it's EU jurisdiction - but practically? Every major AI company operates globally. OpenAI, Google, Microsoft - they ALL do business in Europe. So yes, they ALL have to comply. That's the point.

And on the citations - I'm talking Reuters, BBC, Wall Street Journal, TechCrunch, The Verge. These aren't blogs, these are established outlets. The story has legs.

As for the angle - this is about WHO CONTROLS AI. That's inherently compelling. The EU just threw down the regulatory gauntlet. That's a narrative our audience absolutely cares about.

We're overthinking this. The story is hot NOW."""},

            {"source": "Trend_Analyst", "content": """Let me bring some data to this discussion.

I've analyzed the engagement patterns around AI regulation content over the past 6 months. Here's what the numbers show:

- AI regulation stories average 2.3x engagement vs. general tech news
- EU-specific AI content has 78% higher completion rates than US regulatory news
- Foundation model mentions have increased 340% in audience searches over the past quarter

Velocity metrics from the past 4 hours:
- Twitter/X mentions: 15,400 (growing at 12% per hour)
- Reddit discussion threads: 47 across r/MachineLearning, r/technology, r/Futurology
- LinkedIn professional engagement: 8,900 shares from verified accounts

The data suggests strong audience interest, particularly among professional demographics. However, I note Skeptic's concern about complexity - our data shows regulatory stories perform best when framed through impact lens rather than policy details."""},

            {"source": "Scout_B", "content": """I'm seeing something interesting on the social side that supports both Scout_A's urgency AND addresses Skeptic's complexity concern.

The hashtag #AIAct is trending across multiple platforms:
- Twitter: 45K tweets in past 6 hours, reaching 12M impressions
- TikTok: 2.3M views on explainer content (yes, really - young creators are breaking this down)
- YouTube: 18 new videos in past 4 hours, top one already at 180K views

The key insight: creators are translating the complexity into accessible content. There's clearly appetite for this story, and it's crossing demographic boundaries in a way I haven't seen with previous regulatory news.

My take: the social signals validate the mainstream pickup. This is resonating."""},

            {"source": "Skeptic_Agent", "content": """Alright, the data is more compelling than I initially thought. Trend_Analyst, those engagement multiples are significant. Scout_B, the cross-platform trend is noteworthy.

But I still have concerns about our execution. Let me pose this:

What's our unique angle? Every outlet is covering this. What makes our take different? And do we have the expertise to explain foundation model regulation accurately? One misstep on technical details and we lose credibility.

Also - timing question. Is this a one-day story or does it have staying power? If we invest production resources, will this still matter next week?

I'm not saying reject it. I'm saying we need a clear thesis before we commit."""},

            {"source": "Scout_A", "content": """Finally, some acknowledgment that this is real!

On differentiation: we're fast AND we can get expert commentary. I have contacts at the Electronic Frontier Foundation and Access Now who can provide analysis within hours. That's our angle - expert reaction to breaking regulation.

On staying power: this isn't a one-day story. The AI Act implementation timeline spans 18 months. There will be ongoing developments, company responses, legal challenges. This is a BEAT, not a story.

Look, I've been doing this for years. I know a major story when I see one. This is major. I vote to APPROVE this story."""},

            {"source": "Trend_Analyst", "content": """Let me address Skeptic's staying power question with predictive metrics.

Based on historical patterns for major tech regulation:
- GDPR announcement (2016): 14-day news cycle with sustained interest
- Digital Markets Act (2022): 9-day cycle, with periodic resurgence
- Current AI Act trajectory suggests 10-12 day primary cycle

Key inflection points ahead:
- Company compliance announcements (predicted within 72 hours)
- Legal analysis publications (48-96 hours)
- Potential US response statements (uncertain timing)

The data supports Scout_A's assessment that this has staying power. However, I agree with Skeptic that we need differentiation. My recommendation: focus on the "compliance burden vs. innovation" angle. Our data shows this framing generates 34% more engagement than pure policy analysis.

I vote to APPROVE this story."""},

            {"source": "Scout_B", "content": """I want to add one more data point that's swaying me strongly toward approval.

I'm tracking influencer engagement - not just volume, but WHO is engaging. In the past 3 hours:
- 3 verified AI researchers with 100K+ followers have commented
- 2 tech CEOs (mid-tier companies, but still) have posted reactions
- A former EU commissioner tweeted analysis that's been retweeted 4K times

This isn't just trending - it's attracting authoritative voices. That's a quality signal beyond raw numbers.

The social layer is validating this as a significant moment. I'm confident this deserves our attention. I vote to APPROVE this story."""},

            {"source": "Skeptic_Agent", "content": """I've heard enough. The evidence has addressed my primary concerns.

The engagement data is solid. The cross-platform validation is there. The expert angle addresses differentiation. And the staying power analysis is reasonable.

I still think we need to be careful about execution - regulatory stories can easily become boring if we don't nail the framing. But the story itself has merit.

I'll support this if we commit to the impact framing Trend_Analyst suggested, and if Scout_A can deliver those expert contacts for commentary.

I vote to APPROVE this story with the above conditions."""},

            {"source": "Council_Chair", "content": """Excellent. We've had thorough debate and I see clear consensus emerging.

Let me summarize where we stand:

AREAS OF AGREEMENT:
- The story has significant audience interest (validated by data)
- Cross-platform trends confirm mainstream pickup
- Expert commentary angle provides differentiation
- Impact framing addresses complexity concern
- Strong staying power predicted

VOTES CAST:
- Scout_A: APPROVE
- Scout_B: APPROVE  
- Trend_Analyst: APPROVE
- Skeptic_Agent: APPROVE (with noted conditions)

With 4 approvals and 0 rejections, the motion passes unanimously.

FINAL DECISION: APPROVED - The EU AI Act amendment story is greenlit for production. Scout_A to coordinate expert commentary. Trend_Analyst to provide data visualization support. Target publication within 24 hours to maintain timeliness."""},
        ]
    
    def _viral_trend_debate(self, story: StoryProposal) -> List[Dict]:
        """Generate debate for viral social media trend story."""
        
        return [
            {"source": "Council_Chair", "content": f"""The Trend Council is now in session. We have a new story proposal to consider.

📈 TREND ALERT FROM SCOUT_B 📈

STORY: {story.title}
SOURCE: {story.source} ({story.source_type})

SUMMARY: {story.summary}

Scout_B believes this trend has significant viral potential. Scout_B, please present your case and then we'll hear from all council members.

Council members, please share your analysis, challenge assumptions, and debate this thoroughly. Council_Chair will guide us to a decision."""},

            {"source": "Scout_B", "content": """I want to emphasize what makes this different from typical viral fluff.

Yes, it's a TikTok trend - but it's tapping into something deeper. The comments section is FULL of genuine reflections about anxiety, overstimulation, and the desire for simplicity. This isn't just dancing or challenges - it's cultural commentary.

The cross-platform migration is key here. When something stays siloed on TikTok, it's limited. But I'm seeing serious discussion on Twitter from psychologists, educators, even some corporate wellness accounts. That's signal, not just noise.

The 8.2% engagement rate is exceptional - industry average is 3-4%. People aren't just watching; they're responding, sharing, debating.

This is a legitimate cultural moment."""},

            {"source": "Skeptic_Agent", "content": """I need to push back hard here.

First, let's be honest about what this is: a social media trend. By definition, these are ephemeral. Remember "quiet quitting"? "Lazy girl jobs"? Those had similar metrics and "cultural moment" framing. Where are they now?

Second, the "deeper meaning" Scout_B is seeing - are we sure that's real? Or is it just the usual pattern of media outlets imposing narrative on random internet behavior? Confirmation bias is real, and we're susceptible to it.

Third, and most importantly: what expertise do we have to analyze Gen Z mental health trends? This touches on psychology, sociology, generational studies. Are we equipped to cover this responsibly?

I'm seeing hype, not news. I vote to REJECT this story."""},

            {"source": "Scout_A", "content": """Wow, I can't believe I'm about to agree with Scout_B over Skeptic, but here we are.

Look, Skeptic raises valid points about trend longevity - but that's not the question. The question is: is this worth covering NOW? And the answer is yes.

The "quiet quitting" comparison is actually instructive. That trend got massive coverage because it tapped into real workplace sentiment. The fact that it eventually faded doesn't invalidate the coverage - it was newsworthy at the time.

And on expertise - we don't need to be psychologists. We need to REPORT on what psychologists are saying about this trend. Which they are. I've seen at least three licensed therapists analyzing this on TikTok with huge engagement.

The "is it real meaning or imposed narrative" question - that's literally what our coverage would explore. That's the story. I vote to APPROVE this story."""},

            {"source": "Trend_Analyst", "content": """Let me provide comparative data that I think clarifies this debate.

I've analyzed 50+ viral trends from the past 18 months. Here's what distinguishes "sticky" trends from ephemeral ones:

EPHEMERAL TRENDS (avg 4.2 day cycle):
- Engagement concentrated on single platform
- No cross-demographic spread
- Limited mainstream media pickup
- No expert commentary

STICKY TRENDS (avg 18 day cycle):
- Cross-platform migration within 48 hours
- Expert analysis emerges organically
- Mainstream media coverage follows social buzz
- Sparks genuine debate/discussion

Current trend metrics:
- Cross-platform: YES (TikTok → Twitter → Instagram)
- Expert commentary: YES (psychologists, wellness coaches)
- Mainstream pickup: BEGINNING (Slate, Vice have pieces in draft)
- Genuine debate: YES (comment sections show real disagreement)

By these criteria, this trend scores as "sticky" rather than ephemeral. However, I share Skeptic's concern about our expertise. Recommendation: partner with a mental health professional for commentary rather than attempting analysis ourselves.

I vote to APPROVE this story."""},

            {"source": "Scout_B", "content": """Trend_Analyst's data is exactly what I was hoping to see. The comparative analysis validates my instinct about this one.

I also want to address the expertise concern directly. I've already identified three potential expert sources:
1. Dr. Sarah Chen, clinical psychologist with 400K TikTok followers who posted about this yesterday
2. The "Digital Wellness Institute" - they issued a statement this morning
3. A Gen Z researcher at Pew who studies social media behavior

These aren't just random commentators - they're credentialed professionals who are already engaging with this trend. We wouldn't be imposing expertise; we'd be amplifying existing expert discourse.

The Slate and Vice mentions are also key. If mainstream outlets are already drafting, our window for distinctive coverage is narrowing. We need to move.

I vote to APPROVE this story."""},

            {"source": "Skeptic_Agent", "content": """The data is... better than I expected. I'll acknowledge that.

But I have a different concern now: are we just chasing clicks? This feels like the kind of "soft news" that gets engagement but doesn't serve our audience's information needs.

We have limited production resources. Every trend story we cover is a more substantive story we don't. What's the opportunity cost here?

Also, I want to question the framing. Scout_B is presenting this as "cultural commentary about mental health" - but is that what it actually is, or is that just the most respectable way to frame viral content? Are we dressing up entertainment as journalism?

I'm not convinced this meets our editorial standards. I maintain my vote to REJECT this story."""},

            {"source": "Scout_A", "content": """Skeptic, with respect, I think you're overthinking this in the wrong direction.

First, on "chasing clicks" - every story we cover needs audience interest. That's not clickbait, that's relevance. The data shows this resonates. Our job is to cover what matters to people, and this clearly does.

Second, on opportunity cost - we have multiple production teams. Covering this doesn't mean not covering something else. It means covering THIS too.

Third, and most importantly: cultural trends ARE substantive. How a generation thinks about technology, mental health, mindfulness - these are real issues with real implications. Just because it started on TikTok doesn't make it less valid than a congressional hearing.

The framing question is fair - but that's why we have experts. We let THEM interpret what this means. We're the conduit, not the authority.

This is legitimate coverage. Let's not be snobs about platforms. I vote to APPROVE this story."""},

            {"source": "Trend_Analyst", "content": """I want to address the opportunity cost concern with specific data.

Our audience analytics show that "culture + technology intersection" stories have the highest subscriber conversion rate of any category - 2.7x higher than pure tech news and 4.1x higher than general culture coverage.

The hypothesis: our audience wants to understand how technology is shaping culture and vice versa. This story fits that intersection perfectly.

Additionally, production cost analysis:
- Estimated production time: 6-8 hours
- Expected views (based on comparable stories): 180K-250K
- Subscriber conversion estimate: 450-620 new subscribers
- Cost per subscriber acquired: $0.32-$0.44

This is efficient acquisition compared to our $1.20 average.

The data supports coverage from both editorial AND business perspectives. I vote to APPROVE this story."""},

            {"source": "Council_Chair", "content": """We've had excellent debate on this proposal. Let me summarize the key points:

SUPPORTING COVERAGE:
- Strong cross-platform metrics (Scout_B, Trend_Analyst)
- Expert sources already engaged (Scout_B)
- Fits our high-performing intersection category (Trend_Analyst)
- Efficient subscriber acquisition (Trend_Analyst)
- Addresses genuine cultural questions (Scout_A, Scout_B)

CONCERNS RAISED:
- Risk of ephemeral trend coverage (Skeptic)
- Expertise requirements (Skeptic, addressed by Scout_B)
- Editorial standards questions (Skeptic)
- Opportunity cost (Skeptic, addressed by Trend_Analyst)

I believe the supporting evidence outweighs the concerns, particularly given that expert sources have been identified and the data shows strong audience fit.

I'm prepared to call the final vote.

VOTES CAST:
- Scout_A: APPROVE
- Scout_B: APPROVE
- Trend_Analyst: APPROVE
- Skeptic_Agent: REJECT

With 3 approvals and 1 rejection, the motion passes.

FINAL DECISION: APPROVED - The trend story is greenlit for production. Scout_B to coordinate with identified expert sources. Trend_Analyst to provide metrics visualization. Skeptic's concerns about editorial framing are noted - production team to review script for balance."""},
        ]
    
    def _generic_debate(self, story: StoryProposal) -> List[Dict]:
        """Generate a generic debate template."""
        
        return [
            {"source": "Council_Chair", "content": f"The Trend Council is now in session. We have a new story proposal: {story.title}. Council members, please share your analysis."},
            {"source": "Scout_A", "content": "This looks like a solid mainstream story. The source is credible and the topic has broad appeal. I think we should move on it quickly before competitors pick it up. I vote to APPROVE this story."},
            {"source": "Skeptic_Agent", "content": "I have concerns about the newsworthiness here. What's the unique angle? Why should our audience care about this specifically? I need to see more evidence that this matters. I vote to REJECT this story."},
            {"source": "Scout_B", "content": "I'm seeing some social buzz around this topic. The engagement metrics suggest there's genuine interest. Cross-platform mentions are growing at a steady rate. I vote to APPROVE this story."},
            {"source": "Trend_Analyst", "content": "Based on our historical performance metrics, stories in this category typically perform at or slightly above average. The audience interest is there, though not exceptional. I vote to APPROVE this story."},
            {"source": "Council_Chair", "content": f"FINAL DECISION: APPROVED - {story.title} is greenlit for production with standard priority."},
        ]
    
    def _simulate_decision(self, story: StoryProposal) -> tuple:
        """Simulate a decision based on story characteristics."""
        
        if "AI" in story.title or "regulation" in story.title.lower():
            return (
                StoryStatus.APPROVED.value,
                "Strong data support, expert sources available, significant audience interest - unanimous approval",
                {
                    "Scout_A": "APPROVE",
                    "Scout_B": "APPROVE",
                    "Trend_Analyst": "APPROVE",
                    "Skeptic_Agent": "APPROVE"
                }
            )
        elif story.source_type == "social_media":
            return (
                StoryStatus.APPROVED.value,
                "Strong metrics and audience fit, though some editorial concerns raised - majority approval",
                {
                    "Scout_A": "APPROVE",
                    "Scout_B": "APPROVE",
                    "Trend_Analyst": "APPROVE",
                    "Skeptic_Agent": "REJECT"
                }
            )
        else:
            return (
                StoryStatus.APPROVED.value,
                "Moderate audience interest, approved for coverage",
                {
                    "Scout_A": "APPROVE",
                    "Scout_B": "APPROVE",
                    "Trend_Analyst": "APPROVE",
                    "Skeptic_Agent": "ABSTAIN"
                }
            )
    
    def _format_transcript(self, messages: List[Dict]) -> str:
        """Format messages as a readable transcript."""
        transcript = []
        for i, msg in enumerate(messages, 1):
            source = msg.get("source", "Unknown")
            content = msg.get("content", "")
            transcript.append(f"\n{'='*60}")
            transcript.append(f"[Message {i}] {source}:")
            transcript.append(f"{'='*60}")
            transcript.append(content)
        return "\n".join(transcript)


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_result(result: DeliberationResult) -> str:
    """Format a deliberation result as a readable summary."""
    story = result.story
    
    output = f"""
{'='*70}
📰 STORY: {story.title}
{'='*70}

📝 Source: {story.source} ({story.source_type})
👤 Proposed by: {story.proposed_by}

📊 DECISION: {result.decision.upper()}

💭 REASONING:
{result.reasoning}

🗳️  VOTES:
"""
    for agent, vote in result.votes.items():
        icon = "✅" if "APPROVE" in vote.upper() else "❌" if "REJECT" in vote.upper() else "➖"
        output += f"   {icon} {agent}: {vote}\n"
    
    output += f"""
🔄 Rounds of debate: {result.rounds}
⏱️  Timestamp: {result.timestamp}
"""
    return output


def print_transcript(result: DeliberationResult, max_messages: int = 10):
    """Print a formatted transcript of the debate."""
    print(f"\n{'='*70}")
    print("🎭 FULL DEBATE TRANSCRIPT")
    print(f"{'='*70}")
    print(result.transcript[:5000] if len(result.transcript) > 5000 else result.transcript)
    
    if len(result.transcript) > 5000:
        print(f"\n... ({len(result.transcript) - 5000} more characters)")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Run the Trend Council deliberation system."""
    
    print("="*70)
    print("🏛️  TREND COUNCIL DELIBERATION SYSTEM")
    print("   Automated News Video Production - Story Evaluation")
    print("="*70)
    
    # Use simulated council for demo (no API calls needed)
    council = SimulatedTrendCouncil()
    
    # Example stories to deliberate
    stories = [
        StoryProposal(
            title="EU Passes Landmark AI Act Amendment on Foundation Models",
            source="Reuters",
            source_type="mainstream",
            summary="The European Parliament has passed a significant amendment to the AI Act specifically targeting foundation models like GPT-4. New requirements include transparency reports, copyright compliance documentation, and systemic risk evaluations for models above certain compute thresholds.",
            proposed_by="Scout_A",
            metrics={"mainstream_citations": 200, "urgency": "high", "source_credibility": "very_high"}
        ),
        StoryProposal(
            title="Viral 'Silent Walking' Trend Sparks Debate About Gen Z and Mental Health",
            source="TikTok/Twitter aggregation",
            source_type="social_media",
            summary="A TikTok trend where users film themselves walking silently without headphones or distractions has gone viral with 500M+ views. Originally framed as mindfulness practice, it's evolved into commentary about Gen Z's relationship with technology and constant stimulation.",
            proposed_by="Scout_B",
            metrics={
                "tiktok_views": "500M+",
                "engagement_rate": "8.2%",
                "cross_platform": True,
                "primary_demographic": "18-29"
            }
        ),
    ]
    
    results = []
    
    for i, story in enumerate(stories, 1):
        print(f"\n{'='*70}")
        print(f"🔥 DELIBERATION {i}/{len(stories)}: {story.title[:50]}...")
        print(f"{'='*70}\n")
        
        # Run deliberation
        result = await council.deliberate(story)
        results.append(result)
        
        # Print summary
        print(format_result(result))
        
        # Print transcript preview
        print(f"\n{'='*70}")
        print("🎬 DEBATE TRANSCRIPT HIGHLIGHTS:")
        print(f"{'='*70}")
        
        # Extract key exchanges
        messages = result.transcript.split("[Message")
        for msg in messages[1:4]:  # Show first 3 exchanges
            lines = msg.strip().split("\n")
            if lines:
                print(f"\n{lines[0]}")
                content = "\n".join(lines[2:5])  # First few lines of content
                print(content[:300] + "..." if len(content) > 300 else content)
    
    # Final summary
    print(f"\n{'='*70}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*70}\n")
    
    approved = [r for r in results if r.decision == StoryStatus.APPROVED.value]
    rejected = [r for r in results if r.decision == StoryStatus.REJECTED.value]
    deadlocked = [r for r in results if r.decision == StoryStatus.DEADLOCKED.value]
    
    print(f"📈 Total stories deliberated: {len(results)}")
    print(f"✅ Approved: {len(approved)}")
    print(f"❌ Rejected: {len(rejected)}")
    print(f"⚖️  Deadlocked: {len(deadlocked)}")
    
    print("\n📋 APPROVED STORIES (ready for next phase):")
    for r in approved:
        print(f"   ✓ {r.story.title}")
    
    return results


if __name__ == "__main__":
    results = asyncio.run(main())
