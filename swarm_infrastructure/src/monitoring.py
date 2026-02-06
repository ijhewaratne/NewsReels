# =============================================================================
# LANGFUSE MONITORING INTEGRATION
# =============================================================================
"""
Langfuse integration for monitoring agent decisions, consensus votes, and costs.

Features:
- Trace agent decision chains
- Track consensus votes and revisions
- Monitor token usage and costs
- Custom dashboards for key metrics
- Alerting on anomalies
"""

import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Try to import langfuse
try:
    from langfuse import Langfuse
    from langfuse.api.resources.commons.types.observation_level import ObservationLevel
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.warning("Langfuse not installed. Monitoring will use fallback implementation.")


# =============================================================================
# MONITORING SCHEMAS
# =============================================================================

class TraceType(str, Enum):
    """Types of traces to monitor"""
    AGENT_DECISION = "agent_decision"
    CONSENSUS_VOTE = "consensus_vote"
    DEBATE_SESSION = "debate_session"
    SCRIPT_GENERATION = "script_generation"
    VIDEO_PRODUCTION = "video_production"
    QUALITY_REVIEW = "quality_review"
    END_TO_END = "end_to_end"


class SpanType(str, Enum):
    """Types of spans within traces"""
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    AGENT_ACTION = "agent_action"
    RETRIEVAL = "retrieval"
    PROCESSING = "processing"
    VALIDATION = "validation"


@dataclass
class AgentDecision:
    """Record of an agent's decision"""
    agent_id: str
    agent_role: str
    decision_type: str
    input_context: Dict[str, Any]
    output_decision: Dict[str, Any]
    confidence: float
    reasoning: str
    alternatives_considered: List[str]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ConsensusVote:
    """Record of a consensus vote"""
    debate_id: str
    agent_id: str
    agent_role: str
    proposal_id: str
    vote: str  # for, against, abstain
    confidence: float
    reasoning: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ConsensusResult:
    """Result of a consensus process"""
    debate_id: str
    topic: str
    total_votes: int
    votes_for: int
    votes_against: int
    winning_proposal: str
    consensus_score: float  # 0-1, higher = more consensus
    participating_agents: List[str]
    duration_seconds: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class RevisionRecord:
    """Record of a revision"""
    content_id: str
    revision_number: int
    revised_by: str
    revision_type: str  # grammar, content, style, fact_check
    changes_made: List[str]
    improvement_score: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


# =============================================================================
# LANGFUSE MONITOR CLASS
# =============================================================================

class SwarmMonitor:
    """
    Central monitoring for the multi-agent swarm using Langfuse.
    
    Tracks:
    - Agent decisions and reasoning
    - Consensus votes and outcomes
    - Script revisions and improvements
    - Token usage and costs
    - End-to-end pipeline performance
    """
    
    def __init__(
        self,
        langfuse_client: Optional[Any] = None,
        redis_client=None,
        enabled: bool = True,
        sample_rate: float = 1.0  # 1.0 = trace everything
    ):
        self.enabled = enabled and LANGFUSE_AVAILABLE
        self.sample_rate = sample_rate
        self.redis = redis_client
        
        # Initialize Langfuse client
        if self.enabled and langfuse_client:
            self.langfuse = langfuse_client
        elif self.enabled:
            self.langfuse = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
            )
        else:
            self.langfuse = None
            self._fallback_traces = []
        
        self._active_traces: Dict[str, Any] = {}
        self._alert_handlers: List[Callable] = []
        
    # ========================================================================
    # TRACE MANAGEMENT
    # ========================================================================
    
    def start_trace(
        self,
        trace_type: TraceType,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new trace.
        
        Returns trace ID for subsequent spans.
        """
        if not self.enabled or self._should_sample() is False:
            return trace_id or f"trace_{datetime.now(timezone.utc).timestamp()}"
        
        trace_id = trace_id or str(uuid.uuid4())
        
        if self.langfuse:
            trace = self.langfuse.trace(
                id=trace_id,
                name=trace_type.value,
                metadata=metadata or {}
            )
            self._active_traces[trace_id] = trace
        else:
            # Fallback: store locally
            self._active_traces[trace_id] = {
                'id': trace_id,
                'type': trace_type.value,
                'metadata': metadata or {},
                'spans': [],
                'start_time': datetime.now(timezone.utc)
            }
        
        logger.debug(f"Started trace: {trace_id} ({trace_type.value})")
        return trace_id
    
    def end_trace(
        self,
        trace_id: str,
        status: str = "success",
        output: Optional[Dict[str, Any]] = None
    ):
        """End a trace"""
        if trace_id not in self._active_traces:
            return
        
        trace = self._active_traces.pop(trace_id)
        
        if self.langfuse:
            trace.update(
                status=status,
                output=output or {}
            )
        else:
            # Fallback
            trace['end_time'] = datetime.now(timezone.utc)
            trace['status'] = status
            trace['output'] = output
            self._fallback_traces.append(trace)
        
        logger.debug(f"Ended trace: {trace_id} ({status})")
    
    def start_span(
        self,
        trace_id: str,
        span_type: SpanType,
        name: str,
        input_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a span within a trace"""
        if trace_id not in self._active_traces:
            return None
        
        span_id = f"{trace_id}:{span_type.value}:{datetime.now(timezone.utc).timestamp()}"
        
        if self.langfuse:
            trace = self._active_traces[trace_id]
            span = trace.span(
                id=span_id,
                name=name,
                input=input_data or {},
                metadata={'span_type': span_type.value}
            )
            # Store span reference
            if not hasattr(trace, '_spans'):
                trace._spans = {}
            trace._spans[span_id] = span
        else:
            # Fallback
            span = {
                'id': span_id,
                'type': span_type.value,
                'name': name,
                'input': input_data,
                'start_time': datetime.now(timezone.utc)
            }
            self._active_traces[trace_id]['spans'].append(span)
        
        return span_id
    
    def end_span(
        self,
        trace_id: str,
        span_id: str,
        output_data: Optional[Dict[str, Any]] = None,
        status: str = "success",
        usage: Optional[Dict[str, int]] = None
    ):
        """End a span"""
        if not self.langfuse:
            return
        
        trace = self._active_traces.get(trace_id)
        if not trace or not hasattr(trace, '_spans'):
            return
        
        span = trace._spans.pop(span_id, None)
        if span:
            span.end(
                output=output_data or {},
                status=status,
                usage=usage
            )
    
    def log_llm_call(
        self,
        trace_id: str,
        model: str,
        prompt: str,
        completion: str,
        tokens: Dict[str, int],
        cost: float,
        duration_ms: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an LLM call as a generation"""
        if not self.enabled or trace_id not in self._active_traces:
            return
        
        if self.langfuse:
            trace = self._active_traces[trace_id]
            trace.generation(
                name=f"llm_{model}",
                model=model,
                prompt=prompt[:1000],  # Truncate for storage
                completion=completion[:1000],
                usage=tokens,
                metadata={
                    **(metadata or {}),
                    'cost_usd': cost,
                    'duration_ms': duration_ms
                }
            )
    
    # ========================================================================
    # AGENT DECISION TRACKING
    # ========================================================================
    
    def track_agent_decision(
        self,
        trace_id: str,
        decision: AgentDecision
    ):
        """Track an agent's decision"""
        if not self.enabled:
            return
        
        # Create span for decision
        span_id = self.start_span(
            trace_id=trace_id,
            span_type=SpanType.AGENT_ACTION,
            name=f"decision_{decision.decision_type}",
            input_data=decision.input_context
        )
        
        # Log to Langfuse as event
        if self.langfuse:
            trace = self._active_traces.get(trace_id)
            if trace:
                trace.event(
                    name="agent_decision",
                    metadata={
                        'agent_id': decision.agent_id,
                        'agent_role': decision.agent_role,
                        'decision_type': decision.decision_type,
                        'confidence': decision.confidence,
                        'reasoning': decision.reasoning,
                        'alternatives': decision.alternatives_considered
                    }
                )
        
        # Store in Redis for quick access
        if self.redis:
            asyncio.create_task(self._store_decision(decision))
        
        self.end_span(
            trace_id=trace_id,
            span_id=span_id,
            output_data=decision.output_decision,
            status="success" if decision.confidence > 0.5 else "warning"
        )
    
    async def _store_decision(self, decision: AgentDecision):
        """Store decision in Redis"""
        key = f"monitor:decisions:{decision.agent_id}:{decision.timestamp.strftime('%Y%m%d')}"
        await self.redis.lpush(key, json.dumps(decision.to_dict()))
        await self.redis.expire(key, 7 * 24 * 3600)
    
    # ========================================================================
    # CONSENSUS TRACKING
    # ========================================================================
    
    def track_consensus_vote(
        self,
        trace_id: str,
        vote: ConsensusVote
    ):
        """Track a consensus vote"""
        if not self.enabled:
            return
        
        if self.langfuse:
            trace = self._active_traces.get(trace_id)
            if trace:
                trace.event(
                    name="consensus_vote",
                    metadata={
                        'debate_id': vote.debate_id,
                        'agent_id': vote.agent_id,
                        'vote': vote.vote,
                        'confidence': vote.confidence
                    }
                )
        
        if self.redis:
            asyncio.create_task(self._store_vote(vote))
    
    async def _store_vote(self, vote: ConsensusVote):
        """Store vote in Redis"""
        key = f"monitor:votes:{vote.debate_id}"
        await self.redis.hset(key, vote.agent_id, json.dumps(vote.to_dict()))
        await self.redis.expire(key, 7 * 24 * 3600)
    
    def track_consensus_result(
        self,
        trace_id: str,
        result: ConsensusResult
    ):
        """Track the result of a consensus process"""
        if not self.enabled:
            return
        
        if self.langfuse:
            trace = self._active_traces.get(trace_id)
            if trace:
                trace.score(
                    name="consensus_score",
                    value=result.consensus_score,
                    metadata={
                        'total_votes': result.total_votes,
                        'votes_for': result.votes_for,
                        'duration_seconds': result.duration_seconds
                    }
                )
        
        if self.redis:
            asyncio.create_task(self._store_consensus_result(result))
    
    async def _store_consensus_result(self, result: ConsensusResult):
        """Store consensus result in Redis"""
        key = f"monitor:consensus:{result.debate_id}"
        await self.redis.set(key, json.dumps(result.to_dict()))
        await self.redis.expire(key, 30 * 24 * 3600)
    
    # ========================================================================
    # REVISION TRACKING
    # ========================================================================
    
    def track_revision(
        self,
        trace_id: str,
        revision: RevisionRecord
    ):
        """Track a content revision"""
        if not self.enabled:
            return
        
        if self.langfuse:
            trace = self._active_traces.get(trace_id)
            if trace:
                trace.event(
                    name="content_revision",
                    metadata={
                        'content_id': revision.content_id,
                        'revision_number': revision.revision_number,
                        'revision_type': revision.revision_type,
                        'improvement_score': revision.improvement_score
                    }
                )
        
        if self.redis:
            asyncio.create_task(self._store_revision(revision))
    
    async def _store_revision(self, revision: RevisionRecord):
        """Store revision in Redis"""
        key = f"monitor:revisions:{revision.content_id}"
        await self.redis.lpush(key, json.dumps(revision.to_dict()))
        await self.redis.expire(key, 30 * 24 * 3600)
    
    # ========================================================================
    # COST TRACKING
    # ========================================================================
    
    def track_cost(
        self,
        trace_id: str,
        agent_id: str,
        task_type: str,
        cost_usd: float,
        tokens: Dict[str, int]
    ):
        """Track cost for a task"""
        if not self.enabled:
            return
        
        if self.langfuse:
            trace = self._active_traces.get(trace_id)
            if trace:
                trace.score(
                    name="cost_usd",
                    value=cost_usd,
                    metadata={
                        'agent_id': agent_id,
                        'task_type': task_type,
                        'tokens': tokens
                    }
                )
    
    # ========================================================================
    # CONTEXT MANAGERS
    # ========================================================================
    
    @contextmanager
    def trace_context(
        self,
        trace_type: TraceType,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracing"""
        tid = self.start_trace(trace_type, trace_id, metadata)
        try:
            yield tid
            self.end_trace(tid, "success")
        except Exception as e:
            self.end_trace(tid, "error", {"error": str(e)})
            raise
    
    # ========================================================================
    # SAMPLING
    # ========================================================================
    
    def _should_sample(self) -> bool:
        """Determine if this trace should be sampled"""
        import random
        return random.random() < self.sample_rate
    
    # ========================================================================
    # ALERTS
    # ========================================================================
    
    def register_alert_handler(self, handler: Callable[[Dict], None]):
        """Register handler for monitoring alerts"""
        self._alert_handlers.append(handler)
    
    async def trigger_alert(self, alert: Dict[str, Any]):
        """Trigger a monitoring alert"""
        for handler in self._alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    async def get_agent_decisions(
        self,
        agent_id: str,
        days: int = 7
    ) -> List[AgentDecision]:
        """Get recent decisions for an agent"""
        if not self.redis:
            return []
        
        decisions = []
        for i in range(days):
            date_str = (datetime.now(timezone.utc) - __import__('datetime').timedelta(days=i)).strftime('%Y%m%d')
            key = f"monitor:decisions:{agent_id}:{date_str}"
            data_list = await self.redis.lrange(key, 0, -1)
            
            for data in data_list:
                decision_dict = json.loads(data)
                decisions.append(AgentDecision(**decision_dict))
        
        return decisions
    
    async def get_consensus_history(
        self,
        debate_id: Optional[str] = None
    ) -> List[ConsensusResult]:
        """Get consensus history"""
        if not self.redis:
            return []
        
        if debate_id:
            key = f"monitor:consensus:{debate_id}"
            data = await self.redis.get(key)
            if data:
                return [ConsensusResult(**json.loads(data))]
            return []
        
        # Get all consensus results
        keys = await self.redis.keys("monitor:consensus:*")
        results = []
        for key in keys:
            data = await self.redis.get(key)
            if data:
                results.append(ConsensusResult(**json.loads(data)))
        
        return results
    
    async def get_revision_count(
        self,
        content_id: str
    ) -> int:
        """Get number of revisions for content"""
        if not self.redis:
            return 0
        
        key = f"monitor:revisions:{content_id}"
        return await self.redis.llen(key)
    
    async def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard"""
        metrics = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'traces_active': len(self._active_traces),
            'total_fallback_traces': len(self._fallback_traces),
        }
        
        if self.redis:
            # Get decision counts
            decision_keys = await self.redis.keys("monitor:decisions:*")
            metrics['total_decisions_tracked'] = len(decision_keys)
            
            # Get consensus counts
            consensus_keys = await self.redis.keys("monitor:consensus:*")
            metrics['total_consensus_tracked'] = len(consensus_keys)
            
            # Get revision counts
            revision_keys = await self.redis.keys("monitor:revisions:*")
            metrics['content_with_revisions'] = len(revision_keys)
        
        return metrics


# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================

LANGFUSE_DASHBOARD_CONFIG = {
    "dashboards": [
        {
            "name": "Swarm Overview",
            "widgets": [
                {
                    "type": "trace_count",
                    "title": "Total Traces",
                    "filters": {"type": "end_to_end"}
                },
                {
                    "type": "average_latency",
                    "title": "Avg Pipeline Latency",
                    "filters": {"type": "end_to_end"}
                },
                {
                    "type": "token_usage",
                    "title": "Daily Token Usage",
                    "aggregation": "sum"
                },
                {
                    "type": "cost_chart",
                    "title": "Daily Costs",
                    "currency": "USD"
                }
            ]
        },
        {
            "name": "Agent Performance",
            "widgets": [
                {
                    "type": "score_distribution",
                    "title": "Decision Confidence",
                    "score_name": "confidence"
                },
                {
                    "type": "event_count",
                    "title": "Decisions per Agent",
                    "event_name": "agent_decision",
                    "group_by": "agent_id"
                },
                {
                    "type": "average_score",
                    "title": "Avg Consensus Score",
                    "score_name": "consensus_score"
                }
            ]
        },
        {
            "name": "Quality Metrics",
            "widgets": [
                {
                    "type": "event_count",
                    "title": "Total Revisions",
                    "event_name": "content_revision"
                },
                {
                    "type": "score_trend",
                    "title": "Improvement Score Trend",
                    "score_name": "improvement_score"
                },
                {
                    "type": "error_rate",
                    "title": "Pipeline Error Rate"
                }
            ]
        }
    ]
}


# =============================================================================
# SETUP INSTRUCTIONS
# =============================================================================

LANGFUSE_SETUP_INSTRUCTIONS = """
# Langfuse Setup for Swarm Monitoring

## 1. Environment Variables

```bash
# Required
export LANGFUSE_PUBLIC_KEY="your_public_key"
export LANGFUSE_SECRET_KEY="your_secret_key"
export LANGFUSE_HOST="http://localhost:3000"  # or cloud URL

# Optional
export LANGFUSE_SAMPLE_RATE="1.0"  # 1.0 = trace everything
```

## 2. Docker Compose

Langfuse is included in the docker-compose.yml:

```yaml
langfuse:
  image: langfuse/langfuse:latest
  ports:
    - "3000:3000"
  environment:
    - DATABASE_URL=postgresql://postgres:password@postgres:5432/langfuse
    - NEXTAUTH_SECRET=your_secret
    - NEXTAUTH_URL=http://localhost:3000
```

## 3. Database Setup

```bash
# Create database
docker-compose up -d postgres

# Langfuse will auto-migrate on first start
docker-compose up -d langfuse
```

## 4. Access Dashboard

Open http://localhost:3000 in your browser.

Default credentials (if not configured):
- Create account on first run

## 5. API Keys

1. Go to Settings > API Keys
2. Generate new key pair
3. Add to environment variables

## 6. Custom Dashboards

Import the dashboard configuration from monitoring.py:

1. Go to Dashboards > Create New
2. Add widgets as defined in LANGFUSE_DASHBOARD_CONFIG
3. Configure alerts for budget thresholds
"""


# Import uuid for trace IDs
import uuid
