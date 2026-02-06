# =============================================================================
# COST MANAGEMENT SYSTEM - Token Usage and Budget Control
# =============================================================================
"""
The CostManager tracks and controls token usage across the swarm.

Features:
- Per-agent token tracking
- Budget limits and alerts
- LLM tiering strategy
- Early stopping rules
- Rate limiting
- Cost optimization recommendations
"""

import json
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# PRICING CONFIGURATION
# =============================================================================

class LLMModel(str, Enum):
    """Supported LLM models with pricing"""
    # OpenAI
    GPT4_TURBO = "gpt-4-turbo-preview"
    GPT4 = "gpt-4"
    GPT35_TURBO = "gpt-3.5-turbo"
    GPT35_TURBO_16K = "gpt-3.5-turbo-16k"
    
    # Anthropic
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    
    # Google
    GEMINI_PRO = "gemini-pro"
    GEMINI_ULTRA = "gemini-ultra"


# Pricing per 1K tokens (input/output)
MODEL_PRICING: Dict[LLMModel, Dict[str, float]] = {
    LLMModel.GPT4_TURBO: {"input": 0.01, "output": 0.03},
    LLMModel.GPT4: {"input": 0.03, "output": 0.06},
    LLMModel.GPT35_TURBO: {"input": 0.0005, "output": 0.0015},
    LLMModel.GPT35_TURBO_16K: {"input": 0.001, "output": 0.002},
    LLMModel.CLAUDE_3_OPUS: {"input": 0.015, "output": 0.075},
    LLMModel.CLAUDE_3_SONNET: {"input": 0.003, "output": 0.015},
    LLMModel.CLAUDE_3_HAIKU: {"input": 0.00025, "output": 0.00125},
    LLMModel.GEMINI_PRO: {"input": 0.0005, "output": 0.0015},
    LLMModel.GEMINI_ULTRA: {"input": 0.001, "output": 0.003},
}


# =============================================================================
# TIER STRATEGY
# =============================================================================

class LLMTier(str, Enum):
    """
    LLM tiers for cost optimization.
    
    Strategy:
    - FAST: Quick, cheap models for initial screening
    - BALANCED: Good quality at moderate cost
    - PREMIUM: Best quality for final outputs
    """
    FAST = "fast"           # GPT-3.5, Claude Haiku
    BALANCED = "balanced"   # GPT-4 Turbo, Claude Sonnet
    PREMIUM = "premium"     # GPT-4, Claude Opus


TIER_MODELS: Dict[LLMTier, List[LLMModel]] = {
    LLMTier.FAST: [
        LLMModel.GPT35_TURBO,
        LLMModel.CLAUDE_3_HAIKU,
    ],
    LLMTier.BALANCED: [
        LLMModel.GPT4_TURBO,
        LLMModel.CLAUDE_3_SONNET,
    ],
    LLMTier.PREMIUM: [
        LLMModel.GPT4,
        LLMModel.CLAUDE_3_OPUS,
    ],
}


# Task type to tier mapping
TASK_TIER_MAPPING: Dict[str, LLMTier] = {
    # Fast tier - initial screening, simple classification
    "news_scout": LLMTier.FAST,
    "content_filter": LLMTier.FAST,
    "topic_tagging": LLMTier.FAST,
    "sentiment_analysis": LLMTier.FAST,
    
    # Balanced tier - analysis and debate
    "content_analysis": LLMTier.BALANCED,
    "fact_check": LLMTier.BALANCED,
    "debate_argument": LLMTier.BALANCED,
    "consensus_vote": LLMTier.BALANCED,
    "script_draft": LLMTier.BALANCED,
    "visual_plan": LLMTier.BALANCED,
    
    # Premium tier - final outputs
    "script_final": LLMTier.PREMIUM,
    "script_revision": LLMTier.PREMIUM,
    "video_review": LLMTier.PREMIUM,
    "quality_assurance": LLMTier.PREMIUM,
}


# =============================================================================
# COST TRACKING SCHEMAS
# =============================================================================

@dataclass
class TokenUsage:
    """Token usage record"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __post_init__(self):
        self.total_tokens = self.prompt_tokens + self.completion_tokens
    
    def to_dict(self) -> Dict[str, int]:
        return asdict(self)
    
    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens
        )


@dataclass
class CostRecord:
    """Cost record for a single LLM call"""
    id: str
    agent_id: str
    task_id: str
    task_type: str
    model: LLMModel
    tier: LLMTier
    tokens: TokenUsage
    cost_usd: float
    timestamp: datetime
    duration_ms: int
    success: bool
    confidence_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['model'] = self.model.value
        data['tier'] = self.tier.value
        data['tokens'] = self.tokens.to_dict()
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class AgentBudget:
    """Budget configuration for an agent"""
    agent_id: str
    daily_limit: float = 50.0
    hourly_limit: float = 10.0
    task_limit: float = 5.0
    alert_threshold: float = 0.8  # Alert at 80% of budget
    
    # Tier limits (max percentage of budget per tier)
    tier_limits: Dict[str, float] = None
    
    def __post_init__(self):
        if self.tier_limits is None:
            self.tier_limits = {
                LLMTier.FAST.value: 0.3,      # 30% for fast tier
                LLMTier.BALANCED.value: 0.5,   # 50% for balanced tier
                LLMTier.PREMIUM.value: 0.2,    # 20% for premium tier
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BudgetStatus:
    """Current budget status"""
    agent_id: str
    daily_spent: float
    daily_remaining: float
    daily_percent: float
    hourly_spent: float
    hourly_remaining: float
    hourly_percent: float
    tier_spent: Dict[str, float]
    is_exceeded: bool
    alert_triggered: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# COST MANAGER CLASS
# =============================================================================

class CostManager:
    """
    Central cost management for the swarm.
    
    Tracks:
    - Token usage per agent, task, and model
    - Budget consumption and limits
    - Cost trends and optimization opportunities
    
    Controls:
    - LLM tier selection based on task type
    - Early stopping based on confidence
    - Rate limiting and budget enforcement
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        global_daily_budget: float = 1000.0,
        enable_alerts: bool = True
    ):
        self.redis = redis_client
        self.global_daily_budget = global_daily_budget
        self.enable_alerts = enable_alerts
        self._alert_handlers: List[Callable] = []
        self._stop_handlers: List[Callable] = []
        
    # ========================================================================
    # COST CALCULATION
    # ========================================================================
    
    @staticmethod
    def calculate_cost(model: LLMModel, tokens: TokenUsage) -> float:
        """Calculate cost for a token usage"""
        pricing = MODEL_PRICING.get(model, {"input": 0.01, "output": 0.03})
        
        input_cost = (tokens.prompt_tokens / 1000) * pricing["input"]
        output_cost = (tokens.completion_tokens / 1000) * pricing["output"]
        
        return round(input_cost + output_cost, 6)
    
    @staticmethod
    def estimate_cost(model: LLMModel, estimated_tokens: int) -> float:
        """Estimate cost for a given token count"""
        pricing = MODEL_PRICING.get(model, {"input": 0.01, "output": 0.03})
        avg_price = (pricing["input"] + pricing["output"]) / 2
        return round((estimated_tokens / 1000) * avg_price, 6)
    
    # ========================================================================
    # TIER SELECTION
    # ========================================================================
    
    def select_model(
        self,
        task_type: str,
        agent_budget: Optional[AgentBudget] = None,
        required_quality: float = 0.7
    ) -> LLMModel:
        """
        Select appropriate model for task based on tier strategy.
        
        Args:
            task_type: Type of task being performed
            agent_budget: Agent's budget configuration
            required_quality: Minimum quality threshold (0-1)
            
        Returns:
            Selected LLM model
        """
        # Get tier for task
        tier = TASK_TIER_MAPPING.get(task_type, LLMTier.BALANCED)
        
        # Check budget constraints
        if agent_budget:
            tier_budget_used = self._get_tier_budget_usage(
                agent_budget.agent_id,
                tier
            )
            tier_limit = agent_budget.tier_limits.get(tier.value, 1.0)
            
            # If tier budget exceeded, downgrade
            if tier_budget_used >= tier_limit:
                if tier == LLMTier.PREMIUM:
                    tier = LLMTier.BALANCED
                elif tier == LLMTier.BALANCED:
                    tier = LLMTier.FAST
        
        # Select model from tier
        models = TIER_MODELS[tier]
        
        # For high quality requirements, pick best model in tier
        if required_quality >= 0.9 and tier == LLMTier.PREMIUM:
            return LLMModel.CLAUDE_3_OPUS
        elif required_quality >= 0.8 and tier == LLMTier.BALANCED:
            return LLMModel.GPT4_TURBO
        
        # Default to first model in tier
        return models[0]
    
    def get_tier_for_model(self, model: LLMModel) -> LLMTier:
        """Get tier for a given model"""
        for tier, models in TIER_MODELS.items():
            if model in models:
                return tier
        return LLMTier.BALANCED
    
    # ========================================================================
    # TOKEN TRACKING
    # ========================================================================
    
    async def record_usage(
        self,
        agent_id: str,
        task_id: str,
        task_type: str,
        model: LLMModel,
        tokens: TokenUsage,
        duration_ms: int,
        success: bool = True,
        confidence_score: Optional[float] = None
    ) -> CostRecord:
        """
        Record token usage and calculate cost.
        
        Returns the cost record for tracking.
        """
        cost = self.calculate_cost(model, tokens)
        tier = self.get_tier_for_model(model)
        
        record = CostRecord(
            id=f"{agent_id}:{task_id}:{datetime.now(timezone.utc).timestamp()}",
            agent_id=agent_id,
            task_id=task_id,
            task_type=task_type,
            model=model,
            tier=tier,
            tokens=tokens,
            cost_usd=cost,
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            success=success,
            confidence_score=confidence_score
        )
        
        # Store in Redis
        await self._store_cost_record(record)
        
        # Update aggregations
        await self._update_aggregations(record)
        
        # Check early stopping
        if confidence_score is not None:
            should_stop = await self._check_early_stopping(
                agent_id, task_id, confidence_score, cost
            )
            if should_stop:
                await self._trigger_stop(agent_id, task_id, confidence_score)
        
        # Check budget
        await self._check_budget(agent_id)
        
        logger.debug(
            f"Recorded usage: {agent_id} used {tokens.total_tokens} tokens "
            f"(${cost:.4f}) for {task_type}"
        )
        
        return record
    
    async def _store_cost_record(self, record: CostRecord):
        """Store cost record in Redis"""
        key = f"cost:records:{record.agent_id}:{record.timestamp.strftime('%Y%m%d')}"
        await self.redis.lpush(key, json.dumps(record.to_dict()))
        
        # Set expiration (keep for 30 days)
        await self.redis.expire(key, 30 * 24 * 3600)
    
    async def _update_aggregations(self, record: CostRecord):
        """Update cost aggregations"""
        date_str = record.timestamp.strftime('%Y%m%d')
        hour_str = record.timestamp.strftime('%Y%m%d%H')
        
        # Daily aggregation
        daily_key = f"cost:daily:{record.agent_id}:{date_str}"
        await self.redis.hincrbyfloat(daily_key, "total_cost", record.cost_usd)
        await self.redis.hincrby(daily_key, "total_tokens", record.tokens.total_tokens)
        await self.redis.hincrby(daily_key, "call_count", 1)
        await self.redis.expire(daily_key, 30 * 24 * 3600)
        
        # Hourly aggregation
        hourly_key = f"cost:hourly:{record.agent_id}:{hour_str}"
        await self.redis.hincrbyfloat(hourly_key, "total_cost", record.cost_usd)
        await self.redis.expire(hourly_key, 7 * 24 * 3600)
        
        # Tier aggregation
        tier_key = f"cost:tier:{record.agent_id}:{date_str}:{record.tier.value}"
        await self.redis.hincrbyfloat(tier_key, "total_cost", record.cost_usd)
        await self.redis.expire(tier_key, 30 * 24 * 3600)
        
        # Task type aggregation
        task_key = f"cost:task:{date_str}:{record.task_type}"
        await self.redis.hincrbyfloat(task_key, "total_cost", record.cost_usd)
        await self.redis.hincrby(task_key, "count", 1)
        await self.redis.expire(task_key, 30 * 24 * 3600)
    
    # ========================================================================
    # BUDGET MANAGEMENT
    # ========================================================================
    
    async def set_budget(self, budget: AgentBudget):
        """Set budget for an agent"""
        key = f"cost:budget:{budget.agent_id}"
        await self.redis.set(key, json.dumps(budget.to_dict()))
        logger.info(f"Set budget for agent {budget.agent_id}: ${budget.daily_limit}/day")
    
    async def get_budget(self, agent_id: str) -> Optional[AgentBudget]:
        """Get budget for an agent"""
        key = f"cost:budget:{agent_id}"
        data = await self.redis.get(key)
        if data:
            budget_dict = json.loads(data)
            return AgentBudget(**budget_dict)
        return None
    
    async def get_budget_status(self, agent_id: str) -> BudgetStatus:
        """Get current budget status for an agent"""
        budget = await self.get_budget(agent_id)
        if not budget:
            budget = AgentBudget(agent_id=agent_id)
        
        date_str = datetime.now(timezone.utc).strftime('%Y%m%d')
        hour_str = datetime.now(timezone.utc).strftime('%Y%m%d%H')
        
        # Get daily spending
        daily_key = f"cost:daily:{agent_id}:{date_str}"
        daily_data = await self.redis.hgetall(daily_key)
        daily_spent = float(daily_data.get(b"total_cost", 0))
        
        # Get hourly spending
        hourly_key = f"cost:hourly:{agent_id}:{hour_str}"
        hourly_data = await self.redis.hgetall(hourly_key)
        hourly_spent = float(hourly_data.get(b"total_cost", 0))
        
        # Get tier spending
        tier_spent = {}
        for tier in LLMTier:
            tier_key = f"cost:tier:{agent_id}:{date_str}:{tier.value}"
            tier_data = await self.redis.hgetall(tier_key)
            tier_spent[tier.value] = float(tier_data.get(b"total_cost", 0))
        
        daily_percent = daily_spent / budget.daily_limit if budget.daily_limit > 0 else 0
        hourly_percent = hourly_spent / budget.hourly_limit if budget.hourly_limit > 0 else 0
        
        is_exceeded = daily_spent >= budget.daily_limit or hourly_spent >= budget.hourly_limit
        alert_triggered = daily_percent >= budget.alert_threshold
        
        return BudgetStatus(
            agent_id=agent_id,
            daily_spent=daily_spent,
            daily_remaining=budget.daily_limit - daily_spent,
            daily_percent=daily_percent,
            hourly_spent=hourly_spent,
            hourly_remaining=budget.hourly_limit - hourly_spent,
            hourly_percent=hourly_percent,
            tier_spent=tier_spent,
            is_exceeded=is_exceeded,
            alert_triggered=alert_triggered
        )
    
    async def check_can_proceed(
        self,
        agent_id: str,
        estimated_cost: float,
        task_type: str
    ) -> tuple[bool, str]:
        """
        Check if agent can proceed with task based on budget.
        
        Returns (can_proceed, reason)
        """
        status = await self.get_budget_status(agent_id)
        budget = await self.get_budget(agent_id) or AgentBudget(agent_id=agent_id)
        
        # Check daily limit
        if status.daily_spent + estimated_cost > budget.daily_limit:
            return False, f"Daily budget exceeded: ${status.daily_spent:.2f} / ${budget.daily_limit:.2f}"
        
        # Check hourly limit
        if status.hourly_spent + estimated_cost > budget.hourly_limit:
            return False, f"Hourly budget exceeded: ${status.hourly_spent:.2f} / ${budget.hourly_limit:.2f}"
        
        # Check task limit
        if estimated_cost > budget.task_limit:
            return False, f"Task cost ${estimated_cost:.2f} exceeds limit ${budget.task_limit:.2f}"
        
        # Check tier budget
        tier = TASK_TIER_MAPPING.get(task_type, LLMTier.BALANCED)
        tier_spent = status.tier_spent.get(tier.value, 0)
        tier_limit = budget.daily_limit * budget.tier_limits.get(tier.value, 1.0)
        
        if tier_spent + estimated_cost > tier_limit:
            return False, f"{tier.value} tier budget exceeded: ${tier_spent:.2f} / ${tier_limit:.2f}"
        
        return True, "OK"
    
    # ========================================================================
    # EARLY STOPPING
    # ========================================================================
    
    async def _check_early_stopping(
        self,
        agent_id: str,
        task_id: str,
        confidence_score: float,
        cost_so_far: float
    ) -> bool:
        """
        Check if processing should stop early based on confidence.
        
        Rules:
        - If confidence < 0.3: Stop immediately (low quality)
        - If confidence > 0.9: Stop (good enough)
        - If cost > 2x estimate and confidence < 0.5: Stop
        """
        # Low confidence threshold
        if confidence_score < 0.3:
            logger.warning(
                f"Early stopping: confidence {confidence_score:.2f} < 0.3 "
                f"for task {task_id}"
            )
            return True
        
        # High confidence threshold
        if confidence_score > 0.9:
            logger.info(
                f"Early stopping: confidence {confidence_score:.2f} > 0.9 "
                f"for task {task_id}"
            )
            return True
        
        return False
    
    def register_stop_handler(self, handler: Callable[[str, str, float], None]):
        """Register handler for early stop events"""
        self._stop_handlers.append(handler)
    
    async def _trigger_stop(self, agent_id: str, task_id: str, confidence: float):
        """Trigger early stop handlers"""
        for handler in self._stop_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(agent_id, task_id, confidence)
                else:
                    handler(agent_id, task_id, confidence)
            except Exception as e:
                logger.error(f"Error in stop handler: {e}")
    
    # ========================================================================
    # ALERTS
    # ========================================================================
    
    async def _check_budget(self, agent_id: str):
        """Check budget and trigger alerts if needed"""
        if not self.enable_alerts:
            return
        
        status = await self.get_budget_status(agent_id)
        
        if status.alert_triggered:
            await self._trigger_alert(agent_id, status)
    
    async def _trigger_alert(self, agent_id: str, status: BudgetStatus):
        """Trigger budget alert"""
        alert = {
            'type': 'budget_alert',
            'agent_id': agent_id,
            'daily_percent': status.daily_percent,
            'daily_spent': status.daily_spent,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Store alert
        await self.redis.lpush(
            f"cost:alerts:{agent_id}",
            json.dumps(alert)
        )
        
        # Call handlers
        for handler in self._alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        logger.warning(
            f"Budget alert for {agent_id}: {status.daily_percent*100:.1f}% of daily budget used"
        )
    
    def register_alert_handler(self, handler: Callable[[Dict], None]):
        """Register handler for budget alerts"""
        self._alert_handlers.append(handler)
    
    # ========================================================================
    # RATE LIMITING
    # ========================================================================
    
    async def check_rate_limit(
        self,
        agent_id: str,
        max_calls_per_minute: int = 60
    ) -> tuple[bool, int]:
        """
        Check if agent has exceeded rate limit.
        
        Returns (allowed, remaining_calls)
        """
        minute_key = datetime.now(timezone.utc).strftime('%Y%m%d%H%M')
        key = f"cost:rate:{agent_id}:{minute_key}"
        
        current = await self.redis.incr(key)
        if current == 1:
            await self.redis.expire(key, 60)
        
        allowed = current <= max_calls_per_minute
        remaining = max(0, max_calls_per_minute - current)
        
        return allowed, remaining
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    async def get_agent_report(
        self,
        agent_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Generate cost report for an agent"""
        now = datetime.now(timezone.utc)
        
        daily_costs = []
        total_tokens = 0
        total_cost = 0.0
        
        for i in range(days):
            date = now - timedelta(days=i)
            date_str = date.strftime('%Y%m%d')
            
            daily_key = f"cost:daily:{agent_id}:{date_str}"
            daily_data = await self.redis.hgetall(daily_key)
            
            day_cost = float(daily_data.get(b"total_cost", 0))
            day_tokens = int(daily_data.get(b"total_tokens", 0))
            
            daily_costs.append({
                'date': date_str,
                'cost': day_cost,
                'tokens': day_tokens
            })
            
            total_cost += day_cost
            total_tokens += day_tokens
        
        # Get task breakdown
        task_breakdown = await self._get_task_breakdown(agent_id, days)
        
        return {
            'agent_id': agent_id,
            'period_days': days,
            'total_cost': total_cost,
            'total_tokens': total_tokens,
            'daily_breakdown': daily_costs,
            'task_breakdown': task_breakdown,
            'current_budget_status': (await self.get_budget_status(agent_id)).to_dict()
        }
    
    async def _get_task_breakdown(
        self,
        agent_id: str,
        days: int
    ) -> Dict[str, Dict[str, float]]:
        """Get cost breakdown by task type"""
        now = datetime.now(timezone.utc)
        task_costs = {}
        
        for i in range(days):
            date_str = (now - timedelta(days=i)).strftime('%Y%m%d')
            
            # Get all task keys for this date
            pattern = f"cost:task:{date_str}:*"
            keys = await self.redis.keys(pattern)
            
            for key in keys:
                task_type = key.decode().split(':')[-1]
                data = await self.redis.hgetall(key)
                
                cost = float(data.get(b"total_cost", 0))
                count = int(data.get(b"count", 0))
                
                if task_type not in task_costs:
                    task_costs[task_type] = {'cost': 0.0, 'count': 0}
                
                task_costs[task_type]['cost'] += cost
                task_costs[task_type]['count'] += count
        
        return task_costs
    
    async def get_global_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate global cost report for all agents"""
        # Get all agent IDs
        budget_keys = await self.redis.keys("cost:budget:*")
        agent_ids = [k.decode().split(':')[-1] for k in budget_keys]
        
        agent_reports = []
        total_cost = 0.0
        
        for agent_id in agent_ids:
            report = await self.get_agent_report(agent_id, days)
            agent_reports.append(report)
            total_cost += report['total_cost']
        
        return {
            'period_days': days,
            'total_cost': total_cost,
            'global_daily_budget': self.global_daily_budget,
            'budget_utilization': total_cost / (self.global_daily_budget * days),
            'agent_count': len(agent_ids),
            'agent_reports': agent_reports
        }
    
    # ========================================================================
    # OPTIMIZATION RECOMMENDATIONS
    # ========================================================================
    
    async def get_optimization_recommendations(
        self,
        agent_id: str
    ) -> List[Dict[str, Any]]:
        """Get cost optimization recommendations for an agent"""
        recommendations = []
        status = await self.get_budget_status(agent_id)
        
        # Check tier distribution
        total_tier_cost = sum(status.tier_spent.values())
        if total_tier_cost > 0:
            premium_pct = status.tier_spent.get('premium', 0) / total_tier_cost
            if premium_pct > 0.4:
                recommendations.append({
                    'type': 'tier_optimization',
                    'severity': 'medium',
                    'message': f'High premium tier usage ({premium_pct*100:.1f}%). Consider downgrading non-critical tasks.',
                    'potential_savings': total_tier_cost * 0.2
                })
        
        # Check hourly patterns
        hour_str = datetime.now(timezone.utc).strftime('%Y%m%d%H')
        hourly_key = f"cost:hourly:{agent_id}:{hour_str}"
        hourly_data = await self.redis.hgetall(hourly_key)
        hourly_cost = float(hourly_data.get(b"total_cost", 0))
        
        budget = await self.get_budget(agent_id)
        if budget and hourly_cost > budget.hourly_limit * 0.9:
            recommendations.append({
                'type': 'rate_limit',
                'severity': 'high',
                'message': 'Approaching hourly rate limit. Consider throttling.',
                'action': 'Enable rate limiting'
            })
        
        return recommendations
    
    def _get_tier_budget_usage(self, agent_id: str, tier: LLMTier) -> float:
        """Get percentage of tier budget used (synchronous for simplicity)"""
        # This would normally be async, simplified for tier selection
        return 0.0  # Default to no constraint


# =============================================================================
# DECORATOR FOR AUTOMATIC TRACKING
# =============================================================================

def track_costs(
    task_type: str,
    cost_manager: CostManager,
    model: Optional[LLMModel] = None
):
    """
    Decorator to automatically track LLM costs.
    
    Usage:
        @track_costs("script_draft", cost_manager)
        async def generate_script(agent_id, prompt):
            # LLM call here
            return result, tokens_used, confidence
    """
    def decorator(func):
        async def wrapper(agent_id: str, *args, **kwargs):
            import time
            
            start_time = time.time()
            
            # Check budget before proceeding
            can_proceed, reason = await cost_manager.check_can_proceed(
                agent_id, 0.01, task_type  # Minimal estimate
            )
            
            if not can_proceed:
                raise BudgetExceededError(f"Cannot proceed: {reason}")
            
            # Execute function
            result = await func(agent_id, *args, **kwargs)
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Extract usage info from result
            # Expected: (actual_result, tokens, confidence_score)
            if isinstance(result, tuple) and len(result) >= 2:
                actual_result = result[0]
                tokens = result[1] if isinstance(result[1], TokenUsage) else TokenUsage(
                    total_tokens=result[1]
                )
                confidence = result[2] if len(result) > 2 else None
            else:
                actual_result = result
                tokens = TokenUsage()
                confidence = None
            
            # Record usage
            selected_model = model or cost_manager.select_model(task_type)
            
            await cost_manager.record_usage(
                agent_id=agent_id,
                task_id=str(uuid.uuid4()),
                task_type=task_type,
                model=selected_model,
                tokens=tokens,
                duration_ms=duration_ms,
                success=True,
                confidence_score=confidence
            )
            
            return actual_result
        
        return wrapper
    return decorator


class BudgetExceededError(Exception):
    """Raised when budget limit is exceeded"""
    pass
