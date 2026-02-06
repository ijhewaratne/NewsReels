# =============================================================================
# SWARM INFRASTRUCTURE - Main Integration Module
# =============================================================================
"""
Main integration module that ties together all infrastructure components:
- Blackboard (shared memory)
- MessageBus (communication)
- CostManager (budget control)
- SwarmMonitor (observability)
- FailureRecovery (resilience)
- ChromaManager (vector memory)

Provides a unified interface for the multi-agent swarm.
"""

import os
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import redis.asyncio as redis
import logging

# Import infrastructure components
from blackboard import Blackboard, BlackboardSection, ContentStatus
from message_bus import MessageBus, Message, MessageType, Channels
from cost_manager import CostManager, TokenUsage, LLMModel, LLMTier
from monitoring import SwarmMonitor, TraceType, AgentDecision, ConsensusVote
from failure_recovery import FailureRecoveryManager, RetryPolicy
from chroma_integration import ChromaManager

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SwarmConfig:
    """Configuration for swarm infrastructure"""
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_auth_token: Optional[str] = None
    
    # Langfuse
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "http://localhost:3000"
    langfuse_enabled: bool = True
    
    # Cost Management
    global_daily_budget: float = 1000.0
    enable_cost_alerts: bool = True
    
    # Monitoring
    monitoring_sample_rate: float = 1.0
    
    # Failure Recovery
    default_max_retries: int = 3
    circuit_breaker_threshold: int = 5
    
    @classmethod
    def from_env(cls) -> 'SwarmConfig':
        """Create config from environment variables"""
        return cls(
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            chroma_host=os.getenv("CHROMA_HOST", "localhost"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            chroma_auth_token=os.getenv("CHROMA_AUTH_TOKEN"),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            langfuse_host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
            langfuse_enabled=os.getenv("LANGFUSE_ENABLED", "true").lower() == "true",
            global_daily_budget=float(os.getenv("GLOBAL_DAILY_BUDGET", "1000.0")),
            enable_cost_alerts=os.getenv("ENABLE_COST_ALERTS", "true").lower() == "true",
            monitoring_sample_rate=float(os.getenv("MONITORING_SAMPLE_RATE", "1.0")),
            default_max_retries=int(os.getenv("DEFAULT_MAX_RETRIES", "3")),
            circuit_breaker_threshold=int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")),
        )


# =============================================================================
# SWARM INFRASTRUCTURE CLASS
# =============================================================================

class SwarmInfrastructure:
    """
    Unified infrastructure interface for the multi-agent swarm.
    
    This is the main entry point for all infrastructure operations.
    Agents interact with the swarm through this class.
    """
    
    def __init__(self, config: Optional[SwarmConfig] = None):
        self.config = config or SwarmConfig.from_env()
        
        # Components (initialized in connect())
        self.redis: Optional[redis.Redis] = None
        self.blackboard: Optional[Blackboard] = None
        self.message_bus: Optional[MessageBus] = None
        self.cost_manager: Optional[CostManager] = None
        self.monitor: Optional[SwarmMonitor] = None
        self.failure_recovery: Optional[FailureRecoveryManager] = None
        self.chroma: Optional[ChromaManager] = None
        
        self._connected = False
        
    async def connect(self):
        """Connect to all infrastructure components"""
        if self._connected:
            return
        
        logger.info("Connecting to swarm infrastructure...")
        
        # Connect to Redis
        await self._connect_redis()
        
        # Initialize components
        await self._init_blackboard()
        await self._init_message_bus()
        await self._init_cost_manager()
        await self._init_monitor()
        await self._init_failure_recovery()
        await self._init_chroma()
        
        self._connected = True
        logger.info("Swarm infrastructure connected successfully")
    
    async def disconnect(self):
        """Disconnect from all components"""
        if not self._connected:
            return
        
        logger.info("Disconnecting from swarm infrastructure...")
        
        if self.message_bus:
            await self.message_bus.stop()
        
        if self.redis:
            await self.redis.close()
        
        self._connected = False
        logger.info("Swarm infrastructure disconnected")
    
    async def _connect_redis(self):
        """Connect to Redis"""
        self.redis = redis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            password=self.config.redis_password,
            db=self.config.redis_db,
            decode_responses=True
        )
        
        # Test connection
        await self.redis.ping()
        logger.info(f"Connected to Redis at {self.config.redis_host}:{self.config.redis_port}")
    
    async def _init_blackboard(self):
        """Initialize blackboard"""
        self.blackboard = Blackboard(
            redis_client=self.redis,
            chroma_client=self.chroma,
            lock_timeout=300
        )
        logger.info("Blackboard initialized")
    
    async def _init_message_bus(self):
        """Initialize message bus"""
        self.message_bus = MessageBus(
            redis_client=self.redis,
            default_timeout=30
        )
        await self.message_bus.start()
        logger.info("Message bus initialized")
    
    async def _init_cost_manager(self):
        """Initialize cost manager"""
        self.cost_manager = CostManager(
            redis_client=self.redis,
            global_daily_budget=self.config.global_daily_budget,
            enable_alerts=self.config.enable_cost_alerts
        )
        logger.info("Cost manager initialized")
    
    async def _init_monitor(self):
        """Initialize monitoring"""
        self.monitor = SwarmMonitor(
            redis_client=self.redis,
            enabled=self.config.langfuse_enabled,
            sample_rate=self.config.monitoring_sample_rate
        )
        logger.info("Monitor initialized")
    
    async def _init_failure_recovery(self):
        """Initialize failure recovery"""
        self.failure_recovery = FailureRecoveryManager(
            redis_client=self.redis,
            message_bus=self.message_bus,
            default_retry_policy=RetryPolicy(
                max_retries=self.config.default_max_retries
            )
        )
        logger.info("Failure recovery initialized")
    
    async def _init_chroma(self):
        """Initialize ChromaDB"""
        self.chroma = ChromaManager(
            host=self.config.chroma_host,
            port=self.config.chroma_port,
            auth_token=self.config.chroma_auth_token
        )
        await self.chroma.connect()
        logger.info("ChromaDB initialized")
    
    # ========================================================================
    # AGENT REGISTRATION
    # ========================================================================
    
    async def register_agent(
        self,
        agent_id: str,
        role: str,
        capabilities: List[str],
        budget: Optional[Dict[str, float]] = None
    ):
        """
        Register an agent with the swarm.
        
        Args:
            agent_id: Unique agent identifier
            role: Agent role (scout, analyst, etc.)
            capabilities: List of agent capabilities
            budget: Optional budget configuration
        """
        from blackboard import AgentState, AgentRole
        
        # Register with blackboard
        agent_state = AgentState(
            agent_id=agent_id,
            role=AgentRole(role),
            status="idle",
            capabilities=capabilities
        )
        await self.blackboard.register_agent(agent_state)
        
        # Set budget if provided
        if budget:
            from cost_manager import AgentBudget
            agent_budget = AgentBudget(
                agent_id=agent_id,
                **budget
            )
            await self.cost_manager.set_budget(agent_budget)
        
        # Subscribe to agent channel
        await self.message_bus._pubsub.subscribe(Channels.agent_channel(agent_id))
        
        logger.info(f"Registered agent: {agent_id} ({role})")
    
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        await self.blackboard.update_agent_state(agent_id, {"status": "offline"})
        await self.message_bus._pubsub.unsubscribe(Channels.agent_channel(agent_id))
        logger.info(f"Unregistered agent: {agent_id}")
    
    # ========================================================================
    # AGENT OPERATIONS
    # ========================================================================
    
    async def agent_heartbeat(self, agent_id: str):
        """Send agent heartbeat"""
        await self.blackboard.heartbeat(agent_id)
    
    async def claim_task(
        self,
        agent_id: str,
        section: BlackboardSection,
        item_id: str
    ) -> bool:
        """
        Claim a task for processing.
        
        Returns True if claim successful.
        """
        try:
            async with self.blackboard.acquire_lock(item_id, agent_id):
                item = await self.blackboard.read(section, item_id)
                if not item:
                    return False
                
                if item.get('status') != ContentStatus.PENDING.value:
                    return False
                
                # Update status
                await self.blackboard.update(section, item_id, {
                    'status': ContentStatus.PROCESSING.value,
                    'claimed_by': agent_id,
                    'claimed_at': datetime.now(timezone.utc).isoformat()
                })
                
                # Update agent state
                await self.blackboard.update_agent_state(agent_id, {
                    'status': 'working',
                    'current_task_id': item_id
                })
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to claim task: {e}")
            return False
    
    async def complete_task(
        self,
        agent_id: str,
        section: BlackboardSection,
        item_id: str,
        result: Dict[str, Any],
        move_to: Optional[BlackboardSection] = None
    ):
        """Mark task as complete and optionally move to next section"""
        # Update item
        await self.blackboard.update(section, item_id, {
            'status': ContentStatus.CONSENSUS_REACHED.value if move_to else ContentStatus.APPROVED.value,
            'result': result,
            'completed_by': agent_id,
            'completed_at': datetime.now(timezone.utc).isoformat()
        })
        
        # Move if specified
        if move_to:
            await self.blackboard.move(item_id, section, move_to)
        
        # Update agent state
        await self.blackboard.update_agent_state(agent_id, {
            'status': 'idle',
            'current_task_id': None,
            'tasks_completed': (await self.blackboard.get_agent_state(agent_id) or {}).get('tasks_completed', 0) + 1
        })
    
    async def fail_task(
        self,
        agent_id: str,
        section: BlackboardSection,
        item_id: str,
        error: Exception
    ):
        """Mark task as failed"""
        # Record failure
        await self.failure_recovery.record_failure(
            task_id=item_id,
            agent_id=agent_id,
            exception=error
        )
        
        # Update item status
        await self.blackboard.transition_status(
            section=section,
            item_id=item_id,
            new_status=ContentStatus.FAILED,
            agent_id=agent_id,
            reason=str(error)
        )
        
        # Update agent state
        await self.blackboard.update_agent_state(agent_id, {
            'status': 'idle',
            'current_task_id': None,
            'tasks_failed': (await self.blackboard.get_agent_state(agent_id) or {}).get('tasks_failed', 0) + 1
        })
    
    # ========================================================================
    # COST TRACKING
    # ========================================================================
    
    async def track_llm_usage(
        self,
        agent_id: str,
        task_id: str,
        task_type: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: int,
        confidence: Optional[float] = None
    ):
        """Track LLM usage for cost management"""
        tokens = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        
        await self.cost_manager.record_usage(
            agent_id=agent_id,
            task_id=task_id,
            task_type=task_type,
            model=LLMModel(model),
            tokens=tokens,
            duration_ms=duration_ms,
            confidence_score=confidence
        )
    
    async def check_budget(self, agent_id: str) -> Dict[str, Any]:
        """Check agent's budget status"""
        return (await self.cost_manager.get_budget_status(agent_id)).to_dict()
    
    # ========================================================================
    # MONITORING
    # ========================================================================
    
    def start_trace(
        self,
        trace_type: TraceType,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a monitoring trace"""
        return self.monitor.start_trace(trace_type, trace_id, metadata)
    
    def end_trace(self, trace_id: str, status: str = "success"):
        """End a monitoring trace"""
        self.monitor.end_trace(trace_id, status)
    
    def track_decision(
        self,
        trace_id: str,
        agent_id: str,
        role: str,
        decision_type: str,
        input_context: Dict[str, Any],
        output: Dict[str, Any],
        confidence: float,
        reasoning: str
    ):
        """Track an agent decision"""
        decision = AgentDecision(
            agent_id=agent_id,
            agent_role=role,
            decision_type=decision_type,
            input_context=input_context,
            output_decision=output,
            confidence=confidence,
            reasoning=reasoning,
            alternatives_considered=[]
        )
        self.monitor.track_agent_decision(trace_id, decision)
    
    # ========================================================================
    # CHECKPOINTING
    # ========================================================================
    
    async def create_checkpoint(
        self,
        task_id: str,
        stage: str,
        data: Dict[str, Any],
        agent_states: Dict[str, Any]
    ):
        """Create a pipeline checkpoint"""
        await self.failure_recovery.create_checkpoint(
            task_id=task_id,
            stage=stage,
            data=data,
            agent_states=agent_states
        )
    
    async def resume_from_checkpoint(
        self,
        task_id: str
    ) -> Optional[Dict[str, Any]]:
        """Resume from checkpoint"""
        checkpoint = await self.failure_recovery.resume_from_checkpoint(task_id)
        return checkpoint.to_dict() if checkpoint else None
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get infrastructure statistics"""
        stats = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'connected': self._connected,
        }
        
        if self.blackboard:
            stats['blackboard'] = await self.blackboard.get_all_stats()
        
        if self.chroma:
            stats['chroma'] = await self.chroma.get_all_stats()
        
        if self.failure_recovery:
            stats['health'] = await self.failure_recovery.get_health_status()
        
        if self.monitor:
            stats['monitoring'] = await self.monitor.get_dashboard_metrics()
        
        return stats
    
    # ========================================================================
    # CONTEXT MANAGER
    # ========================================================================
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


# =============================================================================
# DECORATORS
# =============================================================================

def with_infrastructure(swarm_infra: SwarmInfrastructure):
    """
    Decorator to inject infrastructure into agent methods.
    
    Usage:
        @with_infrastructure(swarm)
        async def process_news(agent_id, item_id, infra):
            # infra is SwarmInfrastructure instance
            await infra.claim_task(agent_id, BlackboardSection.RAW_FEED, item_id)
    """
    def decorator(func):
        async def wrapper(agent_id: str, *args, **kwargs):
            return await func(agent_id, *args, infrastructure=swarm_infra, **kwargs)
        return wrapper
    return decorator


def with_retry_and_tracking(
    swarm_infra: SwarmInfrastructure,
    task_type: str,
    trace_type: TraceType = TraceType.AGENT_DECISION
):
    """
    Decorator combining retry, cost tracking, and monitoring.
    
    Usage:
        @with_retry_and_tracking(swarm, "script_generation")
        async def generate_script(agent_id, news_item, infra):
            # Automatically tracked and retried
            return await llm.generate(prompt)
    """
    def decorator(func):
        async def wrapper(agent_id: str, *args, **kwargs):
            task_id = f"{agent_id}:{task_type}:{int(time.time())}"
            
            # Start trace
            trace_id = swarm_infra.start_trace(trace_type, metadata={
                'agent_id': agent_id,
                'task_type': task_type
            })
            
            try:
                # Execute with retry
                retry_decorator = swarm_infra.failure_recovery.with_retry()
                result = await retry_decorator(func)(agent_id, *args, **kwargs)
                
                # End trace successfully
                swarm_infra.end_trace(trace_id, "success")
                
                return result
                
            except Exception as e:
                # End trace with error
                swarm_infra.end_trace(trace_id, "error")
                raise
                
        return wrapper
    return decorator


# Import datetime
from datetime import datetime, timezone
import time
