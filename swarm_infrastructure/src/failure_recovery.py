# =============================================================================
# FAILURE RECOVERY SYSTEM - Resilience for Multi-Agent Swarm
# =============================================================================
"""
The FailureRecovery system handles agent failures, service outages, and
pipeline interruptions with automatic retry, fallback, and checkpointing.

Features:
- Automatic retry with exponential backoff
- Alternative agent spawning on failure
- Checkpoint and resume capabilities
- Circuit breaker pattern for external services
- Health monitoring and self-healing
"""

import json
import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Coroutine, Type
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager
import redis.asyncio as redis
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# FAILURE SCHEMAS
# =============================================================================

class FailureType(str, Enum):
    """Types of failures"""
    AGENT_CRASH = "agent_crash"
    AGENT_TIMEOUT = "agent_timeout"
    LLM_ERROR = "llm_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RATE_LIMIT = "rate_limit"
    VALIDATION_FAILED = "validation_failed"
    VISUAL_GEN_FAILED = "visual_gen_failed"
    VOICE_GEN_FAILED = "voice_gen_failed"
    VIDEO_RENDER_FAILED = "video_render_failed"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


class RecoveryAction(str, Enum):
    """Recovery actions"""
    RETRY = "retry"
    FALLBACK_AGENT = "fallback_agent"
    SKIP_TASK = "skip_task"
    ESCALATE = "escalate"
    ARCHIVE = "archive"
    MANUAL_REVIEW = "manual_review"


@dataclass
class FailureRecord:
    """Record of a failure"""
    id: str
    task_id: str
    agent_id: str
    failure_type: FailureType
    error_message: str
    stack_trace: Optional[str]
    timestamp: datetime
    retry_count: int = 0
    recovery_action: Optional[RecoveryAction] = None
    recovered: bool = False
    recovered_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['failure_type'] = self.failure_type.value
        data['recovery_action'] = self.recovery_action.value if self.recovery_action else None
        data['timestamp'] = self.timestamp.isoformat()
        data['recovered_at'] = self.recovered_at.isoformat() if self.recovered_at else None
        return data


@dataclass
class Checkpoint:
    """Pipeline checkpoint for resumption"""
    id: str
    task_id: str
    stage: str
    data: Dict[str, Any]
    agent_states: Dict[str, Any]
    timestamp: datetime
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['expires_at'] = self.expires_at.isoformat() if self.expires_at else None
        return data


@dataclass
class RetryPolicy:
    """Retry policy configuration"""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    retryable_failures: List[FailureType] = None
    
    def __post_init__(self):
        if self.retryable_failures is None:
            self.retryable_failures = [
                FailureType.LLM_ERROR,
                FailureType.SERVICE_UNAVAILABLE,
                FailureType.RATE_LIMIT,
                FailureType.NETWORK_ERROR,
            ]


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    half_open_max_calls: int = 3


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker pattern for external services.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Service considered down, fast fail
    - HALF_OPEN: Testing if service recovered
    """
    
    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half_open"
    
    def __init__(
        self,
        service_name: str,
        redis_client: redis.Redis,
        config: CircuitBreakerConfig = None
    ):
        self.service_name = service_name
        self.redis = redis_client
        self.config = config or CircuitBreakerConfig()
        self._state = self.STATE_CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        
    @property
    def state(self) -> str:
        """Get current circuit state"""
        return self._state
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Raises CircuitOpenError if circuit is open.
        """
        await self._update_state()
        
        if self._state == self.STATE_OPEN:
            raise CircuitOpenError(f"Circuit open for {self.service_name}")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _update_state(self):
        """Update circuit state based on time"""
        if self._state == self.STATE_OPEN:
            if self._last_failure_time:
                elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
                if elapsed >= self.config.recovery_timeout:
                    self._state = self.STATE_HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"Circuit {self.service_name} entering half-open state")
    
    async def _on_success(self):
        """Handle successful call"""
        if self._state == self.STATE_HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls >= self.config.half_open_max_calls:
                self._state = self.STATE_CLOSED
                self._failure_count = 0
                logger.info(f"Circuit {self.service_name} closed")
        else:
            self._failure_count = max(0, self._failure_count - 1)
    
    async def _on_failure(self):
        """Handle failed call"""
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)
        
        if self._state == self.STATE_HALF_OPEN:
            self._state = self.STATE_OPEN
            logger.warning(f"Circuit {self.service_name} opened (half-open failure)")
        elif self._failure_count >= self.config.failure_threshold:
            self._state = self.STATE_OPEN
            logger.warning(
                f"Circuit {self.service_name} opened ({self._failure_count} failures)"
            )


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


# =============================================================================
# FAILURE RECOVERY MANAGER
# =============================================================================

class FailureRecoveryManager:
    """
    Central failure recovery for the swarm.
    
    Handles:
    - Retry with exponential backoff
    - Alternative agent spawning
    - Checkpoint management
    - Circuit breakers for external services
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        message_bus=None,
        default_retry_policy: RetryPolicy = None
    ):
        self.redis = redis_client
        self.message_bus = message_bus
        self.default_retry_policy = default_retry_policy or RetryPolicy()
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._recovery_handlers: Dict[FailureType, Callable] = {}
        self._fallback_agents: Dict[str, List[str]] = {}
        
    # ========================================================================
    # CIRCUIT BREAKER MANAGEMENT
    # ========================================================================
    
    def get_circuit_breaker(
        self,
        service_name: str,
        config: CircuitBreakerConfig = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker for service"""
        if service_name not in self._circuit_breakers:
            self._circuit_breakers[service_name] = CircuitBreaker(
                service_name=service_name,
                redis_client=self.redis,
                config=config
            )
        return self._circuit_breakers[service_name]
    
    # ========================================================================
    # RETRY DECORATOR
    # ========================================================================
    
    def with_retry(
        self,
        policy: RetryPolicy = None,
        on_retry: Callable[[int, Exception], None] = None,
        on_failure: Callable[[Exception], None] = None
    ):
        """
        Decorator for automatic retry with exponential backoff.
        
        Usage:
            @failure_recovery.with_retry()
            async def generate_visual(agent_id, prompt):
                # May raise exceptions
                return await visual_gen.generate(prompt)
        """
        policy = policy or self.default_retry_policy
        
        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(policy.max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        failure_type = self._classify_failure(e)
                        
                        # Check if failure is retryable
                        if failure_type not in policy.retryable_failures:
                            logger.warning(f"Non-retryable failure: {failure_type}")
                            break
                        
                        if attempt < policy.max_retries:
                            delay = self._calculate_delay(policy, attempt)
                            logger.info(
                                f"Retry {attempt + 1}/{policy.max_retries} "
                                f"after {delay:.1f}s: {e}"
                            )
                            
                            if on_retry:
                                if asyncio.iscoroutinefunction(on_retry):
                                    await on_retry(attempt, e)
                                else:
                                    on_retry(attempt, e)
                            
                            await asyncio.sleep(delay)
                
                # All retries exhausted
                if on_failure:
                    if asyncio.iscoroutinefunction(on_failure):
                        await on_failure(last_exception)
                    else:
                        on_failure(last_exception)
                
                raise last_exception
            
            return wrapper
        return decorator
    
    def _calculate_delay(self, policy: RetryPolicy, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        delay = policy.base_delay * (policy.exponential_base ** attempt)
        return min(delay, policy.max_delay)
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify exception into failure type"""
        error_str = str(exception).lower()
        exception_type = type(exception).__name__.lower()
        
        if "timeout" in error_str or "timeout" in exception_type:
            return FailureType.AGENT_TIMEOUT
        elif "rate limit" in error_str or "rate_limit" in error_str:
            return FailureType.RATE_LIMIT
        elif "visual" in error_str or "image" in error_str:
            return FailureType.VISUAL_GEN_FAILED
        elif "voice" in error_str or "audio" in error_str:
            return FailureType.VOICE_GEN_FAILED
        elif "video" in error_str or "render" in error_str:
            return FailureType.VIDEO_RENDER_FAILED
        elif "network" in error_str or "connection" in error_str:
            return FailureType.NETWORK_ERROR
        elif "validation" in error_str:
            return FailureType.VALIDATION_FAILED
        elif "llm" in error_str or "openai" in error_str or "anthropic" in error_str:
            return FailureType.LLM_ERROR
        else:
            return FailureType.UNKNOWN
    
    # ========================================================================
    # FALLBACK AGENTS
    # ========================================================================
    
    def register_fallback(
        self,
        agent_role: str,
        fallback_agents: List[str]
    ):
        """
        Register fallback agents for a role.
        
        Usage:
            failure_recovery.register_fallback(
                "visual_designer",
                ["visual_designer_backup", "visual_designer_simple"]
            )
        """
        self._fallback_agents[agent_role] = fallback_agents
        logger.info(f"Registered {len(fallback_agents)} fallbacks for {agent_role}")
    
    async def spawn_fallback_agent(
        self,
        failed_agent_id: str,
        agent_role: str,
        task_id: str,
        task_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Spawn a fallback agent when primary fails.
        
        Returns ID of spawned agent or None if no fallback available.
        """
        fallbacks = self._fallback_agents.get(agent_role, [])
        
        for fallback_id in fallbacks:
            # Check if fallback is available
            is_available = await self._check_agent_available(fallback_id)
            
            if is_available:
                logger.info(
                    f"Spawning fallback agent {fallback_id} for failed "
                    f"agent {failed_agent_id} on task {task_id}"
                )
                
                # Notify via message bus
                if self.message_bus:
                    from message_bus import Message, MessageType
                    
                    message = Message.create(
                        msg_type=MessageType.TASK_ASSIGN,
                        sender="failure_recovery",
                        recipient=fallback_id,
                        payload={
                            'task_id': task_id,
                            'task_data': task_data,
                            'is_fallback': True,
                            'original_agent': failed_agent_id
                        }
                    )
                    await self.message_bus.send_to_agent(fallback_id, message)
                
                return fallback_id
        
        logger.error(f"No fallback agents available for role {agent_role}")
        return None
    
    async def _check_agent_available(self, agent_id: str) -> bool:
        """Check if agent is available"""
        # Check Redis for agent state
        state_key = f"bb:agent:{agent_id}"
        state_data = await self.redis.get(state_key)
        
        if not state_data:
            # Agent not registered, assume available
            return True
        
        state = json.loads(state_data)
        return state.get('status') == 'idle'
    
    # ========================================================================
    # CHECKPOINT MANAGEMENT
    # ========================================================================
    
    async def create_checkpoint(
        self,
        task_id: str,
        stage: str,
        data: Dict[str, Any],
        agent_states: Dict[str, Any],
        ttl_hours: int = 24
    ) -> Checkpoint:
        """
        Create a checkpoint for resumption.
        
        Usage:
            checkpoint = await failure_recovery.create_checkpoint(
                task_id="video_123",
                stage="script_written",
                data={"script": script_content},
                agent_states={"writer": writer_state}
            )
        """
        checkpoint = Checkpoint(
            id=f"chk:{task_id}:{stage}:{int(time.time())}",
            task_id=task_id,
            stage=stage,
            data=data,
            agent_states=agent_states,
            timestamp=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
        )
        
        # Store in Redis
        key = f"recovery:checkpoint:{task_id}"
        await self.redis.hset(key, stage, json.dumps(checkpoint.to_dict()))
        await self.redis.expire(key, ttl_hours * 3600)
        
        logger.debug(f"Created checkpoint {checkpoint.id} for task {task_id}")
        return checkpoint
    
    async def get_checkpoint(
        self,
        task_id: str,
        stage: Optional[str] = None
    ) -> Optional[Checkpoint]:
        """Get checkpoint for a task"""
        key = f"recovery:checkpoint:{task_id}"
        
        if stage:
            data = await self.redis.hget(key, stage)
            if data:
                return self._dict_to_checkpoint(json.loads(data))
        else:
            # Get latest checkpoint
            stages = await self.redis.hkeys(key)
            if not stages:
                return None
            
            # Find most recent
            latest = None
            latest_time = 0
            
            for stage_key in stages:
                stage_str = stage_key.decode() if isinstance(stage_key, bytes) else stage_key
                data = await self.redis.hget(key, stage_str)
                if data:
                    checkpoint_dict = json.loads(data)
                    checkpoint_time = datetime.fromisoformat(
                        checkpoint_dict['timestamp'].replace('Z', '+00:00')
                    ).timestamp()
                    if checkpoint_time > latest_time:
                        latest_time = checkpoint_time
                        latest = checkpoint_dict
            
            if latest:
                return self._dict_to_checkpoint(latest)
        
        return None
    
    async def list_checkpoints(self, task_id: str) -> List[Checkpoint]:
        """List all checkpoints for a task"""
        key = f"recovery:checkpoint:{task_id}"
        all_data = await self.redis.hgetall(key)
        
        checkpoints = []
        for stage, data in all_data.items():
            checkpoints.append(self._dict_to_checkpoint(json.loads(data)))
        
        return sorted(checkpoints, key=lambda c: c.timestamp, reverse=True)
    
    def _dict_to_checkpoint(self, data: Dict[str, Any]) -> Checkpoint:
        """Convert dict to Checkpoint"""
        return Checkpoint(
            id=data['id'],
            task_id=data['task_id'],
            stage=data['stage'],
            data=data['data'],
            agent_states=data['agent_states'],
            timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
            expires_at=datetime.fromisoformat(data['expires_at'].replace('Z', '+00:00')) if data.get('expires_at') else None
        )
    
    async def resume_from_checkpoint(
        self,
        task_id: str,
        stage: Optional[str] = None
    ) -> Optional[Checkpoint]:
        """
        Resume task from checkpoint.
        
        Returns checkpoint data or None if no checkpoint exists.
        """
        checkpoint = await self.get_checkpoint(task_id, stage)
        
        if not checkpoint:
            logger.warning(f"No checkpoint found for task {task_id}")
            return None
        
        # Check if expired
        if checkpoint.expires_at and datetime.now(timezone.utc) > checkpoint.expires_at:
            logger.warning(f"Checkpoint for task {task_id} has expired")
            return None
        
        logger.info(f"Resuming task {task_id} from checkpoint at stage {checkpoint.stage}")
        
        # Notify via message bus
        if self.message_bus:
            from message_bus import Message, MessageType
            
            message = Message.create(
                msg_type=MessageType.TASK_ASSIGN,
                sender="failure_recovery",
                recipient=None,
                payload={
                    'task_id': task_id,
                    'checkpoint': checkpoint.to_dict(),
                    'is_resume': True
                }
            )
            await self.message_bus.broadcast(message)
        
        return checkpoint
    
    # ========================================================================
    # FAILURE RECORDING
    # ========================================================================
    
    async def record_failure(
        self,
        task_id: str,
        agent_id: str,
        exception: Exception,
        recovery_action: Optional[RecoveryAction] = None
    ) -> FailureRecord:
        """Record a failure"""
        failure_type = self._classify_failure(exception)
        
        # Get retry count
        retry_key = f"recovery:retries:{task_id}:{agent_id}"
        retry_count = int(await self.redis.get(retry_key) or 0)
        
        record = FailureRecord(
            id=f"fail:{task_id}:{int(time.time())}",
            task_id=task_id,
            agent_id=agent_id,
            failure_type=failure_type,
            error_message=str(exception),
            stack_trace=None,  # Could capture traceback here
            timestamp=datetime.now(timezone.utc),
            retry_count=retry_count,
            recovery_action=recovery_action
        )
        
        # Store in Redis
        key = f"recovery:failures:{task_id}"
        await self.redis.lpush(key, json.dumps(record.to_dict()))
        await self.redis.expire(key, 7 * 24 * 3600)
        
        # Increment retry count
        await self.redis.incr(retry_key)
        await self.redis.expire(retry_key, 24 * 3600)
        
        logger.warning(
            f"Recorded failure for task {task_id}, agent {agent_id}: "
            f"{failure_type.value}"
        )
        
        return record
    
    async def mark_recovered(
        self,
        task_id: str,
        failure_id: str
    ):
        """Mark a failure as recovered"""
        key = f"recovery:failures:{task_id}"
        failures = await self.redis.lrange(key, 0, -1)
        
        for i, data in enumerate(failures):
            failure_dict = json.loads(data)
            if failure_dict['id'] == failure_id:
                failure_dict['recovered'] = True
                failure_dict['recovered_at'] = datetime.now(timezone.utc).isoformat()
                await self.redis.lset(key, i, json.dumps(failure_dict))
                break
    
    async def get_failure_history(
        self,
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        failure_type: Optional[FailureType] = None
    ) -> List[FailureRecord]:
        """Get failure history"""
        records = []
        
        if task_id:
            key = f"recovery:failures:{task_id}"
            data_list = await self.redis.lrange(key, 0, -1)
            for data in data_list:
                records.append(self._dict_to_failure_record(json.loads(data)))
        else:
            # Get all failures
            keys = await self.redis.keys("recovery:failures:*")
            for key in keys:
                data_list = await self.redis.lrange(key, 0, -1)
                for data in data_list:
                    record = self._dict_to_failure_record(json.loads(data))
                    if agent_id and record.agent_id != agent_id:
                        continue
                    if failure_type and record.failure_type != failure_type:
                        continue
                    records.append(record)
        
        return sorted(records, key=lambda r: r.timestamp, reverse=True)
    
    def _dict_to_failure_record(self, data: Dict[str, Any]) -> FailureRecord:
        """Convert dict to FailureRecord"""
        return FailureRecord(
            id=data['id'],
            task_id=data['task_id'],
            agent_id=data['agent_id'],
            failure_type=FailureType(data['failure_type']),
            error_message=data['error_message'],
            stack_trace=data.get('stack_trace'),
            timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
            retry_count=data.get('retry_count', 0),
            recovery_action=RecoveryAction(data['recovery_action']) if data.get('recovery_action') else None,
            recovered=data.get('recovered', False),
            recovered_at=datetime.fromisoformat(data['recovered_at'].replace('Z', '+00:00')) if data.get('recovered_at') else None
        )
    
    # ========================================================================
    # RECOVERY HANDLERS
    # ========================================================================
    
    def register_recovery_handler(
        self,
        failure_type: FailureType,
        handler: Callable[[FailureRecord], Coroutine]
    ):
        """Register custom recovery handler for failure type"""
        self._recovery_handlers[failure_type] = handler
    
    async def handle_failure(
        self,
        task_id: str,
        agent_id: str,
        exception: Exception,
        agent_role: Optional[str] = None
    ) -> RecoveryAction:
        """
        Handle a failure with appropriate recovery action.
        
        Returns the recovery action taken.
        """
        failure_type = self._classify_failure(exception)
        
        # Record failure
        record = await self.record_failure(task_id, agent_id, exception)
        
        # Check for custom handler
        handler = self._recovery_handlers.get(failure_type)
        if handler:
            try:
                await handler(record)
            except Exception as e:
                logger.error(f"Recovery handler failed: {e}")
        
        # Determine recovery action
        action = await self._determine_recovery_action(record, agent_role)
        
        # Execute recovery
        if action == RecoveryAction.RETRY:
            # Retry is handled by decorator
            pass
        elif action == RecoveryAction.FALLBACK_AGENT:
            if agent_role:
                await self.spawn_fallback_agent(agent_id, agent_role, task_id, {})
        elif action == RecoveryAction.ESCALATE:
            await self._escalate_failure(record)
        elif action == RecoveryAction.ARCHIVE:
            await self._archive_failed_task(task_id)
        
        # Update record with action
        record.recovery_action = action
        
        return action
    
    async def _determine_recovery_action(
        self,
        record: FailureRecord,
        agent_role: Optional[str]
    ) -> RecoveryAction:
        """Determine appropriate recovery action"""
        # Check retry count
        if record.retry_count < self.default_retry_policy.max_retries:
            if record.failure_type in self.default_retry_policy.retryable_failures:
                return RecoveryAction.RETRY
        
        # Try fallback agent
        if agent_role and agent_role in self._fallback_agents:
            return RecoveryAction.FALLBACK_AGENT
        
        # Critical failures escalate
        if record.failure_type in [
            FailureType.VIDEO_RENDER_FAILED,
            FailureType.VALIDATION_FAILED
        ]:
            return RecoveryAction.ESCALATE
        
        # Default to archive
        return RecoveryAction.ARCHIVE
    
    async def _escalate_failure(self, record: FailureRecord):
        """Escalate failure for manual review"""
        logger.error(f"Escalating failure: {record.id}")
        
        # Store in escalation queue
        key = "recovery:escalated"
        await self.redis.lpush(key, json.dumps(record.to_dict()))
        
        # Notify via message bus
        if self.message_bus:
            from message_bus import Message, MessageType, MessagePriority
            
            message = Message.create(
                msg_type=MessageType.TASK_FAILED,
                sender="failure_recovery",
                recipient=None,
                payload={
                    'failure_record': record.to_dict(),
                    'requires_manual_review': True
                },
                priority=MessagePriority.HIGH
            )
            await self.message_bus.broadcast(message)
    
    async def _archive_failed_task(self, task_id: str):
        """Archive a failed task"""
        logger.info(f"Archiving failed task: {task_id}")
        
        # Move to archive (would integrate with blackboard)
        # This is a placeholder - actual implementation would
        # update blackboard section
        pass
    
    # ========================================================================
    # VISUAL GENERATION FAILURE HANDLING
    # ========================================================================
    
    async def handle_visual_gen_failure(
        self,
        task_id: str,
        agent_id: str,
        prompt: str,
        failure_reason: str
    ) -> Dict[str, Any]:
        """
        Handle visual generation failure with fallback strategies.
        
        Strategies:
        1. Retry with simplified prompt
        2. Use alternative provider
        3. Use stock image
        4. Skip visual
        """
        logger.warning(f"Visual generation failed for task {task_id}: {failure_reason}")
        
        # Strategy 1: Simplified prompt
        simplified_prompt = self._simplify_visual_prompt(prompt)
        
        return {
            'fallback_strategy': 'simplified_prompt',
            'original_prompt': prompt,
            'simplified_prompt': simplified_prompt,
            'can_retry': True
        }
    
    def _simplify_visual_prompt(self, prompt: str) -> str:
        """Simplify a visual generation prompt"""
        # Remove complex instructions
        # Keep main subject and style
        words = prompt.split()
        if len(words) > 20:
            return ' '.join(words[:20]) + "..."
        return prompt
    
    # ========================================================================
    # HEALTH MONITORING
    # ========================================================================
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        status = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'circuit_breakers': {},
            'recent_failures': 0,
            'pending_recoveries': 0
        }
        
        # Circuit breaker states
        for name, cb in self._circuit_breakers.items():
            status['circuit_breakers'][name] = {
                'state': cb.state,
                'failure_count': cb._failure_count
            }
        
        # Recent failures
        failure_keys = await self.redis.keys("recovery:failures:*")
        status['recent_failures'] = len(failure_keys)
        
        # Pending escalations
        status['pending_recoveries'] = await self.redis.llen("recovery:escalated")
        
        return status


# =============================================================================
# VISUAL GENERATION SERVICE WITH CIRCUIT BREAKER
# =============================================================================

class VisualGenerationService:
    """
    Visual generation service with circuit breaker protection.
    
    Supports multiple providers with automatic fallback.
    """
    
    def __init__(
        self,
        failure_recovery: FailureRecoveryManager,
        primary_provider: str = "runway",
        fallback_providers: List[str] = None
    ):
        self.failure_recovery = failure_recovery
        self.primary_provider = primary_provider
        self.fallback_providers = fallback_providers or ["heygen", "d-id"]
        
        # Create circuit breakers for each provider
        self.primary_cb = failure_recovery.get_circuit_breaker(primary_provider)
        self.fallback_cbs = {
            p: failure_recovery.get_circuit_breaker(p)
            for p in self.fallback_providers
        }
    
    async def generate_visual(
        self,
        prompt: str,
        style: str = "news",
        duration: int = 5
    ) -> Dict[str, Any]:
        """
        Generate visual with automatic fallback.
        
        Returns:
            Dict with 'url', 'provider', 'metadata'
        """
        # Try primary provider
        try:
            return await self.primary_cb.call(
                self._generate_with_provider,
                self.primary_provider,
                prompt,
                style,
                duration
            )
        except CircuitOpenError:
            logger.warning(f"Primary provider {self.primary_provider} circuit open")
        except Exception as e:
            logger.warning(f"Primary provider failed: {e}")
        
        # Try fallback providers
        for provider, cb in self.fallback_cbs.items():
            try:
                return await cb.call(
                    self._generate_with_provider,
                    provider,
                    prompt,
                    style,
                    duration
                )
            except Exception as e:
                logger.warning(f"Fallback provider {provider} failed: {e}")
                continue
        
        # All providers failed
        raise VisualGenerationError("All visual generation providers failed")
    
    async def _generate_with_provider(
        self,
        provider: str,
        prompt: str,
        style: str,
        duration: int
    ) -> Dict[str, Any]:
        """Generate visual with specific provider"""
        # This would integrate with actual provider APIs
        # Placeholder implementation
        
        if provider == "runway":
            # RunwayML integration
            pass
        elif provider == "heygen":
            # HeyGen integration
            pass
        elif provider == "d-id":
            # D-ID integration
            pass
        
        # Return mock response
        return {
            'url': f'https://example.com/visual/{provider}/generated.mp4',
            'provider': provider,
            'metadata': {
                'prompt': prompt,
                'style': style,
                'duration': duration
            }
        }


class VisualGenerationError(Exception):
    """Raised when visual generation fails"""
    pass
