# =============================================================================
# Swarm Infrastructure Package
# =============================================================================
"""
Multi-Agent News Video Production Swarm - Infrastructure Package

This package provides the core infrastructure for orchestrating multi-agent
systems that collaboratively produce news videos.

Components:
- Blackboard: Shared memory for agent state
- MessageBus: Communication layer between agents
- CostManager: Token usage tracking and budget control
- SwarmMonitor: Observability with Langfuse
- FailureRecovery: Resilience and retry mechanisms
- ChromaManager: Vector memory for semantic search
"""

from .blackboard import (
    Blackboard,
    BlackboardSection,
    ContentStatus,
    AgentRole,
    NewsItem,
    DebateSession,
    ProductionJob,
    AgentState,
    LockAcquisitionError,
)

from .message_bus import (
    MessageBus,
    Message,
    MessageType,
    MessagePriority,
    TaskMessage,
    Channels,
    RPCError,
)

from .cost_manager import (
    CostManager,
    TokenUsage,
    CostRecord,
    AgentBudget,
    BudgetStatus,
    LLMModel,
    LLMTier,
    MODEL_PRICING,
    TIER_MODELS,
    TASK_TIER_MAPPING,
    track_costs,
    BudgetExceededError,
)

from .monitoring import (
    SwarmMonitor,
    TraceType,
    SpanType,
    AgentDecision,
    ConsensusVote,
    ConsensusResult,
    RevisionRecord,
    LANGFUSE_DASHBOARD_CONFIG,
    LANGFUSE_SETUP_INSTRUCTIONS,
)

from .failure_recovery import (
    FailureRecoveryManager,
    FailureRecord,
    FailureType,
    RecoveryAction,
    Checkpoint,
    RetryPolicy,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitOpenError,
    VisualGenerationService,
    VisualGenerationError,
)

from .chroma_integration import (
    ChromaManager,
    EmbeddingModel,
    EMBEDDING_DIMENSIONS,
    COLLECTION_SCHEMAS,
    EmbeddingProvider,
)

from .swarm_infrastructure import (
    SwarmInfrastructure,
    SwarmConfig,
    with_infrastructure,
    with_retry_and_tracking,
)

__version__ = "1.0.0"
__all__ = [
    # Infrastructure
    "SwarmInfrastructure",
    "SwarmConfig",
    
    # Blackboard
    "Blackboard",
    "BlackboardSection",
    "ContentStatus",
    "AgentRole",
    "NewsItem",
    "DebateSession",
    "ProductionJob",
    "AgentState",
    "LockAcquisitionError",
    
    # Message Bus
    "MessageBus",
    "Message",
    "MessageType",
    "MessagePriority",
    "TaskMessage",
    "Channels",
    "RPCError",
    
    # Cost Management
    "CostManager",
    "TokenUsage",
    "CostRecord",
    "AgentBudget",
    "BudgetStatus",
    "LLMModel",
    "LLMTier",
    "MODEL_PRICING",
    "TIER_MODELS",
    "TASK_TIER_MAPPING",
    "track_costs",
    "BudgetExceededError",
    
    # Monitoring
    "SwarmMonitor",
    "TraceType",
    "SpanType",
    "AgentDecision",
    "ConsensusVote",
    "ConsensusResult",
    "RevisionRecord",
    "LANGFUSE_DASHBOARD_CONFIG",
    "LANGFUSE_SETUP_INSTRUCTIONS",
    
    # Failure Recovery
    "FailureRecoveryManager",
    "FailureRecord",
    "FailureType",
    "RecoveryAction",
    "Checkpoint",
    "RetryPolicy",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitOpenError",
    "VisualGenerationService",
    "VisualGenerationError",
    
    # ChromaDB
    "ChromaManager",
    "EmbeddingModel",
    "EMBEDDING_DIMENSIONS",
    "COLLECTION_SCHEMAS",
    "EmbeddingProvider",
    
    # Decorators
    "with_infrastructure",
    "with_retry_and_tracking",
]
