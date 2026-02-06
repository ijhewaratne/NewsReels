# =============================================================================
# BLACKBOARD ARCHITECTURE - Shared Memory System for Multi-Agent Swarm
# =============================================================================
"""
The Blackboard is the central shared memory where all agents read and write state.
It provides:
- Structured data sections for different pipeline stages
- Duplicate work detection via content hashing
- State transitions with locking mechanisms
- Integration with ChromaDB for vector-based memory
"""

import json
import hashlib
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import redis.asyncio as redis
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class BlackboardSection(str, Enum):
    """Main sections of the blackboard"""
    RAW_FEED = "raw_feed"           # Incoming news items
    DEBATE_QUEUE = "debate_queue"   # Items awaiting agent debate
    CONSENSUS = "consensus"         # Agreed-upon story angles
    PRODUCTION_FLOOR = "production_floor"  # Active video production
    REVIEW_QUEUE = "review_queue"   # Completed videos for review
    PUBLISHED = "published"         # Published content
    ARCHIVE = "archive"             # Historical data
    AGENT_STATE = "agent_state"     # Current agent statuses


class ContentStatus(str, Enum):
    """Status states for content items"""
    PENDING = "pending"
    PROCESSING = "processing"
    DEBATING = "debating"
    CONSENSUS_REACHED = "consensus_reached"
    PRODUCING = "producing"
    REVIEWING = "reviewing"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUBLISHED = "published"
    FAILED = "failed"
    ARCHIVED = "archived"


class AgentRole(str, Enum):
    """Agent roles in the swarm"""
    SCOUT = "scout"
    ANALYST = "analyst"
    DEBATER = "debater"
    SCRIPT_WRITER = "script_writer"
    VISUAL_DESIGNER = "visual_designer"
    VIDEO_EDITOR = "video_editor"
    REVIEWER = "reviewer"
    PUBLISHER = "publisher"


# =============================================================================
# DATA SCHEMAS
# =============================================================================

@dataclass
class NewsItem:
    """Schema for raw news feed items"""
    id: str
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    topics: List[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    credibility_score: float = 0.5
    content_hash: str = ""
    status: ContentStatus = ContentStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute content hash for duplicate detection"""
        content = f"{self.title}:{self.content[:500]}:{self.source}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert enums to strings
        data['status'] = self.status.value
        # Convert datetime to ISO format
        for field_name in ['published_at', 'created_at', 'updated_at']:
            if isinstance(data[field_name], datetime):
                data[field_name] = data[field_name].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsItem':
        # Convert string back to enum
        data['status'] = ContentStatus(data.get('status', 'pending'))
        # Convert ISO format back to datetime
        for field_name in ['published_at', 'created_at', 'updated_at']:
            if field_name in data and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name].replace('Z', '+00:00'))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DebateSession:
    """Schema for agent debate sessions"""
    id: str
    news_item_id: str
    topic: str
    agent_roles: List[AgentRole] = field(default_factory=list)
    arguments: List[Dict[str, Any]] = field(default_factory=list)
    votes: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    consensus_reached: bool = False
    winning_angle: Optional[str] = None
    status: ContentStatus = ContentStatus.DEBATING
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        data['agent_roles'] = [r.value for r in self.agent_roles]
        for field_name in ['started_at', 'completed_at']:
            if data[field_name] and isinstance(data[field_name], datetime):
                data[field_name] = data[field_name].isoformat()
        return data


@dataclass
class ProductionJob:
    """Schema for video production jobs"""
    id: str
    debate_session_id: str
    script: Dict[str, Any] = field(default_factory=dict)
    visual_plan: Dict[str, Any] = field(default_factory=dict)
    voice_over: Dict[str, Any] = field(default_factory=dict)
    video_segments: List[Dict[str, Any]] = field(default_factory=list)
    final_video_url: Optional[str] = None
    status: ContentStatus = ContentStatus.PRODUCING
    progress_percent: float = 0.0
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        for field_name in ['created_at', 'started_at', 'completed_at']:
            if data[field_name] and isinstance(data[field_name], datetime):
                data[field_name] = data[field_name].isoformat()
        return data


@dataclass
class AgentState:
    """Schema for tracking individual agent state"""
    agent_id: str
    role: AgentRole
    status: str = "idle"  # idle, working, error, paused
    current_task_id: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tokens_used: int = 0
    total_cost: float = 0.0
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    capabilities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['role'] = self.role.value
        data['last_heartbeat'] = self.last_heartbeat.isoformat()
        return data


# =============================================================================
# BLACKBOARD CLASS
# =============================================================================

class Blackboard:
    """
    Central shared memory for the multi-agent swarm.
    
    Uses Redis for fast in-memory storage and ChromaDB for vector-based
    semantic search and memory retrieval.
    """
    
    # Redis key prefixes for different sections
    KEY_PREFIXES = {
        BlackboardSection.RAW_FEED: "bb:feed",
        BlackboardSection.DEBATE_QUEUE: "bb:debate",
        BlackboardSection.CONSENSUS: "bb:consensus",
        BlackboardSection.PRODUCTION_FLOOR: "bb:prod",
        BlackboardSection.REVIEW_QUEUE: "bb:review",
        BlackboardSection.PUBLISHED: "bb:published",
        BlackboardSection.ARCHIVE: "bb:archive",
        BlackboardSection.AGENT_STATE: "bb:agent",
    }
    
    def __init__(
        self,
        redis_client: redis.Redis,
        chroma_client=None,
        lock_timeout: int = 300  # 5 minutes
    ):
        self.redis = redis_client
        self.chroma = chroma_client
        self.lock_timeout = lock_timeout
        self._local_cache: Dict[str, Any] = {}
        
    # ========================================================================
    # CORE OPERATIONS
    # ========================================================================
    
    async def write(
        self,
        section: BlackboardSection,
        item_id: str,
        data: Dict[str, Any],
        check_duplicate: bool = True
    ) -> bool:
        """
        Write data to the blackboard.
        
        Args:
            section: Which blackboard section to write to
            item_id: Unique identifier for the item
            data: The data to store
            check_duplicate: Whether to check for duplicates first
            
        Returns:
            True if write was successful, False if duplicate detected
        """
        key = f"{self.KEY_PREFIXES[section]}:{item_id}"
        
        # Check for duplicates if requested
        if check_duplicate and 'content_hash' in data:
            is_dup = await self._check_duplicate(section, data['content_hash'])
            if is_dup:
                logger.info(f"Duplicate detected for item {item_id}, skipping write")
                return False
        
        # Add metadata
        data['_bb_section'] = section.value
        data['_bb_updated_at'] = datetime.now(timezone.utc).isoformat()
        data['_bb_version'] = data.get('_bb_version', 0) + 1
        
        # Write to Redis
        await self.redis.set(key, json.dumps(data))
        
        # Add to section index
        await self.redis.sadd(f"{self.KEY_PREFIXES[section]}:index", item_id)
        
        # Add content hash to deduplication set if present
        if 'content_hash' in data:
            await self.redis.hset(
                f"{self.KEY_PREFIXES[section]}:hashes",
                data['content_hash'],
                item_id
            )
        
        logger.debug(f"Wrote item {item_id} to {section.value}")
        return True
    
    async def read(
        self,
        section: BlackboardSection,
        item_id: str
    ) -> Optional[Dict[str, Any]]:
        """Read data from the blackboard"""
        key = f"{self.KEY_PREFIXES[section]}:{item_id}"
        data = await self.redis.get(key)
        if data:
            return json.loads(data)
        return None
    
    async def read_all(
        self,
        section: BlackboardSection,
        status: Optional[ContentStatus] = None
    ) -> List[Dict[str, Any]]:
        """Read all items from a section, optionally filtered by status"""
        index_key = f"{self.KEY_PREFIXES[section]}:index"
        item_ids = await self.redis.smembers(index_key)
        
        items = []
        for item_id in item_ids:
            item = await self.read(section, item_id.decode() if isinstance(item_id, bytes) else item_id)
            if item:
                if status is None or item.get('status') == status.value:
                    items.append(item)
        
        return items
    
    async def update(
        self,
        section: BlackboardSection,
        item_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update specific fields of an existing item"""
        key = f"{self.KEY_PREFIXES[section]}:{item_id}"
        
        # Get existing data
        existing = await self.redis.get(key)
        if not existing:
            logger.warning(f"Cannot update non-existent item: {item_id}")
            return False
        
        data = json.loads(existing)
        data.update(updates)
        data['_bb_updated_at'] = datetime.now(timezone.utc).isoformat()
        data['_bb_version'] = data.get('_bb_version', 0) + 1
        
        await self.redis.set(key, json.dumps(data))
        logger.debug(f"Updated item {item_id} in {section.value}")
        return True
    
    async def delete(
        self,
        section: BlackboardSection,
        item_id: str
    ) -> bool:
        """Delete an item from the blackboard"""
        key = f"{self.KEY_PREFIXES[section]}:{item_id}"
        
        # Get content hash for cleanup
        data = await self.redis.get(key)
        if data:
            parsed = json.loads(data)
            if 'content_hash' in parsed:
                await self.redis.hdel(
                    f"{self.KEY_PREFIXES[section]}:hashes",
                    parsed['content_hash']
                )
        
        # Remove from index and delete
        await self.redis.srem(f"{self.KEY_PREFIXES[section]}:index", item_id)
        await self.redis.delete(key)
        
        logger.debug(f"Deleted item {item_id} from {section.value}")
        return True
    
    async def move(
        self,
        item_id: str,
        from_section: BlackboardSection,
        to_section: BlackboardSection,
        updates: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Move an item from one section to another"""
        # Read from source
        data = await self.read(from_section, item_id)
        if not data:
            logger.warning(f"Cannot move non-existent item: {item_id}")
            return False
        
        # Apply updates if provided
        if updates:
            data.update(updates)
        
        # Write to destination
        success = await self.write(to_section, item_id, data, check_duplicate=False)
        if success:
            # Delete from source
            await self.delete(from_section, item_id)
            logger.info(f"Moved item {item_id} from {from_section.value} to {to_section.value}")
        
        return success
    
    # ========================================================================
    # LOCKING MECHANISM
    # ========================================================================
    
    @asynccontextmanager
    async def acquire_lock(
        self,
        resource_id: str,
        agent_id: str,
        timeout: Optional[int] = None
    ):
        """
        Distributed lock for preventing concurrent modifications.
        
        Usage:
            async with blackboard.acquire_lock("item_123", "agent_1"):
                # Critical section - only one agent can execute this
                await blackboard.update(...)
        """
        lock_key = f"lock:{resource_id}"
        lock_value = f"{agent_id}:{datetime.now(timezone.utc).isoformat()}"
        timeout = timeout or self.lock_timeout
        
        # Try to acquire lock
        acquired = await self.redis.set(
            lock_key,
            lock_value,
            nx=True,  # Only set if not exists
            ex=timeout
        )
        
        if not acquired:
            # Check who owns the lock
            current = await self.redis.get(lock_key)
            raise LockAcquisitionError(f"Resource {resource_id} locked by {current}")
        
        try:
            logger.debug(f"Agent {agent_id} acquired lock on {resource_id}")
            yield lock_value
        finally:
            # Release lock (only if we still own it)
            current = await self.redis.get(lock_key)
            if current and current.decode() == lock_value:
                await self.redis.delete(lock_key)
                logger.debug(f"Agent {agent_id} released lock on {resource_id}")
    
    async def extend_lock(self, resource_id: str, additional_time: int = 60) -> bool:
        """Extend the expiration of an existing lock"""
        lock_key = f"lock:{resource_id}"
        return await self.redis.expire(lock_key, additional_time)
    
    # ========================================================================
    # DUPLICATE DETECTION
    # ========================================================================
    
    async def _check_duplicate(
        self,
        section: BlackboardSection,
        content_hash: str
    ) -> bool:
        """Check if content hash already exists in the section"""
        existing = await self.redis.hget(
            f"{self.KEY_PREFIXES[section]}:hashes",
            content_hash
        )
        return existing is not None
    
    async def find_similar(
        self,
        section: BlackboardSection,
        content: str,
        threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Find similar items using vector similarity (requires ChromaDB).
        
        Falls back to simple text matching if ChromaDB not available.
        """
        if self.chroma is None:
            # Fallback: simple keyword matching
            items = await self.read_all(section)
            similar = []
            content_words = set(content.lower().split())
            for item in items:
                item_content = item.get('content', item.get('title', ''))
                item_words = set(item_content.lower().split())
                similarity = len(content_words & item_words) / len(content_words | item_words)
                if similarity >= threshold:
                    similar.append({**item, '_similarity': similarity})
            return sorted(similar, key=lambda x: x['_similarity'], reverse=True)
        
        # Use ChromaDB for semantic similarity
        # This would query the vector collection
        # Implementation depends on specific ChromaDB setup
        return []
    
    # ========================================================================
    # STATE TRANSITIONS
    # ========================================================================
    
    async def transition_status(
        self,
        section: BlackboardSection,
        item_id: str,
        new_status: ContentStatus,
        agent_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Transition an item to a new status with validation.
        
        Valid transitions:
        - PENDING -> PROCESSING
        - PROCESSING -> DEBATING
        - DEBATING -> CONSENSUS_REACHED or FAILED
        - CONSENSUS_REACHED -> PRODUCING
        - PRODUCING -> REVIEWING or FAILED
        - REVIEWING -> APPROVED or REJECTED
        - APPROVED -> PUBLISHED
        """
        VALID_TRANSITIONS = {
            ContentStatus.PENDING: [ContentStatus.PROCESSING],
            ContentStatus.PROCESSING: [ContentStatus.DEBATING, ContentStatus.FAILED],
            ContentStatus.DEBATING: [ContentStatus.CONSENSUS_REACHED, ContentStatus.FAILED],
            ContentStatus.CONSENSUS_REACHED: [ContentStatus.PRODUCING],
            ContentStatus.PRODUCING: [ContentStatus.REVIEWING, ContentStatus.FAILED],
            ContentStatus.REVIEWING: [ContentStatus.APPROVED, ContentStatus.REJECTED],
            ContentStatus.APPROVED: [ContentStatus.PUBLISHED],
            ContentStatus.REJECTED: [ContentStatus.ARCHIVED],
            ContentStatus.FAILED: [ContentStatus.PROCESSING, ContentStatus.ARCHIVED],
        }
        
        async with self.acquire_lock(item_id, agent_id):
            item = await self.read(section, item_id)
            if not item:
                return False
            
            current_status = ContentStatus(item.get('status', 'pending'))
            
            # Validate transition
            if new_status not in VALID_TRANSITIONS.get(current_status, []):
                logger.warning(
                    f"Invalid status transition: {current_status.value} -> {new_status.value}"
                )
                return False
            
            # Record transition
            transition_record = {
                'from': current_status.value,
                'to': new_status.value,
                'by': agent_id,
                'at': datetime.now(timezone.utc).isoformat(),
                'reason': reason
            }
            
            updates = {
                'status': new_status.value,
                '_transitions': item.get('_transitions', []) + [transition_record]
            }
            
            return await self.update(section, item_id, updates)
    
    # ========================================================================
    # AGENT STATE MANAGEMENT
    # ========================================================================
    
    async def register_agent(self, agent_state: AgentState) -> bool:
        """Register an agent with the blackboard"""
        return await self.write(
            BlackboardSection.AGENT_STATE,
            agent_state.agent_id,
            agent_state.to_dict(),
            check_duplicate=False
        )
    
    async def update_agent_state(
        self,
        agent_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update an agent's state"""
        return await self.update(
            BlackboardSection.AGENT_STATE,
            agent_id,
            updates
        )
    
    async def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get an agent's current state"""
        return await self.read(BlackboardSection.AGENT_STATE, agent_id)
    
    async def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get all registered agents"""
        return await self.read_all(BlackboardSection.AGENT_STATE)
    
    async def heartbeat(self, agent_id: str) -> bool:
        """Update agent's last heartbeat timestamp"""
        return await self.update_agent_state(agent_id, {
            'last_heartbeat': datetime.now(timezone.utc).isoformat()
        })
    
    # ========================================================================
    # QUERY AND SEARCH
    # ========================================================================
    
    async def query_by_status(
        self,
        section: BlackboardSection,
        status: ContentStatus,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query items by status"""
        items = await self.read_all(section, status)
        return items[:limit]
    
    async def query_by_agent(
        self,
        agent_id: str,
        section: Optional[BlackboardSection] = None
    ) -> List[Dict[str, Any]]:
        """Query items assigned to or modified by a specific agent"""
        sections = [section] if section else list(BlackboardSection)
        results = []
        
        for sec in sections:
            items = await self.read_all(sec)
            for item in items:
                transitions = item.get('_transitions', [])
                if any(t.get('by') == agent_id for t in transitions):
                    results.append(item)
        
        return results
    
    # ========================================================================
    # STATISTICS AND METRICS
    # ========================================================================
    
    async def get_section_stats(self, section: BlackboardSection) -> Dict[str, Any]:
        """Get statistics for a blackboard section"""
        items = await self.read_all(section)
        
        status_counts = {}
        for item in items:
            status = item.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'section': section.value,
            'total_items': len(items),
            'status_breakdown': status_counts,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
    
    async def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all sections"""
        stats = {}
        for section in BlackboardSection:
            stats[section.value] = await self.get_section_stats(section)
        return stats


class LockAcquisitionError(Exception):
    """Raised when unable to acquire a distributed lock"""
    pass
