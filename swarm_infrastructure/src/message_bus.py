# =============================================================================
# REDIS COMMUNICATION LAYER - Message Bus for Multi-Agent Swarm
# =============================================================================
"""
The MessageBus provides fast, reliable communication between agents using Redis.

Features:
- Pub/Sub channels for different swarm tiers
- Request-reply patterns for synchronous operations
- Message queuing for work distribution
- Priority messaging for urgent tasks
- Message persistence and retry logic
"""

import json
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Coroutine
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# MESSAGE SCHEMAS
# =============================================================================

class MessageType(str, Enum):
    """Types of messages in the system"""
    # Task-related
    TASK_ASSIGN = "task.assign"
    TASK_COMPLETE = "task.complete"
    TASK_FAILED = "task.failed"
    TASK_CANCEL = "task.cancel"
    
    # Debate-related
    DEBATE_INVITE = "debate.invite"
    DEBATE_ARGUMENT = "debate.argument"
    DEBATE_VOTE = "debate.vote"
    DEBATE_CONSENSUS = "debate.consensus"
    
    # Production-related
    PROD_START = "prod.start"
    PROD_PROGRESS = "prod.progress"
    PROD_COMPLETE = "prod.complete"
    PROD_FAILED = "prod.failed"
    
    # System-related
    HEARTBEAT = "system.heartbeat"
    AGENT_REGISTER = "system.agent_register"
    AGENT_UNREGISTER = "system.agent_unregister"
    AGENT_STATUS = "system.agent_status"
    
    # Control-related
    PAUSE = "control.pause"
    RESUME = "control.resume"
    SHUTDOWN = "control.shutdown"
    EMERGENCY = "control.emergency"
    
    # Request-reply
    REQUEST = "rpc.request"
    RESPONSE = "rpc.response"


class MessagePriority(int, Enum):
    """Message priority levels"""
    CRITICAL = 0    # Emergency shutdown, failures
    HIGH = 1        # Urgent tasks, consensus votes
    NORMAL = 2      # Regular tasks
    LOW = 3         # Background tasks, logging


@dataclass
class Message:
    """Base message schema"""
    id: str
    type: MessageType
    sender: str
    recipient: Optional[str]  # None = broadcast
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = None
    correlation_id: Optional[str] = None  # For request-reply
    reply_to: Optional[str] = None  # Reply channel for RPC
    ttl: int = 300  # Time to live in seconds
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['type'] = self.type.value
        data['priority'] = self.priority.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def create(
        cls,
        msg_type: MessageType,
        sender: str,
        payload: Dict[str, Any],
        recipient: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None
    ) -> 'Message':
        """Factory method to create a new message"""
        return cls(
            id=str(uuid.uuid4()),
            type=msg_type,
            sender=sender,
            recipient=recipient,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id,
            reply_to=reply_to
        )


@dataclass
class TaskMessage:
    """Specialized message for task assignment"""
    task_id: str
    task_type: str
    parameters: Dict[str, Any]
    deadline: Optional[datetime] = None
    estimated_tokens: int = 0
    max_retries: int = 3
    
    def to_message(
        self,
        sender: str,
        recipient: str,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> Message:
        return Message.create(
            msg_type=MessageType.TASK_ASSIGN,
            sender=sender,
            recipient=recipient,
            payload={
                'task_id': self.task_id,
                'task_type': self.task_type,
                'parameters': self.parameters,
                'deadline': self.deadline.isoformat() if self.deadline else None,
                'estimated_tokens': self.estimated_tokens,
                'max_retries': self.max_retries
            },
            priority=priority
        )


# =============================================================================
# CHANNEL DEFINITIONS
# =============================================================================

class Channels:
    """
    Redis channel definitions for different communication patterns.
    
    Channel Hierarchy:
    - swarm.broadcast.*: All agents receive
    - swarm.tier.{tier}.*: Agents in specific tier
    - swarm.agent.{agent_id}: Direct agent messaging
    - swarm.rpc.{service}: Request-reply pattern
    - swarm.work.{queue}: Work queues for load balancing
    """
    
    # Broadcast channels
    BROADCAST_ALL = "swarm.broadcast.all"
    BROADCAST_CONTROL = "swarm.broadcast.control"
    BROADCAST_EMERGENCY = "swarm.broadcast.emergency"
    
    # Tier-specific channels
    TIER_SCOUT = "swarm.tier.scout"
    TIER_ANALYST = "swarm.tier.analyst"
    TIER_DEBATER = "swarm.tier.debater"
    TIER_PRODUCER = "swarm.tier.producer"
    TIER_REVIEWER = "swarm.tier.reviewer"
    
    # Work queues
    QUEUE_DEBATE = "swarm.work.debate"
    QUEUE_PRODUCTION = "swarm.work.production"
    QUEUE_REVIEW = "swarm.work.review"
    
    # RPC channels
    RPC_BLACKBOARD = "swarm.rpc.blackboard"
    RPC_COST_TRACKER = "swarm.rpc.cost_tracker"
    RPC_MONITOR = "swarm.rpc.monitor"
    
    @classmethod
    def agent_channel(cls, agent_id: str) -> str:
        """Get direct channel for an agent"""
        return f"swarm.agent.{agent_id}"
    
    @classmethod
    def tier_channel(cls, tier: str) -> str:
        """Get channel for a tier"""
        return f"swarm.tier.{tier}"


# =============================================================================
# MESSAGE BUS CLASS
# =============================================================================

class MessageBus:
    """
    Central message bus for agent communication.
    
    Provides:
    - Pub/Sub for real-time communication
    - Work queues for task distribution
    - Request-reply for synchronous operations
    - Message persistence and retry
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        agent_id: Optional[str] = None,
        default_timeout: int = 30
    ):
        self.redis = redis_client
        self.agent_id = agent_id
        self.default_timeout = default_timeout
        self._pubsub = None
        self._handlers: Dict[MessageType, List[Callable]] = {}
        self._rpc_handlers: Dict[str, Callable] = {}
        self._pending_rpcs: Dict[str, asyncio.Future] = {}
        self._running = False
        self._listener_task = None
        
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    async def start(self):
        """Start the message bus listener"""
        if self._running:
            return
        
        self._pubsub = self.redis.pubsub()
        self._running = True
        
        # Subscribe to default channels
        await self._subscribe_default_channels()
        
        # Start listener
        self._listener_task = asyncio.create_task(self._listen())
        logger.info(f"MessageBus started for agent {self.agent_id}")
    
    async def stop(self):
        """Stop the message bus"""
        self._running = False
        
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        
        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()
        
        logger.info(f"MessageBus stopped for agent {self.agent_id}")
    
    async def _subscribe_default_channels(self):
        """Subscribe to default channels"""
        channels = [
            Channels.BROADCAST_ALL,
            Channels.BROADCAST_CONTROL,
            Channels.BROADCAST_EMERGENCY,
        ]
        
        if self.agent_id:
            channels.append(Channels.agent_channel(self.agent_id))
        
        await self._pubsub.subscribe(*channels)
        logger.debug(f"Subscribed to channels: {channels}")
    
    # ========================================================================
    # PUBLISH METHODS
    # ========================================================================
    
    async def publish(
        self,
        message: Message,
        channel: Optional[str] = None
    ) -> int:
        """
        Publish a message to a channel.
        
        Returns number of receivers.
        """
        if channel is None:
            channel = self._determine_channel(message)
        
        # Serialize message
        data = json.dumps(message.to_dict())
        
        # Publish
        receivers = await self.redis.publish(channel, data)
        
        # Also store in history for replay/debugging
        await self._store_message_history(message)
        
        logger.debug(f"Published {message.type.value} to {channel}, {receivers} receivers")
        return receivers
    
    async def broadcast(
        self,
        message: Message,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> int:
        """Broadcast to all agents"""
        if priority == MessagePriority.CRITICAL:
            channel = Channels.BROADCAST_EMERGENCY
        elif priority == MessagePriority.HIGH:
            channel = Channels.BROADCAST_CONTROL
        else:
            channel = Channels.BROADCAST_ALL
        
        return await self.publish(message, channel)
    
    async def send_to_agent(
        self,
        agent_id: str,
        message: Message
    ) -> int:
        """Send message to specific agent"""
        channel = Channels.agent_channel(agent_id)
        message.recipient = agent_id
        return await self.publish(message, channel)
    
    async def send_to_tier(
        self,
        tier: str,
        message: Message
    ) -> int:
        """Send message to all agents in a tier"""
        channel = Channels.tier_channel(tier)
        return await self.publish(message, channel)
    
    def _determine_channel(self, message: Message) -> str:
        """Determine appropriate channel for a message"""
        if message.recipient:
            return Channels.agent_channel(message.recipient)
        
        # Determine by message type
        if message.type in [MessageType.DEBATE_INVITE, MessageType.DEBATE_ARGUMENT, 
                           MessageType.DEBATE_VOTE, MessageType.DEBATE_CONSENSUS]:
            return Channels.TIER_DEBATER
        elif message.type in [MessageType.PROD_START, MessageType.PROD_PROGRESS, 
                             MessageType.PROD_COMPLETE, MessageType.PROD_FAILED]:
            return Channels.TIER_PRODUCER
        elif message.type in [MessageType.EMERGENCY, MessageType.SHUTDOWN]:
            return Channels.BROADCAST_EMERGENCY
        elif message.type in [MessageType.PAUSE, MessageType.RESUME]:
            return Channels.BROADCAST_CONTROL
        
        return Channels.BROADCAST_ALL
    
    # ========================================================================
    # WORK QUEUES
    # ========================================================================
    
    async def enqueue_task(
        self,
        queue: str,
        task: TaskMessage,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """
        Add task to a work queue.
        
        Uses Redis sorted sets for priority queuing.
        """
        message = task.to_message(
            sender=self.agent_id or "system",
            recipient=None,
            priority=priority
        )
        
        # Use priority score (lower = higher priority)
        score = priority.value
        
        # Add to sorted set
        await self.redis.zadd(
            queue,
            {json.dumps(message.to_dict()): score}
        )
        
        # Publish notification
        await self.redis.publish(f"{queue}:notify", "new_task")
        
        logger.debug(f"Enqueued task {task.task_id} to {queue}")
        return True
    
    async def dequeue_task(
        self,
        queue: str,
        timeout: int = 5
    ) -> Optional[TaskMessage]:
        """
        Get next task from work queue.
        
        Blocks until task available or timeout.
        """
        # First, check for notification
        notification = await self.redis.brpop(
            f"{queue}:notify",
            timeout=timeout
        )
        
        if notification is None:
            # No notification, check queue directly
            pass
        
        # Get highest priority task
        tasks = await self.redis.zrange(queue, 0, 0)
        
        if not tasks:
            return None
        
        # Remove from queue
        task_data = tasks[0]
        await self.redis.zrem(queue, task_data)
        
        # Parse task
        message_dict = json.loads(task_data)
        payload = message_dict.get('payload', {})
        
        return TaskMessage(
            task_id=payload.get('task_id'),
            task_type=payload.get('task_type'),
            parameters=payload.get('parameters', {}),
            deadline=datetime.fromisoformat(payload['deadline']) if payload.get('deadline') else None,
            estimated_tokens=payload.get('estimated_tokens', 0),
            max_retries=payload.get('max_retries', 3)
        )
    
    async def get_queue_length(self, queue: str) -> int:
        """Get number of tasks in queue"""
        return await self.redis.zcard(queue)
    
    async def peek_queue(
        self,
        queue: str,
        count: int = 10
    ) -> List[Dict[str, Any]]:
        """Peek at tasks in queue without removing"""
        tasks = await self.redis.zrange(queue, 0, count - 1)
        return [json.loads(t) for t in tasks]
    
    # ========================================================================
    # REQUEST-REPLY (RPC)
    # ========================================================================
    
    async def call(
        self,
        service: str,
        method: str,
        params: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Make synchronous RPC call to a service.
        
        Usage:
            result = await message_bus.call(
                "cost_tracker",
                "get_budget",
                {"agent_id": "agent_1"}
            )
        """
        timeout = timeout or self.default_timeout
        correlation_id = str(uuid.uuid4())
        reply_channel = f"swarm.rpc.reply.{self.agent_id}.{correlation_id}"
        
        # Create future for response
        future = asyncio.Future()
        self._pending_rpcs[correlation_id] = future
        
        # Subscribe to reply channel
        await self._pubsub.subscribe(reply_channel)
        
        try:
            # Send request
            request = Message.create(
                msg_type=MessageType.REQUEST,
                sender=self.agent_id or "anonymous",
                recipient=service,
                payload={
                    'method': method,
                    'params': params
                },
                correlation_id=correlation_id,
                reply_to=reply_channel
            )
            
            await self.publish(request, f"swarm.rpc.{service}")
            
            # Wait for response
            response = await asyncio.wait_for(future, timeout=timeout)
            
            if response.get('error'):
                raise RPCError(response['error'])
            
            return response.get('result', {})
            
        finally:
            # Cleanup
            del self._pending_rpcs[correlation_id]
            await self._pubsub.unsubscribe(reply_channel)
    
    def register_rpc_handler(
        self,
        service: str,
        handler: Callable[[str, Dict[str, Any]], Coroutine]
    ):
        """
        Register handler for RPC requests.
        
        Usage:
            async def handle_cost_request(method, params):
                if method == "get_budget":
                    return {"budget": 100.0}
            
            message_bus.register_rpc_handler("cost_tracker", handle_cost_request)
        """
        self._rpc_handlers[service] = handler
        
        # Subscribe to service channel
        asyncio.create_task(
            self._pubsub.subscribe(f"swarm.rpc.{service}")
        )
        
        logger.info(f"Registered RPC handler for service: {service}")
    
    async def _handle_rpc_request(self, message: Message):
        """Handle incoming RPC request"""
        service = message.recipient
        handler = self._rpc_handlers.get(service)
        
        if not handler:
            logger.warning(f"No handler for RPC service: {service}")
            return
        
        payload = message.payload
        method = payload.get('method')
        params = payload.get('params', {})
        
        try:
            result = await handler(method, params)
            response_payload = {'result': result}
        except Exception as e:
            response_payload = {'error': str(e)}
        
        # Send response
        response = Message.create(
            msg_type=MessageType.RESPONSE,
            sender=service,
            recipient=message.sender,
            payload=response_payload,
            correlation_id=message.correlation_id
        )
        
        await self.publish(response, message.reply_to)
    
    # ========================================================================
    # MESSAGE HANDLERS
    # ========================================================================
    
    def on(
        self,
        msg_type: MessageType,
        handler: Callable[[Message], Coroutine]
    ):
        """Register handler for message type"""
        if msg_type not in self._handlers:
            self._handlers[msg_type] = []
        self._handlers[msg_type].append(handler)
        logger.debug(f"Registered handler for {msg_type.value}")
    
    def off(
        self,
        msg_type: MessageType,
        handler: Callable[[Message], Coroutine]
    ):
        """Unregister handler"""
        if msg_type in self._handlers:
            self._handlers[msg_type] = [
                h for h in self._handlers[msg_type] if h != handler
            ]
    
    # ========================================================================
    # LISTENER
    # ========================================================================
    
    async def _listen(self):
        """Main message listener loop"""
        while self._running:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0
                )
                
                if message is None:
                    continue
                
                # Parse message
                data = json.loads(message['data'])
                msg = self._dict_to_message(data)
                
                # Handle RPC responses
                if msg.type == MessageType.RESPONSE and msg.correlation_id:
                    future = self._pending_rpcs.get(msg.correlation_id)
                    if future and not future.done():
                        future.set_result(msg.payload)
                    continue
                
                # Handle RPC requests
                if msg.type == MessageType.REQUEST:
                    asyncio.create_task(self._handle_rpc_request(msg))
                    continue
                
                # Dispatch to handlers
                handlers = self._handlers.get(msg.type, [])
                for handler in handlers:
                    asyncio.create_task(handler(msg))
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message listener: {e}")
    
    def _dict_to_message(self, data: Dict[str, Any]) -> Message:
        """Convert dict to Message object"""
        return Message(
            id=data['id'],
            type=MessageType(data['type']),
            sender=data['sender'],
            recipient=data.get('recipient'),
            payload=data.get('payload', {}),
            priority=MessagePriority(data.get('priority', 2)),
            timestamp=datetime.fromisoformat(data['timestamp']),
            correlation_id=data.get('correlation_id'),
            reply_to=data.get('reply_to'),
            ttl=data.get('ttl', 300)
        )
    
    # ========================================================================
    # MESSAGE HISTORY
    # ========================================================================
    
    async def _store_message_history(self, message: Message):
        """Store message in history for debugging/replay"""
        history_key = f"swarm:history:{message.type.value}"
        
        # Add to list with timestamp score
        score = message.timestamp.timestamp()
        data = json.dumps(message.to_dict())
        
        await self.redis.zadd(history_key, {data: score})
        
        # Trim to last 1000 messages per type
        await self.redis.zremrangebyrank(history_key, 0, -1001)
    
    async def get_message_history(
        self,
        msg_type: Optional[MessageType] = None,
        count: int = 100
    ) -> List[Message]:
        """Get recent message history"""
        if msg_type:
            history_key = f"swarm:history:{msg_type.value}"
            data_list = await self.redis.zrevrange(history_key, 0, count - 1)
        else:
            # Get from all types
            all_data = []
            for mt in MessageType:
                key = f"swarm:history:{mt.value}"
                data = await self.redis.zrevrange(key, 0, count - 1)
                all_data.extend(data)
            data_list = sorted(all_data, reverse=True)[:count]
        
        return [self._dict_to_message(json.loads(d)) for d in data_list]


class RPCError(Exception):
    """Raised when RPC call fails"""
    pass
