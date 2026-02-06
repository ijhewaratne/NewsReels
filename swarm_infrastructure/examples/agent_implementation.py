# =============================================================================
# AGENT IMPLEMENTATION EXAMPLE - Using Swarm Infrastructure
# =============================================================================
"""
This example shows how to implement agents that use the swarm infrastructure.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from swarm_infrastructure import SwarmInfrastructure
from blackboard import BlackboardSection, ContentStatus, NewsItem
from message_bus import Message, MessageType
from cost_manager import LLMModel, LLMTier
from monitoring import TraceType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# BASE AGENT CLASS
# =============================================================================

class BaseAgent(ABC):
    """Base class for all swarm agents"""
    
    def __init__(
        self,
        agent_id: str,
        role: str,
        swarm: SwarmInfrastructure,
        capabilities: List[str] = None
    ):
        self.agent_id = agent_id
        self.role = role
        self.swarm = swarm
        self.capabilities = capabilities or []
        self._running = False
        self._task_handlers: Dict[str, callable] = {}
        
    async def register(self):
        """Register agent with swarm"""
        await self.swarm.register_agent(
            agent_id=self.agent_id,
            role=self.role,
            capabilities=self.capabilities
        )
        logger.info(f"Agent {self.agent_id} registered")
    
    async def unregister(self):
        """Unregister agent from swarm"""
        await self.swarm.unregister_agent(self.agent_id)
        logger.info(f"Agent {self.agent_id} unregistered")
    
    def register_task_handler(self, task_type: str, handler: callable):
        """Register handler for task type"""
        self._task_handlers[task_type] = handler
    
    async def start(self):
        """Start agent main loop"""
        self._running = True
        logger.info(f"Agent {self.agent_id} started")
        
        # Subscribe to messages
        self.swarm.message_bus.on(
            MessageType.TASK_ASSIGN,
            self._handle_task_message
        )
        
        # Main loop
        while self._running:
            try:
                await self.swarm.agent_heartbeat(self.agent_id)
                await self._process_cycle()
                await asyncio.sleep(5)  # Process every 5 seconds
            except Exception as e:
                logger.error(f"Agent {self.agent_id} error: {e}")
                await asyncio.sleep(10)
    
    async def stop(self):
        """Stop agent"""
        self._running = False
        logger.info(f"Agent {self.agent_id} stopped")
    
    async def _handle_task_message(self, message: Message):
        """Handle incoming task message"""
        if message.recipient != self.agent_id:
            return
        
        payload = message.payload
        task_type = payload.get('task_type')
        
        handler = self._task_handlers.get(task_type)
        if handler:
            try:
                await handler(payload)
            except Exception as e:
                logger.error(f"Task handler error: {e}")
    
    @abstractmethod
    async def _process_cycle(self):
        """Main processing cycle - implement in subclass"""
        pass


# =============================================================================
# SCOUT AGENT
# =============================================================================

class ScoutAgent(BaseAgent):
    """
    Scout agent monitors news sources and adds items to blackboard.
    
    Uses FAST tier LLMs for initial screening.
    """
    
    def __init__(self, agent_id: str, swarm: SwarmInfrastructure):
        super().__init__(
            agent_id=agent_id,
            role="scout",
            swarm=swarm,
            capabilities=["news_monitoring", "source_evaluation", "content_filtering"]
        )
        self.sources = ["TechNews", "WorldNews", "FinanceDaily"]
    
    async def _process_cycle(self):
        """Monitor sources and add news items"""
        # Simulate finding news
        for source in self.sources:
            # In real implementation, this would fetch from RSS/API
            news_item = await self._fetch_news(source)
            if news_item:
                await self._process_news_item(news_item)
    
    async def _fetch_news(self, source: str) -> Optional[Dict]:
        """Fetch news from source (mock)"""
        # Mock implementation
        return {
            "title": f"Breaking news from {source}",
            "content": f"Important update from {source}...",
            "source": source,
            "url": f"https://{source.lower()}.com/article",
            "topics": ["news"]
        }
    
    async def _process_news_item(self, news_data: Dict):
        """Process and add news item to blackboard"""
        # Create news item
        item_id = f"news_{datetime.now(timezone.utc).timestamp()}"
        
        news_item = NewsItem(
            id=item_id,
            title=news_data["title"],
            content=news_data["content"],
            source=news_data["source"],
            url=news_data["url"],
            published_at=datetime.now(timezone.utc),
            topics=news_data["topics"]
        )
        
        # Add to blackboard
        success = await self.swarm.blackboard.write(
            section=BlackboardSection.RAW_FEED,
            item_id=item_id,
            data=news_item.to_dict()
        )
        
        if success:
            logger.info(f"Scout {self.agent_id} added news item: {item_id}")
            
            # Track cost (simulated)
            await self.swarm.track_llm_usage(
                agent_id=self.agent_id,
                task_id=item_id,
                task_type="news_scout",
                model=LLMModel.GPT35_TURBO.value,
                prompt_tokens=200,
                completion_tokens=50,
                duration_ms=500,
                confidence=0.8
            )


# =============================================================================
# ANALYST AGENT
# =============================================================================

class AnalystAgent(BaseAgent):
    """
    Analyst agent evaluates news items and makes coverage decisions.
    
    Uses BALANCED tier LLMs for analysis.
    """
    
    def __init__(self, agent_id: str, swarm: SwarmInfrastructure):
        super().__init__(
            agent_id=agent_id,
            role="analyst",
            swarm=swarm,
            capabilities=["content_analysis", "sentiment_analysis", "trend_detection"]
        )
    
    async def _process_cycle(self):
        """Process pending news items"""
        # Get pending items from raw feed
        items = await self.swarm.blackboard.query_by_status(
            section=BlackboardSection.RAW_FEED,
            status=ContentStatus.PENDING
        )
        
        for item in items[:3]:  # Process up to 3 at a time
            await self._analyze_item(item)
    
    async def _analyze_item(self, item: Dict):
        """Analyze a news item"""
        item_id = item["id"]
        
        # Claim the task
        claimed = await self.swarm.claim_task(
            agent_id=self.agent_id,
            section=BlackboardSection.RAW_FEED,
            item_id=item_id
        )
        
        if not claimed:
            return
        
        logger.info(f"Analyst {self.agent_id} analyzing: {item_id}")
        
        # Start trace
        trace_id = self.swarm.start_trace(
            trace_type=TraceType.AGENT_DECISION,
            metadata={"agent_id": self.agent_id, "task": "news_analysis"}
        )
        
        try:
            # Simulate analysis
            analysis_result = await self._perform_analysis(item)
            
            # Track decision
            self.swarm.track_decision(
                trace_id=trace_id,
                agent_id=self.agent_id,
                role="analyst",
                decision_type="coverage_decision",
                input_context={"news_item": item},
                output=analysis_result,
                confidence=analysis_result["confidence"],
                reasoning=analysis_result["reasoning"]
            )
            
            # Track cost
            await self.swarm.track_llm_usage(
                agent_id=self.agent_id,
                task_id=item_id,
                task_type="content_analysis",
                model=LLMModel.GPT4_TURBO.value,
                prompt_tokens=800,
                completion_tokens=300,
                duration_ms=2000,
                confidence=analysis_result["confidence"]
            )
            
            # Complete task
            if analysis_result["should_cover"]:
                await self.swarm.complete_task(
                    agent_id=self.agent_id,
                    section=BlackboardSection.RAW_FEED,
                    item_id=item_id,
                    result=analysis_result,
                    move_to=BlackboardSection.DEBATE_QUEUE
                )
                logger.info(f"Item {item_id} approved for coverage")
            else:
                await self.swarm.blackboard.transition_status(
                    section=BlackboardSection.RAW_FEED,
                    item_id=item_id,
                    new_status=ContentStatus.REJECTED,
                    agent_id=self.agent_id,
                    reason="Low priority or relevance"
                )
                logger.info(f"Item {item_id} rejected")
            
            self.swarm.end_trace(trace_id, "success")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            await self.swarm.fail_task(
                agent_id=self.agent_id,
                section=BlackboardSection.RAW_FEED,
                item_id=item_id,
                error=e
            )
            self.swarm.end_trace(trace_id, "error")
    
    async def _perform_analysis(self, item: Dict) -> Dict:
        """Perform content analysis (mock)"""
        # In real implementation, this would use LLM
        return {
            "should_cover": True,
            "priority": "high",
            "confidence": 0.85,
            "reasoning": "High relevance to current trends, credible source",
            "suggested_angle": "Impact on industry"
        }


# =============================================================================
# SCRIPT WRITER AGENT
# =============================================================================

class ScriptWriterAgent(BaseAgent):
    """
    Script writer agent generates video scripts.
    
    Uses PREMIUM tier LLMs for final scripts.
    """
    
    def __init__(self, agent_id: str, swarm: SwarmInfrastructure):
        super().__init__(
            agent_id=agent_id,
            role="script_writer",
            swarm=swarm,
            capabilities=["script_writing", "storytelling", "audience_engagement"]
        )
    
    async def _process_cycle(self):
        """Process items in debate queue that have consensus"""
        items = await self.swarm.blackboard.query_by_status(
            section=BlackboardSection.CONSENSUS,
            status=ContentStatus.CONSENSUS_REACHED
        )
        
        for item in items[:2]:
            await self._write_script(item)
    
    async def _write_script(self, item: Dict):
        """Write script for approved topic"""
        item_id = item["id"]
        
        claimed = await self.swarm.claim_task(
            agent_id=self.agent_id,
            section=BlackboardSection.CONSENSUS,
            item_id=item_id
        )
        
        if not claimed:
            return
        
        logger.info(f"Writer {self.agent_id} writing script for: {item_id}")
        
        # Create checkpoint
        await self.swarm.create_checkpoint(
            task_id=f"video_{item_id}",
            stage="script_started",
            data={"item": item},
            agent_states={self.agent_id: {"status": "writing"}}
        )
        
        trace_id = self.swarm.start_trace(
            TraceType.SCRIPT_GENERATION,
            metadata={"agent_id": self.agent_id, "item_id": item_id}
        )
        
        try:
            # Generate script
            script = await self._generate_script(item)
            
            # Track cost (premium model)
            await self.swarm.track_llm_usage(
                agent_id=self.agent_id,
                task_id=item_id,
                task_type="script_final",
                model=LLMModel.GPT4.value,
                prompt_tokens=1500,
                completion_tokens=800,
                duration_ms=5000,
                confidence=0.9
            )
            
            # Update checkpoint
            await self.swarm.create_checkpoint(
                task_id=f"video_{item_id}",
                stage="script_written",
                data={"script": script},
                agent_states={self.agent_id: {"status": "completed"}}
            )
            
            # Complete and move to production
            await self.swarm.complete_task(
                agent_id=self.agent_id,
                section=BlackboardSection.CONSENSUS,
                item_id=item_id,
                result={"script": script},
                move_to=BlackboardSection.PRODUCTION_FLOOR
            )
            
            self.swarm.end_trace(trace_id, "success")
            logger.info(f"Script written for {item_id}")
            
        except Exception as e:
            logger.error(f"Script writing failed: {e}")
            await self.swarm.fail_task(
                agent_id=self.agent_id,
                section=BlackboardSection.CONSENSUS,
                item_id=item_id,
                error=e
            )
            self.swarm.end_trace(trace_id, "error")
    
    async def _generate_script(self, item: Dict) -> Dict:
        """Generate video script (mock)"""
        return {
            "title": f"Video: {item.get('title', 'Breaking News')}",
            "content": "Opening hook...\n\nMain story...\n\nClosing...",
            "word_count": 200,
            "estimated_duration": 60,
            "scenes": [
                {"scene": 1, "visual": "News anchor intro", "duration": 5},
                {"scene": 2, "visual": "B-roll footage", "duration": 45},
                {"scene": 3, "visual": "Outro", "duration": 10}
            ]
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Run agent example"""
    
    # Initialize infrastructure
    swarm = SwarmInfrastructure()
    await swarm.connect()
    
    try:
        # Create agents
        scout = ScoutAgent("scout_1", swarm)
        analyst = AnalystAgent("analyst_1", swarm)
        writer = ScriptWriterAgent("writer_1", swarm)
        
        # Register agents
        await scout.register()
        await analyst.register()
        await writer.register()
        
        # Run agents concurrently for a short time
        logger.info("Starting agents...")
        
        # Run scout once
        await scout._process_cycle()
        
        # Give scout time to add items
        await asyncio.sleep(2)
        
        # Run analyst
        await analyst._process_cycle()
        
        # Give time for processing
        await asyncio.sleep(2)
        
        # Manually move an item to consensus for demo
        items = await swarm.blackboard.read_all(BlackboardSection.DEBATE_QUEUE)
        if items:
            await swarm.blackboard.update(
                BlackboardSection.DEBATE_QUEUE,
                items[0]["id"],
                {"status": ContentStatus.CONSENSUS_REACHED.value}
            )
            await swarm.blackboard.move(
                items[0]["id"],
                BlackboardSection.DEBATE_QUEUE,
                BlackboardSection.CONSENSUS
            )
        
        # Run writer
        await writer._process_cycle()
        
        # Get stats
        stats = await swarm.get_stats()
        logger.info(f"Infrastructure stats: {stats}")
        
        # Cleanup
        await scout.unregister()
        await analyst.unregister()
        await writer.unregister()
        
    finally:
        await swarm.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
