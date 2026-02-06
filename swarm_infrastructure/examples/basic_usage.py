# =============================================================================
# BASIC USAGE EXAMPLE - Multi-Agent Swarm Infrastructure
# =============================================================================
"""
This example demonstrates basic usage of the swarm infrastructure.
"""

import asyncio
import os
from datetime import datetime, timezone

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from swarm_infrastructure import SwarmInfrastructure, SwarmConfig
from blackboard import BlackboardSection, ContentStatus, NewsItem, AgentRole
from message_bus import Message, MessageType, Channels
from cost_manager import TokenUsage, LLMModel
from monitoring import TraceType, AgentDecision


async def main():
    """Main example"""
    
    # =============================================================================
    # 1. INITIALIZE INFRASTRUCTURE
    # =============================================================================
    
    print("=" * 60)
    print("1. INITIALIZING INFRASTRUCTURE")
    print("=" * 60)
    
    # Create config
    config = SwarmConfig(
        redis_host="localhost",
        redis_port=6379,
        redis_password=os.getenv("REDIS_PASSWORD"),
        chroma_host="localhost",
        chroma_port=8000,
        langfuse_enabled=True,
        global_daily_budget=500.0
    )
    
    # Create and connect infrastructure
    swarm = SwarmInfrastructure(config)
    
    try:
        await swarm.connect()
        print("✓ Infrastructure connected")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return
    
    # =============================================================================
    # 2. REGISTER AGENTS
    # =============================================================================
    
    print("\n" + "=" * 60)
    print("2. REGISTERING AGENTS")
    print("=" * 60)
    
    agents = [
        {
            "id": "scout_1",
            "role": "scout",
            "capabilities": ["news_monitoring", "source_evaluation"],
            "budget": {"daily_limit": 20.0, "hourly_limit": 5.0}
        },
        {
            "id": "analyst_1",
            "role": "analyst",
            "capabilities": ["content_analysis", "sentiment_analysis"],
            "budget": {"daily_limit": 50.0, "hourly_limit": 10.0}
        },
        {
            "id": "writer_1",
            "role": "script_writer",
            "capabilities": ["script_writing", "storytelling"],
            "budget": {"daily_limit": 100.0, "hourly_limit": 20.0}
        }
    ]
    
    for agent in agents:
        await swarm.register_agent(
            agent_id=agent["id"],
            role=agent["role"],
            capabilities=agent["capabilities"],
            budget=agent["budget"]
        )
        print(f"✓ Registered agent: {agent['id']} ({agent['role']})")
    
    # =============================================================================
    # 3. ADD NEWS ITEM TO BLACKBOARD
    # =============================================================================
    
    print("\n" + "=" * 60)
    print("3. ADDING NEWS ITEMS TO BLACKBOARD")
    print("=" * 60)
    
    news_items = [
        NewsItem(
            id="news_001",
            title="Major Tech Company Announces Breakthrough AI Model",
            content="In a surprising announcement today, TechCorp revealed their latest AI model "
                    "that promises to revolutionize natural language processing. The model, "
                    "called 'Neural-X', outperforms existing solutions by 40% on benchmark tests.",
            source="TechNews Daily",
            url="https://technews.com/ai-breakthrough",
            published_at=datetime.now(timezone.utc),
            topics=["AI", "Technology", "Innovation"],
            sentiment_score=0.7,
            credibility_score=0.85
        ),
        NewsItem(
            id="news_002",
            title="Global Climate Summit Reaches Historic Agreement",
            content="World leaders have reached a landmark agreement on climate action, "
                    "committing to reduce carbon emissions by 50% by 2030. The agreement "
                    "includes binding targets for all participating nations.",
            source="World News Network",
            url="https://worldnews.com/climate-agreement",
            published_at=datetime.now(timezone.utc),
            topics=["Climate", "Politics", "Environment"],
            sentiment_score=0.8,
            credibility_score=0.9
        )
    ]
    
    for item in news_items:
        success = await swarm.blackboard.write(
            section=BlackboardSection.RAW_FEED,
            item_id=item.id,
            data=item.to_dict()
        )
        if success:
            print(f"✓ Added news item: {item.id} - {item.title[:50]}...")
        else:
            print(f"⚠ Duplicate detected: {item.id}")
    
    # =============================================================================
    # 4. AGENT CLAIMS AND PROCESSES TASK
    # =============================================================================
    
    print("\n" + "=" * 60)
    print("4. AGENT CLAIMING AND PROCESSING TASK")
    print("=" * 60)
    
    # Scout claims a news item
    scout_id = "scout_1"
    news_id = "news_001"
    
    claimed = await swarm.claim_task(
        agent_id=scout_id,
        section=BlackboardSection.RAW_FEED,
        item_id=news_id
    )
    
    if claimed:
        print(f"✓ {scout_id} claimed task: {news_id}")
        
        # Start monitoring trace
        trace_id = swarm.start_trace(
            trace_type=TraceType.AGENT_DECISION,
            metadata={"agent_id": scout_id, "task": "news_evaluation"}
        )
        
        # Simulate LLM usage
        await swarm.track_llm_usage(
            agent_id=scout_id,
            task_id=news_id,
            task_type="content_analysis",
            model=LLMModel.GPT35_TURBO.value,
            prompt_tokens=500,
            completion_tokens=200,
            duration_ms=1500,
            confidence=0.85
        )
        
        # Track decision
        swarm.track_decision(
            trace_id=trace_id,
            agent_id=scout_id,
            role="scout",
            decision_type="news_evaluation",
            input_context={"news_id": news_id},
            output={"should_cover": True, "priority": "high"},
            confidence=0.85,
            reasoning="High credibility source, trending topic, positive sentiment"
        )
        
        # End trace
        swarm.end_trace(trace_id, "success")
        
        # Move to debate queue
        await swarm.complete_task(
            agent_id=scout_id,
            section=BlackboardSection.RAW_FEED,
            item_id=news_id,
            result={"evaluation": "approved", "priority": "high"},
            move_to=BlackboardSection.DEBATE_QUEUE
        )
        print(f"✓ Task completed and moved to debate queue")
    
    # =============================================================================
    # 5. MESSAGE PASSING BETWEEN AGENTS
    # =============================================================================
    
    print("\n" + "=" * 60)
    print("5. MESSAGE PASSING BETWEEN AGENTS")
    print("=" * 60)
    
    # Analyst sends message to writer
    message = Message.create(
        msg_type=MessageType.TASK_ASSIGN,
        sender="analyst_1",
        recipient="writer_1",
        payload={
            "task": "write_script",
            "news_id": "news_001",
            "angle": "AI breakthrough impact on industry",
            "tone": "informative"
        }
    )
    
    receivers = await swarm.message_bus.send_to_agent("writer_1", message)
    print(f"✓ Message sent to writer_1 ({receivers} receivers)")
    
    # Broadcast to all scouts
    broadcast_msg = Message.create(
        msg_type=MessageType.AGENT_STATUS,
        sender="system",
        recipient=None,
        payload={"status": "new_tasks_available", "count": 2}
    )
    
    receivers = await swarm.message_bus.broadcast(broadcast_msg)
    print(f"✓ Broadcast sent ({receivers} receivers)")
    
    # =============================================================================
    # 6. CHECK BUDGET STATUS
    # =============================================================================
    
    print("\n" + "=" * 60)
    print("6. CHECKING BUDGET STATUS")
    print("=" * 60)
    
    for agent in agents:
        budget = await swarm.check_budget(agent["id"])
        print(f"\nAgent: {agent['id']}")
        print(f"  Daily spent: ${budget['daily_spent']:.2f} / ${budget['daily_remaining'] + budget['daily_spent']:.2f}")
        print(f"  Hourly spent: ${budget['hourly_spent']:.2f} / ${budget['hourly_remaining'] + budget['hourly_spent']:.2f}")
        print(f"  Alert triggered: {budget['alert_triggered']}")
    
    # =============================================================================
    # 7. CREATE CHECKPOINT
    # =============================================================================
    
    print("\n" + "=" * 60)
    print("7. CREATING CHECKPOINT")
    print("=" * 60)
    
    await swarm.create_checkpoint(
        task_id="video_001",
        stage="script_written",
        data={
            "script": "Breaking news: TechCorp announces Neural-X...",
            "word_count": 150,
            "estimated_duration": 60
        },
        agent_states={
            "writer_1": {"status": "completed", "output": "script_v1"}
        }
    )
    print("✓ Checkpoint created for video_001")
    
    # =============================================================================
    # 8. GET INFRASTRUCTURE STATS
    # =============================================================================
    
    print("\n" + "=" * 60)
    print("8. INFRASTRUCTURE STATISTICS")
    print("=" * 60)
    
    stats = await swarm.get_stats()
    print(json.dumps(stats, indent=2, default=str))
    
    # =============================================================================
    # 9. CLEANUP
    # =============================================================================
    
    print("\n" + "=" * 60)
    print("9. CLEANUP")
    print("=" * 60)
    
    # Unregister agents
    for agent in agents:
        await swarm.unregister_agent(agent["id"])
        print(f"✓ Unregistered agent: {agent['id']}")
    
    # Disconnect
    await swarm.disconnect()
    print("✓ Infrastructure disconnected")
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY")
    print("=" * 60)


import json

if __name__ == "__main__":
    asyncio.run(main())
