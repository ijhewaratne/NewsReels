"""
Example Usage: Scout & Council Swarm
=====================================

This file demonstrates various ways to use the Scout and Council Swarm system.
"""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from main module
from scout_council_swarm import (
    StoryPipeline,
    StudioSwarmInterface,
    create_scout_crew,
    create_council_crew,
    StoryPitch,
    StoryDecision,
    BatchStoryOutput,
)


# ==============================================================================
# EXAMPLE 1: Basic Usage - Run Full Pipeline
# ==============================================================================

async def example_basic_usage():
    """
    Run the complete pipeline with default settings.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Set your API key
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Initialize pipeline
    pipeline = StoryPipeline(storage_path="./output")
    
    # Run on a topic
    topic = "climate change"
    result = await pipeline.run_full_pipeline(topic=topic)
    
    # Print results
    print(f"\nResults for '{topic}':")
    print(f"  Stories discovered: {result.total_discovered}")
    print(f"  Stories approved: {result.total_approved}")
    print(f"  Approval rate: {result.approval_rate:.1%}")
    print(f"  Processing time: {result.processing_time_seconds:.1f}s")
    
    return result


# ==============================================================================
# EXAMPLE 2: Run Only Scout Phase
# ==============================================================================

async def example_scout_only():
    """
    Run only the discovery phase without council voting.
    Useful for quick story gathering.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Scout-Only Discovery")
    print("="*60)
    
    pipeline = StoryPipeline(storage_path="./output")
    
    # Run just discovery
    stories = await pipeline.run_discovery(topic="technology")
    
    print(f"\nDiscovered {len(stories)} stories:")
    for i, story in enumerate(stories[:5], 1):
        print(f"  {i}. {story.headline}")
        print(f"     Confidence: {story.confidence_score:.2f}")
        print(f"     Source: {story.discovered_by}")
    
    return stories


# ==============================================================================
# EXAMPLE 3: Run Only Council Phase on Existing Stories
# ==============================================================================

async def example_council_only():
    """
    Run council voting on pre-discovered stories.
    Useful for re-evaluating stories with different criteria.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Council-Only Voting")
    print("="*60)
    
    # Create sample stories (in production, load from storage)
    sample_stories = [
        StoryPitch(
            story_id="story_001",
            headline="Major Tech Company Announces Breakthrough AI Model",
            summary="A leading technology company has unveiled a new AI system...",
            category="technology",
            confidence_score=0.85,
            urgency_score=0.90,
            novelty_score=0.75,
            discovered_by="Wire_Scout",
            sources=[],
        ),
        StoryPitch(
            story_id="story_002",
            headline="Viral Video Shows Unusual Weather Phenomenon",
            summary="Social media users are sharing footage of...",
            category="environment",
            confidence_score=0.60,
            urgency_score=0.40,
            novelty_score=0.80,
            discovered_by="Social_Scout",
            sources=[],
        ),
    ]
    
    pipeline = StoryPipeline(storage_path="./output")
    pipeline.discovered_stories = sample_stories
    
    # Run voting
    decisions = await pipeline.run_voting(sample_stories)
    
    print(f"\nVoting results:")
    for decision in decisions:
        status = "✓ APPROVED" if decision.is_approved else "✗ REJECTED"
        print(f"  {status}: {decision.headline[:50]}...")
        print(f"     Score: {decision.overall_score:.2f}")
        print(f"     Priority: {decision.priority}")
    
    return decisions


# ==============================================================================
# EXAMPLE 4: Custom Topic with Handoff to Studio
# ==============================================================================

async def example_with_studio_handoff():
    """
    Run pipeline and automatically handoff approved stories to Studio Swarm.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: With Studio Handoff")
    print("="*60)
    
    pipeline = StoryPipeline(storage_path="./output")
    
    # Run pipeline
    result = await pipeline.run_full_pipeline(topic="space exploration")
    
    # Handoff to Studio Swarm
    if result.approved_stories:
        studio = StudioSwarmInterface(
            input_path="./output",
            output_path="./studio_input"
        )
        handoff_file = studio.handoff_to_studio(result.approved_stories)
        print(f"\nHandoff complete: {handoff_file}")
    else:
        print("\nNo stories approved for handoff")
    
    return result


# ==============================================================================
# EXAMPLE 5: Batch Processing Multiple Topics
# ==============================================================================

async def example_batch_topics():
    """
    Process multiple topics in sequence.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Topic Processing")
    print("="*60)
    
    topics = [
        "artificial intelligence",
        "renewable energy",
        "healthcare innovation",
        "space exploration",
    ]
    
    results = []
    for topic in topics:
        print(f"\n--- Processing: {topic} ---")
        pipeline = StoryPipeline(storage_path="./output")
        result = await pipeline.run_full_pipeline(topic=topic)
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("BATCH SUMMARY")
    print("="*60)
    total_discovered = sum(r.total_discovered for r in results)
    total_approved = sum(r.total_approved for r in results)
    
    print(f"Topics processed: {len(topics)}")
    print(f"Total discovered: {total_discovered}")
    print(f"Total approved: {total_approved}")
    print(f"Overall approval rate: {total_approved/total_discovered:.1%}" if total_discovered > 0 else "N/A")
    
    return results


# ==============================================================================
# EXAMPLE 6: Custom Scout Configuration
# ==============================================================================

async def example_custom_scout_config():
    """
    Create a custom scout crew with modified settings.
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Custom Scout Configuration")
    print("="*60)
    
    from scout_council_swarm import (
        create_wire_scout,
        create_social_scout,
        create_semantic_scout,
        create_scout_tasks,
    )
    from crewai import Crew, Process
    
    # Create custom agents with different models
    wire = create_wire_scout()
    wire.llm = "gpt-4o"  # Use stronger model for wire scout
    
    social = create_social_scout()
    social.max_iter = 5  # Allow more iterations
    
    semantic = create_semantic_scout()
    semantic.max_iter = 10  # Deep research needs more iterations
    
    # Create crew with custom config
    crew = Crew(
        agents=[wire, social, semantic],
        tasks=create_scout_tasks("custom topic"),
        process=Process.parallel,
        verbose=True,
        memory=True,
        cache=True,
        max_rpm=20,  # Lower rate limit
    )
    
    print("Custom crew created with:")
    print(f"  - Wire Scout: {wire.llm}, max_iter={wire.max_iter}")
    print(f"  - Social Scout: max_iter={social.max_iter}")
    print(f"  - Semantic Scout: max_iter={semantic.max_iter}")
    
    # Run crew
    result = await crew.kickoff_async()
    print(f"\nCrew execution complete")
    
    return result


# ==============================================================================
# EXAMPLE 7: Retrieve and Review Past Results
# ==============================================================================

async def example_review_past_results():
    """
    Retrieve and review previously generated results.
    """
    print("\n" + "="*60)
    print("EXAMPLE 7: Review Past Results")
    print("="*60)
    
    studio = StudioSwarmInterface(input_path="./output")
    
    # Get most recent approved stories
    stories = studio.get_approved_stories()
    
    if not stories:
        print("No past results found")
        return []
    
    print(f"Found {len(stories)} approved stories:")
    
    for i, story in enumerate(stories[:5], 1):
        print(f"\n  {i}. {story.headline}")
        print(f"     Priority: {story.priority}")
        print(f"     Overall Score: {story.overall_score:.2f}")
        print(f"     Approved: {story.is_approved}")
        
        # Show vote breakdown
        if story.votes:
            approvals = sum(1 for v in story.votes if v.decision == "approve")
            print(f"     Votes: {approvals}/{len(story.votes)} approved")
    
    return stories


# ==============================================================================
# EXAMPLE 8: Integration with External Systems
# ==============================================================================

async def example_external_integration():
    """
    Example of integrating with external systems (databases, APIs, etc.)
    """
    print("\n" + "="*60)
    print("EXAMPLE 8: External Integration")
    print("="*60)
    
    # Run pipeline
    pipeline = StoryPipeline(storage_path="./output")
    result = await pipeline.run_full_pipeline(topic="cryptocurrency")
    
    # Example: Send to webhook
    print("\nExample webhook payload:")
    webhook_payload = {
        "event": "stories_approved",
        "timestamp": datetime.utcnow().isoformat(),
        "batch_id": result.batch_id,
        "data": {
            "total_discovered": result.total_discovered,
            "total_approved": result.total_approved,
            "stories": [
                {
                    "id": s.story_id,
                    "headline": s.headline,
                    "priority": s.priority,
                    "score": s.overall_score,
                }
                for s in result.approved_stories[:3]
            ]
        }
    }
    
    import json
    print(json.dumps(webhook_payload, indent=2, default=str))
    
    # Example: Save to database
    print("\nExample database insert:")
    for story in result.approved_stories[:2]:
        db_record = {
            "story_id": story.story_id,
            "headline": story.headline,
            "priority": story.priority,
            "overall_score": story.overall_score,
            "is_approved": story.is_approved,
            "created_at": datetime.utcnow(),
        }
        print(f"  INSERT INTO stories: {db_record['headline'][:40]}...")
    
    return result


# ==============================================================================
# MAIN - Run All Examples
# ==============================================================================

async def main():
    """
    Run all examples. In production, you'd run just one.
    """
    print("\n" + "="*60)
    print("SCOUT & COUNCIL SWARM - EXAMPLES")
    print("="*60)
    print("\nNote: These examples require OPENAI_API_KEY to be set.")
    print("Some examples use placeholder data and don't require API calls.")
    
    # Run examples (comment out ones you don't want to run)
    
    # Example 1: Basic usage (requires API key)
    await example_basic_usage()
    
    # Example 2: Scout only (requires API key)
    # await example_scout_only()
    
    # Example 3: Council only (uses sample data, no API needed)
    # await example_council_only()
    
    # Example 4: With studio handoff (requires API key)
    # await example_with_studio_handoff()
    
    # Example 5: Batch topics (requires API key)
    # await example_batch_topics()
    
    # Example 6: Custom config (requires API key)
    # await example_custom_scout_config()
    
    # Example 7: Review past (uses saved files)
    # await example_review_past_results()
    
    # Example 8: External integration (requires API key)
    # await example_external_integration()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
