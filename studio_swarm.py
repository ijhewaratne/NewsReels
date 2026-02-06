
"""
================================================================================
STUDIO SWARM - VIDEO SCRIPT GENERATION
================================================================================
Converts approved story pitches into production-ready news video scripts.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Literal
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

# ✅ ADD THESE TWO LINES - Load .env file
from dotenv import load_dotenv
load_dotenv()  # This loads variables from .env file

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# Set API key explicitly for CrewAI
os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# ==============================================================================
# SCRIPT MODELS - Structured Outputs for Video Production
# ==============================================================================

class ShotType(str, Enum):
    """Types of video shots"""
    A_ROLL = "a_roll"
    B_ROLL = "b_roll"
    GRAPHIC = "graphic"
    STOCK = "stock_footage"
    INTERVIEW = "interview"
    ARCHIVE = "archive"
    ANIMATION = "animation"
    DRONE = "drone"
    SCREEN_CAPTURE = "screen_capture"

class AudioType(str, Enum):
    """Types of audio elements"""
    VOICE_OVER = "voice_over"
    INTERVIEW_CLIP = "interview_clip"
    NAT_SOUND = "natural_sound"
    MUSIC_BED = "music_bed"
    SFX = "sound_effect"
    SILENCE = "silence"

class ScriptSegment(BaseModel):
    """Individual segment of a news script"""
    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(description="Unique ID for this segment")
    timestamp: str = Field(description='Timecode (e.g., "00:00-00:15")')
    duration_seconds: int = Field(ge=1, le=60, description="Length of segment")

    narration: str = Field(description="What the narrator says")
    visual_description: str = Field(description="What appears on screen")
    shot_type: ShotType = Field(description="Type of shot")
    audio_type: AudioType = Field(default=AudioType.VOICE_OVER)

    b_roll_cues: List[str] = Field(default_factory=list, description="Specific footage to source")
    graphics_text: Optional[str] = Field(default=None, description="Lower thirds/text overlays")
    sound_cues: List[str] = Field(default_factory=list, description="SFX or ambient sound")
    source_credit: Optional[str] = Field(default=None, description="On-screen source credit")

class VideoScript(BaseModel):
    """Complete news video script"""
    model_config = ConfigDict(extra="forbid")

    script_id: str = Field(default_factory=lambda: f"script_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    story_id: str = Field(description="Original story ID from Scout phase")
    headline: str = Field(description="Video title/headline")

    total_duration: int = Field(description="Total runtime in seconds")
    target_platform: Literal["youtube", "tiktok", "instagram", "twitter", "broadcast"] = "youtube"
    tone: Literal["urgent", "informative", "conversational", "dramatic", "optimistic"] = "informative"

    segments: List[ScriptSegment] = Field(default_factory=list)

    opening_hook: str = Field(description="First 5 seconds hook")
    closing_cta: str = Field(description="Call to action at end")

    voice_talent: str = Field(default="main_narrator", description="Voice talent assignment")
    music_bed: Optional[str] = Field(default=None, description="Background music suggestion")
    color_grade: Optional[str] = Field(default=None, description="Color grading notes")

    b_roll_sources: List[Dict[str, str]] = Field(default_factory=list)
    stock_footage_needed: List[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    approved_by: Optional[str] = Field(default=None)

# ==============================================================================
# STUDIO AGENTS - Script Creation Team
# ==============================================================================

def create_script_writer() -> Agent:
    """Script_Writer: Creates compelling narration and story structure."""
    return Agent(
        role="Script Writer - Broadcast Journalist",
        goal="""Transform story pitches into compelling video scripts. 
        Write for the ear, not the eye. Every word must earn its place.

        Rules:
        - Hook in first 5 seconds (make it irresistible)
        - One idea per sentence
        - Conversational tone (write like people talk)
        - 130-150 words per minute pacing
        - End with clear call-to-action

        Output structured segments with exact timing.""",

        backstory="""You are an Emmy-nominated news writer who has written for 
        CNN, BBC, and Vice. You know that attention spans are short and 
        competition is fierce. You specialize in scripts that feel like 
        conversations, not lectures.""",

        verbose=True,
        allow_delegation=False,
        max_iter=3,
        llm="gpt-4o",
    )

def create_visual_director() -> Agent:
    """Visual_Director: Plans all visual elements and B-roll."""
    return Agent(
        role="Visual Director - Cinematographer",
        goal="""Design the visual storytelling for every script segment.
        Ensure every word of narration has matching visuals.

        Responsibilities:
        - Specify exact B-roll shots
        - Design lower thirds and graphics
        - Plan shot transitions
        - Source footage requirements
        - Ensure visual variety (avoid talking head fatigue)

        Output detailed visual descriptions and sourcing instructions.""",

        backstory="""You have directed documentaries for Netflix and shot 
        news packages in 40+ countries. You understand that viewers watch 
        with their eyes first, ears second.""",

        verbose=True,
        allow_delegation=False,
        max_iter=3,
        llm="gpt-4o",
    )

def create_voice_caster() -> Agent:
    """Voice_Caster: Determines voice talent and delivery style."""
    return Agent(
        role="Voice Caster - Audio Director",
        goal="""Select voice talent and delivery style for maximum impact.

        Considerations:
        - Story tone (urgent, somber, uplifting, neutral)
        - Target demographic (match voice to audience)
        - Platform constraints (TikTok vs BBC have different norms)
        - Energy arc (start high, sustain, end strong)

        Output voice direction for every segment.""",

        backstory="""You have cast voices for Super Bowl commercials and 
        NPR documentaries. You know that voice is 50% of the emotional 
        impact.""",

        verbose=True,
        allow_delegation=False,
        max_iter=2,
        llm="gpt-4o-mini",
    )

def create_script_editor() -> Agent:
    """Script_Editor: Final polish and timing verification."""
    return Agent(
        role="Script Editor - Final Cut Authority",
        goal="""Polish scripts to broadcast-ready status.
        Ensure accuracy, legality, and perfect timing.

        Final checks:
        - Word count vs runtime accuracy
        - Legal clearance (defamation, copyright)
        - Source attribution completeness
        - Pacing and breathing room
        - Platform-specific requirements

        Output production-ready final script.""",

        backstory="""You have been the final gatekeeper at major news outlets. 
        Nothing airs without your approval. You are paranoid about legal issues
        and timing accuracy.""",

        verbose=True,
        allow_delegation=False,
        max_iter=2,
        llm="gpt-4o",
    )

# ==============================================================================
# SCRIPT GENERATION PIPELINE
# ==============================================================================

class StudioPipeline:
    """Orchestrates the video script creation process."""

    def __init__(self, storage_path: str = "./studio_output"):
        self.storage_path = storage_path
        self.generated_scripts: List[VideoScript] = []

    async def generate_script(self, story: dict, platform: str = "youtube", 
                             duration: int = 60) -> VideoScript:
        """Generate a complete video script from a story pitch."""
        print(f"\n🎬 Generating script for: {story['headline'][:50]}...")

        try:
            # Phase 1: Script Writing
            script_draft = await self._write_script(story, platform, duration)

            # Phase 2: Visual Direction  
            visual_plan = await self._plan_visuals(script_draft, story)

            # Phase 3: Voice Casting
            voice_direction = await self._cast_voice(script_draft, story)

            # Phase 4: Final Edit
            final_script = await self._edit_final(script_draft, visual_plan, voice_direction)

            self.generated_scripts.append(final_script)
            print(f"   ✅ Script complete: {final_script.total_duration}s, {len(final_script.segments)} segments")

            return final_script

        except Exception as e:
            print(f"   ❌ Error generating script: {e}")
            # Return basic fallback script
            return self._create_fallback_script(story, duration)

    def _create_fallback_script(self, story: dict, duration: int) -> VideoScript:
        """Create a basic script if AI generation fails."""
        return VideoScript(
            story_id=story.get("story_id", "unknown"),
            headline=story.get("headline", "News Story"),
            total_duration=duration,
            target_platform="youtube",
            opening_hook=story.get("summary", "")[:100] if story.get("summary") else "Breaking news...",
            closing_cta="Subscribe for more updates.",
            segments=[
                ScriptSegment(
                    segment_id="1",
                    timestamp=f"00:00-00:{duration:02d}",
                    duration_seconds=duration,
                    narration=story.get("summary", "News story details here."),
                    visual_description=f"B-roll footage related to {story.get('category', 'news')}",
                    shot_type=ShotType.B_ROLL,
                    b_roll_cues=["Stock footage related to topic", "News graphics"]
                )
            ],
            stock_footage_needed=["Generic news B-roll", "Topic-related stock footage"],
            voice_talent="Professional narrator"
        )

    async def _write_script(self, story: dict, platform: str, duration: int) -> VideoScript:
        """Create initial script with narration and structure."""
        writer = create_script_writer()
        word_count = (duration // 60) * 140

        task = Task(
            description=f"""
            Write a {duration}-second news video script based on this story:

            HEADLINE: {story["headline"]}
            SUMMARY: {story["summary"]}
            CATEGORY: {story["category"]}
            URGENCY: {story["urgency_score"]}/1.0
            SOURCES: {[s["name"] for s in story["sources"]]}
            KEY QUOTES: {story.get("key_quotes", [])}

            PLATFORM: {platform}
            TARGET WORD COUNT: ~{word_count} words

            STRUCTURE:
            1. HOOK (0-5s): Grab attention immediately
            2. CONTEXT (5-20s): Background and why it matters
            3. DEVELOPMENT (20-40s): Main facts and details
            4. IMPLICATIONS (40-50s): What this means
            5. CLOSING (50-{duration}s): Call to action

            Output as valid JSON matching the VideoScript model.
            """,
            expected_output="JSON object with headline, total_duration, opening_hook, closing_cta, and segments array",
            agent=writer,
        )

        crew = Crew(agents=[writer], tasks=[task], verbose=False)
        result = await crew.kickoff_async()

        # Parse result
        raw_output = result.raw if hasattr(result, "raw") else str(result)

        try:
            # Clean markdown
            clean = raw_output.replace("```json", "").replace("```", "").strip()

            # Try to find JSON
            if "{" in clean and "}" in clean:
                start = clean.find("{")
                end = clean.rfind("}") + 1
                clean = clean[start:end]

            data = json.loads(clean)
            data["story_id"] = story["story_id"]

            return VideoScript(**data)
        except Exception as e:
            print(f"   Warning: Could not parse AI output, using fallback: {e}")
            return self._create_fallback_script(story, duration)

    async def _plan_visuals(self, script: VideoScript, story: dict) -> dict:
        """Add detailed visual direction to script."""
        director = create_visual_director()

        segments_desc = "\n".join([
            f"Segment {s.segment_id} ({s.timestamp}): {s.narration[:100]}..." 
            for s in script.segments
        ])

        task = Task(
            description=f"""
            Plan visuals for this news script:

            STORY: {story["headline"]}
            CATEGORY: {story["category"]}

            SEGMENTS:
            {segments_desc}

            For each segment specify:
            1. Primary shot type (interview, B-roll, graphic, etc.)
            2. Specific B-roll footage needed
            3. Graphics/lower thirds text
            4. Stock footage sources
            5. Transition style

            Output as JSON with visual_segments array.
            """,
            expected_output="JSON with visual plan for each segment",
            agent=director,
        )

        try:
            crew = Crew(agents=[director], tasks=[task], verbose=False)
            result = await crew.kickoff_async()
            raw = result.raw if hasattr(result, "raw") else str(result)
            clean = raw.replace("```json", "").replace("```", "").strip()

            if "{" in clean:
                start = clean.find("{")
                end = clean.rfind("}") + 1
                clean = clean[start:end]

            return json.loads(clean)
        except Exception as e:
            print(f"   Visual planning error: {e}")
            return {"visual_segments": [], "stock_footage_needed": []}

    async def _cast_voice(self, script: VideoScript, story: dict) -> dict:
        """Determine voice talent and delivery style."""
        caster = create_voice_caster()

        task = Task(
            description=f"""
            Cast voice talent for this script:

            HEADLINE: {story["headline"]}
            URGENCY: {story["urgency_score"]}/1.0
            CATEGORY: {story["category"]}

            OPENING: {script.opening_hook}

            Determine voice specs and output as JSON:
            - voice_talent: description (gender, age, tone)
            - pacing_wpm: words per minute
            - energy_level: 1-10
            - delivery_notes: specific directions
            """,
            expected_output="JSON with voice casting details",
            agent=caster,
        )

        try:
            crew = Crew(agents=[caster], tasks=[task], verbose=False)
            result = await crew.kickoff_async()
            raw = result.raw if hasattr(result, "raw") else str(result)
            clean = raw.replace("```json", "").replace("```", "").strip()

            if "{" in clean:
                start = clean.find("{")
                end = clean.rfind("}") + 1
                clean = clean[start:end]

            return json.loads(clean)
        except:
            return {
                "voice_talent": "Neutral professional narrator",
                "pacing_wpm": 140,
                "energy_level": 5,
                "delivery_notes": "Clear, authoritative delivery"
            }

    async def _edit_final(self, script: VideoScript, visuals: dict, voice: dict) -> VideoScript:
        """Compile final production-ready script."""
        # Merge visual direction into segments
        if "visual_segments" in visuals:
            for i, segment in enumerate(script.segments):
                if i < len(visuals["visual_segments"]):
                    vis = visuals["visual_segments"][i]
                    if vis.get("primary_shot"):
                        segment.visual_description = vis["primary_shot"]
                    if vis.get("b_roll_shots"):
                        segment.b_roll_cues = vis["b_roll_shots"]
                    if vis.get("graphics"):
                        segment.graphics_text = vis["graphics"]
                    if vis.get("shot_type"):
                        try:
                            segment.shot_type = ShotType(vis["shot_type"])
                        except:
                            pass

        # Add voice direction
        script.voice_talent = voice.get("voice_talent", "Standard narrator")
        script.stock_footage_needed = visuals.get("stock_footage_needed", [])
        script.color_grade = visuals.get("color_grade", "Standard")

        return script

    def save_script(self, script: VideoScript, format: str = "json"):
        """Save script to file."""
        os.makedirs(self.storage_path, exist_ok=True)
        filename = f"{self.storage_path}/{script.script_id}"

        if format == "json":
            try:
                output = script.model_dump_json(indent=2)
            except:
                output = json.dumps(script.__dict__, indent=2, default=str)

            with open(f"{filename}.json", "w") as f:
                f.write(output)

        elif format == "txt":
            with open(f"{filename}.txt", "w") as f:
                f.write(f"VIDEO SCRIPT: {script.headline}\n")
                f.write(f"Duration: {script.total_duration}s | Platform: {script.target_platform}\n")
                f.write(f"Voice: {script.voice_talent}\n")
                f.write("="*60 + "\n\n")

                for seg in script.segments:
                    f.write(f"[{seg.timestamp}] {seg.shot_type.upper()}\n")
                    f.write(f"NARRATION: {seg.narration}\n")
                    f.write(f"VISUAL: {seg.visual_description}\n")
                    if seg.graphics_text:
                        f.write(f"GRAPHICS: {seg.graphics_text}\n")
                    if seg.b_roll_cues:
                        f.write(f"B-ROLL: {', '.join(seg.b_roll_cues)}\n")
                    f.write("\n")

                f.write(f"\nCLOSING: {script.closing_cta}\n")

                if script.stock_footage_needed:
                    f.write("\n\nSTOCK FOOTAGE NEEDED:\n")
                    for item in script.stock_footage_needed:
                        f.write(f"- {item}\n")

        elif format == "teleprompter":
            with open(f"{filename}_teleprompter.txt", "w") as f:
                f.write(f"{script.headline}\n\n")
                for seg in script.segments:
                    f.write(f"[{seg.timestamp}] {seg.narration}\n\n")

        print(f"   💾 Saved: {filename}.{format}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

async def process_scout_batch(batch_file: str, platform: str = "youtube", duration: int = 60):
    """Process a Scout batch file and generate scripts."""

    # Load batch
    with open(batch_file, "r") as f:
        batch = json.load(f)

    stories = batch.get("all_pitches", [])
    print(f"\n🎬 STUDIO SWARM: Processing {len(stories)} stories")
    print("="*60)

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in environment!")
        print("   Make sure your .env file is loaded or set OPENAI_API_KEY")
        return []
    else:
        print(f"   ✅ API Key found: {api_key[:8]}...")

    pipeline = StudioPipeline()
    scripts = []

    for i, story in enumerate(stories, 1):
        print(f"\n[{i}/{len(stories)}] {story['headline'][:60]}...")

        script = await pipeline.generate_script(story, platform, duration)
        pipeline.save_script(script, format="json")
        pipeline.save_script(script, format="txt")
        pipeline.save_script(script, format="teleprompter")
        scripts.append(script)

    print(f"\n✅ Studio complete: {len(scripts)} scripts generated")
    print(f"📁 Output: {pipeline.storage_path}/")
    return scripts

if __name__ == "__main__":
    import sys

    batch_file = "./output/batch_20260206_071628.json"
    if len(sys.argv) > 1:
        batch_file = sys.argv[1]

    print("\n🎬 NEWS VIDEO STUDIO SWARM")
    print("Converting story pitches to broadcast scripts\n")

    asyncio.run(process_scout_batch(
        batch_file=batch_file,
        platform="youtube",
        duration=60
    ))