from app.agents.script.models import ScriptRequest

SCRIPT_SYSTEM_PROMPT = """You are an expert video scriptwriter and narrative director for high-production AI video content.
Your objective is to adapt structured research into a compelling, tightly timed, scene-by-scene video script.

STRICT INSTRUCTIONS:
1. Output MUST be valid JSON only. Do not include introductory text, explanations, or conversational markdown outside the JSON block.
2. The JSON object must strictly adhere to the following schema:
{
  "title": "<compelling and punchy video title>",
  "scenes": [
    {
      "scene_number": 1,
      "narrative_beat_sequence": 1,
      "beat_type": "hook",
      "narration": "<spoken voiceover narration for this scene>",
      "visual_direction": "<detailed visual, camera movement, subject staging, or b-roll instructions>",
      "target_keywords": [
        "<visual retrieval keyword 1>",
        "<visual retrieval keyword 2>"
      ]
    }
  ]
}

NARRATIVE & PACING RULES:
- Every scene must explicitly map to a narrative beat from the research: "narrative_beat_sequence" must match the beat's sequence number, and "beat_type" must exactly match the beat's beat_type.
- Every narrative beat in the research must be represented by at least one scene; multiple scenes may expand a single beat.
- Scene numbers must be strictly sequential starting at 1 (1, 2, 3, ...).
- Keep total narration word count tightly aligned with the target word count indicated in the prompt (within +/- 15% application tolerance, aim as close to the target as practical) so that spoken pacing matches the target duration.
- The "narration" field contains ONLY spoken dialogue/voiceover. Do not include camera instructions, brackets, or parentheticals in narration.
- The "visual_direction" field contains visual guidance for video generation (lighting, camera movement, composition).
- The "target_keywords" field contains 2 to 5 specific creative retrieval keywords for scene B-roll and generation prompts.
"""


def build_script_user_prompt(request: ScriptRequest) -> str:
    """Build the user prompt for the script agent based on research and request parameters."""
    research = request.research
    target_duration = request.target_duration_seconds
    pacing_wpm = request.pacing_wpm
    target_word_count = round((target_duration * pacing_wpm) / 60)

    lines = [
        f"Topic: {research.topic}",
        f"Core Angle: {research.core_angle}",
        f"Target Audience: {research.target_audience}",
        f"Tone: {request.tone}",
        f"Target Duration: {target_duration} seconds",
        f"Pacing: {pacing_wpm} words per minute",
        f"Target Narration Word Count: approximately {target_word_count} words total",
    ]

    if request.call_to_action:
        lines.append(f"Call to Action: {request.call_to_action}")

    lines.append("\nExecutive Summary:")
    lines.append(f"  {research.executive_summary}")

    lines.append("\nKey Takeaways:")
    for takeaway in research.key_takeaways:
        lines.append(f"  - {takeaway}")

    lines.append("\nKey Facts:")
    for fact in research.facts:
        lines.append(f"  - {fact.claim}")
        if fact.context:
            lines.append(f"    Context: {fact.context}")

    lines.append("\nSuggested Narrative Arc (Scenes MUST strictly reference these beats):")
    for beat in research.suggested_narrative_arc:
        lines.append(
            f"  - Beat #{beat.sequence} [{beat.beat_type}] (suggested ratio: {beat.suggested_duration_ratio:.2f}): {beat.description}"
        )

    lines.append("\nVisual Themes:")
    for theme in research.visual_themes:
        lines.append(f"  - {theme}")

    lines.append(
        "\nGenerate the complete scene-by-scene script formatted as pure JSON following the system instructions."
    )
    return "\n".join(lines)
