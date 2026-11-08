"""
Lesson Content Generator Agent.
Generates beginner-friendly learning content tailored to non-English-medium Indian 12th-graders.
"""

from typing import List, Optional
from src.llm_client import LLMClient
from src.models import AudienceProfile
from src.config import AUDIENCE_PROFILE


class LessonGenerator:
    def __init__(self, llm_client: Optional[LLMClient] = None, audience: AudienceProfile = AUDIENCE_PROFILE):
        self.client = llm_client or LLMClient()
        self.audience = audience

    def generate_lesson(
        self,
        topic: str,
        evolved_guidelines: Optional[List[str]] = None,
        flaw_injection: Optional[str] = None
    ) -> str:
        """Generates an initial lesson draft with audience persona and learned memory."""
        system_prompt = self._build_system_prompt(evolved_guidelines)
        user_prompt = self._build_user_prompt(topic, flaw_injection)

        content = self.client.generate(prompt=user_prompt, system_prompt=system_prompt, temperature=0.6)
        return content

    def _build_system_prompt(self, evolved_guidelines: Optional[List[str]] = None) -> str:
        guidelines_section = ""
        if evolved_guidelines:
            guidelines_section = "\n### Learned Guardrails from Past Runs (Self-Evolved Memory):\n" + "\n".join(
                f"- {g}" for g in evolved_guidelines
            )

        return f"""You are an empathetic, world-class GenAI Educator specializing in teaching complex technical concepts to beginners.

TARGET AUDIENCE:
- Persona: {self.audience.persona}
- Background: Completed 12th standard in India, non-English-medium background, ambitious to start an AI career.
- English Proficiency: Basic to intermediate conversational English. Avoid multi-syllabic academic vocabulary.

PEDAGOGICAL RULES:
1. Grounding Analogy: Always start with an everyday Indian life analogy (e.g., Open-book exam vs closed-book exam, college library, local doctor, or cricket DRS).
2. Zero Unexplained Jargon: If you introduce words like 'retrieval', 'embeddings', 'vectors', 'hallucinations', immediately explain them in plain language.
3. Structure:
   - Welcoming Hook & Encouragement
   - Relatable Real-Life Analogy
   - What it is (Break down the name)
   - Why it matters (Solve hallucination & outdated knowledge)
   - How it works (Step-by-step simple diagram/flow)
   - Quick 3-Question Self-Check Quiz with Answers
   - Encouraging Next Step
4. Readability Target: Flesch-Kincaid Grade Level 6 - 8 (Middle school conversational English). Short sentences (under 20 words).
{guidelines_section}
"""

    def _build_user_prompt(self, topic: str, flaw_injection: Optional[str] = None) -> str:
        flaw_instruction = ""
        if flaw_injection:
            if flaw_injection.lower() == "jargon":
                flaw_instruction = "\n[INJECT_FLAW: JARGON] (For automated testing: output dense, unexplained academic jargon with equations and complex terms)."
            elif flaw_injection.lower() == "missing_how":
                flaw_instruction = "\n[INJECT_FLAW: MISSING_HOW] (For automated testing: omit the entire 'How it works' step-by-step section)."
            elif flaw_injection.lower() == "academic_voice":
                flaw_instruction = "\n[INJECT_FLAW: ACADEMIC_VOICE] (For automated testing: write with dense polysyllabic academic prose like a PhD dissertation)."

        return f"""Please create a standalone beginner lesson for: "{topic}".
Ensure someone with zero prior AI experience finishes it and genuinely understands it.
Cover: What it is, Why it matters, and How it works.
{flaw_instruction}
"""

