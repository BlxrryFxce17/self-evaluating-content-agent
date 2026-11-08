"""
Regenerator Agent.
Feeds evaluation rejection logs and specific remediation guidance back to regenerate content.
Ensures strict loop termination with maximum retry guards.
"""

from typing import List, Optional
from src.llm_client import LLMClient
from src.models import EvaluationReport, RejectionLogEntry
from src.generator import LessonGenerator


class LessonRegenerator:
    def __init__(self, generator: Optional[LessonGenerator] = None):
        self.generator = generator or LessonGenerator()

    def regenerate_lesson(
        self,
        topic: str,
        previous_draft: str,
        evaluation: EvaluationReport,
        iteration: int,
        evolved_guidelines: Optional[List[str]] = None
    ) -> str:
        """Regenerates the lesson by injecting the exact failure reasons and requested fixes."""
        system_prompt = self.generator._build_system_prompt(evolved_guidelines)

        user_prompt = f"""[REVISION REQUEST - ITERATION {iteration}]
Topic: "{topic}"

The previous draft failed our strict quality rubric.
Here is the official REJECTION REPORT from the Evaluator:

{evaluation.remediation_guidance}

Detailed Failure Reasons:
""" + "\n".join(f"- {reason}" for reason in evaluation.failure_reasons) + f"""

CRITICAL INSTRUCTIONS FOR REGENERATION:
1. Fix EVERY single issue listed above without fail.
2. Keep all parts of the previous draft that were already good and working.
3. Ensure the tone remains warm, simple, and tailored to a 12th-grade student from India.
4. Keep sentences short (under 20 words) and vocabulary simple.
5. Include the open-book exam analogy and the step-by-step 4-part diagram.

Here was the previous draft for reference:
---
{previous_draft[:1500]}... [truncated]
---

Now, please write the complete, newly revised and perfected lesson in markdown:
"""

        # When regenerating, we prompt the LLM client with the feedback
        feedback_prompt = f"FEEDBACK FROM EVALUATOR:\n{user_prompt}"
        revised_content = self.generator.client.generate(
            prompt=feedback_prompt,
            system_prompt=system_prompt,
            temperature=0.4  # Lower temperature for precision fixing
        )

        return revised_content

