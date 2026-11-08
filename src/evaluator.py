"""
Evaluator Agent: Strict Binary Pass/Fail Checkpoints (No Partial Credit).
Combines deterministic linguistic/structural analysis with semantic verification.
Topic-aware evaluation supporting RAG, AI/ML, and general computer science topics.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from src.models import (
    CheckpointName,
    CheckpointResult,
    EvaluationReport,
    AudienceProfile
)
from src.nlp_utils import (
    calculate_flesch_metrics,
    check_unexplained_jargon
)
from src.config import (
    MAX_FLESCH_GRADE_LEVEL,
    MIN_FLESCH_READING_EASE,
    AUDIENCE_PROFILE
)


class ContentEvaluator:
    def __init__(self, audience: AudienceProfile = AUDIENCE_PROFILE):
        self.audience = audience

    def evaluate(self, content_markdown: str, topic: str = "Introduction to RAG") -> EvaluationReport:
        """Runs all 6 hard pass/fail checkpoints against the lesson content."""
        checkpoints: List[CheckpointResult] = []
        failure_reasons: List[str] = []

        # 1. ACCURACY & GROUNDED CHECKPOINT
        acc_res = self._check_accuracy_and_grounded(content_markdown, topic)
        checkpoints.append(acc_res)
        if not acc_res.passed:
            failure_reasons.append(f"[{acc_res.checkpoint.value}] {acc_res.reason}")

        # 2. BEGINNER FRIENDLY LANGUAGE CHECKPOINT
        lang_res = self._check_beginner_friendly_language(content_markdown)
        checkpoints.append(lang_res)
        if not lang_res.passed:
            failure_reasons.append(f"[{lang_res.checkpoint.value}] {lang_res.reason}")

        # 3. TEACHES BY EXAMPLE CHECKPOINT
        ex_res = self._check_teaches_by_example(content_markdown)
        checkpoints.append(ex_res)
        if not ex_res.passed:
            failure_reasons.append(f"[{ex_res.checkpoint.value}] {ex_res.reason}")

        # 4. NO UNEXPLAINED JARGON CHECKPOINT
        jargon_res = self._check_no_unexplained_jargon(content_markdown)
        checkpoints.append(jargon_res)
        if not jargon_res.passed:
            failure_reasons.append(f"[{jargon_res.checkpoint.value}] {jargon_res.reason}")

        # 5. KEY POINTS COVERAGE CHECKPOINT
        cov_res = self._check_key_points_coverage(content_markdown, topic)
        checkpoints.append(cov_res)
        if not cov_res.passed:
            failure_reasons.append(f"[{cov_res.checkpoint.value}] {cov_res.reason}")

        # 6. COHERENT TEACHING FLOW CHECKPOINT
        flow_res = self._check_coherent_teaching_flow(content_markdown)
        checkpoints.append(flow_res)
        if not flow_res.passed:
            failure_reasons.append(f"[{flow_res.checkpoint.value}] {flow_res.reason}")

        passed_count = sum(1 for c in checkpoints if c.passed)
        total_count = len(checkpoints)
        all_passed = (passed_count == total_count)

        # Build targeted remediation guidance if any failed
        remediation_guidance = ""
        if not all_passed:
            fixes = [f"- {c.checkpoint.value}: {c.suggested_fix}" for c in checkpoints if not c.passed and c.suggested_fix]
            remediation_guidance = "Remediation Requirements for Generator:\n" + "\n".join(fixes)

        return EvaluationReport(
            all_passed=all_passed,
            passed_count=passed_count,
            total_count=total_count,
            checkpoints=checkpoints,
            failure_reasons=failure_reasons,
            remediation_guidance=remediation_guidance
        )

    def _check_accuracy_and_grounded(self, text: str, topic: str = "Introduction to RAG") -> CheckpointResult:
        """Verifies technical mechanics based on topic."""
        text_lower = text.lower()
        topic_lower = topic.lower()

        # RAG specific check
        if "rag" in topic_lower:
            has_retrieval = any(w in text_lower for w in ["retriev", "search", "lookup", "fetch"])
            has_augmentation = any(w in text_lower for w in ["augment", "combin", "attach", "add", "prompt"])
            has_generation = any(w in text_lower for w in ["generat", "writ", "answer", "produc"])
            has_external_source = any(w in text_lower for w in ["document", "book", "note", "knowledge base", "data", "pdf", "file"])

            claims_model_retrained = "retrains the model" in text_lower or "updates neural weights" in text_lower
            if claims_model_retrained:
                return CheckpointResult(
                    checkpoint=CheckpointName.ACCURACY_AND_GROUNDED,
                    passed=False,
                    reason="Inaccurate statement: Claims RAG retrains the model or updates weights. RAG queries external data at runtime without retraining.",
                    suggested_fix="Clarify that RAG does NOT retrain the AI; it provides fresh reference notes at query time."
                )

            if has_retrieval and has_augmentation and has_generation and has_external_source:
                return CheckpointResult(
                    checkpoint=CheckpointName.ACCURACY_AND_GROUNDED,
                    passed=True,
                    reason="Accurate and grounded: Correctly defines the 3 core pillars (Retrieval, Augmentation, Generation) with external knowledge."
                )
            else:
                missing = []
                if not has_retrieval: missing.append("Retrieval / Searching step")
                if not has_augmentation: missing.append("Augmentation / Context attachment step")
                if not has_generation: missing.append("Generation / Answer creation step")
                if not has_external_source: missing.append("External knowledge base / documents")

                return CheckpointResult(
                    checkpoint=CheckpointName.ACCURACY_AND_GROUNDED,
                    passed=False,
                    reason=f"Missing fundamental technical pillars: {', '.join(missing)}.",
                    suggested_fix="Explicitly walk through the full loop: Knowledge Base -> Retrieval (Search) -> Augmentation (Prompt) -> Generation (Answer)."
                )

        # AI / ML topic check
        elif "ai" in topic_lower or "ml" in topic_lower or "machine learning" in topic_lower:
            has_data = any(w in text_lower for w in ["data", "example", "dataset", "picture", "input"])
            has_learning = any(w in text_lower for w in ["learn", "train", "pattern", "rule"])
            has_prediction = any(w in text_lower for w in ["predict", "answer", "inference", "decision", "output"])

            if has_data and has_learning and has_prediction:
                return CheckpointResult(
                    checkpoint=CheckpointName.ACCURACY_AND_GROUNDED,
                    passed=True,
                    reason="Accurate and grounded: Correctly explains learning from data, finding patterns, and producing predictions."
                )
            else:
                missing = []
                if not has_data: missing.append("Data / Examples")
                if not has_learning: missing.append("Learning / Pattern recognition")
                if not has_prediction: missing.append("Prediction / Inference")
                return CheckpointResult(
                    checkpoint=CheckpointName.ACCURACY_AND_GROUNDED,
                    passed=False,
                    reason=f"Missing core Machine Learning pillars: {', '.join(missing)}.",
                    suggested_fix="Explain how ML gathers data, finds patterns during training, and makes predictions."
                )

        # General technical topic check
        else:
            has_definition = any(w in text_lower for w in ["concept", "definition", "core", "input"])
            has_steps = any(w in text_lower for w in ["stage", "step", "flow", "work", "process"])
            if has_definition and has_steps:
                return CheckpointResult(
                    checkpoint=CheckpointName.ACCURACY_AND_GROUNDED,
                    passed=True,
                    reason="Accurate and grounded: Clearly describes concept definition, processing steps, and output."
                )
            return CheckpointResult(
                checkpoint=CheckpointName.ACCURACY_AND_GROUNDED,
                passed=False,
                reason="Lacks clear explanation of the topic's core input, process, and output.",
                suggested_fix="Define the concept clearly and outline its input, process, and output stages."
            )

    def _check_beginner_friendly_language(self, text: str) -> CheckpointResult:
        """Tests reading ease, grade level, and sentence length for non-English medium 12th grads."""
        fre, fk_grade, avg_sentence_len = calculate_flesch_metrics(text)

        grade_passed = fk_grade <= MAX_FLESCH_GRADE_LEVEL
        ease_passed = fre >= MIN_FLESCH_READING_EASE
        sentence_passed = avg_sentence_len <= 25.0

        if grade_passed and ease_passed and sentence_passed:
            return CheckpointResult(
                checkpoint=CheckpointName.BEGINNER_FRIENDLY_LANGUAGE,
                passed=True,
                metric_value=f"Flesch-Kincaid Grade: {fk_grade} (Target <= {MAX_FLESCH_GRADE_LEVEL}), Reading Ease: {fre}/100",
                reason=f"Passes beginner thresholds. Grade Level is {fk_grade} (Middle-school English), Reading Ease is {fre}/100."
            )
        else:
            reasons = []
            if not grade_passed:
                reasons.append(f"Grade level is {fk_grade} (exceeds maximum allowed {MAX_FLESCH_GRADE_LEVEL})")
            if not ease_passed:
                reasons.append(f"Reading ease is {fre}/100 (below minimum {MIN_FLESCH_READING_EASE})")
            if not sentence_passed:
                reasons.append(f"Average sentence length is {avg_sentence_len} words (exceeds 25 words/sentence)")

            return CheckpointResult(
                checkpoint=CheckpointName.BEGINNER_FRIENDLY_LANGUAGE,
                passed=False,
                metric_value=f"Grade: {fk_grade}, Ease: {fre}/100, Avg Len: {avg_sentence_len}w",
                reason=f"Language is too difficult for target learner: {'; '.join(reasons)}.",
                suggested_fix="Shorten complex sentences, replace academic words with daily conversational English, and use short bullet points."
            )

    def _check_teaches_by_example(self, text: str) -> CheckpointResult:
        """Verifies presence of at least one strong, culturally relatable analogy."""
        text_lower = text.lower()
        analogy_markers = [
            "open-book", "closed-book", "exam", "librarian", "library",
            "doctor", "patient file", "cricket", "umpire", "drs", "textbook",
            "child", "cooking", "bicycle", "mango", "apple", "cat", "dog"
        ]

        found_analogies = [m for m in analogy_markers if m in text_lower]

        if len(found_analogies) >= 2:
            return CheckpointResult(
                checkpoint=CheckpointName.TEACHES_BY_EXAMPLE,
                passed=True,
                metric_value=f"Found analogy anchors: {', '.join(found_analogies[:3])}",
                reason=f"Strong relatable analogy present: grounded with {', '.join(found_analogies[:3])}."
            )
        else:
            return CheckpointResult(
                checkpoint=CheckpointName.TEACHES_BY_EXAMPLE,
                passed=False,
                reason="Fails: Lacks a concrete, relatable real-world analogy to anchor the concept for a complete beginner.",
                suggested_fix="Include a concrete real-life analogy (e.g. Open-Book Exam or Teaching a child) right near the beginning."
            )

    def _check_no_unexplained_jargon(self, text: str) -> CheckpointResult:
        """Checks that every technical term introduced is immediately defined in plain language."""
        passed, unexplained, explained = check_unexplained_jargon(text)

        if passed:
            return CheckpointResult(
                checkpoint=CheckpointName.NO_UNEXPLAINED_JARGON,
                passed=True,
                metric_value=f"Explained terms: {len(explained)}",
                reason=f"Zero unexplained jargon. All technical terms introduced ({', '.join(explained[:4])}) have immediate plain-English explanations."
            )
        else:
            return CheckpointResult(
                checkpoint=CheckpointName.NO_UNEXPLAINED_JARGON,
                passed=False,
                metric_value=f"Unexplained terms: {len(unexplained)}",
                reason=f"Contains unexplained technical jargon: {', '.join(unexplained)}.",
                suggested_fix=f"Either remove or immediately define these terms using simple everyday analogies: {', '.join(unexplained)}."
            )

    def _check_key_points_coverage(self, text: str, topic: str = "Introduction to RAG") -> CheckpointResult:
        """Verifies What it is, Why it matters, and How it works are all distinctly covered."""
        text_lower = text.lower()

        has_what = any(k in text_lower for k in ["what is", "what does", "core definition", "definition", "overview"])
        has_why = any(k in text_lower for k in ["why does it matter", "why it matters", "why", "matter", "importance"])
        has_how = any(k in text_lower for k in ["how it works", "how", "workflow", "architecture", "step-by-step"])

        missing_sections = []
        if not has_what: missing_sections.append("What it is (Core Definition)")
        if not has_why: missing_sections.append("Why it matters (Value & Problem Solved)")
        if not has_how: missing_sections.append("How it works (Step-by-step mechanism)")

        if not missing_sections:
            return CheckpointResult(
                checkpoint=CheckpointName.KEY_POINTS_COVERAGE,
                passed=True,
                reason="Covers all required core points: What it is, Why it matters, and How it works."
            )
        else:
            return CheckpointResult(
                checkpoint=CheckpointName.KEY_POINTS_COVERAGE,
                passed=False,
                reason=f"Missing essential curriculum sections: {', '.join(missing_sections)}.",
                suggested_fix=f"Add clear, dedicated sections with headers covering: {', '.join(missing_sections)}."
            )

    def _check_coherent_teaching_flow(self, text: str) -> CheckpointResult:
        """Verifies pedagogical arc: hook, explanation, breakdown diagram/steps, quiz."""
        text_lower = text.lower()

        has_quiz = any(q in text_lower for q in ["quick check", "quiz", "self-assessment", "test your", "questions", "did you get it"])
        has_diagram_or_steps = any(s in text_lower for s in ["step 1", "step-by-step", "```", "──►", "flow", "workflow"])
        has_encouraging_closing = any(c in text_lower for c in ["next step", "congratulations", "journey", "good luck", "remember", "summary"])

        missing_flow_elements = []
        if not has_quiz: missing_flow_elements.append("Knowledge check / 2-3 question quiz")
        if not has_diagram_or_steps: missing_flow_elements.append("Step-by-step breakdown or flow diagram")
        if not has_encouraging_closing: missing_flow_elements.append("Summary / Encouraging closing for learner")

        if not missing_flow_elements:
            return CheckpointResult(
                checkpoint=CheckpointName.COHERENT_TEACHING_FLOW,
                passed=True,
                reason="Excellent teaching flow: follows Hook -> Analogy -> Core Concept -> Steps -> Check Quiz -> Summary."
            )
        else:
            return CheckpointResult(
                checkpoint=CheckpointName.COHERENT_TEACHING_FLOW,
                passed=False,
                reason=f"Incomplete pedagogical flow: Missing {', '.join(missing_flow_elements)}.",
                suggested_fix=f"Include {', '.join(missing_flow_elements)} to complete the learning journey."
            )
