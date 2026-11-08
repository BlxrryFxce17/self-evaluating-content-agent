"""
Unit tests for ContentEvaluator hard pass/fail checkpoints.
Validates that checkpoints are strictly binary with zero partial credit.
"""

import pytest
from src.evaluator import ContentEvaluator
from src.models import CheckpointName


@pytest.fixture
def evaluator():
    return ContentEvaluator()


def test_evaluator_catches_dense_academic_language(evaluator):
    """Flesch grade level check must fail when academic prose is present."""
    academic_text = """
    The epistemological dilemma confronting contemporary autoregressive transformer architectures
    necessitates an augmentation framework capable of assimilating dynamic, non-parametric corpus
    information into context-dependent generation paradigms.
    """
    report = evaluator.evaluate(academic_text)
    lang_checkpoint = next(c for c in report.checkpoints if c.checkpoint == CheckpointName.BEGINNER_FRIENDLY_LANGUAGE)

    assert not report.all_passed
    assert not lang_checkpoint.passed
    assert "Grade level" in lang_checkpoint.reason or "Reading ease" in lang_checkpoint.reason


def test_evaluator_catches_unexplained_jargon(evaluator):
    """Must strictly fail when technical jargon (e.g., latent, HNSW, MIPS) appears without definition."""
    jargon_text = """
    Welcome to our AI lesson. Today we use HNSW and latent vector embeddings
    with MIPS search to compute cross-entropy perplexity.
    """
    report = evaluator.evaluate(jargon_text)
    jargon_checkpoint = next(c for c in report.checkpoints if c.checkpoint == CheckpointName.NO_UNEXPLAINED_JARGON)

    assert not jargon_checkpoint.passed
    assert "dense academic jargon" in jargon_checkpoint.reason or "without simple" in jargon_checkpoint.reason


def test_evaluator_catches_missing_key_points(evaluator):
    """Must fail when 'How it works' or core sections are missing."""
    incomplete_text = """
    # What is RAG?
    RAG is Retrieval-Augmented Generation. It helps AI.
    # Why it matters
    It stops AI hallucinations. That is all.
    """
    report = evaluator.evaluate(incomplete_text)
    coverage_checkpoint = next(c for c in report.checkpoints if c.checkpoint == CheckpointName.KEY_POINTS_COVERAGE)

    assert not coverage_checkpoint.passed
    assert "How it works" in coverage_checkpoint.reason


def test_evaluator_catches_missing_relatable_analogy(evaluator):
    """Must fail if no relatable anchor analogy (open-book, library, etc.) is included."""
    dry_text = """
    # What is RAG?
    RAG stands for Retrieval-Augmented Generation.
    # Why it matters
    It helps AI avoid hallucinations and updates knowledge.
    # How RAG works
    Step 1: Retrieve documents. Step 2: Augment prompt. Step 3: Generate text.
    """
    report = evaluator.evaluate(dry_text)
    analogy_checkpoint = next(c for c in report.checkpoints if c.checkpoint == CheckpointName.TEACHES_BY_EXAMPLE)

    assert not analogy_checkpoint.passed
    assert "relatable" in analogy_checkpoint.reason.lower()


def test_evaluator_passes_perfect_lesson(evaluator):
    """A fully compliant lesson with simple language, analogies, breakdown, and quiz must pass all 6 checks."""
    from src.llm_client import LLMClient
    client = LLMClient()
    passing_text = client._perfect_remediated_lesson()

    report = evaluator.evaluate(passing_text)

    assert report.all_passed
    assert report.passed_count == 6
    assert report.total_count == 6
    assert len(report.failure_reasons) == 0

