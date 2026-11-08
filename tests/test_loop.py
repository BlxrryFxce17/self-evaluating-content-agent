"""
Integration tests for the LangGraph Autonomous Content System Workflow.
Tests the full loop: Generate -> Evaluate -> Rejection Log -> Regenerate -> Ship / Terminate.
"""

import pytest
from src.graph import ContentSystemWorkflow
from src.models import CheckpointName


def test_full_autonomous_loop_shipped():
    """Tests that a standard run triggers evaluation, identifies flaws, regenerates, and ships."""
    workflow = ContentSystemWorkflow(max_retries=2)
    state = workflow.run(topic="Introduction to RAG")

    assert state["status"] == "SHIPPED"
    assert state["final_lesson"] is not None
    assert len(state["final_lesson"]) > 500
    assert state["evaluation"].all_passed is True
    # Verify that rejection was recorded during the journey
    assert len(state["rejection_history"]) >= 1
    assert state["rejection_history"][0].iteration == 1


def test_deliberate_jargon_flaw_caught_and_remediated():
    """Tests that injecting a deliberate jargon flaw is caught on iteration 1 and corrected on iteration 2."""
    workflow = ContentSystemWorkflow(max_retries=2)
    state = workflow.run(topic="Introduction to RAG", flaw_injection="jargon")

    assert state["status"] == "SHIPPED"
    # Ensure iteration 1 caught the jargon
    first_rejection = state["rejection_history"][0]
    assert CheckpointName.NO_UNEXPLAINED_JARGON.value in first_rejection.failed_checkpoints
    assert state["iteration"] >= 2


def test_max_retries_termination_safety():
    """Verifies that the agentic loop is guaranteed to terminate if issues persist."""
    workflow = ContentSystemWorkflow(max_retries=0)  # 0 retries = single attempt only
    state = workflow.run(topic="Introduction to RAG", flaw_injection="academic_voice")

    assert state["status"] == "TERMINATED_MAX_RETRIES"
    assert not state["evaluation"].all_passed

