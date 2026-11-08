"""
Unit tests for Persistent Memory and Self-Evolving Engine.
"""

import tempfile
from pathlib import Path
from src.memory import PersistentMemoryEngine
from src.models import RejectionLogEntry


def test_memory_persistence_and_guideline_evolution():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_memory.db"
        memory = PersistentMemoryEngine(db_path=db_path)

        # Record a run with a repeated jargon failure
        rejections = [
            RejectionLogEntry(
                iteration=1,
                failed_checkpoints=["NO_UNEXPLAINED_JARGON", "BEGINNER_FRIENDLY_LANGUAGE"],
                reasons=["Contains unexplained jargon", "Reading level too high"],
                remediation_guidance="Define terms immediately in simple Hindi/English cognates",
                changes_demanded="Simplify vocabulary"
            )
        ]

        active_guidelines = memory.record_run(
            run_id="test_run_1",
            topic="Test RAG",
            total_iterations=2,
            final_status="SHIPPED",
            rejection_logs=rejections,
            shipped_content="# Test Content"
        )

        assert len(active_guidelines) >= 2
        assert any("NO_UNEXPLAINED_JARGON" in g or "jargon" in g.lower() for g in active_guidelines)

        # Retrieve guidelines again from disk
        disk_guidelines = memory.get_active_guidelines()
        assert len(disk_guidelines) == len(active_guidelines)

