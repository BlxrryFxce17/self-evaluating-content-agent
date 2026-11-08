"""
Persistent Memory and Self-Evolving Engine.
Persists run history, analyzes repeated rejections across runs,
and automatically sharpens generator prompts and rubrics over time.
"""

import json
import sqlite3
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import MEMORY_DIR
from src.models import RejectionLogEntry, MemoryRecord


class PersistentMemoryEngine:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (MEMORY_DIR / "agent_memory.db")
        self.json_log_path = self.db_path.parent / "run_history.json"
        self._init_db()

    def _execute(self, query: str, params: tuple = ()):
        """Executes a query and immediately closes the connection."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.fetchall()
        finally:
            conn.close()

    def _init_db(self):
        """Initializes SQLite tables for persistent runs and checkpoint failures."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    topic TEXT,
                    total_iterations INTEGER,
                    final_status TEXT,
                    shipped_content TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rejections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    iteration INTEGER,
                    timestamp TEXT,
                    failed_checkpoint TEXT,
                    reason TEXT,
                    remediation TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evolved_guidelines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_text TEXT UNIQUE,
                    source_checkpoint TEXT,
                    frequency INTEGER,
                    created_at TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def record_run(
        self,
        run_id: str,
        topic: str,
        total_iterations: int,
        final_status: str,
        rejection_logs: List[RejectionLogEntry],
        shipped_content: str = ""
    ) -> List[str]:
        """
        Saves run to database, analyzes failure patterns, and returns updated evolved guidelines.
        """
        now = datetime.datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO runs (run_id, timestamp, topic, total_iterations, final_status, shipped_content)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (run_id, now, topic, total_iterations, final_status, shipped_content))

            for entry in rejection_logs:
                for checkpoint in entry.failed_checkpoints:
                    cursor.execute("""
                        INSERT INTO rejections (run_id, iteration, timestamp, failed_checkpoint, reason, remediation)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        run_id,
                        entry.iteration,
                        entry.timestamp,
                        checkpoint,
                        "; ".join(entry.reasons),
                        entry.remediation_guidance
                    ))

            conn.commit()
        finally:
            conn.close()

        # Update JSON audit trail
        self._append_json_history(run_id, topic, total_iterations, final_status, rejection_logs)

        # Trigger self-evolution analysis
        self._evolve_guidelines()
        return self.get_active_guidelines()

    def _evolve_guidelines(self) -> List[str]:
        """
        Self-Evolving Logic:
        Scans historical rejections across all runs. If a checkpoint fails repeatedly,
        synthesizes proactive prompt guardrails to prevent future occurrences.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT failed_checkpoint, COUNT(*) as freq
                FROM rejections
                GROUP BY failed_checkpoint
                HAVING freq >= 1
            """)
            checkpoint_frequencies = cursor.fetchall()

            for checkpoint, freq in checkpoint_frequencies:
                derived_rule = self._map_checkpoint_to_evolved_rule(checkpoint, freq)
                if derived_rule:
                    cursor.execute("""
                        INSERT INTO evolved_guidelines (rule_text, source_checkpoint, frequency, created_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(rule_text) DO UPDATE SET
                            frequency = excluded.frequency
                    """, (derived_rule, checkpoint, freq, datetime.datetime.now().isoformat()))
            conn.commit()
        finally:
            conn.close()

        return self.get_active_guidelines()

    def _map_checkpoint_to_evolved_rule(self, checkpoint: str, frequency: int) -> Optional[str]:
        rules = {
            "NO_UNEXPLAINED_JARGON": (
                "Guardrail (Evolved): Scanned repeatedly for jargon. Always introduce any technical term using: "
                "1) Name in bold, 2) Simple everyday translation, 3) Real-life example immediately in the same sentence."
            ),
            "BEGINNER_FRIENDLY_LANGUAGE": (
                "Guardrail (Evolved): Past drafts exceeded grade level 8. Keep sentences strictly under 18 words. "
                "Use conversational Indian phrasing and break lengthy paragraphs into 3-point bullet lists."
            ),
            "TEACHES_BY_EXAMPLE": (
                "Guardrail (Evolved): Must include the 'Open-Book vs Closed-Book Exam' analogy in Section 1 "
                "before presenting any technical definitions."
            ),
            "KEY_POINTS_COVERAGE": (
                "Guardrail (Evolved): Verify before drafting that all 3 canonical headers are present: "
                "'1. What is RAG', '2. Why it matters', and '3. How it works'."
            ),
            "COHERENT_TEACHING_FLOW": (
                "Guardrail (Evolved): Always end the lesson with a 3-question multiple choice self-check quiz "
                "with clear answer explanations for self-assessment."
            ),
            "ACCURACY_AND_GROUNDED": (
                "Guardrail (Evolved): Clearly emphasize that RAG does NOT retrain the AI model; it simply "
                "provides external documents at query time."
            )
        }
        return rules.get(checkpoint)

    def get_active_guidelines(self) -> List[str]:
        """Retrieves all active self-evolved guidelines from persistent memory."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT rule_text FROM evolved_guidelines ORDER BY frequency DESC")
            rows = cursor.fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def _append_json_history(
        self,
        run_id: str,
        topic: str,
        total_iterations: int,
        final_status: str,
        rejection_logs: List[RejectionLogEntry]
    ):
        history = []
        if self.json_log_path.exists():
            try:
                with open(self.json_log_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []

        record = {
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "topic": topic,
            "total_iterations": total_iterations,
            "final_status": final_status,
            "rejections": [
                {
                    "iteration": r.iteration,
                    "failed_checkpoints": r.failed_checkpoints,
                    "reasons": r.reasons,
                    "remediation": r.remediation_guidance
                }
                for r in rejection_logs
            ]
        }
        history.append(record)

        with open(self.json_log_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
