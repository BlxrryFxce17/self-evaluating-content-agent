"""
Data models and schemas for the Self-Evaluating Lesson Content Generator.
Enforces hard pass/fail types, structured feedback, and persistent logs.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import datetime


class CheckpointName(str, Enum):
    ACCURACY_AND_GROUNDED = "ACCURACY_AND_GROUNDED"
    BEGINNER_FRIENDLY_LANGUAGE = "BEGINNER_FRIENDLY_LANGUAGE"
    TEACHES_BY_EXAMPLE = "TEACHES_BY_EXAMPLE"
    NO_UNEXPLAINED_JARGON = "NO_UNEXPLAINED_JARGON"
    KEY_POINTS_COVERAGE = "KEY_POINTS_COVERAGE"
    COHERENT_TEACHING_FLOW = "COHERENT_TEACHING_FLOW"


class AudienceProfile(BaseModel):
    persona: str = "12th-grade graduate from India, limited English vocabulary, non-English-medium background, aspiring AI engineer"
    max_reading_grade_level: float = 8.0
    preferred_analogies: List[str] = Field(
        default_factory=lambda: [
            "Open-book exam vs closed-book memorization",
            "College library with a helpful assistant fetching specific notes",
            "Cricket umpire consulting third umpire / DRS replays",
            "Doctor checking a patient file before prescribing medicine"
        ]
    )
    language_rules: List[str] = Field(
        default_factory=lambda: [
            "Use everyday words (say 'find' instead of 'retrieve', 'make' instead of 'synthesize')",
            "Keep sentences short (under 20 words where possible)",
            "Every technical term must be introduced like introducing a new friend: Name -> Simple Meaning -> Real Example",
            "Tone: Warm, encouraging, respectful, like a supportive elder sibling ('Bhaiya' / mentor)"
        ]
    )


class CheckpointResult(BaseModel):
    checkpoint: CheckpointName
    passed: bool  # STRICT BINARY PASS / FAIL — NO PARTIAL CREDIT
    metric_value: Optional[str] = None
    reason: str
    suggested_fix: Optional[str] = None


class EvaluationReport(BaseModel):
    all_passed: bool
    passed_count: int
    total_count: int
    checkpoints: List[CheckpointResult]
    failure_reasons: List[str] = Field(default_factory=list)
    remediation_guidance: str = ""


class RejectionLogEntry(BaseModel):
    iteration: int
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    failed_checkpoints: List[str]
    reasons: List[str]
    remediation_guidance: str
    changes_demanded: str


class LessonDraft(BaseModel):
    topic: str
    title: str
    markdown_content: str
    iteration: int = 1
    word_count: int = 0
    estimated_read_time_minutes: int = 0


class MemoryRecord(BaseModel):
    run_id: str
    timestamp: str
    topic: str
    total_iterations: int
    final_status: str  # "SHIPPED" | "REJECTED_AFTER_RETRIES"
    rejection_logs: List[RejectionLogEntry]
    learned_insights: List[str]

