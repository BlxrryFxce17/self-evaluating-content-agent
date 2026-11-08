"""
Configuration settings for the Content Systems Agent.
Reads environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from src.models import AudienceProfile

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MEMORY_DIR = BASE_DIR / "memory_store"

ARTIFACTS_DIR.mkdir(exist_ok=True)
MEMORY_DIR.mkdir(exist_ok=True)

# Agent loop configurations
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
DEFAULT_MODEL_PROVIDER = os.getenv("DEFAULT_MODEL_PROVIDER", "auto").lower()

# Readability & Linguistic Rubric Thresholds
# For 12th-grade Indian non-English-medium learners, target reading grade level should be 6-8 (Middle School English)
MAX_FLESCH_GRADE_LEVEL = 8.5
MIN_FLESCH_READING_EASE = 55.0  # Plain English benchmark (55-70 = accessible conversational English)

# Required technical dimensions for "Introduction to RAG"
REQUIRED_CORE_DIMENSIONS = [
    "what_is_rag",          # Definition: Retrieval + Augmentation + Generation
    "why_it_matters",       # Problem: LLM Hallucinations, knowledge cutoff, private docs
    "how_it_works",         # Flow: Ingestion/Knowledge Base -> Retrieval -> Augmented Prompt -> Output
    "real_world_analogy",   # Analogy grounded in relatable context
    "practical_takeaway",   # Concrete example or career takeaway
    "quiz_check"            # 2-3 quick questions to test understanding
]

# Jargon terms requiring immediate simple explanations if present
MONITORED_JARGON_TERMS = [
    "retrieval",
    "augmented",
    "generation",
    "hallucination",
    "vector",
    "embedding",
    "chunking",
    "knowledge base",
    "token",
    "prompt"
]

AUDIENCE_PROFILE = AudienceProfile()

