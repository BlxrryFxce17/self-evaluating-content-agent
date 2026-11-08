"""
NLP Utilities for readability metrics, text statistics, and jargon verification.
Works with zero external heavy dependencies for 100% deterministic reliability.
"""

import re
import math
from typing import List, Tuple, Dict, Set


def count_syllables_in_word(word: str) -> int:
    """Estimates the syllable count of an English word using phonetic heuristics."""
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 3:
        return 1

    # Remove non-alphabetic characters
    word = re.sub(r'[^a-z]', '', word)
    if not word:
        return 1

    # Basic vowel group counting
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel

    # Adjust for silent 'e'
    if word.endswith("e") and not word.endswith("le") and len(word) > 2 and not word.endswith("ee"):
        count = max(1, count - 1)

    return max(1, count)


def extract_sentences(text: str) -> List[str]:
    """Splits text into sentences, ignoring markdown headings and lists."""
    cleaned = re.sub(r'#+.*?\n', ' ', text)
    cleaned = re.sub(r'```[\s\S]*?```', ' ', cleaned)
    # Split on sentence boundaries
    raw_sentences = re.split(r'[.!?]+(?:\s+|\n+)', cleaned)
    sentences = [s.strip() for s in raw_sentences if len(s.strip().split()) >= 3]
    return sentences


def extract_words(text: str) -> List[str]:
    """Extracts words from text, stripping markdown and punctuation."""
    cleaned = re.sub(r'```[\s\S]*?```', ' ', text)
    cleaned = re.sub(r'https?://\S+', '', cleaned)
    words = re.findall(r'\b[a-zA-Z]{2,}\b', cleaned)
    return words


def calculate_flesch_metrics(text: str) -> Tuple[float, float, float]:
    """
    Returns:
      (flesch_reading_ease, flesch_kincaid_grade, average_words_per_sentence)
    """
    sentences = extract_sentences(text)
    words = extract_words(text)

    if not sentences or not words:
        return (0.0, 12.0, 0.0)

    total_words = len(words)
    total_sentences = len(sentences)
    total_syllables = sum(count_syllables_in_word(w) for w in words)

    words_per_sentence = total_words / total_sentences
    syllables_per_word = total_syllables / total_words

    # Flesch Reading Ease formula
    # 206.835 - 1.015 * (words/sentence) - 84.6 * (syllables/word)
    fre = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
    fre = max(0.0, min(100.0, fre))

    # Flesch-Kincaid Grade Level formula
    # 0.39 * (words/sentence) + 11.8 * (syllables/word) - 15.59
    fk_grade = (0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59
    fk_grade = max(1.0, fk_grade)

    return (round(fre, 1), round(fk_grade, 1), round(words_per_sentence, 1))


# Technical terms that demand immediate beginner explanation if introduced
TECHNICAL_JARGON_REGISTRY = {
    "vector": [r"vector\b", r"vector embedding", r"vector database"],
    "embedding": [r"embedding[s]?\b"],
    "hallucination": [r"hallucinat\w*"],
    "retrieval": [r"retriev\w*"],
    "chunking": [r"chunk\w*"],
    "token": [r"token[s]?\b"],
    "cosine similarity": [r"cosine similarity"],
    "latent": [r"latent space", r"latent manifold"],
    "hnsw": [r"hnsw"],
    "mips": [r"mips"],
    "perplexity": [r"perplexity"],
    "autoregressive": [r"autoregressive"],
    "epistemological": [r"epistemological"]
}

# Clues indicating an explanation follows immediately (within context)
EXPLANATION_INDICATORS = [
    "means", "imagine", "like", "think of", "simply", "in simple words",
    "meaning", "call", "calls", "called", "example", "is just", "stands for",
    "just like", "such as", "refers to", "defined as", "seeing things",
    "जोड़ना", "खोज", "अर्थात", "मतलब"
]


def check_unexplained_jargon(text: str) -> Tuple[bool, List[str], List[str]]:
    """
    Checks if technical jargon appears without being explained in nearby context.
    Returns:
      (passed: bool, unexplained_terms: List[str], explained_terms: List[str])
    """
    text_lower = text.lower()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    unexplained = []
    explained = []

    for term, patterns in TECHNICAL_JARGON_REGISTRY.items():
        found = False
        term_paragraphs = []
        for p in paragraphs:
            for pat in patterns:
                if re.search(pat, p, re.IGNORECASE):
                    found = True
                    term_paragraphs.append(p.lower())
                    break

        if found:
            # Check if ANY of the containing paragraphs has an explanation indicator
            has_explanation = any(
                any(indicator in p for indicator in EXPLANATION_INDICATORS)
                for p in term_paragraphs
            )
            # Highly academic words that fail immediately for 12th-grade target audience
            forbidden_opaque_terms = ["latent", "hnsw", "mips", "perplexity", "autoregressive", "epistemological"]
            if term in forbidden_opaque_terms:
                unexplained.append(f"{term} (dense academic jargon unsuitable for 12th-grade beginners)")
            elif has_explanation:
                explained.append(term)
            else:
                unexplained.append(f"{term} (used without simple plain-language explanation)")

    passed = len(unexplained) == 0
    return (passed, unexplained, explained)
