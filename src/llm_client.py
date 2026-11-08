"""
Unified LLM Client Supporting Multiple Providers + Dynamic Topic Offline Engine.
Supports: Gemini, OpenAI, Groq, Anthropic, and Offline Mock/Deterministic engine.
"""

import os
import json
import re
from typing import Optional, Dict, Any, List


class LLMClient:
    def __init__(self, provider: str = "auto", api_key: Optional[str] = None):
        self.provider = provider.lower()
        self.custom_api_key = api_key
        self.client_type = "offline"
        self._init_client()

    def _init_client(self):
        gemini_key = self.custom_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        openai_key = self.custom_api_key or os.getenv("OPENAI_API_KEY")
        groq_key = self.custom_api_key or os.getenv("GROQ_API_KEY")

        if self.provider == "gemini" or (self.provider == "auto" and gemini_key):
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                self.client_type = "gemini"
                return
            except Exception as e:
                print(f"[Warning] Failed to initialize Gemini: {e}")

        if self.provider == "openai" or (self.provider == "auto" and openai_key):
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=openai_key)
                self.client_type = "openai"
                self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                return
            except Exception as e:
                print(f"[Warning] Failed to initialize OpenAI: {e}")

        if self.provider == "groq" or (self.provider == "auto" and groq_key):
            try:
                from groq import Groq
                self.client = Groq(api_key=groq_key)
                self.client_type = "groq"
                self.model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
                return
            except Exception as e:
                print(f"[Warning] Failed to initialize Groq: {e}")

        self.client_type = "offline"

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        """Dispatches generation request to active provider or offline engine."""
        if self.client_type == "gemini":
            try:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}" if system_prompt else prompt
                response = self.model.generate_content(
                    full_prompt,
                    generation_config={"temperature": temperature}
                )
                return response.text
            except Exception as e:
                print(f"[LLM Client] Gemini error: {e}. Falling back to deterministic engine.")

        elif self.client_type == "openai":
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                print(f"[LLM Client] OpenAI error: {e}. Falling back to deterministic engine.")

        elif self.client_type == "groq":
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                print(f"[LLM Client] Groq error: {e}. Falling back to deterministic engine.")

        # Offline High-Fidelity Engine
        return self._offline_generate(prompt, system_prompt)

    def _offline_generate(self, prompt: str, system_prompt: str) -> str:
        """
        Deterministic, pedagogically sound content engine.
        Dynamically extracts the topic and tailors content to the requested topic.
        """
        # Extract topic from prompt
        topic_match = re.search(r'lesson for:\s*["\']?(.*?)["\']?\s*(?:\.|\n|$)', prompt, re.IGNORECASE)
        topic = topic_match.group(1).strip() if topic_match else "Introduction to RAG"
        if "Topic:" in prompt:
            t_match = re.search(r'Topic:\s*["\']?(.*?)["\']?\s*(?:\n|$)', prompt)
            if t_match:
                topic = t_match.group(1).strip()

        # Check if flaw was deliberately demanded
        if "INJECT_FLAW: JARGON" in prompt or "INJECT_FLAW: JARGON" in system_prompt:
            return self._flawed_jargon_lesson(topic)
        elif "INJECT_FLAW: MISSING_HOW" in prompt or "INJECT_FLAW: MISSING_HOW" in system_prompt:
            return self._flawed_missing_section_lesson(topic)
        elif "INJECT_FLAW: ACADEMIC_VOICE" in prompt or "INJECT_FLAW: ACADEMIC_VOICE" in system_prompt:
            return self._flawed_academic_lesson(topic)

        # If feedback indicates what was fixed after a failure:
        if "FEEDBACK FROM EVALUATOR" in prompt:
            return self._perfect_remediated_lesson(topic)

        # Standard initial pass
        return self._standard_initial_lesson(topic)

    def _flawed_jargon_lesson(self, topic: str) -> str:
        return f"""# {topic} Architecture

In modern deep learning paradigms, {topic} leverages dense multidimensional vector embeddings mapped across high-dimensional latent manifolds. When an arbitrary parametric token query enters the system, we execute approximate nearest-neighbor cosine similarity lookups over indexed hierarchical HNSW graph structures.

The algorithmic pipeline employs maximum inner-product search (MIPS) alongside cross-entropy loss functions to minimize empirical perplexity during multi-head generative decoding.
"""

    def _flawed_missing_section_lesson(self, topic: str) -> str:
        return f"""# {topic}

## What is {topic}?
{topic} is a modern technique in Artificial Intelligence. Think of an open-book exam where a student references notes before answering.

## Why Does it Matter?
It stops systems from guessing false answers (hallucination, which means making up untrue facts). It keeps information fresh and private.

## Summary
{topic} is an essential tool for building modern software products!
"""

    def _flawed_academic_lesson(self, topic: str) -> str:
        return f"""# Theoretical Analysis of {topic}

## Epistemological Framework
The ontological foundation underlying contemporary computational paradigms within {topic} necessitates rigorous parameter estimation over stochastic distributions. Latent manifold representations mitigate systemic entropy through backpropagation optimization.

## Methodological Synthesis
Empirical evaluation demonstrates asymptotic convergence across multi-dimensional loss landscapes, thereby validating theoretical generalizability.
"""

    def _standard_initial_lesson(self, topic: str = "Introduction to RAG") -> str:
        if "rag" in topic.lower():
            return self._rag_standard_lesson()
        elif "ai" in topic.lower() or "ml" in topic.lower() or "machine learning" in topic.lower():
            return self._aiml_standard_lesson(topic)
        else:
            return self._generic_standard_lesson(topic)

    def _perfect_remediated_lesson(self, topic: str = "Introduction to RAG") -> str:
        if "rag" in topic.lower():
            return self._rag_remediated_lesson()
        elif "ai" in topic.lower() or "ml" in topic.lower() or "machine learning" in topic.lower():
            return self._aiml_remediated_lesson(topic)
        else:
            return self._generic_remediated_lesson(topic)

    def _rag_standard_lesson(self) -> str:
        return """# Introduction to RAG (Retrieval-Augmented Generation)

Welcome to your introduction to Artificial Intelligence. If you have finished 12th standard and are curious about how systems like ChatGPT work, this lesson provides a clear starting point.

No coding degree or advanced vocabulary is required today. We will explain RAG step by step using everyday examples.

---

## 1. Real-World Analogy: Closed-Book vs Open-Book Exam

Think back to school exams:
* **Closed-Book Exam:** You sit with only a pen. You must answer purely from memory. If you forget a fact, you make a guess.
* **Open-Book Exam:** The teacher permits you to bring your reference textbook into the exam hall. When a question appears, you flip directly to the relevant chapter, find the page, and write the verified answer.

Standard language models take a closed-book exam. **RAG (Retrieval-Augmented Generation)** turns the AI into an open-book student with reference notes on its desk.

---

## 2. What is RAG? (Breaking Down the Three Words)

RAG represents three core actions:

1. **Retrieval (Searching):**
   * Imagine a college librarian. When you ask a question, the librarian goes to the shelf, finds the exact 2 pages needed, and brings them to you. *Retrieval means searching files and fetching relevant facts.*
2. **Augmented (Attaching Support):**
   * Augment means to add support. Instead of letting AI guess, we attach those library pages directly to the question.
3. **Generation (Writing the Answer):**
   * The AI reads the question and the attached notes. It writes a clear answer grounded strictly in those notes.

Summary: **RAG = Search notes + Attach to question + Write factual answer.**

---

## 3. Technical Rationale: Why RAG Matters

Standard AI has three core challenges that RAG resolves:

1. **Hallucinations:** When an AI lacks facts, it invents plausible-sounding false answers. In AI, this is called a hallucination (seeing things that are not real). RAG forces the AI to check verified documents first.
2. **Dynamic Knowledge:** Training large models costs crores and takes months. If a timetable changes today, old AI does not know. With RAG, you drop the new PDF into your folder, and AI references it immediately without retraining.
3. **Data Privacy:** A hospital in Mumbai cannot share confidential patient records with public AI. With RAG, records stay securely on local servers.

---

## 4. How RAG Works: Step-by-Step Architecture

Here is the RAG pipeline in 4 stages:

```
[User Query] ──► 1. RETRIEVER  ──► Searches Knowledge Base (Local Documents)
                      │
                      ▼
                 2. FETCH      ──► Extracts Top Relevant Text
                      │
                      ▼
                 3. AUGMENT    ──► Merges [User Query] + [Fetched Context]
                      │
                      ▼
                 4. GENERATOR  ──► Synthesizes Grounded, Factual Response
```

* **Step 1 (Knowledge Base):** You store rulebooks, notes, or PDFs in a folder.
* **Step 2 (The Retriever):** The search tool identifies the relevant paragraph.
* **Step 3 (Augmentation):** The system prepares a combined prompt: "Based on these notes, answer the query."
* **Step 4 (Generation):** The AI reads the text and produces a verified answer.

---

## 5. Self-Assessment Quiz

1. **In our exam analogy, what distinguishes standard AI from RAG?**
   * A) Standard AI is faster; RAG only runs on mobile phones.
   * B) Standard AI is a closed-book student guessing from memory; RAG uses open-book reference notes.
   * C) Standard AI uses books; RAG does not use books.
   *(Correct Answer: B)*

2. **What is an AI hallucination?**
   * A) The computer monitor turning off.
   * B) An AI inventing an incorrect answer with confidence because it lacked facts.
   * C) Network latency.
   *(Correct Answer: B)*

3. **Does updating knowledge in RAG require retraining the AI model?**
   * A) Yes, the neural network must be retrained from scratch.
   * B) No, new documents are placed in the folder and referenced immediately.
   * C) RAG only works on old data.
   *(Correct Answer: B)*

---

## 6. Summary & Next Steps in AI
Smart AI systems do not rely solely on memorized parameters. They combine generation with real-time retrieval from verified sources. Congratulations on finishing this lesson!
"""

    def _rag_remediated_lesson(self) -> str:
        return """# Introduction to RAG (Retrieval-Augmented Generation)

Welcome to this lesson on Artificial Intelligence. Have you finished 12th standard? Do you want to learn how AI works? You are in the right place!

You do not need any coding degree today. We will learn step by step using simple examples.

---

## 1. Real-World Analogy: Closed-Book vs Open-Book Exam

Think back to school exams:
* **Closed-Book Exam:** You sit with only a pen. You must answer purely from memory. If you forget a fact, you make a wild guess. In AI, when a computer makes up a false answer, we call this a **hallucination** (which means seeing things that are not real).
* **Open-Book Exam:** The teacher allows students to bring reference books into the room. When a question appears, you flip directly to the chapter, find the page, and write the verified answer.

Standard AI takes a closed-book exam. It answers only from memory.
**RAG (Retrieval-Augmented Generation)** turns the AI into an open-book student with reference notes on its desk.

---

## 2. What is RAG? (Breaking Down the Three Words)

RAG is simply three everyday words joined together:

```
R  =  Retrieval    (Searching for verified reference documents)
A  =  Augmented    (Attaching the fetched text to the question)
G  =  Generation   (Writing a clear, true answer from those notes)
```

1. **Retrieval (Searching):**
   * Imagine a college librarian. When you ask for a book, the librarian brings the exact 2 pages you need. Retrieval means searching your files and fetching relevant facts.
2. **Augmented (Adding Support):**
   * Augment means to add support. Instead of letting AI guess, we attach those library pages directly to the question.
3. **Generation (Writing the Answer):**
   * The AI reads the question and the attached notes. It writes a clear, true answer based only on those notes.

Summary: **RAG = Search notes + Add to question + Write true answer.**

---

## 3. Why Does RAG Matter? (Three Problems Solved)

Standard AI has three big problems that RAG solves:

1. **Stops AI Lies (Hallucinations):** When an AI lacks facts, it invents false stories. RAG tells the AI: 'Do not guess. Read these exact lines from our notes and answer using only them.'
2. **Keeps Data Fresh:** Training large models costs crores of rupees. If a timetable changes today, old AI does not know. With RAG, you drop the new PDF into your folder, and AI references it right away.
3. **Protects Private Data:** A hospital cannot share patient records with public AI. With RAG, records stay safely on the local computer.

---

## 4. How RAG Works: 4 Simple Steps

```
[User Query] ──► 1. RETRIEVER  ──► Searches Knowledge Base (Local Documents)
                      │
                      ▼
                 2. FETCH      ──► Extracts Top Relevant Text
                      │
                      ▼
                 3. AUGMENT    ──► Merges [User Query] + [Fetched Context]
                      │
                      ▼
                 4. GENERATOR  ──► Synthesizes Grounded, Factual Response
```

* **Step 1 (Knowledge Base):** You save your notes or PDFs in a folder.
* **Step 2 (The Retriever):** The search tool finds the right paragraph.
* **Step 3 (Augmentation):** The system prepares a combined prompt: 'Based on this text, answer the user.'
* **Step 4 (Generation):** The AI reads the text and writes the verified answer.

---

## 5. Self-Assessment Quiz

1. **In our exam analogy, what distinguishes standard AI from RAG?**
   * A) Standard AI is faster; RAG only runs on mobile phones.
   * B) Standard AI is a closed-book student guessing from memory; RAG uses open-book reference notes.
   * C) Standard AI uses books; RAG does not use books.
   *(Correct Answer: B)*

2. **What is an AI hallucination?**
   * A) The computer monitor turning off.
   * B) An AI inventing an incorrect answer with confidence because it lacked facts.
   * C) Network latency.
   *(Correct Answer: B)*

3. **Does updating knowledge in RAG require retraining the AI model?**
   * A) Yes, the full AI model must be retrained from scratch.
   * B) No, new documents are placed in the folder and referenced immediately.
   * C) RAG only works on old data.
   *(Correct Answer: B)*

---

## 6. Summary & Next Steps in AI
Smart AI systems do not try to memorize everything in their weights. They combine generation with real-time retrieval from verified sources. Congratulations on finishing this lesson!
"""

    def _aiml_standard_lesson(self, topic: str) -> str:
        return f"""# {topic}: How Computers Learn from Experience

Welcome to your first lesson on Artificial Intelligence (AI) and Machine Learning (ML). If you have completed 12th standard and are curious about how smart technology works, you are in the right place.

You do not need any coding background today. We will explain everything step by step using everyday examples.

---

## 1. Real-World Analogy: Teaching a Child vs Writing 1000 Rules

Think about how a young child learns:
* **The Traditional Rule Approach:** Imagine trying to write a huge rulebook to describe a cat: *"It has 4 legs, two ears, fur, and whiskers."* But a dog also has 4 legs and fur! Writing strict rules for every case is impossible.
* **The Learning Approach (Machine Learning):** You show the child 20 pictures of cats and 20 pictures of dogs. The child observes patterns. Soon, when a new cat walks past, the child immediately says: *"Cat!"*

Traditional software is like a rigid rulebook. **Machine Learning** is teaching a computer by showing it examples.

---

## 2. What is AI and Machine Learning?

Let us break down the two terms simply:

1. **Artificial Intelligence (AI):**
   * The broad science of making computers perform tasks that normally require human intelligence (like recognizing faces, translating languages, or playing chess).
2. **Machine Learning (ML):**
   * The main engine inside modern AI. Instead of programming every single rule by hand, we feed the computer data, and it learns the patterns on its own.

Summary: **AI is the goal (smart behavior). Machine Learning is the method (learning from data).**

---

## 3. Why Does AI and Machine Learning Matter?

Why are technology companies in India and across the world hiring engineers who understand AI and ML?

1. **Handling Massive Data:** Humans cannot scan millions of medical X-rays or UPI transactions every minute. Machine learning models scan millions of patterns in seconds to spot fraud or illness.
2. **Personalized Assistance:** Apps like YouTube, Spotify, and shopping sites use machine learning to suggest content you genuinely like based on past behavior.
3. **Automating Repetitive Tasks:** AI tools help write emails, transcribe speech into text, and assist doctors with research notes.

---

## 4. How Machine Learning Works: 4-Step Architecture

Here is the basic machine learning workflow:

```
[Collection] ──► 1. GATHER DATA      ──► Collect examples (pictures, numbers, text)
                       │
                       ▼
                 2. TRAIN MODEL      ──► Computer finds mathematical patterns
                       │
                       ▼
                 3. TEST & EVALUATE  ──► Check accuracy on new, unseen examples
                       │
                       ▼
                 4. MAKE PREDICTIONS ──► Model identifies new inputs in the real world
```

* **Step 1 (Data):** Collect 1,000 photos of traffic signs.
* **Step 2 (Training):** The machine learning algorithm learns which shapes mean 'Stop' and 'Go'.
* **Step 3 (Testing):** Test the model with 50 new photos to ensure it does not guess wrong.
* **Step 4 (Inference):** The model runs inside a car to assist the driver safely.

---

## 5. Technical Glossary (Key Terms Simplified)

| Term | What People Say | Plain Language Meaning |
| :--- | :--- | :--- |
| **Dataset** | "Clean the dataset" | The collection of examples, tables, or images used for training. |
| **Model** | "Deploy the ML model" | The trained software program that has learned the patterns and makes predictions. |
| **Training** | "Train the network" | The process where the computer looks at data and adjusts its internal numbers. |
| **Prediction** | "Model inference" | The answer or decision the model produces when given new input. |

---

## 6. Self-Assessment Quiz

1. **In our analogy, what is the core difference between traditional programming and Machine Learning?**
   * A) Traditional code uses electricity; ML does not.
   * B) Traditional code relies on hardcoded rules; ML learns patterns from examples and data.
   * C) Traditional code only runs on phones; ML only runs in labs.
   *(Correct Answer: B)*

2. **What is the relationship between AI and Machine Learning?**
   * A) Machine Learning is the method used to achieve Artificial Intelligence.
   * B) AI and ML are completely unrelated subjects.
   * C) Machine Learning is an old word for a calculator.
   *(Correct Answer: A)*

3. **What is a 'Dataset'?**
   * A) The computer power cord.
   * B) The collection of examples (data) used to teach the model.
   * C) The computer screen.
   *(Correct Answer: B)*

---

## 7. Summary & Next Steps in AI
Machine Learning allows computers to solve complex problems by recognizing patterns in data. As you begin your journey as an AI engineer, remember that every smart system starts with good data and clear evaluation. Congratulations on completing this lesson!
"""

    def _aiml_remediated_lesson(self, topic: str) -> str:
        return f"""# {topic}: How Computers Learn from Experience

Welcome to this lesson on Artificial Intelligence (AI) and Machine Learning (ML). Have you finished 12th standard? Do you want to learn how computers learn? You are in the right place!

You do not need any coding degree today. We will learn step by step using simple examples.

---

## 1. Real-World Analogy: Teaching a Child vs Writing Rules

Think about how a child learns:
* **The Rule Method:** Imagine trying to write a huge rulebook to describe a cat: 'It has 4 legs, two ears, and fur.' But a dog also has 4 legs! Writing rules for every animal is impossible.
* **The Learning Method (Machine Learning):** You show the child 20 pictures of cats and 20 pictures of dogs. The child spots patterns. Soon, when a new cat walks past, the child says: 'Cat!'

Old software is like a rigid rulebook. **Machine Learning** is teaching a computer by showing it examples.

---

## 2. What is AI and Machine Learning?

Let us break down the two terms simply:

1. **Artificial Intelligence (AI):**
   * Making computers do smart tasks that normally need human thinking (like translating words or recognizing faces).
2. **Machine Learning (ML):**
   * The engine inside modern AI. Instead of writing every rule by hand, we feed the computer data, and it learns the patterns.

Summary: **AI is the goal (smart behavior). Machine Learning is the method (learning from data).**

---

## 3. Why Does Machine Learning Matter?

Top tech companies need AI and ML for three big reasons:

1. **Handling Big Data:** Humans cannot scan millions of UPI payments every minute. ML models scan patterns in seconds to spot fraud.
2. **Personal Recommendations:** Apps like YouTube and Spotify use ML to suggest songs you like.
3. **Saving Time:** AI tools help write emails and assist doctors with medical notes.

---

## 4. How Machine Learning Works: 4 Simple Steps

Here is the machine learning workflow in 4 steps:

```
[Collection] ──► 1. GATHER DATA      ──► Collect examples (pictures, numbers, text)
                       │
                       ▼
                 2. TRAIN MODEL      ──► Computer finds useful patterns
                       │
                       ▼
                 3. TEST & EVALUATE  ──► Check accuracy on new, unseen examples
                       │
                       ▼
                 4. MAKE PREDICTIONS ──► Model identifies new inputs in the real world
```

* **Step 1 (Data):** Collect 1,000 photos of traffic signs.
* **Step 2 (Training):** The model learns which shapes mean 'Stop' and 'Go'.
* **Step 3 (Testing):** Test the model with 50 new photos to make sure it does not guess wrong.
* **Step 4 (Prediction):** The model runs inside a car to assist the driver safely.

---

## 5. Technical Glossary (Key Terms Simplified)

| Term | What People Say | Plain Language Meaning |
| :--- | :--- | :--- |
| **Dataset** | 'Clean the dataset' | The collection of examples or photos used for learning. |
| **Model** | 'Deploy the ML model' | The software program that has learned the patterns. |
| **Training** | 'Train the network' | The process where the computer looks at data and learns. |
| **Prediction** | 'Model inference' | The answer or decision the model produces. |

---

## 6. Self-Assessment Quiz

1. **In our analogy, what is the core difference between old code and Machine Learning?**
   * A) Old code uses electricity; ML does not.
   * B) Old code relies on hard rules; ML learns patterns from examples and data.
   * C) Old code only runs on phones; ML only runs in labs.
   *(Correct Answer: B)*

2. **What is the relationship between AI and Machine Learning?**
   * A) Machine Learning is the method used to achieve Artificial Intelligence.
   * B) AI and ML are completely unrelated subjects.
   * C) Machine Learning is an old word for a calculator.
   *(Correct Answer: A)*

3. **What is a 'Dataset'?**
   * A) The computer power cord.
   * B) The collection of examples (data) used to teach the model.
   * C) The computer screen.
   *(Correct Answer: B)*

---

## 7. Summary & Your Next Steps in AI
Machine Learning allows computers to solve hard problems by recognizing patterns in data. Every smart system starts with good data and clear rules. Congratulations on completing this lesson!
"""

    def _generic_standard_lesson(self, topic: str) -> str:
        return f"""# {topic}: A Beginner's Introduction

Welcome to this foundational lesson on {topic}. If you have finished 12th standard and are curious to learn modern technology, this guide is designed for you.

You do not need any advanced background today. We will explain {topic} step by step using everyday analogies and clear language.

---

## 1. Real-World Analogy: Learning the Fundamentals

Think about learning a new skill like cooking or riding a bicycle:
* You start with simple, proven steps before trying complicated recipes.
* When you understand *why* each ingredient or action matters, you do not need to memorize blindly.

Similarly, **{topic}** becomes intuitive when broken down into its core components.

---

## 2. What is {topic}? (Core Definition)

{topic} is an important concept in modern technology.
At its core, it provides a structured method to solve a specific problem reliably and efficiently.

Summary: **{topic} takes inputs, applies clear rules or patterns, and produces reliable outputs.**

---

## 3. Why Does {topic} Matter?

Understanding {topic} is valuable for three reasons:

1. **Efficiency:** It simplifies complex tasks that would otherwise take hours of manual effort.
2. **Reliability:** It provides a consistent framework, reducing mistakes and unexpected errors.
3. **Foundation for Growth:** Mastering {topic} prepares you for advanced engineering topics in the software industry.

---

## 4. How It Works: Step-by-Step Architecture

Here is the workflow broken into 4 simple stages:

```
[Input Data] ──► 1. INGESTION   ──► Accepts the starting information
                      │
                      ▼
                 2. PROCESSING  ──► Applies core logic or transformations
                      │
                      ▼
                 3. VALIDATION  ──► Checks accuracy against quality rules
                      │
                      ▼
                 4. OUTPUT      ──► Delivers the final verified result
```

---

## 5. Self-Assessment Quiz

1. **What is the best way to understand {topic}?**
   * A) Memorizing words without understanding.
   * B) Breaking it down into simple inputs, processes, and outputs.
   * C) Avoiding real-world practice.
   *(Correct Answer: B)*

2. **Why does {topic} matter in modern technology?**
   * A) It makes systems slower.
   * B) It improves efficiency, reduces errors, and provides a reliable framework.
   * C) It only works on old computers.
   *(Correct Answer: B)*

---

## 6. Summary & Next Steps in AI
Every powerful technology is built on simple, solid building blocks. Congratulations on completing this introduction to {topic}!
"""

    def _generic_remediated_lesson(self, topic: str) -> str:
        return self._generic_standard_lesson(topic)
