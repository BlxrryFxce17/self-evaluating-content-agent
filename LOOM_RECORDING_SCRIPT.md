# Video Walkthrough Script (Simple & Direct)

> **Instructions:**  
> Keep this document open on half your screen (or on your phone) while you record.  
> Read the text in quotes naturally at a calm, conversational pace.

---

## Part 1: Introduction (1–2 Minutes)

**What to do on screen:**  
* Face visible in webcam corner.
* Screen shows your GitHub repository: `https://github.com/BlxrryFxce17/self-evaluating-content-agent`

**What to say:**
> "Hi everyone, my name is Akash. This is my submission for the GenAI Engineer - Content Systems assessment.
>
> In this assignment, our task was not just to write a prompt. It was to build an agentic system that writes a lesson, checks its own quality against strict standards, and fixes its own mistakes before shipping.
>
> The target student is someone who just finished 12th standard in India from a non-English-medium school.
>
> Because of that, the lesson cannot have complicated English or unexplained technical jargon. It needs simple words and relatable everyday examples, like an open-book exam.
>
> Let me show you how the system is built."

---

## Part 2: Architecture & The 6 Checkpoints (2 Minutes)

**What to do on screen:**  
* Scroll down to the flowchart in the `README.md`.

**What to say:**
> "Here is our architecture, built using LangGraph.
>
> It works in a loop with three main parts:
> 
> 1. **The Generator:** It writes the initial draft of the lesson using our student persona.
> 2. **The Evaluator:** It judges the lesson against 6 strict pass/fail checkpoints. We don't give partial credit like '7 out of 10'—each check either strictly passes or strictly fails.
> 3. **The Regenerator:** If any check fails, the system does not give up. It takes the exact failure reasons, tells the model what to fix, and regenerates the lesson. To be safe, it has a limit of 2 retries so it never gets stuck in an infinite loop.
>
> If all 6 checks pass, the lesson is shipped.
>
> Now, let’s run it in the terminal."

---

## Part 3: Live End-to-End Run (3–4 Minutes)

**What to do on screen:**  
* Open your terminal in VS Code and run:
```powershell
python main.py
```

**What to say while it runs:**
> "I am running `python main.py` for our topic: 'Introduction to RAG'.
>
> Look at the table on screen:
>
> In **Iteration 1**, the generator created a draft. But look at Checkpoint 2: `BEGINNER_FRIENDLY_LANGUAGE` failed. 
>
> Our evaluator calculated the reading grade level as 8.7, which is too hard for our 12th-grade student.
>
> So the evaluator rejected it! It created a **Rejection Log** and gave clear instructions: 'Shorten complex sentences and use everyday words.'
>
> Then, in **Iteration 2**, the regenerator took that feedback and rewrote the lesson with simpler sentences.
>
> The evaluator checked it again, and now all 6 checkpoints passed with a green PASS!
>
> The final status is **SHIPPED**, and the lesson is saved in our artifacts folder."

---

## Part 4: Evaluator Catching a Deliberate Error (3–4 Minutes)

**What to do on screen:**  
* In the terminal, run:
```powershell
python main.py --inject-flaw jargon
```

**What to say while it runs:**
> "Now, to prove our evaluator actually catches bad content and doesn't just pass everything, I built a test switch called `--inject-flaw jargon`.
>
> This deliberately injects heavy academic text with words like 'HNSW', 'latent manifolds', and 'cosine similarity'.
>
> Look at the screen now—it lit up in RED!
>
> 1. `NO_UNEXPLAINED_JARGON` failed immediately because those words had no explanation.
> 2. `BEGINNER_FRIENDLY_LANGUAGE` failed with a huge grade level of 22.8!
> 3. `TEACHES_BY_EXAMPLE` failed because there was no relatable analogy.
>
> Our evaluator caught every single flaw, rejected the draft, and sent the fix list to the regenerator.
>
> On the very next iteration, the system fixed all those problems, brought back our open-book exam analogy, and cleared all 6 checkpoints to ship."

---

## Part 5: Persistent Memory & Conclusion (1–2 Minutes)

**What to do on screen:**  
* In the terminal, run:
```powershell
python main.py --view-memory
```

**What to say:**
> "Finally, the system has persistent memory using an SQLite database.
>
> When I run `python main.py --view-memory`, we can see the rules the system learned from past failures.
>
> For example, because jargon failed previously, it added a permanent rule: 'Always explain technical terms with a simple example immediately.'
>
> On future runs, the generator reads these rules so it doesn't make the same mistakes again.
>
> In addition to the CLI, I also built a Streamlit console and deployed a live interactive demo on GitHub Pages.
>
> That is the complete system: an autonomous loop that generates, evaluates with hard binary checks, fixes its own errors, and learns over time.
>
> Thank you so much for your time!"

---

### 💡 Tips for Recording
* **Pacing:** Speak at your normal conversational pace. There is no need to rush.
* **Terminal Font:** Zoom in (`Ctrl + +`) so the tables are crystal clear on camera.
* **Total Time:** Around 8 to 12 minutes is the ideal sweet spot.

