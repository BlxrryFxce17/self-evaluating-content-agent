"""
Streamlit Web Console: Content Systems Evaluator & Generator
Clean developer interface for running the autonomous evaluation loop,
inspecting binary checkpoints, and viewing rejection audit trails.
"""

import streamlit as st
import json
from pathlib import Path

from src.graph import ContentSystemWorkflow
from src.llm_client import LLMClient
from src.generator import LessonGenerator
from src.evaluator import ContentEvaluator
from src.regenerator import LessonRegenerator
from src.memory import PersistentMemoryEngine
from src.config import ARTIFACTS_DIR

st.set_page_config(
    page_title="Content Systems Agent | Lesson Evaluator",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal, clean CSS styling (neutral dark theme, no AI slop gradients)
st.markdown("""
<style>
    /* Global font and background overrides */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    /* Clean headers */
    .main-header {
        font-size: 1.5rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.875rem;
        color: #8b949e;
        margin-bottom: 1.5rem;
    }
    
    /* Checkpoint cards */
    .checkpoint-card-pass {
        background-color: #0d1e18;
        border: 1px solid #1f4738;
        border-left: 3px solid #238636;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .checkpoint-card-fail {
        background-color: #261114;
        border: 1px solid #541d24;
        border-left: 3px solid #da3633;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    
    .checkpoint-title {
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .checkpoint-reason {
        font-size: 0.8rem;
        color: #c9d1d9;
        margin-top: 4px;
    }
    .checkpoint-meta {
        font-size: 0.75rem;
        color: #8b949e;
        margin-top: 4px;
        font-family: monospace;
    }
    
    /* Metric pill */
    .metric-container {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        font-family: monospace;
        color: #f0f6fc;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Top Header
st.markdown('<div class="main-header">Content Systems Engine • Self-Evaluating Lesson Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Autonomous generation loop with hard binary checkpoints, rubric feedback, and persistent run memory.</div>', unsafe_allow_html=True)

# Sidebar: Controls
st.sidebar.markdown("### Execution Parameters")
topic = st.sidebar.text_input("Lesson Topic", value="Introduction to RAG")
provider = st.sidebar.selectbox("LLM Provider", ["auto", "offline", "gemini", "openai", "groq"], index=0)
api_key = st.sidebar.text_input("API Key (Optional)", type="password", help="Enter Google Gemini, OpenAI, or Groq API key to use live models. Leave empty for dynamic offline engine.")
max_retries = st.sidebar.slider("Max Regeneration Retries", min_value=1, max_value=3, value=2)

st.sidebar.markdown("---")
st.sidebar.markdown("### Flaw Injection (Evaluator Verification)")
flaw_selection = st.sidebar.selectbox(
    "Error Mode",
    [
        "None (Standard Production)",
        "jargon (Unexplained Technical Jargon)",
        "missing_how (Missing Core Mechanism)",
        "academic_voice (Dense Academic English)"
    ],
    index=0
)

flaw_param = None
if "jargon" in flaw_selection:
    flaw_param = "jargon"
elif "missing_how" in flaw_selection:
    flaw_param = "missing_how"
elif "academic_voice" in flaw_selection:
    flaw_param = "academic_voice"

st.sidebar.markdown("---")
st.sidebar.markdown("### Audience Constraints")
st.sidebar.markdown("""
- **Target Persona:** 12th-grade graduate (India, non-English-medium)
- **Flesch-Kincaid Max:** Grade 8.5
- **Flesch Reading Ease Min:** 60.0
- **Sentence Length Max:** 22 words
- **Unexplained Jargon:** 0 tolerance
""")

# Action button
if st.button("Run Pipeline", type="primary"):
    with st.spinner("Executing agent workflow..."):
        memory = PersistentMemoryEngine()
        client = LLMClient(provider=provider, api_key=api_key.strip() if api_key else None)
        generator = LessonGenerator(llm_client=client)
        evaluator = ContentEvaluator()
        regenerator = LessonRegenerator(generator=generator)

        workflow = ContentSystemWorkflow(
            generator=generator,
            evaluator=evaluator,
            regenerator=regenerator,
            memory=memory,
            max_retries=max_retries
        )

        final_state = workflow.run(topic=topic, flaw_injection=flaw_param)

    # Top summary metrics
    status = final_state["status"]
    eval_report = final_state.get("evaluation")
    iterations = final_state["iteration"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Pipeline Status</div>
            <div class="metric-value" style="color: {'#238636' if status == 'SHIPPED' else '#da3633'};">{status}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        passed_text = f"{eval_report.passed_count}/{eval_report.total_count}" if eval_report else "-"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Checkpoints Passed</div>
            <div class="metric-value">{passed_text}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Total Iterations</div>
            <div class="metric-value">{iterations}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        rejections_count = len(final_state.get("rejection_history", []))
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Rejections Intercepted</div>
            <div class="metric-value">{rejections_count}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Tabs for Inspection
    tab_lesson, tab_checkpoints, tab_audit, tab_memory = st.tabs([
        "Lesson Output",
        "Evaluator Checkpoints",
        "Rejection Log",
        "Persistent Memory"
    ])

    with tab_lesson:
        if final_state.get("final_lesson"):
            col_left, col_right = st.columns([0.85, 0.15])
            with col_right:
                st.download_button(
                    label="Download .md",
                    data=final_state["final_lesson"],
                    file_name="lesson_output.md",
                    mime="text/markdown"
                )
            st.markdown(final_state["final_lesson"])

    with tab_checkpoints:
        if eval_report:
            st.markdown(f"**Final Evaluation State (Iteration {iterations}):**")
            for c in eval_report.checkpoints:
                card_class = "checkpoint-card-pass" if c.passed else "checkpoint-card-fail"
                status_label = "PASS" if c.passed else "FAIL"
                status_color = "#3fb950" if c.passed else "#f85149"
                metric_str = f" • Metrics: {c.metric_value}" if c.metric_value else ""
                fix_str = f"<div style='color: #d29922; font-size: 0.75rem; margin-top: 4px;'>Remediation: {c.suggested_fix}</div>" if (not c.passed and c.suggested_fix) else ""

                st.markdown(f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="checkpoint-title" style="color: {status_color};">{c.checkpoint.value}</span>
                        <span style="font-family: monospace; font-size: 0.75rem; color: {status_color}; font-weight: bold;">[{status_label}]</span>
                    </div>
                    <div class="checkpoint-reason">{c.reason}</div>
                    <div class="checkpoint-meta">{metric_str}</div>
                    {fix_str}
                </div>
                """, unsafe_allow_html=True)

    with tab_audit:
        rejections = final_state.get("rejection_history", [])
        if not rejections:
            st.info("No rejections occurred. Draft met all rubric criteria on the first attempt.")
        else:
            for r in rejections:
                with st.expander(f"Iteration {r.iteration} Rejection — Failed: {', '.join(r.failed_checkpoints)}", expanded=True):
                    st.markdown("**Failure Reasons:**")
                    for reason in r.reasons:
                        st.markdown(f"- `{reason}`")
                    st.markdown("**Remediation Prompt Injected to Generator:**")
                    st.code(r.remediation_guidance, language="text")

    with tab_memory:
        guidelines = memory.get_active_guidelines()
        if guidelines:
            st.markdown("**Dynamically Synthesized Guardrails (Stored in SQLite):**")
            for idx, g in enumerate(guidelines, 1):
                st.markdown(f"`{idx}.` {g}")
        else:
            st.info("No guidelines evolved yet. Run iterations with flaw injection to observe automated rule synthesis.")

else:
    # Initial state screen
    st.info("Select parameters in the sidebar and click **'Run Pipeline'** to execute the agentic loop.")
    memory = PersistentMemoryEngine()
    guidelines = memory.get_active_guidelines()
    if guidelines:
        with st.expander("Active Self-Evolved Guardrails in SQLite", expanded=False):
            for idx, g in enumerate(guidelines, 1):
                st.markdown(f"- {g}")
