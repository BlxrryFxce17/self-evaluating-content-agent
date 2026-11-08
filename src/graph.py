"""
LangGraph State Machine for the Self-Evaluating Lesson Content Generator.
Orchestrates: Generate -> Evaluate -> Conditional Decision (Ship / Regenerate / Terminate).
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
import uuid
import datetime
from langgraph.graph import StateGraph, END

from src.models import (
    EvaluationReport,
    RejectionLogEntry,
    CheckpointName
)
from src.generator import LessonGenerator
from src.evaluator import ContentEvaluator
from src.regenerator import LessonRegenerator
from src.memory import PersistentMemoryEngine
from src.config import MAX_RETRIES, ARTIFACTS_DIR


class AgentWorkflowState(TypedDict):
    run_id: str
    topic: str
    flaw_injection: Optional[str]
    iteration: int
    max_retries: int
    current_draft: str
    evaluation: Optional[EvaluationReport]
    rejection_history: List[RejectionLogEntry]
    evolved_guidelines: List[str]
    status: str
    final_lesson: Optional[str]


class ContentSystemWorkflow:
    def __init__(
        self,
        generator: Optional[LessonGenerator] = None,
        evaluator: Optional[ContentEvaluator] = None,
        regenerator: Optional[LessonRegenerator] = None,
        memory: Optional[PersistentMemoryEngine] = None,
        max_retries: int = MAX_RETRIES
    ):
        self.generator = generator or LessonGenerator()
        self.evaluator = evaluator or ContentEvaluator()
        self.regenerator = regenerator or LessonRegenerator(self.generator)
        self.memory = memory or PersistentMemoryEngine()
        self.max_retries = max_retries
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentWorkflowState)

        # Register Workflow Nodes
        workflow.add_node("generate_initial", self._node_generate)
        workflow.add_node("evaluate", self._node_evaluate)
        workflow.add_node("regenerate", self._node_regenerate)
        workflow.add_node("ship", self._node_ship)
        workflow.add_node("terminate_max_retries", self._node_terminate_max_retries)

        # Set Entrypoint
        workflow.set_entry_point("generate_initial")

        # Edge: Generate -> Evaluate
        workflow.add_edge("generate_initial", "evaluate")

        # Conditional Edge: Evaluate -> [Ship | Regenerate | Terminate]
        workflow.add_conditional_edges(
            "evaluate",
            self._route_after_evaluation,
            {
                "ship": "ship",
                "regenerate": "regenerate",
                "terminate": "terminate_max_retries"
            }
        )

        # Edge: Regenerate -> Evaluate
        workflow.add_edge("regenerate", "evaluate")

        # Final Terminal Edges
        workflow.add_edge("ship", END)
        workflow.add_edge("terminate_max_retries", END)

        return workflow.compile()

    # --- Nodes ---

    def _node_generate(self, state: AgentWorkflowState) -> Dict[str, Any]:
        """Generates the initial lesson draft."""
        topic = state["topic"]
        flaw = state.get("flaw_injection")
        evolved_guidelines = state.get("evolved_guidelines", [])

        draft = self.generator.generate_lesson(
            topic=topic,
            evolved_guidelines=evolved_guidelines,
            flaw_injection=flaw
        )

        return {
            "current_draft": draft,
            "status": "initial_draft_generated"
        }

    def _node_evaluate(self, state: AgentWorkflowState) -> Dict[str, Any]:
        """Evaluates draft against 6 hard pass/fail checkpoints."""
        draft = state["current_draft"]
        topic = state.get("topic", "Introduction to RAG")
        report = self.evaluator.evaluate(draft, topic=topic)

        return {
            "evaluation": report,
            "status": "evaluated"
        }

    def _route_after_evaluation(self, state: AgentWorkflowState) -> str:
        """Determines next action based on strict binary pass/fail and retry limit."""
        report = state["evaluation"]
        iteration = state["iteration"]
        max_retries = state["max_retries"]

        if report.all_passed:
            return "ship"

        if iteration > max_retries:
            return "terminate"

        return "regenerate"

    def _node_regenerate(self, state: AgentWorkflowState) -> Dict[str, Any]:
        """Logs rejection and regenerates lesson using remediation guidance."""
        current_iter = state["iteration"]
        evaluation = state["evaluation"]
        topic = state["topic"]
        previous_draft = state["current_draft"]
        rejection_history = list(state.get("rejection_history", []))

        # Build Rejection Log Entry
        failed_checkpoints = [c.checkpoint.value for c in evaluation.checkpoints if not c.passed]
        log_entry = RejectionLogEntry(
            iteration=current_iter,
            failed_checkpoints=failed_checkpoints,
            reasons=evaluation.failure_reasons,
            remediation_guidance=evaluation.remediation_guidance,
            changes_demanded=f"Addressing failures in: {', '.join(failed_checkpoints)}"
        )
        rejection_history.append(log_entry)

        # Trigger Regenerator with diagnostic feedback
        revised_draft = self.regenerator.regenerate_lesson(
            topic=topic,
            previous_draft=previous_draft,
            evaluation=evaluation,
            iteration=current_iter + 1,
            evolved_guidelines=state.get("evolved_guidelines", [])
        )

        return {
            "current_draft": revised_draft,
            "iteration": current_iter + 1,
            "rejection_history": rejection_history,
            "status": f"regenerated_iteration_{current_iter + 1}"
        }

    def _node_ship(self, state: AgentWorkflowState) -> Dict[str, Any]:
        """Persists the passing lesson and updates self-evolving memory."""
        final_lesson = state["current_draft"]
        run_id = state["run_id"]
        topic = state["topic"]
        total_iterations = state["iteration"]
        rejections = state.get("rejection_history", [])

        # Record run and evolve memory guidelines
        updated_guidelines = self.memory.record_run(
            run_id=run_id,
            topic=topic,
            total_iterations=total_iterations,
            final_status="SHIPPED",
            rejection_logs=rejections,
            shipped_content=final_lesson
        )

        # Save artifacts
        self._save_artifacts(run_id, topic, final_lesson, rejections, is_shipped=True)

        return {
            "final_lesson": final_lesson,
            "status": "SHIPPED",
            "evolved_guidelines": updated_guidelines
        }

    def _node_terminate_max_retries(self, state: AgentWorkflowState) -> Dict[str, Any]:
        """Safely terminates loop when max retries are exceeded without passing."""
        run_id = state["run_id"]
        topic = state["topic"]
        total_iterations = state["iteration"]
        rejections = state.get("rejection_history", [])

        self.memory.record_run(
            run_id=run_id,
            topic=topic,
            total_iterations=total_iterations,
            final_status="REJECTED_AFTER_RETRIES",
            rejection_logs=rejections,
            shipped_content=state["current_draft"]
        )

        self._save_artifacts(run_id, topic, state["current_draft"], rejections, is_shipped=False)

        return {
            "status": "TERMINATED_MAX_RETRIES",
            "final_lesson": state["current_draft"]
        }

    def _save_artifacts(
        self,
        run_id: str,
        topic: str,
        content: str,
        rejection_logs: List[RejectionLogEntry],
        is_shipped: bool
    ):
        """Writes deliverables to disk for inspection and reporting."""
        status_prefix = "shipped" if is_shipped else "rejected"
        lesson_file = ARTIFACTS_DIR / f"{status_prefix}_lesson.md"
        log_file = ARTIFACTS_DIR / "rejection_log.json"

        with open(lesson_file, "w", encoding="utf-8") as f:
            f.write(content)

        serializable_logs = [entry.model_dump() for entry in rejection_logs]
        with open(log_file, "w", encoding="utf-8") as f:
            import json
            json.dump(serializable_logs, f, indent=2)

    def run(self, topic: str, flaw_injection: Optional[str] = None) -> AgentWorkflowState:
        """Executes the complete state graph pipeline."""
        initial_state: AgentWorkflowState = {
            "run_id": f"run_{uuid.uuid4().hex[:8]}",
            "topic": topic,
            "flaw_injection": flaw_injection,
            "iteration": 1,
            "max_retries": self.max_retries,
            "current_draft": "",
            "evaluation": None,
            "rejection_history": [],
            "evolved_guidelines": self.memory.get_active_guidelines(),
            "status": "initialized",
            "final_lesson": None
        }

        final_state = self.graph.invoke(initial_state)
        return final_state

