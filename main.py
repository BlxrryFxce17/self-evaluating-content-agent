"""
Main Entry Point & Terminal CLI for Content Systems Agent.
Clean developer console with structured tables, rejection audit logs, and flaw injection.
"""

import argparse
import sys
import io
import json
from pathlib import Path

# Fix Windows console UTF-8 encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule

from src.graph import ContentSystemWorkflow
from src.llm_client import LLMClient
from src.generator import LessonGenerator
from src.evaluator import ContentEvaluator
from src.regenerator import LessonRegenerator
from src.memory import PersistentMemoryEngine
from src.config import ARTIFACTS_DIR

console = Console(force_terminal=True, legacy_windows=False)


def display_welcome():
    console.print(Panel(
        "[bold white]Content Systems Engine[/bold white] | Self-Evaluating Lesson Generator\n"
        "[dim]Target: 12th-grade graduate (India, non-English-medium) | RAG Education Pipeline[/dim]",
        border_style="blue"
    ))


def display_evaluation_report(report, iteration: int):
    table = Table(title=f"Evaluator Checkpoints (Iteration {iteration})", show_header=True, header_style="bold cyan")
    table.add_column("Checkpoint", style="bold", width=32)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Findings & Metrics", width=55)

    for c in report.checkpoints:
        status_text = "[bold green]PASS[/bold green]" if c.passed else "[bold red]FAIL[/bold red]"
        metric_str = f"\n[dim]{c.metric_value}[/dim]" if c.metric_value else ""
        table.add_row(
            c.checkpoint.value,
            status_text,
            f"{c.reason}{metric_str}"
        )

    console.print(table)

    if not report.all_passed:
        console.print(Panel(
            f"[bold red]Quality Gate Rejected ({report.passed_count}/{report.total_count} Passed)[/bold red]\n\n"
            f"[yellow]{report.remediation_guidance}[/yellow]",
            title="Remediation Plan for Regenerator",
            border_style="red"
        ))
    else:
        console.print(Panel(
            f"[bold green]Quality Gate Passed ({report.passed_count}/{report.total_count})[/bold green]\n"
            "[green]Lesson cleared all 6 hard checkpoints. Approved for shipping.[/green]",
            title="Shipping Clearance",
            border_style="green"
        ))


def display_rejection_history(rejections):
    if not rejections:
        console.print("[dim]No rejections recorded. Cleared on iteration 1.[/dim]")
        return

    table = Table(title="Rejection Audit Trail", show_header=True, header_style="bold red")
    table.add_column("Iter", justify="center", width=6)
    table.add_column("Failed Checkpoints", style="bold yellow", width=28)
    table.add_column("Diagnostic Findings", width=40)
    table.add_column("Remediation Injected", style="cyan", width=25)

    for entry in rejections:
        table.add_row(
            str(entry.iteration),
            "\n".join(entry.failed_checkpoints),
            "\n".join(entry.reasons),
            entry.changes_demanded
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Autonomous Content Systems Agent")
    parser.add_argument("--topic", type=str, default="Introduction to RAG", help="Lesson topic")
    parser.add_argument("--provider", type=str, default="auto", choices=["auto", "gemini", "openai", "groq", "offline"], help="LLM Provider")
    parser.add_argument("--inject-flaw", type=str, default=None, choices=["jargon", "missing_how", "academic_voice"], help="Inject synthetic flaw for evaluator testing")
    parser.add_argument("--max-retries", type=int, default=2, help="Maximum regeneration attempts")
    parser.add_argument("--view-memory", action="store_true", help="Display persistent self-evolved guidelines")

    args = parser.parse_args()

    display_welcome()

    memory = PersistentMemoryEngine()

    if args.view_memory:
        rules = memory.get_active_guidelines()
        console.print(Rule("[bold white]Self-Evolved Memory Guidelines[/bold white]"))
        if not rules:
            console.print("[dim]No guidelines stored yet. Run iterations to trigger synthesis.[/dim]")
        for i, r in enumerate(rules, 1):
            console.print(f"[cyan]{i}.[/cyan] {r}")
        return

    # Initialize workflow components
    llm_client = LLMClient(provider=args.provider)
    console.print(f"[dim]Provider: [bold white]{llm_client.client_type.upper()}[/bold white][/dim]")

    generator = LessonGenerator(llm_client=llm_client)
    evaluator = ContentEvaluator()
    regenerator = LessonRegenerator(generator=generator)

    workflow = ContentSystemWorkflow(
        generator=generator,
        evaluator=evaluator,
        regenerator=regenerator,
        memory=memory,
        max_retries=args.max_retries
    )

    console.print(f"\n[bold]Target Topic:[/bold] [yellow]{args.topic}[/yellow]")
    if args.inject_flaw:
        console.print(f"[bold red]Flaw Injection Active:[/bold red] [yellow]{args.inject_flaw}[/yellow]")

    console.print(Rule("[dim]Workflow Execution[/dim]"))

    final_state = workflow.run(topic=args.topic, flaw_injection=args.inject_flaw)

    console.print(Rule("[dim]Evaluation Report[/dim]"))

    # Display evaluation details
    if final_state.get("evaluation"):
        display_evaluation_report(final_state["evaluation"], final_state["iteration"])

    # Display rejection log
    console.print(Rule("[dim]Audit Trail[/dim]"))
    display_rejection_history(final_state.get("rejection_history", []))

    # Summary
    status = final_state["status"]
    console.print(Rule(f"[bold]{'Final Pipeline Status: ' + status}[/bold]"))

    if status == "SHIPPED":
        console.print(f"[bold green]Artifact generated:[/bold green] [cyan]{ARTIFACTS_DIR / 'shipped_lesson.md'}[/cyan]")
        console.print(f"[bold green]Audit log saved:[/bold green] [cyan]{ARTIFACTS_DIR / 'rejection_log.json'}[/cyan]")
    else:
        console.print(f"[bold yellow]Terminated with status: {status}. Artifacts in {ARTIFACTS_DIR}[/bold yellow]")


if __name__ == "__main__":
    main()
