from __future__ import annotations

import asyncio
import json
import shutil
import sys
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer

import click
from rich.console import Console
from rich.table import Table

from agenteval.reporter import write_report
from agenteval.runner import build_summary, load_test_cases, run_scenarios
from agenteval.schema.test_case import load_test_case
from agenteval.simulation.groq_simulator import GROQ_MODEL


console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="agenteval")
def cli() -> None:
    pass


@cli.command()
@click.argument("path", type=click.Path(exists=True))
def validate(path: str) -> None:
    target = Path(path)

    if target.is_file():
        files = [target]
    else:
        files = sorted(target.rglob("*.yaml"))

    valid_count = 0
    invalid_count = 0

    for file_path in files:
        try:
            load_test_case(file_path)
        except Exception as error:
            invalid_count += 1
            console.print(f"[red]✗ {file_path}[/red]")
            console.print(str(error))
        else:
            valid_count += 1
            console.print(f"[green]✓ {file_path}[/green]")

    console.print(f"{valid_count} valid, {invalid_count} invalid")

    if invalid_count:
        sys.exit(1)


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--mode",
    type=click.Choice(["scripted", "groq"]),
    default="scripted",
    show_default=True,
)
@click.option("--concurrency", type=int, default=4, show_default=True)
@click.option("--fail-on-threshold", type=float, default=None)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("reports"),
    show_default=True,
)
def run(
    path: str,
    mode: str,
    concurrency: int,
    fail_on_threshold: float | None,
    output_dir: Path,
) -> None:
    test_cases = load_test_cases(Path(path))
    if not test_cases:
        console.print("[red]No valid test cases found[/red]")
        sys.exit(1)

    results = asyncio.run(
        run_scenarios(test_cases, mode=mode, concurrency=concurrency)
    )
    scores = [
        result.aggregate_score
        for result in results
        if not getattr(result, "errored", False)
    ]
    summary = build_summary(results, scores, fail_on_threshold)
    model = GROQ_MODEL if mode == "groq" else None
    report_path = write_report(results, summary, mode, model, output_dir)

    table = Table(title="AgentEval Results")
    table.add_column("Scenario")
    table.add_column("Score", justify="right")
    table.add_column("Passed")
    table.add_column("Turns", justify="right")
    table.add_column("Termination")

    for result in results:
        passed = "yes" if result.passed else "no"
        style = "green" if result.passed else "red"
        table.add_row(
            result.scenario_name,
            f"{result.aggregate_score:.3f}",
            f"[{style}]{passed}[/{style}]",
            str(getattr(result, "turns_used", 0)),
            getattr(result, "termination_reason", "agent_error"),
        )

    console.print(table)
    console.print(f"Report written to {report_path}")

    if summary["overall_pass"]:
        console.print("[green]Overall pass[/green]")
    else:
        console.print("[red]Overall fail[/red]")

    if fail_on_threshold is not None and not summary["overall_pass"]:
        sys.exit(1)


@cli.command()
@click.argument(
    "reports_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def dashboard(reports_dir: Path) -> None:
    dashboard_dist = Path(__file__).parent / "dashboard_dist"
    reports_output = dashboard_dist / "reports"
    reports_output.mkdir(parents=True, exist_ok=True)

    report_files = sorted(reports_dir.glob("agenteval_report_*.json"))
    manifest = {"reports": []}
    for report_file in report_files:
        destination = reports_output / report_file.name
        shutil.copy2(report_file, destination)
        manifest["reports"].append(report_file.name)

    (reports_output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    url = "http://localhost:8080"
    handler = partial(SimpleHTTPRequestHandler, directory=str(dashboard_dist))
    console.print(f"Serving dashboard at {url}")
    webbrowser.open(url)

    with TCPServer(("", 8080), handler) as httpd:
        httpd.serve_forever()
