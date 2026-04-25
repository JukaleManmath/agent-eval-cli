from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

from agenteval.schema.test_case import load_test_case


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
