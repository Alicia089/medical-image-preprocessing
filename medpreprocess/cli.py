import json
from pathlib import Path

import click

from medpreprocess.pipeline import run_pipeline


@click.group()
def cli():
    """medpreprocess — config-driven medical image preprocessing CLI."""


@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to YAML pipeline config")
@click.option("--input", "input_path", required=True, type=click.Path(), help="Input image path")
@click.option("--output", "output_dir", required=True, type=click.Path(), help="Output directory")
def run(config, input_path, output_dir):
    """Run a preprocessing pipeline on an image."""
    try:
        result = run_pipeline(config, input_path, output_dir)
    except (ValueError, FileNotFoundError, IOError) as e:
        raise click.ClickException(str(e))

    meta = result["metadata"]
    click.echo(f"Output:  {meta['output']}")
    click.echo(f"Steps:   {[s['step'] for s in meta['steps_applied']]}")
    click.echo(f"Elapsed: {meta['elapsed_seconds']}s")

    if meta.get("artifact_warnings"):
        for w in meta["artifact_warnings"]:
            click.echo(f"WARNING: {w}", err=True)
