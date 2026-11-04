"""Command-line interface: ``pulse-qc run | compare | report``."""

from __future__ import annotations

from pathlib import Path

import typer

from pulse_qc import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Data-quality gating and beat-detection evaluation for pulsatile waveforms.",
)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"pulse-qc {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True),
) -> None:
    pass


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, help="YAML config"),
    root: Path = typer.Option(Path("."), help="Directory that cache/ and runs/ are relative to"),
    no_figures: bool = typer.Option(False, help="Skip figure generation"),
) -> None:
    """Run the pipeline for CONFIG (cached stages are skipped) and write a report."""
    from pulse_qc.pipeline import load_config
    from pulse_qc.pipeline import run as run_pipeline
    from pulse_qc.report import write_report

    result = run_pipeline(load_config(config), root=root)
    report = write_report(result.run_dir, figures=not no_figures)
    typer.echo(f"run directory: {result.run_dir}")
    for stage, c in result.cache_summary.items():
        typer.echo(f"  {stage:<22} hits={c['hits']:<3} misses={c['misses']}")
    if not result.summary.empty:
        overall = result.summary[(result.summary["activity"] == "all")]
        for _, r in overall.iterrows():
            typer.echo(
                f"  {r['detector']:<12} {r['gating']:<7} F1={r['f1']:.3f} "
                f"sens={r['sensitivity']:.3f} ppv={r['ppv']:.3f} n={int(r['n_ref'])}"
            )
    typer.echo(f"report: {report}")


@app.command()
def compare(
    config_a: Path = typer.Argument(..., exists=True),
    config_b: Path = typer.Argument(..., exists=True),
    root: Path = typer.Option(Path("."), help="Directory that cache/ and runs/ are relative to"),
    output: Path | None = typer.Option(None, help="Write the comparison markdown here"),
) -> None:
    """Run both configs (sharing the cache) and print what changed between them."""
    from pulse_qc.pipeline import load_config
    from pulse_qc.pipeline import run as run_pipeline
    from pulse_qc.report import compare_runs

    ra = run_pipeline(load_config(config_a), root=root)
    rb = run_pipeline(load_config(config_b), root=root)
    md = compare_runs(ra.run_dir, rb.run_dir)
    if output:
        output.write_text(md)
        typer.echo(f"wrote {output}")
    else:
        typer.echo(md)


@app.command()
def report(
    run_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="A runs/<name>-<hash> directory"
    ),
    no_figures: bool = typer.Option(False, help="Skip figure generation"),
) -> None:
    """Regenerate report.md and figures from a saved run directory."""
    from pulse_qc.report import write_report

    out = write_report(run_dir, figures=not no_figures)
    typer.echo(f"report: {out}")


@app.command()
def holdout(
    run_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="A runs/<name>-<hash> directory"
    ),
    rule: list[str] = typer.Option(["periodicity"], help="Single-threshold rule(s) to tune"),
    bad_f1: float = typer.Option(0.9, help="A window is 'bad' below this mean beat F1"),
) -> None:
    """Choose rule thresholds on held-out subjects (leave-one-subject-out) and score gating
    on subjects the threshold never saw."""
    from pulse_qc.holdout import write_holdout

    out = write_holdout(run_dir, list(rule), bad_f1=bad_f1)
    typer.echo(out.read_text())
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
