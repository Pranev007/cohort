"""Command line interface.

cohort generate     build the synthetic enterprise corpus
cohort scan         run the full pipeline and write findings
cohort evaluate     score the pipeline against ground truth
cohort show         print a single finding in full
cohort demo         generate -> scan -> evaluate in one command
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from cohort import __version__
from cohort.config import CohortConfig
from cohort.evaluate.report import run_evaluation, write_report
from cohort.pipeline import run_pipeline, write_artifacts
from cohort.synthorg import generate_corpus

app = typer.Typer(
    add_completion=False, help="Peer-baseline anomaly detection for unstructured data."
)
console = Console()


def _load(config: Path | None) -> CohortConfig:
    return CohortConfig.load(config)


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"cohort {__version__}")


@app.command()
def generate(
    config: Path = typer.Option(None, "--config", "-c", help="YAML config path."),
    out: Path = typer.Option(Path("artifacts/corpus"), "--out", "-o"),
    documents: int = typer.Option(None, "--documents", "-n", help="Override document count."),
    seed: int = typer.Option(None, "--seed", help="Override the RNG seed."),
) -> None:
    """Build the synthetic enterprise corpus with labelled anomalies."""
    cfg = _load(config)
    if documents:
        cfg.synthorg.n_documents = documents
    if seed is not None:
        cfg.synthorg.seed = seed

    with console.status(f"generating {cfg.synthorg.n_documents:,} documents..."):
        paths = generate_corpus(cfg.synthorg, out)

    t = Table(title="Corpus generated", show_header=True)
    t.add_column("artifact")
    t.add_column("path")
    for k, v in paths.items():
        t.add_row(k, str(v))
    console.print(t)


@app.command()
def scan(
    config: Path = typer.Option(None, "--config", "-c"),
    corpus: Path = typer.Option(Path("artifacts/corpus/corpus.parquet"), "--corpus"),
    findings: int = typer.Option(500, "--findings", help="How many top findings to explain."),
    no_remediation: bool = typer.Option(False, "--no-remediation"),
    html_report: bool = typer.Option(
        False, "--html", help="Also write a self-contained HTML report."
    ),
) -> None:
    """Run the full pipeline: embed, cluster, baseline, score, explain."""
    cfg = _load(config)
    if not corpus.exists():
        console.print(f"[red]corpus not found:[/red] {corpus}\nRun `cohort generate` first.")
        raise typer.Exit(1)

    with console.status("scanning..."):
        result = run_pipeline(
            cfg, corpus, max_findings=findings, with_remediation=not no_remediation
        )
        out = write_artifacts(result, cfg)

    console.print(f"\n[bold]{result.summary()}[/bold]\n")

    t = Table(title="Stage timings", show_header=True)
    t.add_column("stage")
    t.add_column("seconds", justify="right")
    for k, v in result.timings.items():
        t.add_row(k, f"{v:.2f}")
    console.print(t)

    t = Table(title="Discovered cohorts", show_header=True)
    t.add_column("id", justify="right")
    t.add_column("size", justify="right")
    t.add_column("label")
    for n in sorted(result.names.values(), key=lambda x: -x.size)[:20]:
        t.add_row(str(n.cohort_id), f"{n.size:,}", n.label)
    console.print(t)

    t = Table(title="Top findings", show_header=True)
    t.add_column("doc")
    t.add_column("cohort")
    t.add_column("risk", justify="right")
    t.add_column("p", justify="right")
    t.add_column("drivers")
    for _, r in result.findings.head(10).iterrows():
        t.add_row(
            r.doc_id,
            r.cohort_name[:26],
            f"{r.risk_score:.2f}",
            f"{r.conformal_p:.4f}",
            r.top_features[:52],
        )
    console.print(t)

    if html_report:
        from cohort.report_html import write_report as write_html

        out["html"] = write_html(
            result.findings,
            result.stats,
            cfg.paths.reports / "cohorts.json",
            cfg.paths.reports / "scan_report.html",
            result.timings,
        )

    console.print(f"\nwrote: {', '.join(str(p) for p in out.values())}")


@app.command()
def evaluate(
    config: Path = typer.Option(None, "--config", "-c"),
    corpus: Path = typer.Option(Path("artifacts/corpus/corpus.parquet"), "--corpus"),
    truth: Path = typer.Option(Path("artifacts/corpus/ground_truth.parquet"), "--truth"),
    out: Path = typer.Option(Path("artifacts/reports"), "--out", "-o"),
    quick: bool = typer.Option(False, "--quick", help="Skip ablations and sensitivity."),
) -> None:
    """Evaluate detection against ground truth and write the metrics report."""
    cfg = _load(config)
    for p in (corpus, truth):
        if not p.exists():
            console.print(f"[red]missing:[/red] {p}\nRun `cohort generate` first.")
            raise typer.Exit(1)

    with console.status("evaluating..."):
        report = run_evaluation(
            cfg,
            corpus,
            truth,
            with_ablations=not quick,
            with_sensitivity=not quick,
        )
        paths = write_report(report, out)

    console.print(report.to_markdown())
    console.print(f"\nwrote: {paths['markdown']}, {paths['json']}")


@app.command()
def show(
    doc_id: str = typer.Argument(..., help="Document id, e.g. doc-009314"),
    findings: Path = typer.Option(Path("artifacts/findings/findings.parquet"), "--findings"),
) -> None:
    """Print one finding in full: narrative, drivers, remediation plan."""
    import pandas as pd

    if not findings.exists():
        console.print(f"[red]no findings at[/red] {findings}\nRun `cohort scan` first.")
        raise typer.Exit(1)

    df = pd.read_parquet(findings)
    row = df[df.doc_id == doc_id]
    if row.empty:
        console.print(f"[yellow]{doc_id} is not among the top findings.[/yellow]")
        raise typer.Exit(1)

    r = row.iloc[0]
    console.print(f"\n[bold]{r.doc_id}[/bold]  {r.title}")
    console.print(
        f"[dim]cohort:[/dim] {r.cohort_name}   "
        f"[dim]risk:[/dim] {r.risk_score:.2f} nats   "
        f"[dim]conformal p:[/dim] {r.conformal_p:.4f}   "
        f"[dim]flagged:[/dim] {r.is_flagged}"
    )
    console.print(f"\n[bold]Why[/bold]\n{r.narrative}")
    console.print(f"\n[bold]Drivers[/bold]\n{r.top_features}")
    if r.remediation:
        console.print(f"\n[bold]Remediation[/bold]\n{r.remediation}")


@app.command()
def demo(
    config: Path = typer.Option(None, "--config", "-c"),
    documents: int = typer.Option(15000, "--documents", "-n"),
) -> None:
    """Generate, scan and evaluate in one command."""
    cfg = _load(config)
    cfg.synthorg.n_documents = documents
    cfg.paths.ensure()

    console.rule("[bold]1/3 generate")
    with console.status(f"generating {documents:,} documents..."):
        paths = generate_corpus(cfg.synthorg, cfg.paths.corpus)
    console.print(f"corpus: {paths['corpus']}")

    console.rule("[bold]2/3 scan")
    with console.status("scanning..."):
        result = run_pipeline(cfg, paths["corpus"])
        write_artifacts(result, cfg)
    console.print(result.summary())

    console.rule("[bold]3/3 evaluate")
    with console.status("evaluating..."):
        report = run_evaluation(cfg, paths["corpus"], paths["ground_truth"])
        out = write_report(report, cfg.paths.reports)
    console.print(report.to_markdown())
    console.print(f"\n[green]done[/green] — report at {out['markdown']}")


@app.command("real-eval")
def real_eval(
    config: Path = typer.Option(None, "--config", "-c"),
    data: Path = typer.Option(Path("data"), "--data", help="Where corpora are cached."),
    out: Path = typer.Option(Path("artifacts/reports"), "--out", "-o"),
    limit: int = typer.Option(40000, "--limit", help="Enron messages to sample."),
    skip_enron: bool = typer.Option(False, "--skip-enron"),
    skip_newsgroups: bool = typer.Option(False, "--skip-newsgroups"),
) -> None:
    """Run the real-corpus experiments (20 Newsgroups, Enron).

    Downloads roughly 450 MB on first run. Results are written separately from
    the synthetic benchmark and are never blended into it.
    """
    from cohort.real.evaluate import run_enron, run_newsgroups, to_markdown, write_reports

    cfg = _load(config)
    data.mkdir(parents=True, exist_ok=True)
    reports = []

    if not skip_newsgroups:
        from cohort.real.newsgroups import download as dl_news

        with console.status("20 Newsgroups: fetching..."):
            parquet = dl_news(data / "20news_train.parquet")
        with console.status("20 Newsgroups: clustering..."):
            reports.append(run_newsgroups(cfg, parquet))
        console.print("[green]20 Newsgroups done[/green]")

    if not skip_enron:
        from cohort.real.enron import download as dl_enron

        with console.status("Enron: fetching ~443 MB (first run only)..."):
            archive = dl_enron(data / "enron_mail.tar.gz")
        with console.status(f"Enron: streaming archive, sampling {limit:,} messages..."):
            reports.append(run_enron(cfg, archive, limit=limit))
        console.print("[green]Enron done[/green]")

    if not reports:
        console.print("[yellow]nothing to run[/yellow]")
        raise typer.Exit(1)

    paths = write_reports(reports, out)
    console.print(to_markdown(reports))
    console.print(f"\nwrote: {paths['markdown']}, {paths['json']}")


@app.command("dump-config")
def dump_config(
    out: Path = typer.Option(Path("configs/default.yaml"), "--out", "-o"),
) -> None:
    """Write the full default configuration to a YAML file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    CohortConfig().dump(out)
    console.print(f"wrote {out}")


@app.command("show-cohorts")
def show_cohorts(
    path: Path = typer.Option(Path("artifacts/reports/cohorts.json"), "--path"),
) -> None:
    """List the discovered cohorts and their keyphrases."""
    if not path.exists():
        console.print(f"[red]not found:[/red] {path}")
        raise typer.Exit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    t = Table(show_header=True)
    t.add_column("id", justify="right")
    t.add_column("size", justify="right")
    t.add_column("label")
    t.add_column("keyphrases")
    for c in data:
        t.add_row(str(c["cohort_id"]), f"{c['size']:,}", c["label"], ", ".join(c["keyphrases"][:5]))
    console.print(t)


if __name__ == "__main__":
    app()
