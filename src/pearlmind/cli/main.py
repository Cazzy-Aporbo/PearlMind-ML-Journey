"""Commands report actual work; no placeholder metrics or imaginary servers."""

from pathlib import Path
import json
import typer
from pearlmind import __version__
from pearlmind.utils.config import Config

app = typer.Typer(help="PearlMind: run, inspect and question a machine-learning experiment.")


def version_callback(value: bool):
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=version_callback, is_eager=True),
):
    pass


@app.command()
def train(
    config_path: Path = typer.Argument(Path("configs/default.yaml")),
    data: Path = typer.Option(None),
    target: str = "target",
    output: Path = Path("outputs/tabular"),
):
    """Train on a local CSV or seeded synthetic data; save held-out metrics and JSON model."""
    from pearlmind.lessons.tabular import run

    report = run(output, Config.from_yaml(config_path), data, target)
    typer.echo(json.dumps(report, indent=2))


@app.command()
def evaluate(
    model_path: Path,
    data_path: Path,
    target: str = "target",
    output: Path = Path("outputs/evaluation.json"),
):
    """Evaluate against actual labels in a separate CSV. The group column is optional."""
    import pandas as pd
    from pearlmind.models import load_model

    frame = pd.read_csv(data_path)
    y = frame.pop(target)
    groups = frame.pop("group") if "group" in frame else None
    model = load_model(model_path)
    report = model.audit_fairness(frame, y, groups)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    typer.echo(output.read_text())


@app.command()
def audit(data_path: Path, output: Path = Path("outputs/audit.json")):
    """Read a CSV containing actual, prediction and group columns; report observed gaps."""
    import pandas as pd
    from pearlmind.evaluation import FairnessAuditor

    d = pd.read_csv(data_path)
    report = FairnessAuditor().audit(d.actual, d.prediction, d.group)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    typer.echo(output.read_text())


@app.command()
def serve(model_path: Path, host: str = "127.0.0.1", port: int = 8000):
    """Run the local teaching API. Install the api extra first."""
    import uvicorn
    from pearlmind.deployment.api import create_app

    uvicorn.run(create_app(model_path), host=host, port=port)


@app.command(name="list")
def list_models(directory: Path = Path("outputs")):
    """List model metadata that actually exists on disk."""
    for path in sorted(directory.rglob("*.metadata.json")):
        typer.echo(str(path))


@app.command()
def config(action: str, path: Path = typer.Option(None), output: Path = typer.Option(None)):
    """Create, show or validate configuration."""
    if action == "create" and output:
        Config().save_yaml(output)
    elif action in ("show", "validate") and path:
        typer.echo(Config.from_yaml(path).model_dump_json(indent=2))
    else:
        raise typer.BadParameter("Use create --output FILE or show/validate --path FILE")


if __name__ == "__main__":
    app()
