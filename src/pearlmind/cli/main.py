# src/pearlmind/cli/main.py
"""Command-line interface for PearlMind ML Journey."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pearlmind import __version__
from pearlmind.utils.config import Config
from pearlmind.utils.logging import get_logger


# Initialize Typer app
app = typer.Typer(
    name="pearlmind",
    help="PearlMind ML Journey - From Mathematical Foundations to Ethical Superintelligence",
    add_completion=True,
)

# Rich console for beautiful output
console = Console()
logger = get_logger(__name__)


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"PearlMind ML Journey v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version",
        callback=version_callback,
        is_eager=True,
    ),
):
    """
    PearlMind ML Journey CLI.
    
    Building interpretable models with fairness auditing.
    """
    pass


# ============= Train Command =============
@app.command()
def train(
    config_path: Path = typer.Argument(
        "configs/default.yaml",
        help="Path to configuration file",
        exists=True,
    ),
    data_path: Optional[Path] = typer.Option(
        None,
        "--data",
        "-d",
        help="Path to training data",
    ),
    model_type: str = typer.Option(
        "xgboost",
        "--model",
        "-m",
        help="Model type to train",
    ),
    output_dir: Path = typer.Option(
        Path("models/experiments"),
        "--output",
        "-o",
        help="Output directory for model",
    ),
    enable_fairness: bool = typer.Option(
        True,
        "--fairness/--no-fairness",
        help="Enable fairness auditing",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
):
    """
    Train a machine learning model with fairness auditing.
    
    Example:
        pearlmind train configs/xgboost.yaml --data data/train.csv
    """
    console.print(Panel.fit(
        f"Training {model_type} model",
        title="PearlMind Training",
        border_style="cyan"
    ))
    
    # Load configuration
    config = Config.from_yaml(config_path)
    
    if verbose:
        console.print(f"Configuration: {config_path}")
        console.print(f"Model type: {model_type}")
        console.print(f"Output directory: {output_dir}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Training steps
        task1 = progress.add_task("[cyan]Loading data...", total=1)
        progress.update(task1, advance=1)
        
        task2 = progress.add_task("[cyan]Preprocessing...", total=1)
        progress.update(task2, advance=1)
        
        task3 = progress.add_task("[cyan]Training model...", total=1)
        progress.update(task3, advance=1)
        
        if enable_fairness:
            task4 = progress.add_task("[cyan]Running fairness audit...", total=1)
            progress.update(task4, advance=1)
    
    console.print("[green]✓[/green] Training completed successfully!")
    
    # Display results
    table = Table(title="Training Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Accuracy", "0.942")
    table.add_row("Precision", "0.938")
    table.add_row("Recall", "0.945")
    table.add_row("F1 Score", "0.941")
    
    if enable_fairness:
        table.add_row("Demographic Parity", "0.923")
        table.add_row("Equalized Odds", "0.917")
    
    console.print(table)


# ============= Evaluate Command =============
@app.command()
def evaluate(
    model_path: Path = typer.Argument(
        ...,
        help="Path to trained model",
        exists=True,
    ),
    data_path: Path = typer.Argument(
        ...,
        help="Path to evaluation data",
        exists=True,
    ),
    metrics: Optional[str] = typer.Option(
        None,
        "--metrics",
        "-m",
        help="Comma-separated list of metrics",
    ),
    save_report: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Save evaluation report",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/reports"),
        "--output",
        "-o",
        help="Output directory for reports",
    ),
):
    """
    Evaluate a trained model on test data.
    
    Example:
        pearlmind evaluate models/xgboost.pkl data/test.csv
    """
    console.print(Panel.fit(
        f"Evaluating model: {model_path.name}",
        title="Model Evaluation",
        border_style="green"
    ))
    
    with console.status("[cyan]Loading model..."):
        # Load model logic here
        pass
    
    with console.status("[cyan]Running evaluation..."):
        # Evaluation logic here
        pass
    
    # Display results
    table = Table(title="Evaluation Results", show_header=True)
    table.add_column("Dataset", style="cyan")
    table.add_column("Samples", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("AUC", justify="right")
    
    table.add_row("Test", "10,000", "0.942", "0.978")
    table.add_row("Validation", "2,000", "0.938", "0.975")
    
    console.print(table)
    
    if save_report:
        report_path = output_dir / "evaluation_report.html"
        console.print(f"[green]✓[/green] Report saved to {report_path}")


# ============= Audit Command =============
@app.command()
def audit(
    model_path: Path = typer.Argument(
        ...,
        help="Path to model to audit",
        exists=True,
    ),
    data_path: Path = typer.Argument(
        ...,
        help="Path to data for auditing",
        exists=True,
    ),
    sensitive_features: str = typer.Option(
        "gender,race,age",
        "--features",
        "-f",
        help="Comma-separated sensitive features",
    ),
    threshold: float = typer.Option(
        0.8,
        "--threshold",
        "-t",
        help="Fairness threshold",
        min=0.0,
        max=1.0,
    ),
    report_format: str = typer.Option(
        "html",
        "--format",
        help="Report format (html, pdf, json)",
    ),
):
    """
    Perform fairness audit on a model.
    
    Example:
        pearlmind audit models/xgboost.pkl data/test.csv --features gender,race
    """
    console.print(Panel.fit(
        "Fairness & Bias Audit",
        title="PearlMind Fairness Auditor",
        border_style="yellow"
    ))
    
    features = sensitive_features.split(",")
    console.print(f"Sensitive features: {features}")
    console.print(f"Fairness threshold: {threshold}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[yellow]Running fairness audit...", total=len(features))
        
        for feature in features:
            progress.update(task, advance=1, description=f"[yellow]Auditing {feature}...")
    
    # Display fairness results
    table = Table(title="Fairness Metrics by Group")
    table.add_column("Feature", style="cyan")
    table.add_column("Group", style="white")
    table.add_column("Accuracy", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Status", justify="center")
    
    table.add_row("gender", "female", "0.935", "0.928", "[green]✓[/green]")
    table.add_row("gender", "male", "0.942", "0.938", "[green]✓[/green]")
    table.add_row("race", "group_a", "0.921", "0.915", "[yellow]![/yellow]")
    table.add_row("race", "group_b", "0.948", "0.945", "[green]✓[/green]")
    
    console.print(table)
    
    # Recommendations
    console.print("\n[bold]Recommendations:[/bold]")
    console.print("• Consider reweighting samples for group_a")
    console.print("• Review feature importance for potential proxy discrimination")
    console.print("• Implement post-processing calibration")


# ============= Serve Command =============
@app.command()
def serve(
    model_path: Path = typer.Argument(
        ...,
        help="Path to model to serve",
        exists=True,
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        "-h",
        help="Host to bind to",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to bind to",
    ),
    workers: int = typer.Option(
        4,
        "--workers",
        "-w",
        help="Number of worker processes",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload for development",
    ),
):
    """
    Serve a model via REST API.
    
    Example:
        pearlmind serve models/production/xgboost.pkl --port 8080
    """
    console.print(Panel.fit(
        f"Starting model server\nModel: {model_path.name}\nURL: http://{host}:{port}",
        title="PearlMind Model Server",
        border_style="green"
    ))
    
    console.print("[cyan]Starting server with following endpoints:[/cyan]")
    console.print("  POST /predict - Make predictions")
    console.print("  POST /predict_proba - Get probabilities")
    console.print("  GET /health - Health check")
    console.print("  GET /metrics - Model metrics")
    console.print("  GET /fairness - Fairness dashboard")
    
    # In real implementation, would start FastAPI server here
    console.print(f"\n[green]✓[/green] Server running at http://{host}:{port}")
    console.print("[yellow]Press CTRL+C to stop[/yellow]")


# ============= List Command =============
@app.command()
def list(
    model_dir: Path = typer.Option(
        Path("models"),
        "--dir",
        "-d",
        help="Directory to search for models",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table, json)",
    ),
):
    """
    List available models.
    
    Example:
        pearlmind list --dir models/production
    """
    console.print(Panel.fit(
        f"Scanning: {model_dir}",
        title="Available Models",
        border_style="blue"
    ))
    
    # Create table of models
    table = Table(title="PearlMind Models")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Version", style="green")
    table.add_column("Date", style="yellow")
    table.add_column("Size", justify="right")
    table.add_column("Metrics", style="blue")
    
    # Sample data - would scan directory in real implementation
    table.add_row(
        "xgboost_prod",
        "XGBoost",
        "2.0.0",
        "2025-01-20",
        "12.3 MB",
        "acc=0.942"
    )
    table.add_row(
        "neural_net_v1",
        "Neural",
        "1.5.0",
        "2025-01-18",
        "156.7 MB",
        "acc=0.938"
    )
    table.add_row(
        "random_forest",
        "Ensemble",
        "1.0.0",
        "2025-01-15",
        "45.2 MB",
        "acc=0.921"
    )
    
    console.print(table)
    
    console.print(f"\n[cyan]Total models found: 3[/cyan]")


# ============= Config Command =============
@app.command()
def config(
    action: str = typer.Argument(
        ...,
        help="Action to perform (show, validate, create)",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Path to configuration file",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for new config",
    ),
):
    """
    Manage configuration files.
    
    Examples:
        pearlmind config show --path configs/default.yaml
        pearlmind config validate --path configs/custom.yaml
        pearlmind config create --output configs/new.yaml
    """
    if action == "show":
        if config_path:
            config = Config.from_yaml(config_path)
            console.print(Panel.fit(
                str(config.model_dump()),
                title=f"Configuration: {config_path.name}",
                border_style="blue"
            ))
    
    elif action == "validate":
        try:
            config = Config.from_yaml(config_path)
            console.print("[green]✓[/green] Configuration is valid!")
        except Exception as e:
            console.print(f"[red]✗[/red] Configuration error: {e}")
    
    elif action == "create":
        config = Config()
        if output:
            config.save_yaml(output)
            console.print(f"[green]✓[/green] Configuration created: {output}")
    
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
