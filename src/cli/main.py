import os
import sys
import click
import threading
from typing import Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.manager import ConfigManager
from app_logging.logger import AprilLogger
from networking.client import NetworkClient
from models.dataset import DatasetMetadata, SearchResult
from search.engine import SearchEngine
from downloader.manager import DownloadManager
from cache.store import CacheStore
from analytics.inspector import DatasetInspector
from filters.engine import FilterEngine
from ai.assistant import AIAssistant
from export.generator import ExportGenerator
import ui.components as ui
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

config_manager = ConfigManager()
logger = AprilLogger.setup()
cache_store = CacheStore(config_manager)
search_engine = SearchEngine(config_manager)
download_manager = DownloadManager(config_manager)
filter_engine = FilterEngine()
ai_assistant = AIAssistant(config_manager)
export_generator = ExportGenerator()

@click.group(invoke_without_command=True)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose log output.")
@click.option("--debug", "-d", is_flag=True, help="Enable debug level log output.")
@click.pass_context
def cli(ctx, verbose: bool, debug: bool):
    """APRIL SET (Search Engine for Training) - Professional CLI Dataset Search Engine."""
    AprilLogger.setup(verbose=verbose, debug=debug)
    if ctx.invoked_subcommand is None:
        ui.print_banner()
        os.environ["APRIL_SET_REPL"] = "1"
        ui.console.print("[bold green]Entering interactive mode. Type 'help' for commands, or 'exit' to quit.[/bold green]\n")

        import shlex
        while True:
            try:
                user_input = click.prompt(Text("april-set> ", style="bold green"), prompt_suffix="").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    ui.console.print("[dim green]Goodbye.[/dim green]")
                    break

                try:
                    args = shlex.split(user_input)
                except ValueError as e:
                    ui.console.print(f"[bold red]Error parsing command: {e}[/bold red]")
                    continue

                cmd_name = args[0].lower()
                if cmd_name == "help":
                    ui.console.print("[bold green]Available Commands:[/bold green]")
                    ui.console.print("  [green]search [query][/green]   - Search datasets across providers")
                    ui.console.print("  [green]inspect [path][/green]   - Generate statistical profile of dataset")
                    ui.console.print("  [green]download [id][/green]    - Download dataset with progress tracking")
                    ui.console.print("  [green]filter [rules][/green]    - Apply logical or NL filters on search results")
                    ui.console.print("  [green]ai [id] [query][/green]   - Query Expert Assistant about dataset preprocessing/modeling")
                    ui.console.print("  [green]export [id] [args][/green] - Generate ML loader boilerplate (PyTorch, TensorFlow, etc.)")
                    ui.console.print("  [green]config [key] [val][/green] - Get/Set settings")
                    ui.console.print("  [green]cache [list/clear][/green] - View/delete cached datasets")
                    ui.console.print("  [green]bookmark [args][/green]   - Manage bookmarks")
                    ui.console.print("  [green]history[/green]           - View search history")
                    ui.console.print("  [green]exit[/green]              - Close interactive mode\n")
                    continue

                if cmd_name in cli.commands:
                    try:
                        cli.main(args=args, standalone_mode=False)
                    except click.exceptions.Exit:
                        pass
                    except click.exceptions.ClickException as e:
                        ui.console.print(f"[bold red]Error: {e}[/bold red]")
                    except Exception as e:
                        ui.console.print(f"[bold red]Unexpected error: {e}[/bold red]")
                else:

                    downloads = cache_store.get_downloads()
                    if downloads:
                        last_ds = downloads[0]
                        ds_id = last_ds["id"]
                        filepath = None
                        path = last_ds.get("local_path", "")
                        if os.path.exists(path):
                            if os.path.isdir(path):
                                files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith((".csv", ".parquet", ".json", ".arff"))]
                                if files:
                                    filepath = files[0]
                            else:
                                filepath = path

                        if filepath:
                            ui.console.print(f"[dim green]Routing question to Expert Assistant in context of dataset '{ds_id}'...[/dim green]\n")
                            try:
                                inspector = DatasetInspector(filepath)
                                report = inspector.inspect()
                                response = ai_assistant.analyze_dataset_prompt(report, user_input)
                                ui.console.print(response)
                            except Exception as e:
                                ui.console.print(f"[bold red]Error: {e}[/bold red]")
                        else:
                            ui.console.print(f"[bold red]Command '{cmd_name}' not found. To ask questions, please download a dataset first or use 'ai [dataset_id] [question]'.[/bold red]")
                    else:
                        ui.console.print(f"[bold red]Command '{cmd_name}' not found. Try 'help' to see list of valid commands.[/bold red]")
            except (KeyboardInterrupt, EOFError):
                ui.console.print("\n[dim green]Goodbye.[/dim green]")
                break
        ctx.exit(0)

@cli.command()
@click.argument("query", required=False)
@click.option("--limit", "-l", default=10, help="Maximum number of datasets to return.")
@click.option("--task", "-t", help="Filter results by machine learning task.")
@click.option("--license", "-lic", "license_type", help="Filter results by license.")
@click.option("--size", "-s", help="Filter results by size (e.g. <500MB, <1GB).")
def search(query: Optional[str], limit: int, task: Optional[str], license_type: Optional[str], size: Optional[str]):
    """Search for machine learning datasets across providers."""
    ui.print_banner()

    if not query:

        query = click.prompt("Search Query", type=str)

    cache_store.add_search_history(query)

    ui.console.print("Checking Internet Connection...")
    if not NetworkClient.is_connected():
        ui.console.print("[bold red]✗ Offline Mode: Connected providers cannot be queried. Searching local cache...[/bold red]\n")

        local_datasets = [
            DatasetMetadata(**{k: v for k, v in d.items() if k != "local_path" and k != "downloaded_at" and k != "actual_size"})
            for d in cache_store.get_downloads()
        ]
        results = [SearchResult(dataset=d, score=100.0) for d in local_datasets if query.lower() in d.name.lower()]
    else:
        ui.console.print("[green]✓ Connected[/green]\n")

        ui.console.print("[bold cyan]Querying Providers concurrently...[/bold cyan]")
        with ui.console.status("[bold green]Searching OpenML, HuggingFace, UCI, GitHub, Kaggle...", spinner="dots") as status:
            results = search_engine.search(query, limit_per_provider=limit)

    if task or license_type or size:
        results = filter_engine.filter_results(results, task=task, license_type=license_type, size_cond=size)

    results = results[:limit]

    if not results:
        ui.console.print("\n[bold yellow]No datasets found matching your criteria.[/bold yellow]")
        return

    ui.console.print(f"\n[bold green]{len(results)} datasets found.[/bold green]\n")
    ui.print_dataset_list(results)

    for r in results:
        meta_file = config_manager.metadata_dir / f"{r.dataset.id.replace('/', '_')}.json"
        try:
            with open(meta_file, "w") as f:
                import json
                json.dump(r.dataset.to_dict(), f)
        except Exception:
            pass

    import sys
    if sys.stdin.isatty():
        try:
            selection = click.prompt(
                "Select a dataset number to download (or press Enter to skip)",
                default="",
                show_default=False
            )
            if selection.strip().isdigit():
                idx = int(selection.strip()) - 1
                if 0 <= idx < len(results):
                    selected_ds = results[idx].dataset

                    ctx = click.get_current_context()
                    ctx.invoke(download, dataset_id=selected_ds.id, dest_dir=None)
        except (KeyboardInterrupt, EOFError):
            pass

@cli.command()
@click.argument("dataset_id")
@click.option("--dir", "-d", "dest_dir", help="Custom destination directory to save the dataset.")
def download(dataset_id: str, dest_dir: Optional[str]):
    """Download a dataset using its unique ID (e.g. uci/53, hf/ylecun/mnist)."""

    safe_id = dataset_id.replace("/", "_")
    meta_file = config_manager.metadata_dir / f"{safe_id}.json"

    ds_metadata = None
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                import json
                ds_dict = json.load(f)
                ds_metadata = DatasetMetadata(**ds_dict)
        except Exception:
            pass

    if not ds_metadata:

        ui.console.print(f"Metadata cache missed. Fetching details for '{dataset_id}' from provider...")
        provider_name = dataset_id.split("/")[0]

        for prov in search_engine.providers:
            if prov.name == provider_name:
                if provider_name == "uci":
                    uci_id = int(dataset_id.split("/")[-1])
                    ds_metadata = prov.fetch_metadata(uci_id)
                else:

                    term = dataset_id.split("/")[-1]
                    res = prov.search(term, limit=1)
                    if res:
                        ds_metadata = res[0]
                break

    if not ds_metadata:
        ui.console.print(f"[bold red]Error: Could not resolve metadata for dataset ID '{dataset_id}'.[/bold red]")
        return

    if not dest_dir:
        dest_path = cache_store.get_dataset_dir(ds_metadata.id)
    else:
        dest_path = Path(dest_dir)

    ui.console.print(f"[bold cyan]Connecting to provider for '{ds_metadata.name}'...[/bold cyan]")

    progress = ui.get_download_progress()
    task_id = progress.add_task(f"Downloading {ds_metadata.name}", total=100)

    def progress_callback(downloaded, total):
        if total > 0:
            progress.update(task_id, completed=downloaded, total=total)
        else:

            progress.update(task_id, completed=downloaded, total=100)

    cancel_event = threading.Event()

    matching_provider = None
    for prov in search_engine.providers:
        if prov.name == ds_metadata.provider:
            matching_provider = prov
            break

    if not matching_provider:
        ui.console.print(f"[bold red]Error: Provider '{ds_metadata.provider}' is not enabled or available.[/bold red]")
        return

    progress.start()
    try:
        success = matching_provider.download(
            dataset=ds_metadata,
            dest_dir=str(dest_path),
            progress_callback=progress_callback,
            cancel_event=cancel_event
        )
    except KeyboardInterrupt:
        cancel_event.set()
        success = False
    finally:
        progress.stop()

    if success:
        ui.console.print(f"\n[bold green]✓ Download Completed Successfully.[/bold green]")
        ui.console.print(f"Dataset stored at: [cyan]{dest_path}[/cyan]\n")
        cache_store.track_download(ds_metadata, str(dest_path))
    else:
        ui.console.print(f"\n[bold red]✗ Download Failed or was Cancelled.[/bold red]\n")

@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
def inspect(file_path: str):
    """Inspect and generate a deep statistical profile report of a dataset file."""
    ui.console.print(f"Analyzing structure of {os.path.basename(file_path)}...")
    with ui.console.status("[bold green]Calculating statistics, duplicates, correlations, outliers...", spinner="clock"):
        try:
            inspector = DatasetInspector(file_path)
            report = inspector.inspect()
        except Exception as e:
            report = {"error": str(e)}

    ui.print_inspection_report(report)

@cli.command()
@click.argument("nl_query", required=False)
@click.option("--task", "-t", help="Filter by task.")
@click.option("--license", "-lic", "license_type", help="Filter by license.")
@click.option("--rows", help="Comparison condition for rows (e.g. >10000).")
@click.option("--cols", help="Comparison condition for columns (e.g. >10).")
@click.option("--size", help="Comparison condition for size (e.g. <500MB).")
def filter(nl_query: Optional[str], task: Optional[str], license_type: Optional[str], rows: Optional[str], cols: Optional[str], size: Optional[str]):
    """Filter datasets logically or using natural language."""

    downloads = cache_store.get_downloads()
    if not downloads:
        ui.console.print("[bold yellow]No local datasets available. Querying online providers...[/bold yellow]")

        with ui.console.status("[bold green]Fetching general query for filtering...", spinner="dots"):
            results = search_engine.search("data", limit_per_provider=20)
    else:
        results = [
            SearchResult(
                dataset=DatasetMetadata(**{k: v for k, v in d.items() if k != "local_path" and k != "downloaded_at" and k != "actual_size"}),
                score=100.0
            )
            for d in downloads
        ]

    if nl_query:
        ui.console.print(f"Parsing natural language filter: [italic]\"{nl_query}\"[/italic]...")
        conds = filter_engine.parse_natural_language(nl_query)
        task = conds.get("task", task)
        license_type = conds.get("license_type", license_type)
        rows = conds.get("rows_cond", rows)
        size = conds.get("size_cond", size)

    filtered = filter_engine.filter_results(
        results,
        task=task,
        license_type=license_type,
        rows_cond=rows,
        cols_cond=cols,
        size_cond=size
    )

    ui.console.print(f"[bold green]Filtered {len(filtered)} results.[/bold green]\n")
    ui.print_dataset_list(filtered)

@cli.command()
@click.argument("dataset_source", required=False)
@click.argument("question", required=False)
def ai(dataset_source: Optional[str] = None, question: Optional[str] = None):
    """Ask the Expert Assistant for recommendations (preprocessing, leakage, model suggestions) on a dataset."""
    downloads = cache_store.get_downloads()

    if dataset_source and not question:
        looks_like_dataset = (
            os.path.exists(dataset_source) or
            "/" in dataset_source or
            any(d["id"] == dataset_source or d["name"].lower() == dataset_source.lower() for d in downloads)
        )
        if not looks_like_dataset:
            question = dataset_source
            dataset_source = None

    if not dataset_source:
        if downloads:
            dataset_source = downloads[0]["id"]
        else:
            ui.console.print("[bold red]Error: No cached dataset found. Please specify a dataset source. Example: ai openml/37 'your question'[/bold red]")
            return

    if not question:
        question = click.prompt("Question", type=str)

    filepath = None
    if os.path.exists(dataset_source):
        filepath = dataset_source
    else:

        for d in downloads:
            if d["id"] == dataset_source or d["name"].lower() == dataset_source.lower():
                path = d.get("local_path", "")
                if os.path.exists(path):

                    if os.path.isdir(path):
                        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith((".csv", ".parquet", ".json", ".arff"))]
                        if files:
                            filepath = files[0]
                    else:
                        filepath = path
                break

    if not filepath:
        ui.console.print(f"[bold red]Error: Could not find dataset file/cache for source '{dataset_source}'.[/bold red]")
        return

    with ui.console.status("[bold green]Profiling dataset structure for AI context...", spinner="earth"):
        try:
            inspector = DatasetInspector(filepath)
            report = inspector.inspect()
        except Exception as e:
            ui.console.print(f"[bold red]Failed to analyze file: {e}[/bold red]")
            return

    ui.console.print(f"Connecting to Assistant ({ai_assistant.provider})...\n")
    with ui.console.status("[bold green]Assistant is analyzing...", spinner="dots"):
        response = ai_assistant.analyze_dataset_prompt(report, question)

    ui.console.print(response)

@cli.command()
@click.argument("dataset_source")
@click.option("--framework", "-f", help="Framework target: pytorch, tensorflow, sklearn, xgboost, duckdb, r, julia, github, aws.")
def export(dataset_source: str, framework: Optional[str]):
    """Generate ready-to-run code loaders for PyTorch, TensorFlow, Scikit-learn, etc."""

    filepath = None
    name = "Dataset"
    target_col = "target"

    if os.path.exists(dataset_source):
        filepath = dataset_source
        name = os.path.splitext(os.path.basename(filepath))[0]
    else:
        downloads = cache_store.get_downloads()
        for d in downloads:
            if d["id"] == dataset_source or d["name"].lower() == dataset_source.lower():
                path = d.get("local_path", "")
                if os.path.exists(path):
                    name = d["name"]
                    target_col = "target"
                    if os.path.isdir(path):
                        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith((".csv", ".parquet", ".json", ".arff"))]
                        if files:
                            filepath = files[0]
                    else:
                        filepath = path
                break

    if not filepath:
        ui.console.print(f"[bold red]Error: Could not find local dataset for '{dataset_source}'.[/bold red]")
        return

    try:
        inspector = DatasetInspector(filepath)
        report = inspector.inspect()
        if "error" not in report and report.get("target_candidates"):
            target_col = report["target_candidates"][0]["column"]
    except Exception:
        pass

    if not framework:

        frameworks = ["pytorch", "tensorflow", "scikit-learn", "xgboost", "lightgbm", "catboost", "duckdb", "parquet", "r", "julia", "github", "aws"]
        ui.console.print("[bold cyan]Available Targets:[/bold cyan] " + ", ".join(frameworks))
        framework = click.prompt("Select framework/platform", type=click.Choice(frameworks))

    snippet = export_generator.generate_snippet(framework, name, filepath, target_col)

    ui.console.print(Panel(
        Text.from_ansi(snippet) if hasattr(Text, "from_ansi") else Text(snippet),
        title=f"[bold green]Boilerplate Code Loader: {framework.upper()}[/bold green]",
        border_style="green"
    ))

@cli.command()
@click.argument("key", required=False)
@click.argument("value", required=False)
def config(key: Optional[str], value: Optional[str]):
    """Get or set configuration values (keys, cache dirs)."""
    if not key:

        import yaml
        yaml_str = yaml.safe_dump(config_manager.config, default_flow_style=False)
        ui.console.print(Panel(yaml_str, title="[bold white]Current Configuration[/bold white]"))
        ui.console.print(f"Config file located at: [cyan]{config_manager.config_path}[/cyan]")
    elif key and value is None:

        val = config_manager.get(key)
        ui.console.print(f"{key}: {val}")
    else:

        if value.lower() == "true":
            parsed_val = True
        elif value.lower() == "false":
            parsed_val = False
        else:
            try:
                parsed_val = int(value)
            except ValueError:
                try:
                    parsed_val = float(value)
                except ValueError:
                    parsed_val = value

        config_manager.set(key, parsed_val)
        ui.console.print(f"[green]Set config '{key}' to: {parsed_val}[/green]")

@cli.command()
@click.argument("action", type=click.Choice(["add", "remove", "list"]))
@click.argument("dataset_id", required=False)
def bookmark(action: str, dataset_id: Optional[str]):
    """Bookmark, remove, or list bookmarked datasets."""
    if action == "list":
        bookmarks = cache_store.get_bookmarks()
        if not bookmarks:
            ui.console.print("[bold yellow]No bookmarked datasets found.[/bold yellow]")
            return
        ui.console.print(f"[bold green]Bookmarked Datasets ({len(bookmarks)}):[/bold green]\n")
        results = [SearchResult(dataset=DatasetMetadata(**b), score=100.0) for b in bookmarks]
        ui.print_dataset_list(results)
    elif action == "add":
        if not dataset_id:
            ui.console.print("[bold red]Error: dataset_id is required to bookmark.[/bold red]")
            return

        safe_id = dataset_id.replace("/", "_")
        meta_file = config_manager.metadata_dir / f"{safe_id}.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    import json
                    ds = DatasetMetadata(**json.load(f))
                    cache_store.add_bookmark(ds)
                    ui.console.print(f"[green]✓ Bookmarked {dataset_id}[/green]")
            except Exception as e:
                ui.console.print(f"[bold red]Error: Failed to add bookmark: {e}[/bold red]")
        else:
            ui.console.print(f"[bold red]Error: Metadata for dataset '{dataset_id}' not cached. Search or query it first.[/bold red]")
    elif action == "remove":
        if not dataset_id:
            ui.console.print("[bold red]Error: dataset_id is required to remove bookmark.[/bold red]")
            return
        cache_store.remove_bookmark(dataset_id)
        ui.console.print(f"[green]✓ Removed bookmark: {dataset_id}[/green]")

@cli.command()
def history():
    """Display history of search queries."""
    hist = cache_store.get_search_history()
    if not hist:
        ui.console.print("[bold yellow]No search history found.[/bold yellow]")
        return
    ui.console.print("[bold green]Recent Search History:[/bold green]")
    for idx, item in enumerate(hist):
        ui.console.print(f" {idx + 1}. {item}")

@cli.command()
@click.argument("action", type=click.Choice(["list", "clear"]))
def cache(action: str):
    """List or clear cached/downloaded datasets."""
    if action == "list":
        downloads = cache_store.get_downloads()
        if not downloads:
            ui.console.print("[bold yellow]Cache is empty.[/bold yellow]")
            return

        table = Table(title="[bold white]Cached Datasets[/bold white]", border_style="blue")
        table.add_column("Dataset ID", style="cyan")
        table.add_column("Name")
        table.add_column("Provider", style="green")
        table.add_column("Local Path")

        for d in downloads:
            table.add_row(d["id"], d["name"], d["provider"], d["local_path"])
        ui.console.print(table)

    elif action == "clear":

        downloads = cache_store.get_downloads()
        import shutil
        for d in downloads:
            path = d.get("local_path")
            if path and os.path.exists(path):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                except Exception as e:
                    ui.console.print(f"[red]Error deleting {path}: {e}[/red]")

        for f in [cache_store.bookmarks_file, cache_store.history_file, cache_store.downloads_file]:
            with open(f, "w") as fh:
                json.dump([], fh)
        ui.console.print("[bold green]✓ Cache cleared successfully.[/bold green]")

if __name__ == "__main__":
    cli()
