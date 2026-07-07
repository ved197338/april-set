import math
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from models.dataset import DatasetMetadata, SearchResult

console = Console()

ASCII_LOGO = """
[green] █████╗ ██████╗ ██████╗ ██╗██╗     ███████╗███████╗████████╗[/green]
[green]██╔══██╗██╔══██╗██╔══██╗██║██║     ██╔════╝██╔════╝╚══██╔══╝[/green]
[green]███████║██████╔╝██████╔╝██║██║     ███████╗█████╗     ██║   [/green]
[green]██╔══██║██╔═══╝ ██╔══██╗██║██║     ╚════██║██╔══╝     ██║   [/green]
[green]██║  ██║██║     ██║  ██║██║███████╗███████║███████╗   ██║   [/green]
[green]╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚══════╝   ╚═╝   [/green]
[dim green]                 Search Engine for Training[/dim green]
"""

def print_banner():
    import os
    if "APRIL_SET_REPL" in os.environ:
        return
    console.print(Panel.fit(
        ASCII_LOGO,
        border_style="green",
        title="[bold green]v1.0.0[/bold green]",
        subtitle="[green]Ready[/green]"
    ))

def render_dataset_card(ds: DatasetMetadata, index: Optional[int] = None) -> Panel:
    idx_str = f"[{index}] " if index is not None else ""

    pop_color = "green" if ds.popularity > 75 else ("dim green" if ds.popularity > 40 else "bright_black")
    qual_color = "green" if ds.quality_score > 75 else ("dim green" if ds.quality_score > 40 else "bright_black")

    rows_str = f"{ds.rows:,}" if isinstance(ds.rows, int) else "?"
    cols_str = f"{ds.columns}" if ds.columns is not None else "?"
    rows_cols = f"{rows_str} rows x {cols_str} cols"
    size_str = ds.formatted_size

    card_text = Text()
    card_text.append("Provider: ", style="dim green")
    card_text.append(f"{ds.provider.upper()}", style="bold green")
    card_text.append("  |  Task: ", style="dim green")
    card_text.append(f"{ds.task}", style="green")
    card_text.append("  |  Shape: ", style="dim green")
    card_text.append(f"{rows_cols}", style="green")
    card_text.append("  |  Size: ", style="dim green")
    card_text.append(f"{size_str}", style="green")
    card_text.append("\nLicense: ", style="dim green")
    card_text.append(f"{ds.license}", style="green")
    card_text.append("  |  Updated: ", style="dim green")
    card_text.append(f"{ds.last_updated[:10]}", style="green")

    import re
    desc_clean = re.sub(r"https?://\S+", "", ds.description)
    desc_clean = re.sub(r"www\.\S+", "", desc_clean).strip()
    if not desc_clean:
        desc_clean = ds.name
    else:
        desc_clean = f"{desc_clean[:200]}..."

    card_text.append("\n\n")
    card_text.append(desc_clean, style="dim green")

    if ds.tags:
        card_text.append("\n\nTags: ", style="dim green")
        card_text.append(", ".join(ds.tags[:5]), style="green")

    cmd_str = f"aset download {ds.id}"
    inner_width = max(len(cmd_str) + 14, 46)
    cmd_pad = inner_width - 14
    card_text.append("\n\n")
    card_text.append(" ┌" + "─" * inner_width + "┐\n")
    card_text.append(" │ ")
    card_text.append(" DOWNLOAD ", style="bold black on green")
    card_text.append(" ")
    card_text.append(f"{cmd_str:<{cmd_pad}}", style="green")
    card_text.append(" │\n")
    card_text.append(" └" + "─" * inner_width + "┘")

    title_text = f"{idx_str}{ds.name}"
    subtitle_text = f"ID: {ds.id} | Pop: [{pop_color}]{ds.popularity:.1f}%[/{pop_color}] | Ready: [{qual_color}]{ds.quality_score:.1f}%[/{qual_color}]"

    return Panel(
        card_text,
        title=f"[bold green]{title_text}[/bold green]",
        title_align="left",
        subtitle=subtitle_text,
        subtitle_align="right",
        border_style="green",
        padding=(1, 2)
    )

def print_dataset_list(results: List[SearchResult]):
    for i, r in enumerate(results):
        console.print(render_dataset_card(r.dataset, i + 1))
        console.print("")

def print_inspection_report(data: Dict[str, Any]):
    if "error" in data:
        console.print(Panel(f"[bold red]Inspection Error: {data['error']}[/bold red]", border_style="red"))
        return

    console.print(Panel.fit(
        f"[bold green]Dataset Profile: {data['filename']}[/bold green]", 
        border_style="green",
        padding=(0, 4)
    ))

    shape_table = Table(title="[bold green]Dataset Overview[/bold green]", show_header=False, border_style="green")
    shape_table.add_column("Key", style="dim green")
    shape_table.add_column("Value", style="green")

    shape_table.add_row("File Location", data["filepath"])
    shape_table.add_row("Rows", f"{data['rows']:,}")
    shape_table.add_row("Columns", f"{data['columns']:,}")
    shape_table.add_row("Memory Footprint", f"{data['memory_usage_mb']:.2f} MB")
    shape_table.add_row("Duplicates", f"{data['duplicate_rows']:,} rows ({data['duplicate_percentage']:.2f}%)")
    shape_table.add_row("Encoding", data["encoding"])
    shape_table.add_row("CSV Delimiter", data["delimiter"])
    shape_table.add_row("Total Missing cells", f"{data['total_missing']:,} ({data['overall_missing_pct']:.2f}%)")
    console.print(shape_table)

    cols_table = Table(title="[bold green]Schema Information[/bold green]", border_style="green")
    cols_table.add_column("Index", justify="right", style="dim green")
    cols_table.add_column("Column Name", justify="left", style="green")
    cols_table.add_column("Data Type", justify="center", style="green")
    cols_table.add_column("Cardinality", justify="right", style="green")
    cols_table.add_column("Missing Counts", justify="right", style="green")
    cols_table.add_column("Missing %", justify="right", style="green")
    cols_table.add_column("Null Distribution Graph", justify="left")

    for idx, col in enumerate(data["columns_info"]):
        missing_pct = col["missing_pct"]

        bar_len = int(missing_pct / 5)
        bar_graph = "█" * bar_len + "░" * (20 - bar_len)
        bar_color = "dim green" if missing_pct > 20 else "green"

        cols_table.add_row(
            str(idx + 1),
            col["name"],
            col["type"],
            f"{col['cardinality']:,}",
            f"{col['missing_count']:,}",
            f"{missing_pct:.1f}%",
            f"[{bar_color}]{bar_graph}[/{bar_color}]"
        )
    console.print(cols_table)

    if data["numerical_summary"]:
        num_table = Table(title="[bold green]Statistical Profile (Numerical Columns)[/bold green]", border_style="green")
        num_table.add_column("Column Name", style="green")
        num_table.add_column("Mean", style="green")
        num_table.add_column("Std Dev", style="green")
        num_table.add_column("Min", style="green")
        num_table.add_column("Median (50%)", style="green")
        num_table.add_column("Max", style="green")
        num_table.add_column("Outliers (Count / %)", style="green")

        for col, stats in data["numerical_summary"].items():
            outlier_desc = "None"
            if col in data["outliers"]:
                out = data["outliers"][col]
                if out["count"] > 0:
                    outlier_desc = f"{out['count']:,} ({out['percentage']:.1f}%)"

            num_table.add_row(
                col,
                f"{stats['mean']:.2f}",
                f"{stats['std']:.2f}",
                f"{stats['min']:.2f}",
                f"{stats['50%']:.2f}",
                f"{stats['max']:.2f}",
                outlier_desc
            )
        console.print(num_table)

    if data.get("correlations") and "top_correlated_pairs" in data["correlations"]:
        corr_table = Table(title="[bold green]Highly Correlated Features[/bold green]", border_style="green")
        corr_table.add_column("Feature 1", style="green")
        corr_table.add_column("Feature 2", style="green")
        corr_table.add_column("Pearson Coefficient", justify="right", style="green")

        for f1, f2, coef in data["correlations"]["top_correlated_pairs"]:
            color = "green" if abs(coef) > 0.8 else "dim green"
            corr_table.add_row(f1, f2, f"[{color}]{coef:+.4f}[/{color}]")
        console.print(corr_table)

    target_text = Text()
    if data["target_candidates"]:
        for cand in data["target_candidates"]:
            target_text.append("• Column: ", style="dim green")
            target_text.append(f"'{cand['column']}'", style="bold green")
            target_text.append("  |  Score: ", style="dim green")
            target_text.append(f"{cand['score']:.0f}/100", style="green")
            target_text.append("  |  Inferred Task: ", style="dim green")
            target_text.append(f"{cand['inferred_task']}\n", style="green")
    else:
        target_text.append("No clear target variable detected.")

    console.print(Panel(target_text, title="[bold green]Inferred ML Target Variable Candidates[/bold green]", border_style="green"))

    if data.get("class_balance") and "distribution" in data["class_balance"]:
        cb = data["class_balance"]
        cb_table = Table(title=f"[bold green]Class Distribution for target '{cb['column']}'[/bold green]", border_style="green")
        cb_table.add_column("Class / Value", style="green")
        cb_table.add_column("Instances Count", justify="right", style="green")
        cb_table.add_column("Percentage", justify="right", style="green")
        cb_table.add_column("Distribution Graph", justify="left")

        for val, stats in cb["distribution"].items():
            pct = stats["percentage"]
            graph_len = int(pct / 4)
            graph = "█" * graph_len
            cb_table.add_row(
                val,
                f"{stats['count']:,}",
                f"{pct:.1f}%",
                f"[green]{graph}[/green]"
            )
        console.print(cb_table)

def get_download_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(bar_width=40, style="dim green", complete_style="green"),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console
    )
