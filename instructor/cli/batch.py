import os
from rich.console import Console
from rich.table import Table
from rich.live import Live
import typer
import time
import json
from typing import Any, Optional
from instructor.batch import BatchProcessor
from instructor.auto_client import from_provider
import warnings

app = typer.Typer()

console = Console()


def generate_table(batch_jobs: list[Any], provider: str):
    """Generate table for batch jobs based on provider"""
    table = Table(title=f"{provider.title()} Batch Jobs")

    table.add_column("Batch ID", style="dim", min_width=30, no_wrap=True)
    table.add_column("Created At")
    table.add_column("Status")

    # Add provider-specific columns
    if provider == "openai":
        table.add_column("Failed")
        table.add_column("Completed")
        table.add_column("Total")
    elif provider == "anthropic":
        table.add_column("Request Count")

    for batch_job in batch_jobs:
        if provider == "openai":
            table.add_row(
                str(batch_job.id),
                str(batch_job.created_at),
                str(batch_job.status),
                str(getattr(batch_job, "failed", "N/A")),
                str(getattr(batch_job, "completed", "N/A")),
                str(getattr(batch_job, "total", "N/A")),
            )
        elif provider == "anthropic":
            table.add_row(
                str(batch_job.id),
                str(batch_job.created_at),
                str(batch_job.processing_status),
                str(getattr(batch_job, "request_counts", {}).get("processing", "N/A")),
            )

    return table


def get_jobs(limit: int = 10, provider: str = "openai"):
    """Get batch jobs for the specified provider"""

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        return client.batches.list(limit=limit).data
    elif provider == "anthropic":
        from anthropic import Anthropic

        client = Anthropic()
        try:
            batches_client = client.messages.batches
        except AttributeError:
            batches_client = client.beta.messages.batches
        response = batches_client.list(limit=limit)
        return response.data
    else:
        raise ValueError(f"Unsupported provider: {provider}")


@app.command(name="list", help="See all existing batch jobs")
def watch(
    limit: int = typer.Option(10, help="Total number of batch jobs to show"),
    poll: int = typer.Option(
        10, help="Time in seconds to wait for the batch job to complete"
    ),
    screen: bool = typer.Option(False, help="Enable or disable screen output"),
    live: bool = typer.Option(
        False, help="Enable live polling to continuously update the table"
    ),
    model: str = typer.Option(
        "openai/gpt-4o-mini",
        help="Model in format 'provider/model-name' (e.g., 'openai/gpt-4', 'anthropic/claude-3-sonnet')",
    ),
    # Deprecated flag for backward compatibility
    use_anthropic: bool = typer.Option(
        None,
        help="[DEPRECATED] Use --model instead. Use Anthropic API instead of OpenAI",
    ),
):
    """
    Monitor the status of the most recent batch jobs
    """
    # Handle deprecated flag
    if use_anthropic is not None:
        warnings.warn(
            "--use-anthropic is deprecated. Use --model 'anthropic/claude-3-sonnet' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if use_anthropic:
            model = "anthropic/claude-3-sonnet"

    provider, _ = model.split("/", 1)
    batch_jobs = get_jobs(limit, provider)
    table = generate_table(batch_jobs, provider)

    if not live:
        # Show table once and exit
        console.print(table)
        return

    # Live polling mode
    with Live(table, refresh_per_second=2, screen=screen) as live_table:
        while True:
            batch_jobs = get_jobs(limit, provider)
            table = generate_table(batch_jobs, provider)
            live_table.update(table)
            time.sleep(poll)


@app.command(
    help="Create a batch job from a file",
)
def create_from_file(
    file_path: str = typer.Option(help="File containing the batch job requests"),
    model: str = typer.Option(
        "openai/gpt-4o-mini",
        help="Model in format 'provider/model-name' (e.g., 'openai/gpt-4', 'anthropic/claude-3-sonnet')",
    ),
    # Deprecated flag for backward compatibility
    use_anthropic: bool = typer.Option(
        None,
        help="[DEPRECATED] Use --model instead. Use Anthropic API instead of OpenAI",
    ),
):
    # Handle deprecated flag
    if use_anthropic is not None:
        warnings.warn(
            "--use-anthropic is deprecated. Use --model 'anthropic/claude-3-sonnet' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if use_anthropic:
            model = "anthropic/claude-3-sonnet"

    provider, _ = model.split("/", 1)

    try:
        # Create a dummy response model (not used for direct file submission)
        from pydantic import BaseModel

        class DummyModel(BaseModel):
            dummy: str = "dummy"

        # Create BatchProcessor instance
        processor = BatchProcessor(model, DummyModel)

        # Prepare metadata
        metadata = {
            "description": "Instructor batch job",
        }

        with console.status(f"[bold green]Submitting batch job...", spinner="dots"):
            batch_id = processor.submit_batch(file_path, metadata=metadata)

        console.print(f"[bold green]Batch job created with ID: {batch_id}[/bold green]")

        # Show updated batch list
        provider_name = model.split("/", 1)[0]
        watch(limit=5, poll=2, screen=False, live=False, model=model)

    except Exception as e:
        console.print(f"[bold red]Error creating batch job: {e}[/bold red]")


@app.command(help="Cancel a batch job")
def cancel(
    batch_id: str = typer.Option(help="Batch job ID to cancel"),
    provider: str = typer.Option(
        "openai",
        help="Provider to use (e.g., 'openai', 'anthropic')",
    ),
    # Deprecated flag for backward compatibility
    use_anthropic: bool = typer.Option(
        None,
        help="[DEPRECATED] Use --provider 'anthropic' instead. Use Anthropic API instead of OpenAI",
    ),
):
    """Cancel a batch job using the unified BatchProcessor"""
    # Handle deprecated flag
    if use_anthropic is not None:
        warnings.warn(
            "--use-anthropic is deprecated. Use --provider 'anthropic' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if use_anthropic:
            provider = "anthropic"

    try:
        if provider == "anthropic":
            from anthropic import Anthropic

            client = Anthropic()
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches
            batches_client.cancel(batch_id)
        else:
            from openai import OpenAI

            client = OpenAI()
            client.batches.cancel(batch_id)
        watch(limit=5, poll=2, screen=False, live=False, use_anthropic=use_anthropic)
        console.log(f"[bold red]Job {batch_id} cancelled successfully!")
    except Exception as e:
        console.print(f"[bold red]Error cancelling batch {batch_id}: {e}[/bold red]")


@app.command(help="Delete a completed batch job")
def delete(
    batch_id: str = typer.Option(help="Batch job ID to delete"),
    provider: str = typer.Option(
        "openai",
        help="Provider to use (e.g., 'openai', 'anthropic')",
    ),
):
    """Delete a batch job using the unified BatchProcessor"""
    try:
        from pydantic import BaseModel

        class DummyModel(BaseModel):
            dummy: str = "dummy"

        model_map = {
            "openai": "openai/gpt-4o-mini",
            "anthropic": "anthropic/claude-3-sonnet",
        }
        if provider not in model_map:
            console.print(f"[red]Unsupported provider: {provider}[/red]")
            return
        processor = BatchProcessor(model_map[provider], DummyModel)
        with console.status(
            f"[bold yellow]Deleting {provider} batch job...", spinner="dots"
        ):
            processor.delete_batch(batch_id)
        console.print(
            f"[bold green]Batch {batch_id} deleted successfully![/bold green]"
        )
        watch(limit=5, poll=2, screen=False, live=False, model=model_map[provider])
    except NotImplementedError as e:
        console.print(f"[yellow]Note: {e}[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error deleting batch {batch_id}: {e}[/bold red]")


@app.command(help="Download the file associated with a batch job")
def download_file(
    batch_id: str = typer.Option(help="Batch job ID to download"),
    download_file_path: str = typer.Option(help="Path to download file to"),
    provider: str = typer.Option(
        "openai",
        help="Provider to use (e.g., 'openai', 'anthropic')",
    ),
):
    try:
        if provider == "anthropic":
            from anthropic import Anthropic

            client = Anthropic()
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches
            batch = batches_client.retrieve(batch_id)
            if batch.processing_status != "ended":
                raise ValueError("Only completed Jobs can be downloaded")
            results_url = batch.results_url
            if not results_url:
                raise ValueError("Results URL not available")
            with open(download_file_path, "w") as file:
                for result in client.messages.batches.results(batch_id):
                    file.write(json.dumps(result.model_dump()) + "\n")
        else:
            from openai import OpenAI

            client = OpenAI()
            batch = client.batches.retrieve(batch_id=batch_id)
            status = batch.status
            if status != "completed":
                raise ValueError("Only completed Jobs can be downloaded")
            file_id = batch.output_file_id
            assert file_id, f"Equivalent Output File not found for {batch_id}"
            file_response = client.files.content(file_id)
            with open(download_file_path, "w") as file:
                file.write(file_response.text)
    except Exception as e:
        console.log(f"[bold red]Error downloading file for {batch_id}: {e}")


@app.command(help="Retrieve results from a batch job")
def results(
    batch_id: str = typer.Option(help="Batch job ID to get results from"),
    output_file: str = typer.Option(help="File to save the results to"),
    model: str = typer.Option(
        "openai/gpt-4o-mini",
        help="Model in format 'provider/model-name' (e.g., 'openai/gpt-4', 'anthropic/claude-3-sonnet')",
    ),
):
    """Retrieve and save batch job results"""
    provider, _ = model.split("/", 1)
    try:
        if provider == "openai":
            from openai import OpenAI

            client = OpenAI()
            batch = client.batches.retrieve(batch_id=batch_id)
            if batch.status != "completed":
                console.print(
                    f"[yellow]Batch status is '{batch.status}', not completed[/yellow]"
                )
                return
            file_id = batch.output_file_id
            if not file_id:
                console.print("[red]No output file available[/red]")
                return
            file_response = client.files.content(file_id)
            with open(output_file, "w") as f:
                f.write(file_response.text)
            console.print(f"[bold green]Results saved to: {output_file}[/bold green]")
        elif provider == "anthropic":
            from anthropic import Anthropic

            client = Anthropic()
            batch = client.beta.messages.batches.retrieve(batch_id)
            if batch.processing_status != "ended":
                console.print(
                    f"[yellow]Batch status is '{batch.processing_status}', not ended[/yellow]"
                )
                return
            results_iter = client.beta.messages.batches.results(batch_id)
            with open(output_file, "w") as f:
                for result in results_iter:
                    f.write(json.dumps(result.model_dump()) + "\n")
            console.print(f"[bold green]Results saved to: {output_file}[/bold green]")
        else:
            console.print(f"[red]Unsupported provider: {provider}[/red]")
    except Exception as e:
        console.log(f"[bold red]Error retrieving results for {batch_id}: {e}")


@app.command(help="Create batch job using BatchProcessor")
def create(
    messages_file: str = typer.Option(help="JSONL file with message conversations"),
    model: str = typer.Option(
        "openai/gpt-4o-mini",
        help="Model in format 'provider/model-name' (e.g., 'openai/gpt-4', 'anthropic/claude-3-sonnet')",
    ),
    response_model: str = typer.Option(
        help="Python class path for response model (e.g., 'examples.User')"
    ),
    output_file: str = typer.Option(
        "batch_requests.jsonl", help="Output file for batch requests"
    ),
    max_tokens: int = typer.Option(1000, help="Maximum tokens per request"),
    temperature: float = typer.Option(0.1, help="Temperature for generation"),
):
    """Create a batch job using the unified BatchProcessor"""
    try:
        module_path, class_name = response_model.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        response_class = getattr(module, class_name)
        messages_list = []
        with open(messages_file, "r") as f:
            for line in f:
                if line.strip():
                    messages_list.append(json.loads(line))
        processor = BatchProcessor(model, response_class)
        with console.status(
            f"[bold green]Creating batch file with {len(messages_list)} requests...",
            spinner="dots",
        ):
            processor.create_batch_from_messages(
                messages_list, output_file, max_tokens, temperature
            )
        console.print(f"[bold green]Batch file created: {output_file}[/bold green]")
        console.print(
            f"[yellow]Use 'instructor batch create-from-file --file-path {output_file}' to submit the batch[/yellow]"
        )
    except Exception as e:
        console.log(f"[bold red]Error creating batch: {e}")
