import typer
from typing import Optional
from ingestion_pipeline.pipelines.ingest_and_embed import ingest_stage, chunk_stage, embed_stage, run_all
from ingestion_pipeline.preprocessing.fulltext_enricher import run_fulltext
from ingestion_pipeline.pipelines.chunk_from_fulltext import chunk_from_fulltext
import os 

app = typer.Typer(help="Ingestion + Fulltext + Chunking + Embeddings pipeline")

# ---------- FULLTEXT (batched, full-text-only by default) ----------
@app.command()
def fulltext(
    input_path: str = typer.Argument(..., help="Path to .bib or .csv with a 'doi' column"),
    output_dir: str = typer.Option("data/processed", "--output-dir", help="Folder for outputs"),
    # batching for PMC
    idconv_chunk: int = typer.Option(150, "--idconv-chunk", help="Batch size for DOI→PMCID"),
    efetch_chunk: int = typer.Option(80, "--efetch-chunk", help="Batch size for PMC EFetch"),
    batch_workers: int = typer.Option(4, "--batch-workers", help="Parallel workers for PMC batches"),
    batch_throttle_sec: float = typer.Option(0.10, "--batch-throttle-sec", help="Delay after each PMC batch (s)"),
    # progress & timing
    progress: bool = typer.Option(True, "--progress/--no-progress"),
    throttle_sec: float = typer.Option(0.0, "--throttle-sec", help="Per-article delay in Assemble"),
    request_timeout: int = typer.Option(45, "--request-timeout", help="HTTP timeout (s)"),
    # full-text policy
    require_fulltext: bool = typer.Option(True, "--require-fulltext/--allow-abstract-only"),
    min_fulltext_chars: int = typer.Option(200, "--min-fulltext-chars", help="Min body chars for PMC records"),
    # performance tradeoff
    skip_pmc_single_fallback: bool = typer.Option(True, "--skip-pmc-single-fallback/--allow-pmc-single-fallback",
                                                  help="Skip slow per-item PMC fallback if batch missed it"),
):
    """Build fulltext_articles.json (+ summary/CSV) from DOIs using PMC only."""
    summary = run_fulltext(
        input_path=input_path,
        output_dir=output_dir,
        throttle_sec=throttle_sec,
        request_timeout=request_timeout,
        show_progress=progress,
        idconv_chunk=idconv_chunk,
        efetch_chunk=efetch_chunk,
        batch_workers=batch_workers,
        batch_throttle_sec=batch_throttle_sec,
        require_fulltext=require_fulltext,
        min_fulltext_chars=min_fulltext_chars,
        skip_pmc_single_fallback=skip_pmc_single_fallback,
    )
    for k, v in summary.items():
        typer.echo(f"{k}: {v}")

@app.command()
def chunk_fulltext(fulltext_json: str = "data/processed/fulltext_articles.json", output_dir: str = "data/processed", model_name: Optional[str] = None):
    out = chunk_from_fulltext(fulltext_json, output_dir, model_name=model_name)
    typer.echo(out)

@app.command()
def ingest(input_path: str, output_dir: str = "data"):
    out = ingest_stage(input_path, output_dir)
    typer.echo(out)

@app.command()
def chunk(input_path: str, output_dir: str = "data", model_name: Optional[str] = None):
    out = chunk_stage(input_path, output_dir, model_name=model_name)
    typer.echo(out)

@app.command()
def embed(chunks_path: str, output_dir: str = "data", model_name: Optional[str] = None):
    out = embed_stage(chunks_path, output_dir, model_name=model_name)
    typer.echo(out)

@app.command()
def run(input_path: str, output_dir: str = "data", model_name: Optional[str] = None):
    paths = run_all(input_path, output_dir, model_name=model_name)
    for k, v in paths.items():
        typer.echo(f"{k}: {v}")

if __name__ == "__main__":
    app()
