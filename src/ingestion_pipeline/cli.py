import typer
from typing import Optional
from ingestion_pipeline.pipelines.ingest_and_embed import ingest_stage, chunk_stage, embed_stage, run_all
from ingestion_pipeline.preprocessing.fulltext_enricher import run_fulltext
from ingestion_pipeline.pipelines.chunk_from_fulltext import chunk_from_fulltext

app = typer.Typer(help="Ingestion + Fulltext + Chunking + Embeddings pipeline")

@app.command()
def fulltext(input_path: str, output_dir: str = "data/processed", throttle_sec: float = 1.0, request_timeout: int = 45):
    """Produce fulltext_articles.json + diagnostics (matches your notebook)."""
    summary = run_fulltext(input_path=input_path, output_dir=output_dir,
                           throttle_sec=throttle_sec, request_timeout=request_timeout)
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
