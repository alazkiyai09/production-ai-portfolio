#!/usr/bin/env python3
"""
Command-Line Interface for FraudDocs-RAG.

This module provides a CLI for interacting with the RAG system including:
- Querying the knowledge base
- Ingesting documents
- Managing the vector store
- System health checks

Usage:
    python -m fraud_docs_rag.main query "What are SAR requirements?"
    python -m fraud_docs_rag.main ingest ./documents/
    python -m fraud_docs_rag.main serve
    python -m fraud_docs_rag.main health
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fraud_docs_rag.config import settings
from fraud_docs_rag.generation.rag_chain import RAGChain
from fraud_docs_rag.ingestion.document_processor import DocumentProcessor
from fraud_docs_rag.retrieval.hybrid_retriever import HybridRetriever

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOGS_DIR / "frauddocs_rag.log"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# CLI Commands
# ============================================================================


@click.group()
@click.version_option(version=settings.APP_VERSION)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-error output")
def cli(verbose: bool, quiet: bool):
    """
    FraudDocs-RAG: Financial fraud detection document Q&A system.

    A production-grade RAG system for querying financial regulations
    and fraud detection documents.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.ERROR)


@cli.command()
@click.argument("question", required=True)
@click.option(
    "--filter", "-f",
    "doc_type",
    type=click.Choice(["aml", "kyc", "fraud", "regulation", "general"], case_sensitive=False),
    help="Filter by document type"
)
@click.option("--top-k", "-k", type=int, default=None, help="Number of documents to retrieve")
@click.option("--no-rerank", is_flag=True, help="Disable cross-encoder reranking")
@click.option("--format", type=click.Choice(["text", "json"]), default="text", help="Output format")
def query(question: str, doc_type: Optional[str], top_k: Optional[int],
          no_rerank: bool, format: str):
    """
    Query the knowledge base.

    Example:
        frauddocs query "What are SAR filing requirements?" --filter aml
    """
    click.echo(f"🔍 Querying: {question[:100]}...")
    if doc_type:
        click.echo(f"   Filter: {doc_type}")

    try:
        # Initialize retriever
        retriever = HybridRetriever(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            chroma_path=settings.CHROMA_PERSIST_DIRECTORY,
            top_k=top_k or settings.TOP_K_RETRIEVAL,
            rerank_top_n=settings.RERANK_TOP_N,
        )

        if not retriever.load_index():
            click.echo("❌ No vector index found. Please ingest documents first.", err=True)
            click.echo("   Run: frauddocs ingest <directory>")
            sys.exit(1)

        # Initialize RAG chain
        rag_chain = RAGChain(
            retriever=retriever,
            environment=settings.ENVIRONMENT,
        )

        # Process query
        answer, sources = rag_chain.query(
            question=question,
            doc_type_filter=doc_type,
            use_rerank=not no_rerank,
            top_k=top_k,
        )

        # Format output
        if format == "json":
            import json
            output = {
                "question": question,
                "answer": answer,
                "sources": sources,
            }
            click.echo(json.dumps(output, indent=2))
        else:
            click.echo("\n" + "=" * 80)
            click.echo("ANSWER")
            click.echo("=" * 80)
            click.echo(answer)
            click.echo("\n" + "=" * 80)
            click.echo(f"SOURCES ({len(sources)})")
            click.echo("=" * 80)
            for i, source in enumerate(sources, 1):
                click.echo(f"\n[{i}] {source['file_name']}")
                click.echo(f"    Category: {source['category']}")
                click.echo(f"    Score: {source['score']:.3f}")
                click.echo(f"    Preview: {source['text_preview'][:150]}...")

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("path", type=click.Path(exists=True), required=True)
@click.option("--recursive", "-r", is_flag=True, help="Process directories recursively")
@click.option("--no-context", is_flag=True, help="Don't add contextual headers to chunks")
@click.option("--format", type=click.Choice(["text", "json"]), default="text", help="Output format")
def ingest(path: str, recursive: bool, no_context: bool, format: str):
    """
    Ingest documents into the knowledge base.

    PATH can be a file or directory.

    Example:
        frauddocs ingest ./documents/ --recursive
    """
    path_obj = Path(path)

    if path_obj.is_file():
        click.echo(f"📄 Ingesting file: {path_obj.name}")
    else:
        click.echo(f"📁 Ingesting directory: {path_obj}")
        if recursive:
            click.echo("   Recursive: enabled")

    try:
        # Initialize document processor
        processor = DocumentProcessor(
            embed_model_name=settings.EMBEDDING_MODEL,
        )

        # Process documents
        if path_obj.is_file():
            nodes = processor.process_document(path_obj, add_context=not no_context)
        else:
            nodes = processor.process_directory(
                path_obj,
                recursive=recursive,
                add_context=not no_context,
            )

        if not nodes or len(nodes) == 0:
            click.echo("⚠️  No documents were processed.", err=True)
            sys.exit(1)

        # Build or update index
        click.echo(f"\n📊 Processed {len(nodes)} chunks")

        retriever = HybridRetriever(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            chroma_path=settings.CHROMA_PERSIST_DIRECTORY,
        )

        # Check if index exists
        index_existed = retriever.load_index()

        if index_existed:
            click.echo("📦 Adding to existing index...")
        else:
            click.echo("📦 Creating new index...")

        retriever.build_index(nodes)

        # Get statistics
        stats = retriever.get_collection_stats()

        if format == "json":
            import json
            output = {
                "status": "success",
                "chunks_created": len(nodes),
                "total_docs": stats.get("total_docs", 0),
                "collection": settings.CHROMA_COLLECTION_NAME,
            }
            click.echo(json.dumps(output, indent=2))
        else:
            click.echo("\n✅ Ingestion complete!")
            click.echo(f"   Total documents in collection: {stats.get('total_docs', 0)}")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--host", "-h", default=settings.HOST, help="Host to bind to")
@click.option("--port", "-p", default=settings.PORT, type=int, help="Port to bind to")
@click.option("--workers", "-w", default=settings.WORKERS, type=int, help="Number of workers")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, workers: int, reload: bool):
    """
    Start the FastAPI server.

    Example:
        frauddocs serve --host 0.0.0.0 --port 8000 --reload
    """
    click.echo(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    click.echo(f"   Environment: {settings.ENVIRONMENT}")
    click.echo(f"   Host: {host}")
    click.echo(f"   Port: {port}")
    click.echo(f"   Workers: {workers}")
    click.echo(f"   Reload: {reload}")

    try:
        import uvicorn

        uvicorn.run(
            "fraud_docs_rag.api.main:app",
            host=host,
            port=port,
            workers=workers if not reload else 1,
            reload=reload,
            log_level=settings.LOG_LEVEL.lower(),
        )
    except ImportError:
        click.echo("❌ uvicorn is required. Install with: pip install uvicorn", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--format", type=click.Choice(["text", "json"]), default="text", help="Output format")
def health(format: str):
    """
    Check system health.

    Example:
        frauddocs health
    """
    try:
        health_status = {
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "components": {},
        }

        # Check ChromaDB
        try:
            retriever = HybridRetriever(
                collection_name=settings.CHROMA_COLLECTION_NAME,
                chroma_path=settings.CHROMA_PERSIST_DIRECTORY,
            )
            retriever.load_index()

            stats = retriever.get_collection_stats()
            health_status["components"]["retriever"] = {
                "status": "healthy" if retriever.index else "not_loaded",
                "total_docs": stats.get("total_docs", 0),
            }
        except Exception as e:
            health_status["components"]["retriever"] = {
                "status": "error",
                "error": str(e),
            }

        # Check LLM configuration
        llm_config = settings.get_llm_config()
        health_status["components"]["llm"] = {
            "provider": llm_config["provider"],
            "model": llm_config["model"],
            "configured": bool(llm_config.get("api_key") or llm_config["provider"] == "ollama"),
        }

        # Overall status
        components_ok = all(
            c.get("status") in ["healthy", "configured"]
            for c in health_status["components"].values()
        )
        health_status["overall"] = "healthy" if components_ok else "unhealthy"

        if format == "json":
            import json
            click.echo(json.dumps(health_status, indent=2))
        else:
            click.echo(f"\n🏥 {settings.APP_NAME} Health Check")
            click.echo("=" * 60)
            click.echo(f"Version: {health_status['version']}")
            click.echo(f"Environment: {health_status['environment']}")
            click.echo(f"Overall: {health_status['overall'].upper()}")
            click.echo()

            for component, status in health_status["components"].items():
                status_icon = "✅" if status.get("status") in ["healthy", "configured"] else "❌"
                click.echo(f"{status_icon} {component.upper()}")
                for key, value in status.items():
                    if key != "status":
                        click.echo(f"   {key}: {value}")
                click.echo()

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("collection-name", default=settings.CHROMA_COLLECTION_NAME)
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
def delete_collection(collection_name: str, confirm: bool):
    """
    Delete a vector store collection.

    WARNING: This operation is irreversible.

    Example:
        frauddocs delete-collection financial_documents --confirm
    """
    if not confirm:
        click.echo(f"⚠️  You are about to DELETE the collection: {collection_name}")
        click.echo("   This operation cannot be undone!")
        if not click.confirm("Are you sure?"):
            click.echo("Aborted.")
            sys.exit(0)

    try:
        retriever = HybridRetriever(
            collection_name=collection_name,
            chroma_path=settings.CHROMA_PERSIST_DIRECTORY,
        )

        if retriever.delete_index():
            click.echo(f"✅ Collection '{collection_name}' deleted successfully")
        else:
            click.echo(f"❌ Failed to delete collection", err=True)
            sys.exit(1)

    except Exception as e:
        logger.error(f"Delete failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def stats():
    """
    Display system statistics.

    Example:
        frauddocs stats
    """
    try:
        click.echo(f"\n📊 {settings.APP_NAME} Statistics")
        click.echo("=" * 60)

        # Configuration
        click.echo("\nConfiguration:")
        click.echo(f"  Environment: {settings.ENVIRONMENT}")
        click.echo(f"  LLM Provider: {settings.llm_provider}")
        click.echo(f"  LLM Model: {settings.llm_model}")
        click.echo(f"  Embedding Model: {settings.EMBEDDING_MODEL}")

        # Collection stats
        retriever = HybridRetriever(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            chroma_path=settings.CHROMA_PERSIST_DIRECTORY,
        )

        if retriever.load_index():
            stats = retriever.get_collection_stats()
            click.echo("\nVector Store:")
            click.echo(f"  Collection: {stats.get('collection_name')}")
            click.echo(f"  Status: {stats.get('status')}")
            click.echo(f"  Total Documents: {stats.get('total_docs', 0)}")
        else:
            click.echo("\nVector Store: Not initialized")

        # Paths
        click.echo("\nPaths:")
        click.echo(f"  Data: {settings.DATA_DIR}")
        click.echo(f"  Documents: {settings.DOCUMENTS_DIR}")
        click.echo(f"  ChromaDB: {settings.CHROMA_PERSIST_DIRECTORY}")
        click.echo(f"  Logs: {settings.LOGS_DIR}")

        # Retrieval settings
        click.echo("\nRetrieval Settings:")
        click.echo(f"  Top K: {settings.TOP_K_RETRIEVAL}")
        click.echo(f"  Reranking: {settings.RERANK_ENABLED}")
        click.echo(f"  Rerank Model: {settings.RERANK_MODEL}")
        click.echo(f"  Rerank Top N: {settings.RERANK_TOP_N}")

        click.echo()

    except Exception as e:
        logger.error(f"Stats failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def config(output: Optional[str]):
    """
    Display current configuration.

    Example:
        frauddocs config --output config.json
    """
    config_data = {
        "environment": settings.ENVIRONMENT,
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "llm": settings.get_llm_config(),
        "retrieval": {
            "top_k": settings.TOP_K_RETRIEVAL,
            "rerank_enabled": settings.RERANK_ENABLED,
            "rerank_model": settings.RERANK_MODEL,
            "rerank_top_n": settings.RERANK_TOP_N,
        },
        "paths": {
            "data": str(settings.DATA_DIR),
            "documents": str(settings.DOCUMENTS_DIR),
            "chroma_db": str(settings.CHROMA_PERSIST_DIRECTORY),
            "logs": str(settings.LOGS_DIR),
        },
        "chromadb": {
            "host": settings.CHROMA_HOST,
            "port": settings.CHROMA_PORT,
            "collection_name": settings.CHROMA_COLLECTION_NAME,
        },
    }

    if output:
        import json
        with open(output, "w") as f:
            json.dump(config_data, f, indent=2)
        click.echo(f"✅ Configuration saved to: {output}")
    else:
        import json
        click.echo(json.dumps(config_data, indent=2))


# ============================================================================
# Interactive Mode
# ============================================================================


@cli.command()
def interactive():
    """
    Start interactive Q&A mode.

    Example:
        frauddocs interactive
    """
    click.echo(f"\n🤖 {settings.APP_NAME} Interactive Mode")
    click.echo("=" * 60)
    click.echo("Type your questions below. Type 'quit' or 'exit' to stop.")
    click.echo()

    try:
        # Initialize components
        retriever = HybridRetriever(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            chroma_path=settings.CHROMA_PERSIST_DIRECTORY,
        )

        if not retriever.load_index():
            click.echo("❌ No vector index found. Please ingest documents first.")
            sys.exit(1)

        rag_chain = RAGChain(
            retriever=retriever,
            environment=settings.ENVIRONMENT,
        )

        # Interactive loop
        while True:
            try:
                question = click.prompt("\n❓ Question", default="", show_default=False)

                if not question.strip():
                    continue

                if question.lower() in ["quit", "exit", "q"]:
                    click.echo("Goodbye! 👋")
                    break

                click.echo("🔍 Processing...")

                answer, sources = rag_chain.query(question)

                click.echo("\n" + "=" * 60)
                click.echo("ANSWER")
                click.echo("=" * 60)
                click.echo(answer)

                if sources:
                    click.echo("\n" + "-" * 60)
                    click.echo(f"SOURCES ({len(sources)})")
                    click.echo("-" * 60)
                    for i, source in enumerate(sources, 1):
                        click.echo(f"\n[{i}] {source['file_name']} ({source['category']}) - {source['score']:.2f}")

            except KeyboardInterrupt:
                click.echo("\n\nGoodbye! 👋")
                break
            except Exception as e:
                click.echo(f"\n❌ Error: {e}", err=True)

    except Exception as e:
        logger.error(f"Interactive mode failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    cli()
