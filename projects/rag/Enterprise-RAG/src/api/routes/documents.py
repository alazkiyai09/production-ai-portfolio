# ============================================================
# Enterprise-RAG: Document Routes
# ============================================================
"""
Document ingestion and management endpoints.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ============================================================
# Request/Response Models
# ============================================================

class IngestResponse(BaseModel):
    """Response model for document ingestion."""

    document_id: str
    filename: str
    chunks_created: int
    processing_time: float
    status: str


class DocumentInfo(BaseModel):
    """Information about an ingested document."""

    doc_id: str
    filename: str
    file_type: str
    chunks: int
    metadata: dict


class DocumentsListResponse(BaseModel):
    """Response model for documents list."""

    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int


class DeleteResponse(BaseModel):
    """Response model for document deletion."""

    deleted: bool
    doc_id: str
    chunks_deleted: int


# ============================================================
# Routes
# ============================================================

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(..., description="Document file (PDF, DOCX, MD, TXT)"),
):
    """
    Upload and ingest a document into the knowledge base.

    Supported formats:
    - PDF (.pdf)
    - Word (.docx)
    - Markdown (.md)
    - Text (.txt)

    The document will be:
    1. Text extraction based on format
    2. Chunked with overlap
    3. Embedded into vectors
    4. Stored in vector database

    Args:
        file: Uploaded document file

    Returns:
        Document ID and chunk count

    Example:
        POST /api/v1/documents/ingest
        Content-Type: multipart/form-data
        file: document.pdf
    """
    from fastapi import Request

    request_obj = Request.scope()["app"]
    document_processor = getattr(request_obj.state, "document_processor", None)
    rag_chain = getattr(request_obj.state, "rag_chain", None)

    if document_processor is None or rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized",
        )

    # Validate file size
    max_size = settings.MAX_FILE_SIZE * 1024 * 1024  # Convert to bytes
    content = await file.read()

    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE}MB",
        )

    # Validate file type (both extension and MIME type)
    file_ext = Path(file.filename).suffix.lower()
    supported_formats = settings.supported_formats_list

    if file_ext not in supported_formats:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file_ext}. Supported: {supported_formats}",
        )

    # Validate MIME type for security (prevent extension spoofing)
    try:
        import magic
        detected_mime = magic.from_buffer(content, mime=True)

        # Map extensions to expected MIME types
        mime_map = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".md": "text/markdown",
            ".txt": "text/plain",
        }

        expected_mime = mime_map.get(file_ext)
        if expected_mime and not detected_mime.startswith(expected_mime.split("/")[0]):
            logger.warning(
                f"MIME type mismatch for {file.filename}: "
                f"expected {expected_mime}, detected {detected_mime}"
            )
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"File content doesn't match extension {file_ext}. Detected: {detected_mime}",
            )
    except ImportError:
        logger.warning("python-magic not installed, skipping MIME validation")
    except Exception as e:
        logger.warning(f"MIME validation failed: {e}, continuing with extension validation")

    try:
        logger.info(f"Ingesting document: {file.filename} ({len(content)} bytes)")

        # Process document
        import time

        start_time = time.time()

        result = document_processor.process_bytes(
            content=content,
            filename=file.filename,
            metadata={"source": file.filename},
        )

        processing_time = time.time() - start_time

        if result.errors:
            logger.warning(f"Document processed with errors: {result.errors}")

        if not result.documents:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to extract content from {file.filename}",
            )

        # Add to vector store and BM25
        rag_chain.retriever.add_documents(result.documents)

        # Get document ID from first chunk
        doc_id = result.documents[0].doc_id if result.documents else "unknown"

        logger.info(
            f"Document ingested successfully: {file.filename} -> {doc_id} ({result.total_chunks} chunks)"
        )

        return IngestResponse(
            document_id=doc_id,
            filename=file.filename,
            chunks_created=result.total_chunks,
            processing_time=processing_time,
            status="success",
        )

    except Exception as e:
        logger.error(f"Document ingestion failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}",
        )


@router.get("", response_model=DocumentsListResponse)
async def list_documents():
    """
    List all ingested documents.

    Returns information about all documents in the knowledge base.

    Example:
        GET /api/v1/documents
    """
    from fastapi import Request
    from collections import defaultdict

    request_obj = Request.scope()["app"]
    vector_store = getattr(request_obj.state, "vector_store", None)

    if vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not initialized",
        )

    try:
        stats = vector_store.get_stats()

        # For now, return summary info
        # In production, you'd want to track individual documents
        return DocumentsListResponse(
            documents=[],
            total_documents=stats.total_documents,
            total_chunks=stats.total_chunks,
        )

    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}",
        )


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    """
    Delete a document from the knowledge base.

    Args:
        doc_id: Document ID to delete

    Returns:
        Deletion status

    Example:
        DELETE /api/v1/documents/doc_abc123
    """
    from fastapi import Request

    request_obj = Request.scope()["app"]
    vector_store = getattr(request_obj.state, "vector_store", None)
    rag_chain = getattr(request_obj.state, "rag_chain", None)

    if vector_store is None or rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized",
        )

    try:
        logger.info(f"Deleting document: {doc_id}")

        # Delete from vector store
        chunks_deleted = vector_store.delete([doc_id])

        # Note: BM25 index doesn't support efficient deletion
        # In production, you'd want to track this and rebuild periodically

        logger.info(f"Document deleted: {doc_id} ({chunks_deleted} chunks)")

        return DeleteResponse(
            deleted=True,
            doc_id=doc_id,
            chunks_deleted=chunks_deleted,
        )

    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}",
        )


@router.post("/batch-ingest")
async def batch_ingest(files: list[UploadFile] = File(...)):
    """
    Batch ingest multiple documents.

    Args:
        files: List of document files

    Returns:
        Summary of ingestion results

    Example:
        POST /api/v1/documents/batch-ingest
        Files: [doc1.pdf, doc2.docx, doc3.txt]
    """
    from fastapi import Request

    request_obj = Request.scope()["app"]
    document_processor = getattr(request_obj.state, "document_processor", None)
    rag_chain = getattr(request_obj.state, "rag_chain", None)

    if document_processor is None or rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized",
        )

    results = []
    total_chunks = 0
    errors = []

    for file in files:
        try:
            content = await file.read()

            result = document_processor.process_bytes(
                content=content,
                filename=file.filename,
                metadata={"source": file.filename},
            )

            if result.documents:
                rag_chain.retriever.add_documents(result.documents)

                results.append(
                    {
                        "filename": file.filename,
                        "status": "success",
                        "chunks": result.total_chunks,
                    }
                )
                total_chunks += result.total_chunks
            else:
                errors.append({"filename": file.filename, "error": "No content extracted"})

        except Exception as e:
            logger.error(f"Failed to ingest {file.filename}: {str(e)}")
            errors.append({"filename": file.filename, "error": str(e)})

    return {
        "processed": len(results),
        "total_chunks": total_chunks,
        "errors": errors,
        "timestamp": datetime.utcnow().isoformat(),
    }
