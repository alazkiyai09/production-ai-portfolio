"""
Multi-Modal API Endpoints for RAG System.

FastAPI endpoints supporting:
- Multi-modal querying with content type filters
- Document ingestion with image/table extraction
- Image and table retrieval
- Multi-modal citation support
"""

import io
import base64
import uuid
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    Query,
    Depends,
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..multimodal import (
    MultiModalRetriever,
    MultiModalRAGChain,
    ImageProcessor,
    TableExtractor,
    ImageData,
    TableData,
    VisionLLM,
    VisionProvider,
    MultiModalResponse,
    ImageCitation,
    TableCitation,
    TextCitation,
)

logger = logging.getLogger(__name__)

# Global router
router = APIRouter(prefix="/api/v1/multimodal", tags=["multimodal"])

# Global instances (initialized in main.py)
retriever: Optional[MultiModalRetriever] = None
rag_chain: Optional[MultiModalRAGChain] = None
image_processor: Optional[ImageProcessor] = None
table_extractor: Optional[TableExtractor] = None
vision_llm: Optional[VisionLLM] = None

# Storage for images and tables
images_store: Dict[str, ImageData] = {}
tables_store: Dict[str, TableData] = {}


# ==================== Pydantic Models ====================

class MultiModalQueryRequest(BaseModel):
    """Request model for multi-modal query."""
    question: str = Field(..., description="Question to ask")
    content_types: List[str] = Field(
        default=["text", "image", "table"],
        description="Content types to include in search"
    )
    include_images: bool = Field(default=True, description="Include images in response")
    include_tables: bool = Field(default=True, description="Include tables in response")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum relevance score")
    stream: bool = Field(default=False, description="Stream the response")


class MultiModalQueryResponse(BaseModel):
    """Response model for multi-modal query."""
    answer: str
    query: str
    sources_used: List[str]
    text_citations: List[Dict[str, Any]]
    image_citations: List[Dict[str, Any]]
    table_citations: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    extract_images: bool = Field(default=True, description="Extract images from documents")
    extract_tables: bool = Field(default=True, description="Extract tables from documents")
    generate_captions: bool = Field(default=True, description="Generate image captions")
    extract_ocr: bool = Field(default=True, description="Extract OCR text from images")
    generate_embeddings: bool = Field(default=True, description="Generate embeddings")


class ImageDataResponse(BaseModel):
    """Response model for image data."""
    id: str
    caption: Optional[str]
    ocr_text: Optional[str]
    source_doc: str
    page_number: int
    image_index: int
    format: str
    size: tuple
    base64: Optional[str]
    metadata: Dict[str, Any]


class TableDataResponse(BaseModel):
    """Response model for table data."""
    id: str
    description: Optional[str]
    source_doc: str
    page_number: int
    table_index: int
    row_count: int
    col_count: int
    markdown: str
    csv: str
    json: str
    metadata: Dict[str, Any]


# ==================== Dependencies ====================

def get_retriever() -> MultiModalRetriever:
    """Get retriever instance."""
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    return retriever


def get_rag_chain() -> MultiModalRAGChain:
    """Get RAG chain instance."""
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG chain not initialized")
    return rag_chain


# ==================== Query Endpoints ====================

@router.post("/query", response_model=MultiModalQueryResponse)
async def query_multimodal(
    request: MultiModalQueryRequest,
    rag: MultiModalRAGChain = Depends(get_rag_chain),
):
    """
    Query the multi-modal RAG system.

    Supports querying across text, images, and tables with configurable filters.
    """
    try:
        logger.info(f"Multi-modal query: {request.question[:100]}")

        # Process query
        response: MultiModalResponse = rag.query(
            question=request.question,
            include_images=request.include_images,
            include_tables=request.include_tables,
            content_types=request.content_types,
            top_k=request.top_k,
            min_score=request.min_score,
        )

        # Convert citations to dicts
        text_citations = [c.to_dict() for c in response.text_citations]
        image_citations = [c.to_dict() for c in response.image_citations]
        table_citations = [c.to_dict() for c in response.table_citations]

        return MultiModalQueryResponse(
            answer=response.answer,
            query=response.query,
            sources_used=response.sources_used,
            text_citations=text_citations,
            image_citations=image_citations,
            table_citations=table_citations,
            metadata=response.metadata,
        )

    except Exception as e:
        logger.error(f"Multi-modal query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/image")
async def query_with_image(
    question: str = Form(...),
    image: UploadFile = File(...),
    include_text: bool = Form(True),
    include_tables: bool = Form(True),
    top_k: int = Form(10),
    rag: MultiModalRAGChain = Depends(get_rag_chain),
):
    """
    Query using an image (cross-modal search).

    Upload an image and find similar content across all modalities.
    """
    try:
        # Read image
        image_bytes = await image.read()

        logger.info(f"Image query: {question[:100]}, image size: {len(image_bytes)}")

        # Process query
        response: MultiModalResponse = rag.query_with_image(
            question=question,
            query_image=image_bytes,
            include_text=include_text,
            include_tables=include_tables,
            top_k=top_k,
        )

        # Convert citations
        text_citations = [c.to_dict() for c in response.text_citations]
        image_citations = [c.to_dict() for c in response.image_citations]
        table_citations = [c.to_dict() for c in response.table_citations]

        return MultiModalQueryResponse(
            answer=response.answer,
            query=response.query,
            sources_used=response.sources_used,
            text_citations=text_citations,
            image_citations=image_citations,
            table_citations=table_citations,
            metadata=response.metadata,
        )

    except Exception as e:
        logger.error(f"Image query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Ingestion Endpoints ====================

@router.post("/ingest")
async def ingest_multimodal(
    file: UploadFile = File(..., description="Document to ingest"),
    extract_images: bool = Form(True),
    extract_tables: bool = Form(True),
    generate_captions: bool = Form(True),
    extract_ocr: bool = Form(True),
    generate_embeddings: bool = Form(True),
    retriev: MultiModalRetriever = Depends(get_retriever),
):
    """
    Ingest a document with multi-modal extraction.

    Extracts images and tables, generates captions and OCR text,
    and indexes all content for retrieval.
    """
    try:
        # Read file
        file_bytes = await file.read()
        file_path = Path(file.filename)

        logger.info(f"Ingesting {file.filename} ({len(file_bytes)} bytes)")

        results = {
            "filename": file.filename,
            "file_size": len(file_bytes),
            "extracted_images": 0,
            "extracted_tables": 0,
            "indexed_nodes": 0,
            "image_ids": [],
            "table_ids": [],
        }

        # Save to temporary file
        temp_path = Path(f"/tmp/{uuid.uuid4()}_{file_path.name}")
        temp_path.write_bytes(file_bytes)

        try:
            # Extract images
            if extract_images and image_processor:
                if file_path.suffix.lower() == ".pdf":
                    images = image_processor.extract_images_from_pdf(
                        temp_path,
                        extract_embeddings=generate_embeddings,
                        generate_captions=generate_captions,
                        extract_ocr=extract_ocr,
                    )
                elif file_path.suffix.lower() in [".docx", ".doc"]:
                    images = image_processor.extract_images_from_docx(
                        temp_path,
                        extract_embeddings=generate_embeddings,
                        generate_captions=generate_captions,
                        extract_ocr=extract_ocr,
                    )
                else:
                    images = []

                for img in images:
                    img_id = str(uuid.uuid4())
                    img.source_doc = file.filename
                    images_store[img_id] = img
                    results["image_ids"].append(img_id)

                    # Index in retriever
                    retriev.add_image_node(
                        node_id=img_id,
                        image=img.image_bytes,
                        content=img.caption or "",
                        caption=img.caption,
                        ocr_text=img.ocr_text,
                        metadata={
                            "source": file.filename,
                            "page_number": img.page_number,
                            "image_index": img.image_index,
                        },
                    )

                results["extracted_images"] = len(images)

            # Extract tables
            if extract_tables and table_extractor:
                if file_path.suffix.lower() == ".pdf":
                    tables = table_extractor.extract_tables_from_pdf(
                        temp_path,
                        extract_embeddings=generate_embeddings,
                        extract_descriptions=generate_captions,
                    )
                else:
                    tables = []

                for tbl in tables:
                    tbl_id = str(uuid.uuid4())
                    tbl.source_doc = file.filename
                    tables_store[tbl_id] = tbl
                    results["table_ids"].append(tbl_id)

                    # Index in retriever
                    retriev.add_table_node(
                        node_id=tbl_id,
                        dataframe=tbl.dataframe,
                        description=tbl.description,
                        metadata={
                            "source": file.filename,
                            "page_number": tbl.page_number,
                            "table_index": tbl.table_index,
                        },
                    )

                results["extracted_tables"] = len(tables)

            results["indexed_nodes"] = results["extracted_images"] + results["extracted_tables"]

            logger.info(f"Ingestion complete: {results}")

        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

        return JSONResponse(content=results)

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/image")
async def ingest_image(
    file: UploadFile = File(...),
    content: str = Form(""),
    generate_caption: bool = Form(True),
    extract_ocr: bool = Form(True),
    retriev: MultiModalRetriever = Depends(get_retriever),
):
    """
    Ingest a single image.

    Upload an image, optionally generate caption and OCR text,
    and index for retrieval.
    """
    try:
        image_bytes = await file.read()
        img_id = str(uuid.uuid4())

        logger.info(f"Ingesting image {file.filename}: {len(image_bytes)} bytes")

        # Process image
        if image_processor:
            img_data = image_processor.process_image(
                image_bytes,
                source_doc=file.filename,
                generate_caption=generate_caption,
                extract_ocr=extract_ocr,
                generate_embedding=True,
            )
        else:
            # Minimal processing
            from ..multimodal import ImageData
            img_data = ImageData(image_bytes=image_bytes, source_doc=file.filename)

        img_data.source_doc = file.filename
        images_store[img_id] = img_data

        # Index in retriever
        retriev.add_image_node(
            node_id=img_id,
            image=img_data.image_bytes,
            content=content or img_data.caption or "",
            caption=img_data.caption,
            ocr_text=img_data.ocr_text,
            metadata={"source": file.filename},
        )

        return {
            "id": img_id,
            "filename": file.filename,
            "caption": img_data.caption,
            "ocr_text": img_data.ocr_text,
        }

    except Exception as e:
        logger.error(f"Image ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/table")
async def ingest_table(
    file: UploadFile = File(..., description="CSV file with table data"),
    description: str = Form(..., description="Table description"),
    retriev: MultiModalRetriever = Depends(get_retriever),
):
    """
    Ingest a table from CSV.

    Upload a CSV file and index for retrieval.
    """
    try:
        import pandas as pd

        content = await file.read()
        tbl_id = str(uuid.uuid4())

        logger.info(f"Ingesting table {file.filename}: {len(content)} bytes")

        # Parse CSV
        df = pd.read_csv(io.BytesIO(content))

        # Store
        from ..multimodal import TableData
        tbl_data = TableData(
            dataframe=df,
            description=description,
            source_doc=file.filename,
        )
        tables_store[tbl_id] = tbl_data

        # Index
        retriev.add_table_node(
            node_id=tbl_id,
            dataframe=df,
            description=description,
            metadata={"source": file.filename},
        )

        return {
            "id": tbl_id,
            "filename": file.filename,
            "row_count": len(df),
            "col_count": len(df.columns),
            "description": description,
        }

    except Exception as e:
        logger.error(f"Table ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Retrieval Endpoints ====================

@router.get("/images/{image_id}", response_model=ImageDataResponse)
async def get_image(image_id: str):
    """
    Get image data by ID.

    Returns image metadata and base64-encoded image.
    """
    if image_id not in images_store:
        raise HTTPException(status_code=404, detail="Image not found")

    img = images_store[image_id]

    return ImageDataResponse(
        id=image_id,
        caption=img.caption,
        ocr_text=img.ocr_text,
        source_doc=img.source_doc,
        page_number=img.page_number,
        image_index=img.image_index,
        format=img.format,
        size=img.size,
        base64=img.to_base64(),
        metadata=img.metadata,
    )


@router.get("/images/{image_id}/preview")
async def get_image_preview(image_id: str):
    """
    Get image preview as image file.

    Returns the actual image file for display.
    """
    if image_id not in images_store:
        raise HTTPException(status_code=404, detail="Image not found")

    img = images_store[image_id]

    return StreamingResponse(
        io.BytesIO(img.image_bytes),
        media_type=f"image/{img.format}",
        headers={
            "Content-Disposition": f"inline; filename={image_id}.{img.format}"
        }
    )


@router.get("/tables/{table_id}", response_model=TableDataResponse)
async def get_table(table_id: str):
    """
    Get table data by ID.

    Returns table metadata and data in multiple formats.
    """
    if table_id not in tables_store:
        raise HTTPException(status_code=404, detail="Table not found")

    tbl = tables_store[table_id]

    return TableDataResponse(
        id=table_id,
        description=tbl.description,
        source_doc=tbl.source_doc,
        page_number=tbl.page_number,
        table_index=tbl.table_index,
        row_count=tbl.row_count,
        col_count=tbl.col_count,
        markdown=tbl.to_markdown(),
        csv=tbl.to_csv(),
        json=tbl.to_json(),
        metadata=tbl.metadata,
    )


@router.get("/tables/{table_id}/html")
async def get_table_html(table_id: str):
    """
    Get table as HTML.

    Returns table formatted as HTML for display.
    """
    if table_id not in tables_store:
        raise HTTPException(status_code=404, detail="Table not found")

    tbl = tables_store[table_id]

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Table {table_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h2>Table from {tbl.source_doc}</h2>
        <p><strong>Description:</strong> {tbl.description or 'N/A'}</p>
        <p><strong>Source:</strong> {tbl.source_doc}, Page {tbl.page_number}</p>
        {tbl.to_html()}
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@router.get("/images")
async def list_images(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    """List all stored images."""
    items = list(images_store.values())
    items.sort(key=lambda x: x.source_doc)

    paginated = items[skip:skip + limit]

    return {
        "total": len(items),
        "skip": skip,
        "limit": limit,
        "images": [
            {
                "id": img_id,
                "source_doc": img.source_doc,
                "page_number": img.page_number,
                "caption": img.caption,
                "has_ocr": bool(img.ocr_text),
                "format": img.format,
                "size": img.size,
            }
            for img_id, img in list(images_store.items())[skip:skip + limit]
        ],
    }


@router.get("/tables")
async def list_tables(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    """List all stored tables."""
    items = list(tables_store.values())
    items.sort(key=lambda x: x.source_doc)

    return {
        "total": len(items),
        "skip": skip,
        "limit": limit,
        "tables": [
            {
                "id": tbl_id,
                "source_doc": tbl.source_doc,
                "page_number": tbl.page_number,
                "description": tbl.description,
                "row_count": tbl.row_count,
                "col_count": tbl.col_count,
            }
            for tbl_id, tbl in list(tables_store.items())[skip:skip + limit]
        ],
    }


@router.delete("/images/{image_id}")
async def delete_image(image_id: str, retriev: MultiModalRetriever = Depends(get_retriever)):
    """Delete an image."""
    if image_id not in images_store:
        raise HTTPException(status_code=404, detail="Image not found")

    del images_store[image_id]
    # Note: Would need to remove from retriever index too
    # retriev.image_nodes = [n for n in retriev.image_nodes if n["id"] != image_id]

    return {"message": "Image deleted"}


@router.delete("/tables/{table_id}")
async def delete_table(table_id: str, retriev: MultiModalRetriever = Depends(get_retriever)):
    """Delete a table."""
    if table_id not in tables_store:
        raise HTTPException(status_code=404, detail="Table not found")

    del tables_store[table_id]
    # Note: Would need to remove from retriever index too

    return {"message": "Table deleted"}


# ==================== Utility Endpoints ====================

@router.get("/stats")
async def get_stats(retriev: MultiModalRetriever = Depends(get_retriever)):
    """Get multi-modal index statistics."""
    stats = retriever.stats()

    return {
        **stats,
        "stored_images": len(images_store),
        "stored_tables": len(tables_store),
    }


# ==================== HTML Response ====================

from fastapi.responses import HTMLResponse
