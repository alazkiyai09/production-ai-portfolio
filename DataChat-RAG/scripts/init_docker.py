#!/usr/bin/env python3
"""
Docker Initialization Script for DataChat-RAG

Waits for databases to be ready, runs migrations, and seeds data.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def wait_for_postgres(max_retries=30, retry_interval=2):
    """Wait for PostgreSQL to be ready."""
    import psycopg2
    from psycopg2 import OperationalError

    db_host = os.getenv("DB_HOST", "postgres")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_name = os.getenv("DB_NAME", "datachat_rag")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "postgres")

    logger.info(f"Waiting for PostgreSQL at {db_host}:{db_port}...")

    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=db_host,
                port=db_port,
                database=db_name,
                user=db_user,
                password=db_password,
            )
            logger.info("✓ PostgreSQL is ready!")
            conn.close()
            return True
        except OperationalError as e:
            if attempt < max_retries - 1:
                logger.info(f"PostgreSQL not ready yet, retrying in {retry_interval}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_interval)
            else:
                logger.error(f"✗ Failed to connect to PostgreSQL after {max_retries} attempts")
                logger.error(f"Error: {e}")
                return False

    return False


def wait_for_chromadb(max_retries=30, retry_interval=2):
    """Wait for ChromaDB to be ready."""
    import requests

    chroma_host = os.getenv("CHROMA_HOST", "chromadb")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
    chroma_url = f"http://{chroma_host}:{chroma_port}"

    logger.info(f"Waiting for ChromaDB at {chroma_url}...")

    for attempt in range(max_retries):
        try:
            response = requests.get(f"{chroma_url}/api/v1/heartbeat", timeout=5)
            if response.status_code == 200:
                logger.info("✓ ChromaDB is ready!")
                return True
        except Exception as e:
            if attempt < max_retries - 1:
                logger.info(f"ChromaDB not ready yet, retrying in {retry_interval}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_interval)
            else:
                logger.error(f"✗ Failed to connect to ChromaDB after {max_retries} attempts")
                logger.error(f"Error: {e}")
                return False

    return False


def setup_database():
    """Set up database tables and seed data."""
    try:
        from scripts.setup_database import (
            create_database,
            create_tables,
            seed_database,
        )

        logger.info("Setting up database...")

        # Create tables
        create_tables()
        logger.info("✓ Database tables created")

        # Seed sample data
        seed_database(
            num_campaigns=100,
            days_to_generate=30,
            generate_raw_events=False,
        )
        logger.info("✓ Database seeded with sample data")

        return True

    except Exception as e:
        logger.error(f"✗ Failed to set up database: {e}")
        import traceback
        traceback.print_exc()
        return False


def setup_documents():
    """Ingest sample documents into ChromaDB."""
    try:
        from llama_index.core import Document
        from src.retrievers import DocumentRetriever, DocumentType

        # Check if OPENAI_API_KEY is available
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("⚠ OPENAI_API_KEY not set, skipping document ingestion")
            return True

        logger.info("Setting up document retriever...")

        # Create document retriever
        retriever = DocumentRetriever(
            chroma_path=os.getenv("CHROMA_PERSIST_DIR", "./data/chromadb"),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "datachat_documents"),
        )

        # Check if documents already exist
        stats = retriever.get_stats()
        if stats["total_chunks"] > 0:
            logger.info(f"✓ Documents already exist ({stats['total_chunks']} chunks)")
            return True

        # Create sample documents
        sample_docs = [
            Document(
                text="""HIPAA Compliance Requirements for Healthcare Advertising

All healthcare advertising campaigns must comply with HIPAA regulations regarding protected health information (PHI).

Key Requirements:
1. Never use PHI in ad creatives without explicit authorization
2. Implement proper data encryption for all patient data
3. Ensure business associate agreements are in place with all partners
4. Train all staff on HIPAA privacy practices annually
5. Maintain audit logs for all PHI access for 6 years

For pharmaceutical campaigns, additional FDA regulations apply regarding fair balance and risk disclosure.""",
                metadata={
                    "doc_type": "compliance",
                    "department": "legal",
                    "source": "HIPAA_Compliance_Guide.txt",
                    "date": "2024-01-15",
                    "author": "Legal Department",
                    "version": "2.1",
                }
            ),
            Document(
                text="""Ad Approval Process for Healthcare Clients

All healthcare and pharmaceutical advertisements must undergo a multi-stage review process:

Stage 1: Creative Review (24-48 hours)
- Check for prohibited claims
- Verify fair balance in pharmaceutical ads
- Ensure proper disclaimer placement
- Review imagery for appropriateness

Stage 2: Compliance Review (24 hours)
- Legal team validates regulatory compliance
- HIPAA verification for patient testimonials
- FDA guidelines check for pharma products

Stage 3: Client Approval (varies)
- Client receives compliance-approved creative
- Revisions submitted through ticket system
- Final sign-off required before launch

Expedited review available for urgent campaigns with 24-hour turnaround.""",
                metadata={
                    "doc_type": "sop",
                    "department": "operations",
                    "source": "Ad_Approval_Process.txt",
                    "date": "2024-02-01",
                    "author": "Operations Team",
                    "version": "3.0",
                }
            ),
            Document(
                text="""Healthcare Campaign Best Practices

Industry Benchmarks (2023-2024):
- Average CTR: 0.8-1.5% (lower than general due to regulations)
- Average CVR: 2-4% (higher due to targeted audiences)
- Average CPA: $150-500 (varies by treatment type)

Targeting Best Practices:
1. Focus on healthcare professionals (HCPs) for pharma
2. Use condition-based targeting rather than behavioral
3. Geographic targeting should align with prescribing data
4. Avoid age/gender targeting that could be discriminatory

Creative Guidelines:
- Include clear disclaimers for all pharmaceutical products
- Use professional, trustworthy imagery
- Avoid fear-based messaging
- Testimonials require proper authorization

Budget Allocation:
- 60% programmatic/managed placements
- 25% premium health publisher sites (WebMD, Healthline)
- 10% social media (LinkedIn only for HCP campaigns)
- 5% contingency for optimization""",
                metadata={
                    "doc_type": "best_practice",
                    "department": "marketing",
                    "source": "Healthcare_Campaign_Best_Practices.txt",
                    "date": "2024-03-10",
                    "author": "Marketing Strategy Team",
                    "version": "1.0",
                }
            ),
            Document(
                text="""Attribution Models for Healthcare Campaigns

Choosing the right attribution model is critical for healthcare campaigns due to long patient journeys.

Recommended Models by Campaign Type:

1. Lead Generation (Doctor Lookup, Appointment Booking):
   - Use: Last Click with 30-day lookback
   - Rationale: Direct response actions need clear attribution

2. Brand Awareness (Condition Education, Treatment Info):
   - Use: Time Decay with 90-day lookback
   - Rationale: Healthcare decisions have long consideration periods

3. Pharma Product Launches:
   - Use: Linear attribution with 60-day lookback
   - Rationale: Multiple touchpoints contribute to prescribing decisions

Important Considerations:
- Track both HCP (healthcare professional) and patient conversions separately
- Include offline conversions (call center, office visits) when possible
- Exclude retargeting from attribution to avoid double counting

Custom Model Configuration:
Contact the analytics team to set up custom models for specialty campaigns.""",
                metadata={
                    "doc_type": "guideline",
                    "department": "analytics",
                    "source": "Attribution_Model_Guidelines.txt",
                    "date": "2024-02-20",
                    "author": "Analytics Team",
                    "version": "1.5",
                }
            ),
            Document(
                text="""Frequently Asked Questions: Healthcare Advertising

Q: Can we use before/after patient photos in ads?
A: Only with explicit written authorization. Photos must not reveal protected health information.

Q: What disclaimers are required for pharmaceutical ads?
A: All pharma ads must include:
   - Brief summary of risks (TV/radio) or Full prescribing information (print)
   - Fair balance of benefits and risks
   - "Individual results may vary" for testimonial ads
   - Generic name pronunciation for TV ads

Q: Can we target by health condition on social media?
A: Condition-based targeting is permitted but must be carefully validated to avoid discrimination.
   Consult legal before implementation.

Q: How long does compliance review take?
A: Standard review: 48-72 hours. Expedited review: 24 hours. Rush review: 4 hours (requires VP approval).

Q: What should we do if we receive a cease and desist letter?
A: Immediately contact legal@datachat.com and pause all related campaigns. Do not respond directly.""",
                metadata={
                    "doc_type": "faq",
                    "department": "legal",
                    "source": "Healthcare_Advertising_FAQ.txt",
                    "date": "2024-01-25",
                    "author": "Legal Department",
                    "version": "4.0",
                }
            ),
        ]

        # Ingest documents
        result = retriever.add_documents(sample_docs)

        logger.info(f"✓ Ingested {result.num_documents} documents → {result.num_chunks} chunks")

        return True

    except Exception as e:
        logger.error(f"✗ Failed to set up documents: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_health_checks():
    """Run health checks on all services."""
    logger.info("\n" + "=" * 60)
    logger.info("HEALTH CHECKS")
    logger.info("=" * 60)

    # Check PostgreSQL
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "postgres"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "datachat_rag"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
        )
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM campaigns")
        count = cur.fetchone()[0]
        logger.info(f"✓ PostgreSQL: {count} campaigns in database")
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"✗ PostgreSQL health check failed: {e}")

    # Check ChromaDB
    try:
        import requests
        response = requests.get(
            f"http://{os.getenv('CHROMA_HOST', 'chromadb')}:{os.getenv('CHROMA_PORT', '8000')}/api/v1/heartbeat",
            timeout=5
        )
        if response.status_code == 200:
            logger.info("✓ ChromaDB: Healthy")
        else:
            logger.warning(f"⚠ ChromaDB: Status {response.status_code}")
    except Exception as e:
        logger.error(f"✗ ChromaDB health check failed: {e}")

    logger.info("=" * 60 + "\n")


def main():
    """Main initialization function."""
    logger.info("\n" + "=" * 60)
    logger.info("DataChat-RAG Docker Initialization")
    logger.info("=" * 60 + "\n")

    # Wait for databases
    postgres_ready = wait_for_postgres()
    chroma_ready = wait_for_chromadb()

    if not postgres_ready:
        logger.error("✗ PostgreSQL is not available. Exiting.")
        sys.exit(1)

    if not chroma_ready:
        logger.warning("⚠ ChromaDB is not available. Continuing anyway...")

    # Setup database
    if postgres_ready:
        if not setup_database():
            logger.error("✗ Database setup failed. Exiting.")
            sys.exit(1)

    # Setup documents (only if ChromaDB is ready)
    if chroma_ready:
        if not setup_documents():
            logger.warning("⚠ Document setup failed. Continuing anyway...")

    # Run health checks
    run_health_checks()

    logger.info("\n" + "=" * 60)
    logger.info("✓ INITIALIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info("\nDataChat-RAG is ready to use!")
    logger.info("\nAccess the application at:")
    logger.info(f"  - API: http://localhost:{os.getenv('API_PORT', '8000')}")
    logger.info(f"  - UI: http://localhost:{os.getenv('STREAMLIT_PORT', '8501')}")
    logger.info("\n")


if __name__ == "__main__":
    main()
