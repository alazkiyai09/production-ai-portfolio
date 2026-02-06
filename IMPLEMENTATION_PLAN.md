# Implementation Plan: Generate REPO_REFERENCE.md & CONTEXT_MANIFEST.yaml

> **Target Repository:** `/home/ubuntu/AIEngineerProject/`
> **Guide:** `/home/ubuntu/Context_Extraction/REPO_SUMMARY_PROMPT.md` (Sections A-F)
> **YAML Template:** `/home/ubuntu/Context_Extraction/CONTEXT_MANIFEST.yaml`
> **Date:** 2026-02-06

---

## Output Files

| File | Path | Purpose |
|------|------|---------|
| `REPO_REFERENCE.md` | `/home/ubuntu/AIEngineerProject/REPO_REFERENCE.md` | Master LLM-consumable reference document |
| `CONTEXT_MANIFEST.yaml` | `/home/ubuntu/AIEngineerProject/CONTEXT_MANIFEST.yaml` | Machine-readable project registry |

---

## Phase 1: Read Repository-Level Context

**Goal:** Gather all information needed for the REPO_INDEX (Section A) and GLOBAL PATTERNS sections.

### Files to Read (11 files)

| # | File | Purpose |
|---|------|---------|
| 1 | `README.md` | Repo overview, project list, tech stack, metrics |
| 2 | `PROJECT_CATEGORIES.md` | Category definitions, shared utilities list |
| 3 | `TECHNICAL.md` | Environment setup, API keys, modes, testing guide |
| 4 | `CODE_REVIEW_SUMMARY.md` | Known issues for P-01, P-05, P-08 |
| 5 | `CRITICAL_ISSUES_FIX_STATUS.md` | Security fixes status across projects |
| 6 | `PROJECT_STRUCTURE.md` | Architecture documentation |
| 7 | `shared/security.py` | Shared security module (API key redaction) |
| 8 | `shared/rate_limit.py` | Shared rate limiting module (slowapi) |
| 9 | `pyproject.toml` (if exists) | Root-level dependency/tooling config |
| 10 | `docker-compose.yml` (if exists) | Container orchestration |
| 11 | `.env.example` (if exists) | Global env var template |

**Parallelization:** All 11 reads can be done in parallel.

### Extract From These Reads

- Repo name, URL, total projects, primary language, domain
- 4 categories with project counts and IDs
- Shared module inventory (`security.py`, `rate_limit.py`)
- Global coding patterns (formatter, linter, testing framework)
- Security posture summary

---

## Phase 2: Read Per-Project Files (3 Parallel Batches)

**Goal:** Gather all source code needed to generate 10 PROJECT CARDs (Section B).

### Per-Project File Read Order

For **each** project, read files in this priority:

1. `README.md` — Purpose, overview
2. `requirements.txt` — External dependencies with versions
3. `src/config.py` (or equivalent config file) — Configuration parameters
4. `.env.example` — Additional env-based config
5. `src/api/main.py` — FastAPI app setup, routes, middleware
6. All remaining `src/**/*.py` (non-`__init__`) — Components, logic, data flow
7. `tests/**/*.py` (non-`__init__`) — Testing info
8. `Dockerfile` / `docker-compose.yml` — Deployment info

### Batch 1 — RAG Systems (4 projects, 66 Python files)

#### P-01: Enterprise-RAG — 20 Python files
**Path:** `projects/rag/Enterprise-RAG/`

| Priority | File | Info to Extract |
|----------|------|-----------------|
| 1 | `README.md` | Purpose, architecture overview |
| 2 | `requirements.txt` | Dependencies table |
| 3 | `src/config.py` | Config parameters (Pydantic BaseSettings) |
| 4 | `src/api/main.py` | FastAPI app, routes, middleware setup |
| 5 | `src/api/routes/query.py` | Query endpoint logic |
| 6 | `src/api/routes/documents.py` | Document ingestion endpoints |
| 7 | `src/api/routes/evaluation.py` | RAGAS evaluation endpoints |
| 8 | `src/retrieval/hybrid_retriever.py` | Core retrieval logic (dense + sparse) |
| 9 | `src/retrieval/embedding_service.py` | Embedding generation |
| 10 | `src/retrieval/vector_store.py` | ChromaDB/Qdrant integration |
| 11 | `src/retrieval/sparse_retriever.py` | BM25 sparse retrieval |
| 12 | `src/retrieval/reranker.py` | Cross-encoder reranking |
| 13 | `src/generation/rag_chain.py` | LLM generation chain |
| 14 | `src/ingestion/document_processor.py` | Multi-format doc processing |
| 15 | `src/evaluation/rag_evaluator.py` | RAGAS metrics |
| 16 | `src/exceptions.py` | Custom exception classes |
| 17 | `src/logging_config.py` | Logging setup |
| 18 | `src/ui/app.py` | Streamlit UI |
| 19 | `tests/conftest.py` | Test fixtures |
| 20 | `tests/test_*.py` (3 files) | Test coverage info |

#### P-02: MultiModal-RAG — 30 Python files
**Path:** `projects/rag/MultiModal-RAG/`

| Priority | File | Info to Extract |
|----------|------|-----------------|
| 1 | `README.md` | Purpose |
| 2 | `requirements.txt` | Dependencies |
| 3 | `src/config.py` | Config params |
| 4 | `src/api/main.py` | FastAPI app |
| 5 | `src/api/multimodal_endpoints.py` | Multimodal-specific routes |
| 6 | `src/api/routes/query.py` | Query endpoint |
| 7 | `src/api/routes/documents.py` | Document endpoints |
| 8 | `src/api/routes/evaluation.py` | Eval endpoints |
| 9 | `src/multimodal/multimodal_rag.py` | Core multimodal RAG logic |
| 10 | `src/multimodal/image_processor.py` | CLIP image processing |
| 11 | `src/multimodal/multimodal_retriever.py` | Cross-modal retrieval |
| 12 | `src/multimodal/vision_llm.py` | Vision-language model |
| 13 | `src/multimodal/table_extractor.py` | Table extraction |
| 14 | `src/extraction/image_extractor.py` | Image extraction |
| 15 | `src/extraction/ocr_processor.py` | OCR processing |
| 16 | `src/extraction/table_extractor.py` | Table extraction |
| 17 | `src/retrieval/hybrid_retriever.py` | Hybrid retrieval |
| 18 | `src/retrieval/embedding_service.py` | Embedding service |
| 19 | `src/retrieval/vector_store.py` | Vector store |
| 20 | `src/retrieval/sparse_retriever.py` | Sparse retrieval |
| 21 | `src/retrieval/reranker.py` | Reranker |
| 22 | `src/generation/rag_chain.py` | Generation chain |
| 23 | `src/ingestion/document_processor.py` | Document processing |
| 24 | `src/evaluation/rag_evaluator.py` | Evaluation |
| 25 | `src/exceptions.py` | Exceptions |
| 26 | `src/logging_config.py` | Logging |
| 27 | `src/ui/app.py` | Streamlit UI |
| 28 | `src/ui/multimodal_app.py` | Multimodal UI |
| 29 | `tests/conftest.py` | Test fixtures |
| 30 | `tests/test_*.py` (3 files) | Tests |

#### P-03: DataChat-RAG — 8 Python files
**Path:** `projects/rag/DataChat-RAG/`

| Priority | File | Info to Extract |
|----------|------|-----------------|
| 1 | `README.md` | Purpose |
| 2 | `requirements.txt` | Dependencies |
| 3 | `src/api/main.py` | FastAPI app |
| 4 | `src/core/rag_chain.py` | NL-to-SQL chain logic |
| 5 | `src/retrievers/doc_retriever.py` | Document retrieval |
| 6 | `src/routers/query_router.py` | Query routing |
| 7 | `src/ui/chat_app.py` | Chat UI |
| 8 | `scripts/setup_database.py` | DB setup |
| 9 | `scripts/init_docker.py` | Docker init |
| 10 | `tests/test_datachat.py` | Tests |

#### P-04: fraud-docs-rag — 8 Python files
**Path:** `projects/rag/fraud-docs-rag/`

| Priority | File | Info to Extract |
|----------|------|-----------------|
| 1 | `README.md` | Purpose |
| 2 | `requirements.txt` | Dependencies |
| 3 | `src/fraud_docs_rag/config.py` | Config params |
| 4 | `src/fraud_docs_rag/api/main.py` | FastAPI app |
| 5 | `src/fraud_docs_rag/main.py` | App entrypoint |
| 6 | `src/fraud_docs_rag/retrieval/hybrid_retriever.py` | Retrieval logic |
| 7 | `src/fraud_docs_rag/generation/rag_chain.py` | Generation chain |
| 8 | `src/fraud_docs_rag/ingestion/document_processor.py` | Doc processing |
| 9 | `tests/conftest.py` | Test fixtures |
| 10 | `tests/integration/test_integration.py` | Integration tests |

### Batch 2 — Agent Systems (3 projects, 53 Python files)

#### P-05: CustomerSupport-Agent — 16 Python files
**Path:** `projects/agents/CustomerSupport-Agent/`

| Priority | File | Info to Extract |
|----------|------|-----------------|
| 1 | `README.md` | Purpose |
| 2 | `requirements.txt` | Dependencies |
| 3 | `src/config.py` | Config params |
| 4 | `src/api/main.py` | FastAPI + WebSocket setup |
| 5 | `src/conversation/support_agent.py` | LangGraph state machine |
| 6 | `src/knowledge/faq_store.py` | ChromaDB FAQ store |
| 7 | `src/memory/conversation_memory.py` | SQLite + summarization memory |
| 8 | `src/sentiment/analyzer.py` | TextBlob sentiment analysis |
| 9 | `src/tools/support_tools.py` | Agent tools |
| 10 | `examples/rest_client.py` | REST example |
| 11 | `examples/websocket_client.py` | WebSocket example |
| 12 | `tests/test_support_agent.py` | Agent tests |
| 13 | `tests/unit/test_api.py` | API tests |
| 14 | `tests/unit/test_conversation_memory.py` | Memory tests |
| 15 | `tests/unit/test_faq_store.py` | FAQ tests |
| 16 | `tests/unit/test_sentiment_analyzer.py` | Sentiment tests |
| 17 | `tests/unit/test_support_agent.py` | Agent unit tests |
| 18 | `tests/unit/test_support_tools.py` | Tools tests |

#### P-06: FraudTriage-Agent — 29 Python files
**Path:** `projects/agents/FraudTriage-Agent/`

| Priority | File | Info to Extract |
|----------|------|-----------------|
| 1 | `README.md` | Purpose |
| 2 | `requirements.txt` | Dependencies |
| 3 | `src/config/settings.py` | Config params |
| 4 | `src/api/main.py` | FastAPI app |
| 5 | `src/agents/graph.py` | LangGraph workflow graph |
| 6 | `src/agents/fraud_triage_agent.py` | Main agent class |
| 7 | `src/agents/state.py` | State definitions |
| 8 | `src/agents/nodes.py` | Graph nodes |
| 9 | `src/agents/triage_nodes.py` | Triage-specific nodes |
| 10 | `src/agents/workflow.py` | Workflow orchestration |
| 11 | `src/models/alert.py` | Alert model |
| 12 | `src/models/agent.py` | Agent model |
| 13 | `src/models/review.py` | Review model |
| 14 | `src/models/state.py` | State model |
| 15 | `src/tools/fraud_tools.py` | Fraud analysis tools |
| 16 | `src/tools/customer_tools.py` | Customer lookup tools |
| 17 | `src/tools/device_tools.py` | Device fingerprint tools |
| 18 | `src/tools/transaction_tools.py` | Transaction tools |
| 19 | `src/tools/utils.py` | Tool utilities |
| 20 | `src/utils/formatting.py` | Output formatting |
| 21 | `src/utils/logging.py` | Logging config |
| 22 | `src/utils/visualize.py` | Visualization |
| 23 | `scripts/test_api.py` | API test script |
| 24 | `scripts/test_fraud_tools.py` | Tool test script |
| 25 | `scripts/test_sample_alert.py` | Sample alert test |
| 26 | `tests/conftest.py` | Test fixtures |
| 27 | `tests/unit/test_agents.py` | Agent tests |
| 28 | `tests/unit/test_models.py` | Model tests |
| 29 | `tests/unit/test_tools.py` | Tool tests |
| 30 | `tests/integration/test_api_integration.py` | API integration |
| 31 | `tests/integration/test_workflow_integration.py` | Workflow integration |

#### P-07: AdInsights-Agent — 8 Python files
**Path:** `projects/agents/AdInsights-Agent/`

| Priority | File | Info to Extract |
|----------|------|-----------------|
| 1 | `README.md` | Purpose |
| 2 | `requirements.txt` | Dependencies |
| 3 | `src/api/main.py` | FastAPI app |
| 4 | `src/agents/insights_agent.py` | LangGraph agent |
| 5 | `src/tools/analysis_tools.py` | Analysis tools |
| 6 | `src/analytics/cohort.py` | Cohort analysis |
| 7 | `src/analytics/statistics.py` | Statistical analysis |
| 8 | `src/analytics/time_series.py` | Time series analysis |
| 9 | `src/visualization/report_generator.py` | Report generation |
| 10 | `tests/test_insights_agent.py` | Tests |

### Batch 3 — Evaluation + Infrastructure (3 projects, 56 Python files)

#### P-08: LLMOps-Eval — 28 Python files
**Path:** `projects/evaluation/LLMOps-Eval/`

| Priority | File | Info to Extract |
|----------|------|-----------------|
| 1 | `README.md` | Purpose |
| 2 | `requirements.txt` | Dependencies |
| 3 | `src/config.py` | Config params |
| 4 | `src/api/main.py` | FastAPI app |
| 5 | `src/evaluation/metrics.py` | 9 evaluation metrics |
| 6 | `src/models/llm_providers.py` | Multi-provider LLM support |
| 7 | `src/runners/eval_runner.py` | Evaluation runner |
| 8 | `src/datasets/dataset_manager.py` | Dataset management |
| 9 | `src/reporting/report_generator.py` | Report generation |
| 10 | `src/monitoring/metrics.py` | Monitoring metrics |
| 11 | `src/monitoring/prometheus_metrics.py` | Prometheus integration |
| 12 | `src/dashboard/app.py` | Streamlit dashboard |
| 13 | `src/prompt_optimizer/config.py` | Optimizer config |
| 14 | `src/prompt_optimizer/experiments/ab_testing.py` | A/B testing |
| 15 | `src/prompt_optimizer/experiments/ab_test.py` | A/B test model |
| 16 | `src/prompt_optimizer/experiments/framework.py` | Experiment framework |
| 17 | `src/prompt_optimizer/api/endpoints.py` | Optimizer API |
| 18 | `src/prompt_optimizer/api/routes.py` | Optimizer routes |
| 19 | `src/prompt_optimizer/dashboard/app.py` | Optimizer dashboard |
| 20 | `src/prompt_optimizer/history/history.py` | History tracking |
| 21 | `src/prompt_optimizer/selection/ranking.py` | Prompt ranking |
| 22 | `src/prompt_optimizer/selection/selector.py` | Prompt selection |
| 23 | `src/prompt_optimizer/statistics/analyzer.py` | Statistical analysis |
| 24 | `src/prompt_optimizer/statistics/corrections.py` | Statistical corrections |
| 25 | `src/prompt_optimizer/statistics/tests.py` | Statistical tests |
| 26 | `src/prompt_optimizer/templates/jinja_env.py` | Jinja templating |
| 27 | `src/prompt_optimizer/templates/template_manager.py` | Template management |
| 28 | `src/prompt_optimizer/variations/variation_generator.py` | Variation generation |
| 29 | `tests/test_llmops_eval.py` | Eval tests |
| 30 | `tests/test_prompt_optimizer.py` | Optimizer tests |

#### P-09: StreamProcess-Pipeline — 20 Python files
**Path:** `projects/infrastructure/StreamProcess-Pipeline/`

| Priority | File | Info to Extract |
|----------|------|-----------------|
| 1 | `README.md` | Purpose |
| 2 | `requirements.txt` | Dependencies |
| 3 | `src/ingestion/ingest_service.py` | Ingestion service |
| 4 | `src/ingestion/producer.py` | Event producer |
| 5 | `src/ingestion/consumer.py` | Event consumer |
| 6 | `src/ingestion/validators.py` | Input validation |
| 7 | `src/processing/worker.py` | Celery worker |
| 8 | `src/processing/batcher.py` | Batch processing |
| 9 | `src/processing/transformer.py` | Data transformation |
| 10 | `src/embedding/embed_service.py` | Embedding service |
| 11 | `src/embedding/generator.py` | Embedding generation |
| 12 | `src/embedding/cache.py` | Embedding cache |
| 13 | `src/storage/vector_store.py` | Vector storage |
| 14 | `src/storage/database.py` | Database connection |
| 15 | `src/storage/models.py` | ORM models |
| 16 | `src/storage/repositories.py` | Repository pattern |
| 17 | `src/api/metrics_endpoint.py` | Prometheus metrics endpoint |
| 18 | `src/monitoring/metrics.py` | Monitoring metrics |
| 19 | `tests/conftest.py` | Test fixtures |
| 20 | `tests/unit/test_ingestion.py` | Ingestion tests |
| 21 | `tests/integration/test_pipeline.py` | Pipeline integration tests |
| 22 | `docker-compose.yml` | Container orchestration |
| 23 | `k8s/` directory (if exists) | Kubernetes manifests |

#### P-10: aiguard — 8 Python files
**Path:** `projects/infrastructure/aiguard/`

| Priority | File | Info to Extract |
|----------|------|-----------------|
| 1 | `README.md` | Purpose |
| 2 | `requirements.txt` | Dependencies |
| 3 | `src/middleware/config.py` | Config params |
| 4 | `src/middleware/security_middleware.py` | Security middleware |
| 5 | `src/guardrails/prompt_injection/prompt_injection.py` | Prompt injection detection |
| 6 | `src/guardrails/jailbreak/jailbreak_detector.py` | Jailbreak detection |
| 7 | `src/guardrails/pii/pii_detector.py` | PII detection |
| 8 | `src/guardrails/output_filter/output_guard.py` | Output filtering |
| 9 | `src/demo/app.py` | Demo Streamlit app |
| 10 | `src/tests/adversarial_tests.py` | Adversarial test suite |

### Additional File Reads for Known Issues

| # | File | Purpose |
|---|------|---------|
| 1 | `CODE_REVIEW_SUMMARY.md` | Known issues for P-01, P-05, P-08 |
| 2 | `CRITICAL_ISSUES_FIX_STATUS.md` | Security fix tracking |
| 3 | Scan all `src/**/*.py` for `TODO`/`FIXME` comments | Issues for remaining projects |

**Command for TODO/FIXME scan:**
```bash
grep -rn "TODO\|FIXME\|HACK\|XXX\|WARN" projects/ --include="*.py"
```

---

## Phase 3: Generate Content (Sequential, Write-Heavy)

### Step 3.1: Write REPO_INDEX Block

Using data from Phase 1, write:

```
# REPO_INDEX
- repo_name: "AIEngineerProject"
- repo_url: "https://github.com/yourusername/production-ai-portfolio"
- total_projects: 10
- primary_language: "Python"
- domain: "Production AI / LLM Applications"
- last_analyzed: "2026-02-06"

## CATEGORIES
| Category ID | Category Name       | Project Count | Projects (IDs)          |
|-------------|---------------------|---------------|-------------------------|
| CAT-01      | RAG Systems         | 4             | P-01, P-02, P-03, P-04 |
| CAT-02      | LangGraph Agents    | 3             | P-05, P-06, P-07       |
| CAT-03      | LLMOps / Evaluation | 1             | P-08                    |
| CAT-04      | Infrastructure      | 2             | P-09, P-10              |

## PROJECT REGISTRY
[10-row table with: Project ID, Name, Category, Path, Status, Key Files, LOC]

## DEPENDENCY MAP
[10-row table with: Project ID, Depends On, Shared Modules, External Libs]

## ARCHITECTURE NOTES
[Bullet list of cross-cutting patterns]
```

### Step 3.2: Write 10 PROJECT CARDs

For each project (P-01 through P-10), produce a card with these **10 sections**:

1. **PURPOSE** — 2-3 sentences from README
2. **ARCHITECTURE** — ASCII tree of internal structure derived from file listing + code analysis
3. **KEY COMPONENTS** — Table: Component | File | Lines | Responsibility
4. **DATA FLOW** — ASCII pipeline: Input -> Processing -> Output
5. **CONFIGURATION & PARAMETERS** — Table: Parameter | Default | Location | Purpose (from config.py)
6. **EXTERNAL DEPENDENCIES** — Table: Library | Version | Used For (from requirements.txt)
7. **KNOWN ISSUES & IMPROVEMENT OPPORTUNITIES** — Checklist from CODE_REVIEW_SUMMARY.md + TODO/FIXME scan
8. **TESTING** — Test file paths, coverage info, run commands
9. **QUICK MODIFICATION GUIDE** — Table: Want to... | Modify | Notes
10. **CODE SNIPPETS** — Only non-obvious/critical logic (1-2 snippets per project max)

**Agent parallelization strategy:**
- Launch 3 subagents in parallel (one per batch)
- Each subagent reads all files for its batch and generates the PROJECT CARDs
- Each subagent writes its output to a temporary file in the scratchpad directory

### Step 3.3: Write GLOBAL PATTERNS & CONVENTIONS

Synthesize from all projects:

| Pattern | Detail | Evidence |
|---------|--------|----------|
| Config | Pydantic `BaseSettings` + `.env` + `lru_cache` singleton | `src/config.py` in P-01, P-02, P-04, P-05, P-08 |
| API | FastAPI with `/health`, `/docs`, versioned routes `/api/v1/` | `src/api/main.py` in all projects |
| Logging | `logging` module + `loguru` in some projects | `src/logging_config.py` in P-01, P-02 |
| Error Handling | Global exception handlers, custom exception classes | `src/exceptions.py` in P-01, P-02 |
| Code Style | Black (line-length=100), ruff, isort, mypy | `.pre-commit-config.yaml`, TECHNICAL.md |
| Security | `shared/security.py` (API key redaction), `shared/rate_limit.py` (slowapi) | `shared/` directory |
| Containerization | Docker multi-stage builds, docker-compose, K8s manifests | Dockerfile, docker-compose.yml, k8s/ |
| Testing | pytest + pytest-asyncio + pytest-cov | `tests/` in each project |

### Step 3.4: Generate CHANGELOG

```bash
cd /home/ubuntu/AIEngineerProject && git log --oneline --since="2025-12-01" --pretty=format:"%ad | %s" --date=short | head -30
```

---

## Phase 4: Assemble REPO_REFERENCE.md

Combine all sections into the final document following Section C template:

```markdown
# REPOSITORY REFERENCE: AIEngineerProject
> Auto-generated on 2026-02-06 | Projects: 10/10 | Version: [commit hash]

## QUICK NAVIGATION

### By Category
- **[CAT-01] RAG Systems** -> P-01, P-02, P-03, P-04
- **[CAT-02] LangGraph Agents** -> P-05, P-06, P-07
- **[CAT-03] LLMOps / Evaluation** -> P-08
- **[CAT-04] Infrastructure** -> P-09, P-10

### By Complexity
- High: P-02 (MultiModal-RAG), P-06 (FraudTriage-Agent), P-08 (LLMOps-Eval)
- Medium: P-01 (Enterprise-RAG), P-05 (CustomerSupport-Agent), P-09 (StreamProcess-Pipeline)
- Low: P-03 (DataChat-RAG), P-04 (fraud-docs-rag), P-07 (AdInsights-Agent), P-10 (aiguard)

### Cross-Project Dependencies
[dependency diagram]

---

## REPO_INDEX
[from Step 3.1]

---

## PROJECT CARDS
[P-01 through P-10, separated by ---]

---

## GLOBAL PATTERNS & CONVENTIONS
[from Step 3.3]

## CHANGELOG
[from Step 3.4]
```

---

## Phase 5: Generate CONTEXT_MANIFEST.yaml

Adapt the template from `/home/ubuntu/Context_Extraction/CONTEXT_MANIFEST.yaml`:

```yaml
repository:
  name: "AIEngineerProject"
  url: "https://github.com/yourusername/production-ai-portfolio"
  total_projects: 10
  primary_language: "Python"
  domain: "Production AI / LLM Applications"
  last_analyzed: "2026-02-06"

categories:
  CAT-01:
    name: "RAG Systems"
    description: "Retrieval-Augmented Generation systems"
    projects: [P-01, P-02, P-03, P-04]
  CAT-02:
    name: "LangGraph Agents"
    description: "LangGraph-based AI agents"
    projects: [P-05, P-06, P-07]
  CAT-03:
    name: "LLMOps / Evaluation"
    description: "LLM evaluation and testing frameworks"
    projects: [P-08]
  CAT-04:
    name: "Infrastructure"
    description: "Supporting infrastructure and utilities"
    projects: [P-09, P-10]

projects:
  P-01:
    name: "Enterprise-RAG"
    category: CAT-01
    path: "projects/rag/Enterprise-RAG/"
    status: complete
    score: 8.5
    key_files: [src/api/main.py, src/retrieval/hybrid_retriever.py, src/config.py]
    depends_on: []
    shared_libs: [shared/security.py, shared/rate_limit.py]
    external: [llama-index, chromadb, qdrant-client, sentence-transformers, fastapi]
    summary: "..."
    tags: [rag, hybrid-retrieval, reranking, fastapi, chromadb]
  # ... P-02 through P-10 ...

# DEPENDENCY GRAPH
# shared/security.py <- used by P-01, P-02, P-04, P-05, P-08
# shared/rate_limit.py <- used by P-01, P-05
# No cross-project code dependencies (all projects are independent)
```

---

## Phase 6: Verification Checklist

Run these checks after generation:

### 6.1 Completeness
- [ ] All 10 PROJECT CARDs present (P-01 through P-10)
- [ ] All 10 sections populated per card (Purpose through Code Snippets)
- [ ] REPO_INDEX contains all 4 tables (Categories, Registry, Dependency Map, Architecture Notes)
- [ ] QUICK NAVIGATION has all 3 subsections

### 6.2 Path Accuracy
```bash
# Verify every file path referenced in REPO_REFERENCE.md actually exists
grep -oP '`[^`]+\.py`' REPO_REFERENCE.md | tr -d '`' | while read f; do
  [ ! -f "$f" ] && echo "MISSING: $f"
done
```

### 6.3 Dependency Accuracy
```bash
# For each project, cross-check requirements.txt versions against EXTERNAL DEPENDENCIES tables
for proj_dir in projects/*/*; do
  [ -f "$proj_dir/requirements.txt" ] && echo "=== $(basename $proj_dir) ===" && cat "$proj_dir/requirements.txt"
done
```

### 6.4 Config Accuracy
```bash
# Verify listed config parameters match actual config.py fields
grep -rn "class.*Settings\|Field(" projects/*/src/config.py projects/*/src/config/*.py 2>/dev/null
```

### 6.5 YAML Validity
```bash
python3 -c "import yaml; yaml.safe_load(open('CONTEXT_MANIFEST.yaml')); print('YAML OK')"
```

### 6.6 Navigation Links
```bash
# Check all markdown anchor links resolve
grep -oP '\[.*?\]\(#.*?\)' REPO_REFERENCE.md | grep -oP '#[^)]+' | while read anchor; do
  heading=$(echo "$anchor" | sed 's/#//;s/-/ /g')
  grep -qi "$heading" REPO_REFERENCE.md || echo "BROKEN LINK: $anchor"
done
```

---

## Execution Summary

| Phase | Action | Files Read | Files Written | Parallelizable |
|-------|--------|-----------|---------------|----------------|
| 1 | Read repo-level context | 11 | 0 | Yes (all 11 in parallel) |
| 2 | Read per-project files | ~175 | 0 | Yes (3 batches in parallel) |
| 3.1 | Write REPO_INDEX | 0 | 1 (temp) | No |
| 3.2 | Write 10 PROJECT CARDs | 0 | 3 (temp, 1 per batch) | Yes (3 subagents) |
| 3.3 | Write GLOBAL PATTERNS | 0 | 1 (temp) | No |
| 3.4 | Generate CHANGELOG | 0 | 0 | Yes (with 3.3) |
| 4 | Assemble REPO_REFERENCE.md | 5 (temp) | 1 | No |
| 5 | Generate CONTEXT_MANIFEST.yaml | 0 | 1 | Yes (with Phase 4) |
| 6 | Verification | 2 | 0 | Yes (all checks in parallel) |

**Total files to read:** ~186 (11 repo-level + ~175 project-level)
**Total files to write:** 2 final output files
**Estimated final REPO_REFERENCE.md size:** ~12,000-15,000 tokens
