"""Compatibility helpers for legacy imports.

Historically the RAG pipeline lived at :mod:`agentic.pipeline`.  The
implementation now resides under :mod:`agentic.rag.pipeline`, but some user
code – including older Streamlit builds – still imports from the original
module path.  This file simply re-exports the public classes so those imports
continue to work without modification.
"""

from .rag.pipeline import HealthcareRAGPipeline, RetrievedContext

__all__ = ["HealthcareRAGPipeline", "RetrievedContext"]
