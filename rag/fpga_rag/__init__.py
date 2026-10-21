"""fpga_rag: local retrieval (RAG) over this repository and the documentation you add.

    from fpga_rag import Searcher, format_hits
    hits = Searcher().search("rerun a synthesis that is already done", k=6)
    print(format_hits(hits))
"""

from .config import Settings
from .search import Hit, Searcher, format_hits
from .store import RagError, Store

__all__ = ["Settings", "Searcher", "Hit", "format_hits", "Store", "RagError"]
__version__ = "0.1.0"
