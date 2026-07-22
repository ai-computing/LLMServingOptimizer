"""DSE server routes mounted under /api/dse and /dse on the main webapp."""
from .routes import dse_router

__all__ = ["dse_router"]
